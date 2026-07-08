import torch
from torch import nn
import torchvision.models as models
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# ==========================================
# 1. GENERATOR COMPONENTS
# ==========================================


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return out + residual


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = (
            self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)

        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return self.gamma * out + x


class UpsampleBlock(nn.Module):
    """
    Replaces PixelShuffle. Uses Bilinear interpolation to physically expand the
    image grid safely, preventing the 'checkerboard' artifacts.
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.upsample(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 128,
        num_classes: int = 5,
        embed_size: int = 50,
    ):
        super().__init__()
        self.number_of_residual_blocks = 12
        self.number_of_upsample_blocks = 6
        self.base_channels = base_channels

        self.label_embedding = nn.Embedding(num_classes, embed_size)

        # Input is Latent Space + Label Embedding
        self.fc = nn.Linear(latent_dim + embed_size, 512 * 4 * 4)

        self.initial_convolution = nn.Sequential(
            nn.Conv2d(512, self.base_channels, 3, padding=1), nn.PReLU()
        )

        self.resBlocks = nn.Sequential(
            *[
                ResidualBlock(self.base_channels)
                for _ in range(self.number_of_residual_blocks)
            ]
        )

        self.postResConv = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.InstanceNorm2d(self.base_channels),
        )

        # Using the new artifact-free UpsampleBlock
        self.upsample_low = nn.Sequential(
            *[UpsampleBlock(self.base_channels, 2) for _ in range(3)]
        )

        # 2. Inject Self-Attention at the 32x32 bottleneck (Very low VRAM cost)
        self.attn = SelfAttention(in_channels=self.base_channels)

        # 3. Upsample from 32x32 to 256x256 (Remaining 3 blocks: 32->64, 64->128, 128->256)
        self.upsample_high = nn.Sequential(
            *[UpsampleBlock(self.base_channels, 2) for _ in range(3)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Tanh(),  # Outputs [-1, 1]
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)  # Stitch noise and condition
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)

        # Ensure the starting tensor requires gradients
        if not x.requires_grad:
            x.requires_grad_()

        x = self.initial_convolution(x)
        residual = x

        # Checkpoint the 12 Residual Blocks
        x = checkpoint(self.resBlocks, x, use_reentrant=False)

        x = self.postResConv(x)
        x = x + residual

        # Checkpoint the low-resolution upsampling
        x = checkpoint(self.upsample_low, x, use_reentrant=False)

        x = self.attn(x)  # Learns global vessel routing

        # Checkpoint the high-resolution upsampling
        x = checkpoint(self.upsample_high, x, use_reentrant=False)

        return self.output_conv(x)


# ==========================================
# 2. CRITIC COMPONENT (Formerly Discriminator)
# ==========================================


class Critic(nn.Module):
    def __init__(
        self, base_channels: int = 128, num_classes: int = 5, image_size: int = 256
    ):
        super().__init__()
        self.image_size = image_size

        # Embed label to map to a full 256x256 image channel
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        # Notice: WGAN-GP authors recommend NO BatchNorm in the critic.
        # Spectral Norm is okay, but often redundant with GP. We'll use standard convolutions.
        def conv_block(in_channels: int, out_channels: int, stride: int):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            # Input channels: 4 (3 RGB + 1 Label Mask)
            nn.Conv2d(4, base_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Full depth for the ADA 6000
            conv_block(base_channels, base_channels, 2),
            conv_block(base_channels, base_channels * 2, 1),
            conv_block(base_channels * 2, base_channels * 2, 2),
            conv_block(base_channels * 2, base_channels * 4, 1),
            conv_block(base_channels * 4, base_channels * 4, 2),
            conv_block(base_channels * 4, base_channels * 8, 1),
            conv_block(base_channels * 8, base_channels * 8, 2),
            conv_block(base_channels * 8, base_channels * 8, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            # CRITICAL WGAN CHANGE: No Sigmoid activation here.
            # It must output a raw, unbounded score.
            nn.Linear(1024, 1),
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels).view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([x, c], dim=1)  # 4-Channel input

        # MANDATORY: The input tensor must require gradients for checkpointing to trigger
        if not x.requires_grad:
            x.requires_grad_()

        # Wrap the heavy sequential block
        x = checkpoint(self.features, x, use_reentrant=False)

        return self.classifier(x)


# 3. WGAN-GP LOSS FUNCTIONS
# ==========================================


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    Calculates the gradient penalty to enforce the Lipschitz constraint.
    This prevents the Critic gradients from exploding or vanishing.
    """
    # Random weight term for interpolation between real and fake
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )

    # Calculate critic output for interpolated samples
    d_interpolates = critic(interpolates, labels)

    # Fake tensor to calculate gradients
    fake = torch.ones(real_samples.shape[0], 1, device=device)

    # Get gradients of critic output with respect to interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten the gradients to 2D
    gradients = gradients.flatten(start_dim=1)

    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ==========================================
# 4. PERCEPTUAL LOSS (VGG16)
# ==========================================


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Extract features from the relu2_2 layer
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()
        # Normalization values required by VGG
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, real_img, fake_img):
        # Denormalize GAN output [-1, 1] -> [0, 1]
        real_img = (real_img + 1) / 2
        fake_img = (fake_img + 1) / 2

        # Normalize for VGG
        real_img = (real_img - self.mean) / self.std
        fake_img = (fake_img - self.mean) / self.std

        real_features = self.feature_extractor(real_img)
        fake_features = self.feature_extractor(fake_img)

        return self.criterion(fake_features, real_features)
