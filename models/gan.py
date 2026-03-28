import torch
from torch import nn
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, channels:int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class PixelShuffleLayer(nn.Module):
    def __init__(self, in_channels:int = 64, scale_factor:int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*(scale_factor**2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim:int = 256, base_channels:int = 64):
        super().__init__()
        self.number_of_residual_blocks = 12
        self.number_of_pixel_shuffles = 6
        self.base_channels = base_channels

        self.fc = nn.Linear(latent_dim, 512*4*4)

        self.initial_convolution = nn.Sequential(
            nn.Conv2d(512, self.base_channels, 9, padding=4),
            nn.PReLU()
        )

        self.resBlocks = nn.Sequential(
            *[ResidualBlock(self.base_channels) for _ in range(self.number_of_residual_blocks)]
        )

        self.postResConv = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.BatchNorm2d(self.base_channels)
        )

        self.upsample = nn.Sequential(
            *[PixelShuffleLayer(self.base_channels, 2) for _ in range(self.number_of_pixel_shuffles)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(self.base_channels, 3, 9, padding=4),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.initial_convolution(x)

        residual = x
        x = self.resBlocks(x)
        x = self.postResConv(x)
        x = x + residual

        x = self.upsample(x)

        return self.output_conv(x)

class Discriminator(nn.Module):
    def __init__(self, base_channels:int=64):
        super().__init__()

        def conv_block(in_channels:int, out_channels:int, stride:int):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(base_channels, base_channels, 2),
            conv_block(base_channels, base_channels*2, 1),
            conv_block(base_channels*2, base_channels*2, 2),
            conv_block(base_channels*2, base_channels*4, 1),
            conv_block(base_channels*4, base_channels*4, 2),
            conv_block(base_channels*4, base_channels*8, 1),
            conv_block(base_channels*8, base_channels*8, 2),
            conv_block(base_channels*8, base_channels*8, 2),
            conv_block(base_channels*8, base_channels*8, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
                *list(vgg.features)[:36]
        ).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, real, synthetic):
        real_features = self.feature_extractor(real)
        synth_features = self.feature_extractor(synthetic)
        return nn.functional.mse_loss(real_features, synth_features)

def generator_loss(disc_preds, synthetic_data, real_data, vgg_loss_fn, lambda_percept=0.006):
    adversarial_loss = nn.functional.binary_cross_entropy(disc_preds, torch.ones_like(disc_preds))
    percept_loss = vgg_loss_fn(real_data, synthetic_data)
    return adversarial_loss + lambda_percept*percept_loss, adversarial_loss, percept_loss

def discriminator_loss(real_preds, synth_preds):
    real_loss = nn.functional.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
    synth_loss = nn.functional.binary_cross_entropy(synth_preds, torch.ones_like(synth_preds))
    return (real_loss + synth_loss)/2
