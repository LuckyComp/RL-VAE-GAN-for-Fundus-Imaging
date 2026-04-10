import torch
from torch import nn
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, channels:int = 64):
        super().__init__()
        self.block = nn.Sequential(
            # Layer 1: Depthwise + Pointwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels), 
            nn.Conv2d(channels, channels, 1), 
            nn.InstanceNorm2d(channels),
            nn.PReLU(),
            
            # Layer 2: Depthwise + Pointwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class PixelShuffleLayer(nn.Module):
    def __init__(self, in_channels:int = 64, scale_factor:int = 2):
        super().__init__()
        mid_channels = in_channels * (scale_factor**2)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        
        self.expansion = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.InstanceNorm2d(in_channels) # Essential for BS=4 stability
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.expansion(x)
        x = self.shuffle(x)
        x = self.norm(x)
        return self.prelu(x)

def warm_start_generator(generator, vae_path, device):
    print(f"Attempting warm start from: {vae_path}")
    vae_state = torch.load(vae_path, map_location=device, weights_only=True)
    gen_state = generator.state_dict()
    
    mapped_weights = {}
    
    if "decoder.fc.weight" in vae_state:
        mapped_weights["fc.weight"] = vae_state["decoder.fc.weight"]
        mapped_weights["fc.bias"]   = vae_state["decoder.fc.bias"]
    else:
        print("Could not find decoder.fc in checkpoint.")

    generator.load_state_dict(mapped_weights, strict=False)
    
    print(f"Warm Start Complete. Generator is now 'seeded' with VAE structure.")
    return generator

class Generator(nn.Module):
    def __init__(self, latent_dim:int = 256, base_channels:int = 64, num_classes:int = 5, embed_size:int = 50):
        super().__init__()
        self.number_of_residual_blocks = 12
        self.number_of_pixel_shuffles = 6
        self.base_channels = base_channels

        # NEW: Embedding layer for the 5 clusters
        self.label_embedding = nn.Embedding(num_classes, embed_size)

        # CHANGED: The input size is now latent_dim + embed_size (256 + 50 = 306)
        self.fc = nn.Linear(latent_dim + embed_size, 512*4*4)  

        self.initial_convolution = nn.Sequential(
            nn.Conv2d(512, self.base_channels, 3, padding=1), 
            nn.PReLU()
        )

        self.resBlocks = nn.Sequential(
            *[ResidualBlock(self.base_channels) for _ in range(self.number_of_residual_blocks)] 
        )

        self.postResConv = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1), 
            nn.InstanceNorm2d(self.base_channels)
        )

        self.upsample = nn.Sequential(
            *[PixelShuffleLayer(self.base_channels, 2) for _ in range(self.number_of_pixel_shuffles)] 
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(self.base_channels, 3, 3, padding=1), 
            nn.Tanh() 
        )

    def forward(self, z, labels):
        # 1. Embed the integer label into a vector of size 50
        c = self.label_embedding(labels)
        
        # 2. Concatenate noise and label vector along the feature dimension
        x = torch.cat([z, c], dim=1)
        
        # 3. Standard forward pass
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.initial_convolution(x)
        residual = x
        x = self.resBlocks(x)
        x = self.postResConv(x)
        x = x + residual 
        x = self.upsample(x)

        return self.output_conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, base_channels:int=64, num_classes:int=5, image_size:int=256):
        super().__init__()
        self.image_size = image_size
        
        # NEW: Embed the label to match the total pixels of one image channel
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        def conv_block(in_channels:int, out_channels:int, stride:int):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            # CHANGED: Input channels is now 4 (3 RGB + 1 Label Mask)
            nn.Conv2d(4, base_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # UN-NERFED: Full depth for the ADA 6000
            conv_block(base_channels, base_channels, 2),
            conv_block(base_channels, base_channels*2, 1),
            conv_block(base_channels*2, base_channels*2, 2),
            conv_block(base_channels*2, base_channels*4, 1),
            conv_block(base_channels*4, base_channels*4, 2),
            conv_block(base_channels*4, base_channels*8, 1),
            conv_block(base_channels*8, base_channels*8, 2),
            conv_block(base_channels*8, base_channels*8, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    # CHANGED: Forward pass now requires the labels
    def forward(self, x, labels):
        # 1. Embed label and reshape it to match the (Batch, 1, H, W) of the image
        c = self.label_embedding(labels).view(-1, 1, self.image_size, self.image_size)
        
        # 2. Concatenate along the channel dimension (dim=1)
        x = torch.cat([x, c], dim=1)
        
        # 3. Pass 4-channel input to the features block
        x = self.features(x)
        return self.classifier(x)
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
                *list(vgg.features)[:18]
        ).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, real, synthetic):
            # SANITY CHECK: Ensure they are the same spatial size
            if real.shape[-2:] != synthetic.shape[-2:]:
                # If they differ, the Generator isn't upsampling correctly
                raise ValueError("Error: Conflicting vector sizes") 
            real_features = self.feature_extractor(real)
            synth_features = self.feature_extractor(synthetic)
        
            # This will now throw an error if they STILL don't match, preventing 'fake' success
            return nn.functional.mse_loss(real_features, synth_features)

def generator_loss(disc_preds, synthetic_data, real_data, vgg_loss_fn, lambda_percept=0.006):
    adversarial_loss = nn.functional.binary_cross_entropy_with_logits(disc_preds, torch.ones_like(disc_preds))
    percept_loss = vgg_loss_fn(real_data, synthetic_data)
    return adversarial_loss + lambda_percept*percept_loss, adversarial_loss, percept_loss

def discriminator_loss(real_preds, synth_preds):
    real_loss = nn.functional.binary_cross_entropy_with_logits(real_preds, torch.full_like(real_preds, 0.9))
    synth_loss = nn.functional.binary_cross_entropy_with_logits(synth_preds, torch.zeros_like(synth_preds))
    return (real_loss + synth_loss)/2
