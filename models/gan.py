import torch
from torch import nn
from torchvision.models import models

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
            nn.Conv2d(in_channels, in_channels*(scale**2), 3, padding=1),
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
            *[PixelShuffleBlock(self.base_channels, 2) for _ in range(self.number_of_pixel_shuffles)]
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


