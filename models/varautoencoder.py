import torch
from torch import nn
from pytorch_msssim import ssim

class ResidualBlock(nn.Module): #Learns the difference between the transformation and original image
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), #Transform
            nn.BatchNorm2d(channels), #Normalize
            nn.LeakyReLU(0.2, True), #Activation
            nn.Conv2d(channels, channels, 3, padding=1), #Transformations on the activated outputs
            nn.BatchNorm2d(channels) #Normalize
        )
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.activation(x + self.block(x)) #Activation with the skip where x is the input and self.block(x) is the residual

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, padding=1), #Block 1
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels),

            nn.Conv2d(base_channels, base_channels*2, 4, 2, padding=1), #Block 2
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels*2),

            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, padding=1), #Block 3
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels*4),

            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, padding=1), #Block 4
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels*8)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4)) #Ensures regardless of input dimensions the final output is 4 x 4 using averaging
        flat_dim = base_channels * 8 * 4 * 4 #What the size of the flattened dimensions should be after transformations
        self.fc_mu = nn.Linear(flat_dim, latent_dim) #Calculates mean of guassian distribution
        self.fc_log_var = nn.Linear(flat_dim, latent_dim) #Calculates +ve variance values using log_var

    def forward(self):
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) #Resizes from 3D vector to 1D vector of size 8192
        return self.fc_mu(x), self.fc_log_var(x)
    
def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var) #Getting the sqrt(var) from log_var
    eps = torch.randn_like(std) #Random number from guassian distribution
    return mu + std*eps #Introduces subtle shift from fixed points in the latent distribution

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, padding=1), #Upscale Block 1
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels*4),

            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, padding=1), #Upscale Block 2
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels*4), 

            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, padding=1), #Upscale Block 3
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(base_channels),

            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, padding=1), #Upscale to out_channels
            nn.Tanh() #Get output in range of [-1, 1]
        )
        flat_dim = base_channels * 8 * 4 * 4
        self.fc = nn.Linear(latent_dim, flat_dim) #Fully connected Linear Layer

    def forward(self, z):
        x = self.fc(z) #Get linear output
        x = x.view(x.size(0), self.base_channels*8, 4, 4) #Resize the output x
        return self.decode(x) #Return the decoded upsampled output
    
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_channels=64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = Decoder(in_channels, latent_dim, base_channels)

    def forward(self, x): #used at pre-train time
        mu, var_log = self.encoder(x)
        z = reparameterize(mu, var_log) #Z value in latent space
        recon = self.decoder(z) #Reconstructed output
        return recon, mu, var_log
    
    def infer(self, x): #Used at inference time
        mu, log_var = self.encoder(x) 
        return reparameterize(mu, log_var), mu, log_var #Only returns Z value and mu, var_log.

def elbo_loss(recon, target, mu, log_var, beta=1.0):
    #Here recon_loss is a mix of mse and ssim loss to check not just pixel to pixel difference but also semantic differences
    recon_loss = 1.0*nn.functional.mse_loss(recon, target, reduction='sum')/target.size(0) +  0.5*(1 - ssim(recon, target)) #Calculate Reconstruction Loss
    kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())/target.size(0) #Calculate KL Divergence Loss

    return recon_loss + beta*kl_loss, recon_loss, kl_loss #Return ELBO Loss

def kl_annealing(epoch, warmup_epochs=50, max_beta=1.0):
    return min(max_beta, (epoch/warmup_epochs)*max_beta) #Calculates the rate of annealing