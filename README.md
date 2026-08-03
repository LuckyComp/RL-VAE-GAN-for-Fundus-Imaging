# RL-VAE-GAN-for-Fundus-Imaging

## Project Overview
RL-VAE-GAN-for-Fundus-Imaging is a retinal fundus image synthesis pipeline that combines a variational autoencoder (VAE) with a conditional GAN. The project is designed to learn a structured latent space from fundus images, cluster that latent space, and use it to generate disease-aware synthetic retinal images.

## Key Components
- `vae_trainer.py`: Pre-trains the VAE and saves encoder weights.
- `gan_trainer.py`: Trains a class-conditional WGAN-GP generator using frozen encoder embeddings.
- `run.py`: Generates synthetic fundus images using the trained encoder and generator.
- `dataloader.py`: Builds PyTorch dataloaders from unified image / CSV dataset splits.
- `models/varautoencoder.py`: VAE encoder/decoder, reparameterization, ELBO loss, KL annealing.
- `models/gan.py`: Conditional generator, critic, WGAN-GP loss, perceptual loss, and self-attention.

## Installation
1. Create and activate a Python virtual environment:
   - Linux/macOS: `python -m venv venv && source ./venv/bin/activate`
   - Windows: `python -m venv venv && .\\venv\\Scripts\\activate`

2. Install dependencies:
   - `pip install -r requirements.txt`

3. If using CUDA 11.8:
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

4. Add a `.env` file in the repository root containing:
   ```bash
   DATASET_PATH=/path/to/Raw_datasets/Combined_DR_Dataset
   ENCODER_PATH=/path/to/models/saved/encoder_pretrained.pth
   ```

## Dataset Structure
The generator and trainer use the unified dataset layout in `DATASET_PATH`:
- `normalized_images_unified/`
- `train_labels.csv`
- `val_labels.csv`
- `test_labels.csv` (optional)

The data loader caches preprocessed images in RAM and applies augmentations during training.

## VAE Pre-training
The VAE encoder uses:
- 4 convolutional blocks with residual connections
- adaptive average pooling to fixed 4×4 feature maps
- 256-dimensional latent space
- decoder with bilinear upsampling and transpose convolutions

Loss components:
- Reconstruction: `1000 * MSE + 0.5 * (1 - SSIM)`
- KL divergence: `-0.5 * mean(1 + log_var - mu^2 - exp(log_var))`
- Total ELBO: `recon_loss + beta * kl_loss`

KL annealing is applied during training via `vae_trainer.py`.

### Run VAE pre-training
```bash
python vae_trainer.py
```

Checkpoint outputs:
- `models/saved/vae_checkpoint_epoch{N}.pth`
- `models/saved/encoder_pretrained.pth`
- `models/saved/vae_full_checkpoint.pth`

## GAN Training
GAN training uses the frozen encoder to generate latent vectors from real images and trains a class-conditional WGAN-GP.

GAN losses include:
- adversarial WGAN-GP loss
- perceptual loss (VGG feature matching)
- L1 reconstruction loss
- SSIM loss
- critic feature-matching loss

The generator conditions on 5 cluster labels and trains with EMA weight tracking.

### Run GAN training
```bash
python gan_trainer.py
```

Checkpoint outputs:
- `models/saved/checkpoints/training_state_epoch{N}.pth`
- `models/saved/checkpoints/generator_epoch{N}.pth`
- `models/saved/checkpoints/generator_ema_epoch{N}.pth`
- `models/saved/checkpoints/discriminator_final.pth`

## Inference
`run.py` generates synthetic fundus images with two modes:
1. Randomized empirical latent sampling from the encoder output
2. Encoder-guided generation from real images

Generated outputs are saved in `inference_outputs/`:
- `inference_random_noise.png`
- `inference_encoded_fakes.png`
- `inference_real_inputs.png`

### Run inference
```bash
python run.py
```

## Recommended Workflow
1. Prepare the dataset and set `DATASET_PATH` in `.env`
2. Train the VAE: `python vae_trainer.py`
3. Train the GAN: `python gan_trainer.py`
4. Generate synthetic outputs: `python run.py`

## License
No license is included. Add a license file before sharing or publishing this repository.
