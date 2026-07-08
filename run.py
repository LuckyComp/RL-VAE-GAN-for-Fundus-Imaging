import torch
import os
from torchvision.utils import save_image
from models.gan import Generator
from models.varautoencoder import Encoder, reparameterize
from dataloader import get_dataloaders
from dotenv import load_dotenv


def generate_random_images(
    generator, device, output_dir, num_images=16, latent_dim=256, num_classes=5
):
    """Generates images from pure random noise using cGAN labels."""
    print(f"Generating {num_images} images from random noise...")
    with torch.no_grad():
        # 1. Sample random noise from a standard normal distribution
        z = torch.randn(num_images, latent_dim).to(device)
        z = torch.clamp(z, min=-1.5, max=1.5)

        # 2. Generate random labels (0 to num_classes-1) for the Conditional GAN
        random_labels = torch.randint(0, num_classes, (num_images,)).to(device)

        # 3. Generate and denormalize [-1, 1] -> [0, 1]
        fake_images = generator(z, random_labels)
        fake_images = (fake_images + 1) / 2

        save_path = os.path.join(output_dir, "inference_random_noise.png")
        save_image(fake_images, save_path, nrow=4)
        print(f"Saved to: {save_path}")


def generate_from_encoder(generator, encoder, dataloader, device, output_dir):
    """Generates images by passing a real batch and its labels through the VAE encoder."""
    print("Generating images from encoded real data...")
    with torch.no_grad():
        # CHANGED: Unpack BOTH the real images and their cluster labels
        real_batch, labels = next(iter(dataloader))
        real_batch = real_batch.to(device)
        labels = labels.to(device)

        # Pass through the encoder
        mu, log_var = encoder(real_batch)
        z = reparameterize(mu, log_var)

        # CHANGED: Pass the labels into the Generator alongside z
        fake_images = generator(z, labels)
        fake_images = (fake_images + 1) / 2

        # Also denormalize the real images for side-by-side comparison
        real_images = (real_batch + 1) / 2

        # Save both grids
        save_image(
            real_images, os.path.join(output_dir, "inference_real_inputs.png"), nrow=4
        )
        save_image(
            fake_images, os.path.join(output_dir, "inference_encoded_fakes.png"), nrow=4
        )
        print(f"Saved real and fake comparison grids to {output_dir}")


if __name__ == "__main__":
    load_dotenv()

    # 1. Configuration
    # Automatically use the ADA 6000 if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Update this to whichever epoch you want to test
    GENERATOR_WEIGHTS = "./models/saved/checkpoints/generator_ema_epoch540.pth"
    ENCODER_WEIGHTS = os.getenv("ENCODER_PATH", "./models/saved/encoder_pretrained.pth")
    DATASET_PATH = os.getenv("DATASET_PATH")
    OUTPUT_DIR = "./inference_outputs"
    NUM_CLASSES = 5  # Matches your K-Means clusters

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running inference on {DEVICE}...")

    # 2. Load the Generator
    # CHANGED: Initialize with num_classes to match your cGAN architecture
    generator = Generator(latent_dim=256, num_classes=NUM_CLASSES).to(DEVICE)
    generator.load_state_dict(
        torch.load(GENERATOR_WEIGHTS, map_location=DEVICE, weights_only=True)
    )
    generator.eval()
    print("Generator weights loaded successfully.")

    # 3. Generate completely random retinas
    generate_random_images(generator, DEVICE, OUTPUT_DIR, num_classes=NUM_CLASSES)

    # 4. Load the Encoder & DataLoader for guided generation
    try:
        encoder = Encoder().to(DEVICE)
        encoder.load_state_dict(
            torch.load(ENCODER_WEIGHTS, map_location=DEVICE, weights_only=True)
        )
        encoder.eval()

        # Grab the validation loader
        _, val_loader, _ = get_dataloaders(DATASET_PATH, batch_size=16, num_workers=4)

        # Generate retinas based on real inputs
        generate_from_encoder(generator, encoder, val_loader, DEVICE, OUTPUT_DIR)

    except Exception as e:
        print(f"Could not run encoder-guided generation. Error: {e}")
