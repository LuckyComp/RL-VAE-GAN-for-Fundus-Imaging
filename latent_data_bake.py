import torch
import os
from tqdm import tqdm
from models.varautoencoder import Encoder
from dataloader import get_dataloaders

def bake_latent_space(encoder_weights, dataset_path, output_file):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Baking Latent Coordinates on {DEVICE}...")

    # Load Encoder
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(encoder_weights, map_location=DEVICE, weights_only=True))
    encoder.eval()

    # Load Data
    train_loader, _, _ = get_dataloaders(dataset_path, batch_size=64, num_workers=8)

    all_mu, all_log_var, all_labels = [], [], []

    # Process entire dataset
    with torch.no_grad():
        for imgs, labels in tqdm(train_loader, desc="Encoding Dataset"):
            imgs = imgs.to(DEVICE)
            mu, log_var = encoder(imgs)
            
            # Move to CPU immediately to save VRAM and store in lists
            all_mu.append(mu.cpu())
            all_log_var.append(log_var.cpu())
            all_labels.append(labels.cpu())

    # Concatenate into massive, single tensors
    baked_data = {
        'mu': torch.cat(all_mu, dim=0),           # Shape: [N, 256, 1, 1]
        'log_var': torch.cat(all_log_var, dim=0), # Shape: [N, 256, 1, 1]
        'labels': torch.cat(all_labels, dim=0)    # Shape: [N]
    }

    torch.save(baked_data, output_file)
    print(f"[*] Success! Baked {len(baked_data['labels'])} latent points to {output_file}")

if __name__ == "__main__":
    bake_latent_space(
        encoder_weights="./models/saved/encoder_pretrained.pth",
        dataset_path="./Raw_datasets/A. RFMiD_All_Classes_Dataset/1. Original Images/",
        output_file="baked_latents.pt"
    )
