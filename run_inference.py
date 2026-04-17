import os
import torch
import argparse
from tqdm import tqdm
from torch.amp import autocast
from torchvision.utils import save_image
from models.gan import Generator

def generate_from_baked(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_classes):
        os.makedirs(os.path.join(args.output_dir, f"class_{i}"), exist_ok=True)

    # 1. Load the Generator
    generator = Generator(base_channels=128).to(DEVICE)
    checkpoint = torch.load(args.gen_weights, map_location=DEVICE, weights_only=True)
    generator.load_state_dict(checkpoint.get('generator_state_dict', checkpoint), strict=False)
    generator.eval()

    # 2. Load the Baked Latents (No DataLoader needed!)
    baked = torch.load(args.baked_file, weights_only=True)
    baked_mu = baked['mu'].to(DEVICE)
    baked_log_var = baked['log_var'].to(DEVICE)
    baked_labels = baked['labels'].to(DEVICE)
    
    total_baked_points = len(baked_labels)
    print(f"[*] Loaded {total_baked_points} baked coordinates.")

    images_generated = 0
    pbar = tqdm(total=args.total_images, desc=f"Generating (Temp: {args.temperature})")

    with torch.no_grad():
        while images_generated < args.total_images:
            batch_size = min(args.batch_size, args.total_images - images_generated)

            # --- Randomly select 'batch_size' indices from our baked bank ---
            random_indices = torch.randint(0, total_baked_points, (batch_size,), device=DEVICE)
            
            mu = baked_mu[random_indices]
            log_var = baked_log_var[random_indices]
            labels = baked_labels[random_indices]

            # --- Apply Neighborhood Sampling (Temperature) ---
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z_neighbor = mu + (args.temperature * std * epsilon)

            # --- Generate ---
            with autocast(device_type='cuda'):
                fake_imgs = generator(z_neighbor, labels)

            # --- Save ---
            fake_imgs = (fake_imgs + 1.0) / 2.0
            fake_imgs = fake_imgs.cpu()
            labels_cpu = labels.cpu()

            for i in range(batch_size):
                img_tensor = fake_imgs[i]
                label_val = labels_cpu[i].item()
                filename = f"synth_{label_val}_{images_generated + i:06d}.png"
                save_image(img_tensor, os.path.join(args.output_dir, f"class_{label_val}", filename))

            images_generated += batch_size
            pbar.update(batch_size)

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_weights", type=str, required=True)
    parser.add_argument("--baked_file", type=str, default="baked_latents.pt")
    parser.add_argument("--output_dir", type=str, default="./synthetic_dataset")
    parser.add_argument("--total_images", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128) # A6000 can easily hit 128 here
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    
    generate_from_baked(parser.parse_args())