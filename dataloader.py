import sys
import os
import concurrent.futures
# Fix imports by appending the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from tqdm import tqdm
from torch.amp import autocast
from torchvision.utils import save_image
from models.gan import Generator, Discriminator

# --- Helper function for Async Saving ---
def async_save(img_tensor, filepath):
    save_image(img_tensor, filepath)

def generate_from_baked(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the single main output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("[*] Initializing Models and PyTorch 2.0 Compiler...")
    
    # 1. Load and Compile the Generator
    generator = Generator(base_channels=128).to(DEVICE)
    gen_checkpoint = torch.load(args.gen_weights, map_location=DEVICE, weights_only=True)
    generator.load_state_dict(
        gen_checkpoint.get("generator_state_dict", gen_checkpoint), strict=False
    )
    generator.eval()
    generator = torch.compile(generator)  # 20-30% inference speedup

    # 2. Load and Compile the Critic (Discriminator) for Rejection Sampling
    critic = Discriminator(base_channels=128).to(DEVICE)
    critic_checkpoint = torch.load(args.critic_weights, map_location=DEVICE, weights_only=True)
    critic.load_state_dict(
        critic_checkpoint.get("discriminator_state_dict", critic_checkpoint), strict=False
    )
    critic.eval()
    critic = torch.compile(critic)

    # 3. Load the Baked Latents
    baked = torch.load(args.baked_file, weights_only=True)
    baked_mu = baked["mu"].to(DEVICE)
    baked_log_var = baked["log_var"].to(DEVICE)
    baked_labels = baked["labels"].to(DEVICE)

    total_baked_points = len(baked_labels)
    print(f"[*] Loaded {total_baked_points} baked coordinates.")

    images_generated = 0
    pbar = tqdm(total=args.total_images, desc=f"Generating (Temp: {args.temperature}, Trunc: {args.truncation})")

    # Initialize the ThreadPool for asynchronous I/O saving
    # Max workers can be adjusted based on your CPU threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        with torch.no_grad():
            while images_generated < args.total_images:
                # We pull a full batch, knowing some might be rejected
                batch_size = min(args.batch_size, args.total_images - images_generated)

                # --- Randomly select indices from our baked bank ---
                random_indices = torch.randint(
                    0, total_baked_points, (batch_size,), device=DEVICE
                )

                mu = baked_mu[random_indices]
                log_var = baked_log_var[random_indices]
                labels = baked_labels[random_indices]

                # --- Apply Truncation Trick & Temperature ---
                std = torch.exp(0.5 * log_var)
                epsilon = torch.randn_like(std)
                # Clamp extreme outliers to guarantee high-density structural sampling
                epsilon = torch.clamp(epsilon, -args.truncation, args.truncation)
                z_neighbor = mu + (args.temperature * std * epsilon)

                # --- Generate and Critique ---
                with autocast(device_type="cuda"):
                    fake_imgs = generator(z_neighbor, labels)
                    # WGAN Critic outputs logits (higher is more "real")
                    critic_scores = critic(fake_imgs, labels).squeeze()

                # --- Critic-Based Rejection Sampling ---
                # Keep only images that score higher than the threshold
                valid_mask = critic_scores >= args.critic_threshold
                
                # Handle single-element edge cases correctly
                if valid_mask.dim() == 0:
                    valid_mask = valid_mask.unsqueeze(0)
                
                valid_imgs = fake_imgs[valid_mask]
                valid_labels = labels[valid_mask]
                
                num_accepted = len(valid_imgs)
                
                # If nothing passed the critic, skip saving and generate a new batch
                if num_accepted == 0:
                    continue
                
                # Ensure we do not overshoot the total requested images
                if images_generated + num_accepted > args.total_images:
                    num_accepted = args.total_images - images_generated
                    valid_imgs = valid_imgs[:num_accepted]
                    valid_labels = valid_labels[:num_accepted]

                # --- Asynchronous Saving ---
                valid_imgs = (valid_imgs + 1.0) / 2.0
                valid_imgs = valid_imgs.cpu()
                valid_labels_cpu = valid_labels.cpu()

                for i in range(num_accepted):
                    img_tensor = valid_imgs[i]
                    label_val = valid_labels_cpu[i].item()
                    
                    filename = f"synth_{label_val}_{images_generated + i:06d}.png"
                    filepath = os.path.join(args.output_dir, filename)
                    
                    # Submit the save task to the background CPU threads
                    executor.submit(async_save, img_tensor, filepath)

                images_generated += num_accepted
                pbar.update(num_accepted)

    pbar.close()
    print(f"\n[*] Successfully generated and vetted {args.total_images} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path is required via terminal argument
    parser.add_argument("--gen_weights", type=str, required=True)
    parser.add_argument("--critic_weights", type=str, required=True)
    
    # Existing arguments
    parser.add_argument("--baked_file", type=str, default="baked_latents.pt")
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset")
    parser.add_argument("--total_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--temperature", type=float, default=0.3)
    
    # New quality-control arguments
    parser.add_argument("--truncation", type=float, default=2.0, help="Clamps the gaussian noise to prevent edge-case hallucinations")
    parser.add_argument("--critic_threshold", type=float, default=0.0, help="Minimum WGAN critic score to accept an image")

    generate_from_baked(parser.parse_args())