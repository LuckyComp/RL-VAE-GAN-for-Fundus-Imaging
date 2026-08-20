from torch.optim.lr_scheduler import StepLR
from models.gan import (
    Generator, Critic, PerceptualLoss, compute_gradient_penalty,
    SSIMLoss
)
from models.varautoencoder import Encoder, reparameterize
from dataloader import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from dotenv import load_dotenv
import os
from tqdm import tqdm
import warnings
import copy

warnings.filterwarnings("ignore", message=".*CublasHandlePool.*")

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
ENCODER_PATH = os.getenv("ENCODER_PATH")

def load_encoder(encoder_path, device):
    encoder = Encoder().to(device)
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print(f"Encoder loaded and frozen from {encoder_path}")
    return encoder

def train_one_epoch(
    encoder, generator, generator_ema, critic, 
    perceptual_loss_fn, standard_l1_fn, ssim_fn,
    optimizer_G, optimizer_C, loader, device, scaler, epoch, max_epochs,
):
    generator = generator.to(memory_format=torch.channels_last)
    critic = critic.to(memory_format=torch.channels_last)
    generator.train()
    critic.train()

    total_g_loss, total_c_loss = 0, 0
    total_adv_loss, total_perc_loss = 0, 0
    total_real_preds, total_synth_preds = 0, 0

    loop = tqdm(loader, desc=f"Epoch {epoch:>3} [Train]", leave=False)

    for batch, labels in loop:
        batch = batch.to(device, memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long)

        # --- 1. LATENT GENERATION ---
        with torch.no_grad():
            with autocast(device_type="cuda"):
                mu, log_var = encoder(batch)
                z_encoded = reparameterize(mu, log_var)

                idx = torch.randperm(batch.size(0))
                z_random = reparameterize(mu[idx], log_var[idx])

                mask = torch.rand(batch.size(0), 1).to(device) > 0.5
                z = torch.where(mask, z_encoded, z_random)
                
                mask_1d = mask.view(-1)
                mixed_labels = torch.where(mask_1d, labels, labels[idx])
                
                mask_4d = mask.unsqueeze(-1).unsqueeze(-1)
                mixed_batch = torch.where(mask_4d, batch, batch[idx])
                
        # --- 2. CRITIC STEP ---
        optimizer_C.zero_grad()
        noise_sigma = max(0.0, 0.1 * (1.0 - (epoch / max_epochs)))
        with autocast(device_type="cuda", dtype=torch.float16):
            synthetic = generator(z, mixed_labels)

            if noise_sigma > 0:
                real_noise = torch.randn_like(mixed_batch) * noise_sigma
                fake_noise = torch.randn_like(synthetic) * noise_sigma
            else:
                real_noise, fake_noise = 0, 0

            noisy_real = mixed_batch + real_noise
            noisy_fake = synthetic.detach() + fake_noise

            real_preds = critic(noisy_real, mixed_labels)
            synth_preds = critic(noisy_fake, mixed_labels)

            c_loss_realfake = synth_preds.mean() - real_preds.mean()

        gp = compute_gradient_penalty(critic, mixed_batch, synthetic.detach(), mixed_labels, device)
        lambda_gp = 10.0
        c_loss = c_loss_realfake + (lambda_gp * gp)

        scaler.scale(c_loss).backward()
        scaler.unscale_(optimizer_C)
        scaler.step(optimizer_C)

        # --- 3. GENERATOR STEP ---
        optimizer_G.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16):
            # NEW: Extract feature maps from the Critic for both Real and Fake
            real_validity, real_features = critic(mixed_batch, mixed_labels, return_features=True)
            fake_validity, fake_features = critic(synthetic, mixed_labels, return_features=True)
            
            adv_loss = -fake_validity.mean()
            
            # NEW: Critic Feature Matching Loss
            fm_loss = torch.nn.functional.mse_loss(fake_features, real_features.detach())
            
            is_encoded = mask.view(-1)
            
            # STANDARD RECONSTRUCTION (No Masks)
            if is_encoded.any():
                batch_encoded = mixed_batch[is_encoded]
                synthetic_encoded = synthetic[is_encoded]
                
                perc_loss = perceptual_loss_fn(batch_encoded, synthetic_encoded)
                l1_loss = standard_l1_fn(synthetic_encoded, batch_encoded)
                ssim_loss = ssim_fn(synthetic_encoded, batch_encoded)
            else:
                perc_loss = torch.tensor(0.0, device=device)
                l1_loss = torch.tensor(0.0, device=device)
                ssim_loss = torch.tensor(0.0, device=device)

            if epoch < 10:
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 10.0, 10.0, 5.0, 10.0
            elif epoch < 20:
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 5.0, 5.0, 2.0, 10.0
            else:
                # Keep Feature Matching relatively strong to ensure clinical structures remain stable
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 0.5, 2.0, 1.0, 5.0

        # Inject fm_loss into the total Generator loss
        g_loss = adv_loss + (lambda_perc * perc_loss) + (lambda_l1 * l1_loss) + (lambda_ssim * ssim_loss) + (lambda_fm * fm_loss)

        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizer_G)
        scaler.step(optimizer_G)
        scaler.update()

        ema_decay = 0.999
        with torch.no_grad():
            for param, ema_param in zip(
                generator.parameters(), generator_ema.parameters()
            ):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        # --- 4. LOGGING ---
        total_g_loss += g_loss.item()
        total_c_loss += c_loss.item()
        total_adv_loss += adv_loss.item()
        total_perc_loss += perc_loss.item()
        total_real_preds += real_preds.mean().item()
        total_synth_preds += synth_preds.mean().item()

        # Added fm_loss to the postfix so you can watch it converge
        loop.set_postfix(G_loss=f"{g_loss.item():.3f}", FM_loss=f"{fm_loss.item():.3f}", C_loss=f"{c_loss.item():.3f}")

    n = len(loader)
    return (
        total_g_loss / n, total_c_loss / n, total_adv_loss / n,
        total_perc_loss / n, total_real_preds / n, total_synth_preds / n,
    )


def val_one_epoch(
    encoder, generator, critic, perceptual_loss_fn, standard_l1_fn, ssim_fn, loader, device, epoch
):
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss = 0, 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch:>3} [Val  ]", leave=False)
        for batch, labels in loop:
            batch = batch.to(device)
            labels = labels.to(device, dtype=torch.long)

            mu, log_var = encoder(batch)
            z_encoded = reparameterize(mu, log_var)
            idx = torch.randperm(batch.size(0))
            z_random = reparameterize(mu[idx], log_var[idx])

            mask = torch.rand(batch.size(0), 1).to(device) > 0.5
            z = torch.where(mask, z_encoded, z_random)
            
            mask_1d = mask.view(-1)
            mixed_labels = torch.where(mask_1d, labels, labels[idx])
            mask_4d = mask.unsqueeze(-1).unsqueeze(-1)
            mixed_batch = torch.where(mask_4d, batch, batch[idx])

            synthetic = generator(z, mixed_labels)
            
            # Extract features for validation Feature Matching loss
            real_validity, real_features = critic(mixed_batch, mixed_labels, return_features=True)
            fake_validity, fake_features = critic(synthetic, mixed_labels, return_features=True)

            c_loss = fake_validity.mean() - real_validity.mean()
            adv_loss = -fake_validity.mean()
            
            fm_loss = torch.nn.functional.mse_loss(fake_features, real_features.detach())
            
            is_encoded = mask.view(-1)
            if is_encoded.any():
                batch_encoded = mixed_batch[is_encoded]
                synthetic_encoded = synthetic[is_encoded]
                
                perc_loss = perceptual_loss_fn(batch_encoded, synthetic_encoded)
                l1_loss = standard_l1_fn(synthetic_encoded, batch_encoded)
                ssim_loss = ssim_fn(synthetic_encoded, batch_encoded)
            else:
                perc_loss = torch.tensor(0.0, device=device)
                l1_loss = torch.tensor(0.0, device=device)
                ssim_loss = torch.tensor(0.0, device=device)

            if epoch < 10:
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 10.0, 10.0, 5.0, 10.0
            elif epoch < 20:
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 5.0, 5.0, 2.0, 10.0
            else:
                lambda_perc, lambda_l1, lambda_ssim, lambda_fm = 0.5, 2.0, 1.0, 5.0

            g_loss = adv_loss + (lambda_perc * perc_loss) + (lambda_l1 * l1_loss) + (lambda_ssim * ssim_loss) + (lambda_fm * fm_loss)

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            loop.set_postfix(Val_G=f"{g_loss.item():.3f}", Val_FM=f"{fm_loss.item():.3f}", Val_C=f"{c_loss.item():.3f}")

    n = len(loader)
    return total_g_loss / n, total_c_loss / n


if __name__ == "__main__":
    scaler = GradScaler(device="cuda")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./models/saved/checkpoints"
    SAVE_EVERY = 5
    EPOCHS = int(input("Enter total number of epochs: "))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Training on {DEVICE}")

    train_loader, val_loader, _ = get_dataloaders(
        DATASET_PATH, batch_size=6, num_workers=8
    )

    generator = Generator().to(DEVICE)
    critic = Critic().to(DEVICE)
    encoder = load_encoder(ENCODER_PATH, DEVICE)
    
    perceptual_loss_fn = PerceptualLoss().to(DEVICE)
    standard_l1_fn = nn.L1Loss().to(DEVICE)  # BACK TO NORMAL
    ssim_fn = SSIMLoss(data_range=2.0, device=DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    scheduler_G = StepLR(optimizer_G, step_size=30, gamma=0.8)
    scheduler_C = StepLR(optimizer_C, step_size=30, gamma=0.8)

    start_epoch = 1
    resume_input = input("Enter epoch to resume from (Press Enter or 0 to start fresh): ").strip()

    if resume_input and resume_input.isdigit() and int(resume_input) > 0:
        resume_epoch = int(resume_input)
        unified_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch{resume_epoch}.pth")

        if os.path.exists(unified_path):
            print(f"Loading Unified Checkpoint from Epoch {resume_epoch}...")
            checkpoint = torch.load(unified_path, map_location=DEVICE, weights_only=False)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            critic.load_state_dict(checkpoint["critic_state_dict"])
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            optimizer_C.load_state_dict(checkpoint["optimizer_C_state_dict"])
            scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
            scheduler_C.load_state_dict(checkpoint["scheduler_C_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print("Successfully restored full training momentum.")
        else:
            print(f"Error: No checkpoints found for Epoch {resume_epoch}. Starting fresh.")
            start_epoch = 1

    print("Initializing Generator EMA from current weights...")
    generator_ema = copy.deepcopy(generator).to(DEVICE)

    if resume_input and resume_input.isdigit() and int(resume_input) > 0:
        resume_epoch = int(resume_input)
        ema_path = os.path.join(CHECKPOINT_DIR, f"generator_ema_epoch{resume_epoch}.pth")
        if os.path.exists(ema_path):
            print(f"[*] Successfully loaded saved EMA weights from Epoch {resume_epoch}.")
            generator_ema.load_state_dict(torch.load(ema_path, map_location=DEVICE, weights_only=True))
        else:
            print(f"[!] Warning: No EMA weights found for Epoch {resume_epoch}. Falling back to standard weights.")

    generator_ema.eval()
    for param in generator_ema.parameters():
        param.requires_grad = False

    target_epoch = start_epoch + EPOCHS
    for epoch in range(start_epoch, target_epoch):
        g_loss, c_loss, adv_loss, perc_loss, real_preds, synth_preds = train_one_epoch(
            encoder, generator, generator_ema, critic,
            perceptual_loss_fn, standard_l1_fn, ssim_fn, 
            optimizer_G, optimizer_C, train_loader, DEVICE, scaler, epoch, target_epoch,
        )

        val_g_loss, val_c_loss = val_one_epoch(
            encoder, generator, critic, 
            perceptual_loss_fn, standard_l1_fn, ssim_fn, 
            val_loader, DEVICE, epoch
        )

        scheduler_G.step()
        scheduler_C.step()

        print(
            f"Epoch {epoch:>4} | "
            f"G={g_loss:.4f} (adv={adv_loss:.4f}, perc={perc_loss:.4f}) | "
            f"C={c_loss:.4f} | "
            f"C_real={real_preds:.3f} C_Synthetic={synth_preds:.3f} | "
            f"Val G={val_g_loss:.4f} C={val_c_loss:.4f}"
        )

        if epoch % SAVE_EVERY == 0:
            full_state = {
                "epoch": epoch, "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(), "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_C_state_dict": optimizer_C.state_dict(), "scheduler_G_state_dict": scheduler_G.state_dict(),
                "scheduler_C_state_dict": scheduler_C.state_dict(), "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(full_state, os.path.join(CHECKPOINT_DIR, f"training_state_epoch{epoch}.pth"))
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))
            torch.save(generator_ema.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_ema_epoch{epoch}.pth"))
            print(f"[*] Full training state saved at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_final.pth"))
    torch.save(critic.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_final.pth"))
    print("Training complete. Final models saved.")