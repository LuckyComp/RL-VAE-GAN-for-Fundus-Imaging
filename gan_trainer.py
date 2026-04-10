from torch.optim.lr_scheduler import StepLR
from models.gan import Generator, Discriminator, PerceptualLoss, generator_loss, discriminator_loss
from models.varautoencoder import Encoder, reparameterize
from dataloader import get_dataloaders
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from dotenv import load_dotenv
import os

load_dotenv()
DATASET_PATH  = os.getenv("DATASET_PATH")
ENCODER_PATH  = os.getenv("ENCODER_PATH")

def load_encoder(encoder_path, device):
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    for param in encoder.parameters():
        param.requires_grad = False 
    encoder.eval()
    print(f"Encoder loaded and frozen from {encoder_path}")
    return encoder


def train_one_epoch(encoder, generator, discriminator, perceptual_loss_fn, optimizer_G, optimizer_D, loader, device, scaler):
    generator.train()
    discriminator.train()

    total_g_loss, total_d_loss = 0, 0
    total_adv_loss, total_perc_loss = 0, 0
    total_real_preds, total_synth_preds = 0, 0

    # CHANGED: Unpack batch AND labels
    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)

        # --- 1. LATENT GENERATION ---
        with torch.no_grad():
            with autocast(device_type='cuda'):
                mu, log_var = encoder(batch)
                z = reparameterize(mu, log_var)

        # --- 2. DISCRIMINATOR STEP ---
        optimizer_D.zero_grad()
        
        with autocast(device_type='cuda'):
            # CHANGED: Pass labels to the Generator and Discriminator
            synthetic = generator(z, labels) 
            real_preds = discriminator(batch, labels)
            synth_preds = discriminator(synthetic.detach(), labels)
            d_loss = discriminator_loss(real_preds, synth_preds)

        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_D)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(optimizer_D)                              

        # --- 3. GENERATOR STEP ---
        optimizer_G.zero_grad() 

        with autocast(device_type='cuda'):
            # CHANGED: Pass labels again for the Generator update
            synth_pred_of_gan = discriminator(synthetic, labels)
            g_loss, adv_loss, perc_loss = generator_loss(
                synth_pred_of_gan, synthetic, batch, perceptual_loss_fn, lambda_percept=0.5
            )

        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizer_G)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(optimizer_G)  
        scaler.update()

        # --- 4. LOGGING ---
        total_g_loss     += g_loss.item()
        total_d_loss     += d_loss.item()
        total_adv_loss   += adv_loss.item()
        total_perc_loss  += perc_loss.item()
        total_real_preds += real_preds.mean().item() 
        total_synth_preds += synth_preds.mean().item() 

    n = len(loader)
    return (
        total_g_loss     / n,
        total_d_loss     / n,
        total_adv_loss   / n,
        total_perc_loss  / n,
        total_real_preds / n,
        total_synth_preds / n,
    )


def val_one_epoch(encoder, generator, discriminator, perceptual_loss_fn, loader, device):
    generator.eval()     
    discriminator.eval() 

    total_g_loss, total_d_loss = 0, 0

    with torch.no_grad(): 
        # CHANGED: Unpack batch AND labels
        for batch, labels in loader:
            batch = batch.to(device) 
            labels = labels.to(device)

            mu, log_var = encoder(batch)
            z = reparameterize(mu, log_var)
            
            # CHANGED: Pass labels
            synthetic = generator(z, labels)    
            real_preds = discriminator(batch, labels)
            synth_preds = discriminator(synthetic, labels)

            d_loss           = discriminator_loss(real_preds, synth_preds)  
            g_loss, _, _     = generator_loss(synth_preds, synthetic, batch, perceptual_loss_fn)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

    n = len(loader)
    return total_g_loss / n, total_d_loss / n 


if __name__ == "__main__":
    scaler = GradScaler(device='cuda')
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"  
    CHECKPOINT_DIR = "./models/saved/checkpoints"
    SAVE_EVERY     = 10    
    EPOCHS         = int(input("Enter total number of epochs: "))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
    print(f"Training on {DEVICE}")

    # CHANGED: ADA 6000 scaling! Increased batch_size and num_workers
    train_loader, val_loader, _ = get_dataloaders(
        DATASET_PATH,
        batch_size  = 64, 
        num_workers = 8
    )

    generator          = Generator().to(DEVICE)             
    discriminator      = Discriminator().to(DEVICE)         

    start_epoch = 1
    resume_input = input("Enter epoch to resume from (Press Enter or 0 to start fresh): ").strip()

    if resume_input and resume_input.isdigit() and int(resume_input) > 0:
        resume_epoch = int(resume_input)
        gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch{resume_epoch}.pth")
        disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{resume_epoch}.pth")
        
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            generator.load_state_dict(torch.load(gen_path, map_location=DEVICE, weights_only=True))
            discriminator.load_state_dict(torch.load(disc_path, map_location=DEVICE, weights_only=True))
            print(f"Successfully loaded checkpoints from Epoch {resume_epoch}.")
            start_epoch = resume_epoch + 1 
        else:
            print(f"Error: Checkpoints for Epoch {resume_epoch} not found. Starting fresh.")
            start_epoch = 1

    # REMOVED: warm_start_generator logic. 
    # The new Generator FC layer is Linear(306, ...) so the old VAE Linear(256, ...) 
    # weights will mathematically crash. The ADA 6000 will power through from scratch.

    encoder            = load_encoder(ENCODER_PATH, DEVICE) 
    perceptual_loss_fn = PerceptualLoss().to(DEVICE)        

    optimizer_G = optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=5e-6, betas=(0.9, 0.999))
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.8)
    scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.8)

    # CHANGED: Fixed the loop range to properly account for the resume start_epoch
    target_epoch = start_epoch + EPOCHS
    for epoch in range(start_epoch, target_epoch):

        g_loss, d_loss, adv_loss, perc_loss, real_preds, synth_preds = train_one_epoch(
            encoder, generator, discriminator, perceptual_loss_fn,
            optimizer_G, optimizer_D, train_loader, DEVICE, scaler
        )

        val_g_loss, val_d_loss = val_one_epoch(
            encoder, generator, discriminator, perceptual_loss_fn,
            val_loader, DEVICE
        )

        scheduler_G.step()
        scheduler_D.step()

        print(
            f"Epoch {epoch:>4} | "
            f"G={g_loss:.4f} (adv={adv_loss:.4f}, perc={perc_loss:.4f}) | "
            f"D={d_loss:.4f} | "
            f"D_real={real_preds:.3f} D_Synthetic={synth_preds:.3f} | "
            f"Val G={val_g_loss:.4f} D={val_d_loss:.4f}"
        )

        if epoch % SAVE_EVERY == 0:
            torch.save(
                generator.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth")
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth")
            )
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(generator.state_dict(),     os.path.join(CHECKPOINT_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_final.pth"))
    print("Training complete. Final models saved.")