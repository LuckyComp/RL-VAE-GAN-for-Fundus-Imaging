from torch.optim.lr_scheduler import StepLR
# CHANGED: Import Critic and compute_gradient_penalty instead of old Discriminator/Losses
from models.gan import Generator, Critic, PerceptualLoss, compute_gradient_penalty
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


def train_one_epoch(encoder, generator, critic, perceptual_loss_fn, optimizer_G, optimizer_C, loader, device, scaler):
    generator.train()
    critic.train()

    total_g_loss, total_c_loss = 0, 0
    total_adv_loss, total_perc_loss = 0, 0
    total_real_preds, total_synth_preds = 0, 0

    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)

        # --- 1. LATENT GENERATION ---
        with torch.no_grad():
            with autocast(device_type='cuda'):
                mu, log_var = encoder(batch)
                z = reparameterize(mu, log_var)

        # --- 2. CRITIC STEP (WGAN-GP) ---
        optimizer_C.zero_grad()
        
        with autocast(device_type='cuda'):
            synthetic = generator(z, labels) 
            real_preds = critic(batch, labels)
            synth_preds = critic(synthetic.detach(), labels)
            
            # WGAN Critic Loss: E[fake] - E[real]
            c_loss_realfake = synth_preds.mean() - real_preds.mean()

        # CHANGED: Gradient Penalty MUST be calculated outside autocast to prevent NaN explosions
        gp = compute_gradient_penalty(critic, batch, synthetic.detach(), labels, device)
        lambda_gp = 10.0
        
        c_loss = c_loss_realfake + (lambda_gp * gp)

        scaler.scale(c_loss).backward()
        scaler.unscale_(optimizer_C)
        # CHANGED: Removed clip_grad_norm_. WGAN-GP does not use weight clipping.
        scaler.step(optimizer_C)                              

        # --- 3. GENERATOR STEP (WGAN-GP) ---
        optimizer_G.zero_grad() 

        with autocast(device_type='cuda'):
            synth_pred_of_gan = critic(synthetic, labels)
            
            # WGAN Generator Loss: -E[fake]
            adv_loss = -synth_pred_of_gan.mean()
            perc_loss = perceptual_loss_fn(batch, synthetic)
            
            # Combine losses
            g_loss = adv_loss + (0.5 * perc_loss)

        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizer_G)
        scaler.step(optimizer_G)  
        scaler.update()

        # --- 4. LOGGING ---
        total_g_loss     += g_loss.item()
        total_c_loss     += c_loss.item()
        total_adv_loss   += adv_loss.item()
        total_perc_loss  += perc_loss.item()
        total_real_preds += real_preds.mean().item() 
        total_synth_preds += synth_preds.mean().item() 

    n = len(loader)
    return (
        total_g_loss     / n,
        total_c_loss     / n,
        total_adv_loss   / n,
        total_perc_loss  / n,
        total_real_preds / n,
        total_synth_preds / n,
    )


def val_one_epoch(encoder, generator, critic, perceptual_loss_fn, loader, device):
    generator.eval()     
    critic.eval() 

    total_g_loss, total_c_loss = 0, 0

    with torch.no_grad(): 
        for batch, labels in loader:
            batch = batch.to(device) 
            labels = labels.to(device)

            mu, log_var = encoder(batch)
            z = reparameterize(mu, log_var)
            
            synthetic = generator(z, labels)    
            real_preds = critic(batch, labels)
            synth_preds = critic(synthetic, labels)

            # WGAN Validation Losses
            c_loss    = synth_preds.mean() - real_preds.mean()  
            adv_loss  = -synth_preds.mean()
            perc_loss = perceptual_loss_fn(batch, synthetic)
            g_loss    = adv_loss + (0.5 * perc_loss)

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()

    n = len(loader)
    return total_g_loss / n, total_c_loss / n 


if __name__ == "__main__":
    scaler = GradScaler(device='cuda')
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"  
    CHECKPOINT_DIR = "./models/saved/checkpoints"
    SAVE_EVERY     = 10    
    EPOCHS         = int(input("Enter total number of epochs: "))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
    print(f"Training on {DEVICE}")

    train_loader, val_loader, _ = get_dataloaders(
        DATASET_PATH,
        batch_size  = 4, 
        num_workers = 4
    )

    generator          = Generator().to(DEVICE)             
    critic             = Critic().to(DEVICE)         

    start_epoch = 1
    resume_input = input("Enter epoch to resume from (Press Enter or 0 to start fresh): ").strip()

    if resume_input and resume_input.isdigit() and int(resume_input) > 0:
        resume_epoch = int(resume_input)
        gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch{resume_epoch}.pth")
        disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{resume_epoch}.pth")
        
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            generator.load_state_dict(torch.load(gen_path, map_location=DEVICE, weights_only=True))
            critic.load_state_dict(torch.load(disc_path, map_location=DEVICE, weights_only=True))
            print(f"Successfully loaded checkpoints from Epoch {resume_epoch}.")
            start_epoch = resume_epoch + 1 
        else:
            print(f"Error: Checkpoints for Epoch {resume_epoch} not found. Starting fresh.")
            start_epoch = 1

    encoder            = load_encoder(ENCODER_PATH, DEVICE) 
    perceptual_loss_fn = PerceptualLoss().to(DEVICE)        

    # CHANGED: WGAN-GP strictly requires betas=(0.0, 0.9) to prevent momentum from interfering with the gradient penalty
    # Increased Learning Rate slightly to 1e-4 which is standard for WGAN
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(),    lr=1e-4, betas=(0.0, 0.9))
    
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.8)
    scheduler_C = StepLR(optimizer_C, step_size=20, gamma=0.8)

    target_epoch = start_epoch + EPOCHS
    for epoch in range(start_epoch, target_epoch):

        g_loss, c_loss, adv_loss, perc_loss, real_preds, synth_preds = train_one_epoch(
            encoder, generator, critic, perceptual_loss_fn,
            optimizer_G, optimizer_C, train_loader, DEVICE, scaler
        )

        val_g_loss, val_c_loss = val_one_epoch(
            encoder, generator, critic, perceptual_loss_fn,
            val_loader, DEVICE
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
            torch.save(
                generator.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth")
            )
            torch.save(
                critic.state_dict(),
                # Keeping the old filename pattern to maintain compatibility with your run.py script
                os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth")
            )
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_final.pth"))
    torch.save(critic.state_dict(),    os.path.join(CHECKPOINT_DIR, "discriminator_final.pth"))
    print("Training complete. Final models saved.")
