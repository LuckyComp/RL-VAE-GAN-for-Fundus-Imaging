from models.gan import Generator, Discriminator, PerceptualLoss, generator_loss, discriminator_loss
from models.varautoencoder import Encoder, reparameterize
from dataloader import get_dataloaders
import torch
import torch.optim as optim
from dotenv import load_dotenv
import os

load_dotenv()
DATASET_PATH  = os.getenv("DATASET_PATH")
ENCODER_PATH  = os.getenv("ENCODER_PATH")


def load_encoder(encoder_path, device):
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    for param in encoder.parameters():
        param.requires_grad = False 
    encoder.eval()
    print(f"Encoder loaded and frozen from {encoder_path}")
    return encoder


def train_one_epoch(encoder, generator, discriminator, perceptual_loss_fn,
                    optimizer_G, optimizer_D, loader, device):

    generator.train()
    discriminator.train()

    total_g_loss, total_d_loss = 0, 0
    total_adv_loss, total_perc_loss = 0, 0
    total_real_preds, total_synth_preds = 0, 0

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            mu, log_var = encoder(batch)
            z = reparameterize(mu, log_var)

        synthetic = generator(z) 

        optimizer_D.zero_grad()

        real_preds = discriminator(batch)
        synth_preds = discriminator(synthetic.detach())

        d_loss = discriminator_loss(real_preds, synth_preds)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()                              

        optimizer_G.zero_grad() 

        synth_pred_of_gan = discriminator(synthetic)
        g_loss, adv_loss, perc_loss = generator_loss(
            synth_pred_of_gan, synthetic, batch, perceptual_loss_fn
        )
        g_loss.backward()                               
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()  

        total_g_loss     += g_loss.item()
        total_d_loss     += d_loss.item()
        total_adv_loss   += adv_loss.item()
        total_perc_loss  += perc_loss.item()
        total_real_preds += real_preds.mean().item() 
        total_synth_preds += synth_preds.mean().item() 
    n = len(loader) #number of batches
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
        for batch in loader:
            batch = batch.to(device) 

            mu, log_var  = encoder(batch)
            z = reparameterize(mu, log_var)
            synthetic = generator(z)    

            real_preds = discriminator(batch)
            synth_preds = discriminator(synthetic)

            d_loss           = discriminator_loss(real_preds, synth_preds)  
            g_loss, _, _     = generator_loss(synth_preds, synthetic, batch, perceptual_loss_fn)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

    n = len(loader)
    return total_g_loss / n, total_d_loss / n #return average losses


if __name__ == "__main__":
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu" 
    LEARNING_RATE  = 1e-4  
    CHECKPOINT_DIR = "./models/saved/checkpoints"
    SAVE_EVERY     = 10    
    EPOCHS         = int(input("Enter number of training epochs: "))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
    print(f"Training on {DEVICE}")

    train_loader, val_loader, _ = get_dataloaders(
        DATASET_PATH,
        batch_size  = 8,
        num_workers = 4
    )

    encoder            = load_encoder(ENCODER_PATH, DEVICE) 
    generator          = Generator().to(DEVICE)             
    discriminator      = Discriminator().to(DEVICE)         
    perceptual_loss_fn = PerceptualLoss().to(DEVICE)        

    optimizer_G = optim.Adam(generator.parameters(),     lr=LEARNING_RATE, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    for epoch in range(1, EPOCHS + 1):

        g_loss, d_loss, adv_loss, perc_loss, real_preds, synth_preds = train_one_epoch(
            encoder, generator, discriminator, perceptual_loss_fn,
            optimizer_G, optimizer_D, train_loader, DEVICE
        )

        val_g_loss, val_d_loss = val_one_epoch(
            encoder, generator, discriminator, perceptual_loss_fn,
            val_loader, DEVICE
        )

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
