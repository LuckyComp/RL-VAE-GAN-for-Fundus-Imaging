from models.varautoencoder import VAE, kl_annealing, elbo_loss
from dataloader import get_dataloaders
import torch.optim as optim
import torch.cuda
from dotenv import load_dotenv
import os

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")


def train_one_epoch(
    model, loader, optimizer, beta, device
):  # per epoch training process
    model.train()  # set model to train mode
    total_loss, total_recon, total_kl = 0, 0, 0  # initialize loss values

    for batch, _ in loader:  # for each batch of images from loader
        batch = batch.to(device)  # move batch to device

        optimizer.zero_grad()  # gradient backprop gets cleared

        recon, mu, log_var = model(batch)  # pass the batch of images to model
        loss, recon_loss, kl_loss = elbo_loss(
            recon, batch, mu, log_var, beta=beta
        )  # calculate loss values

        loss.backward()  # back-propogation
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # prevent gradient explosions during back propogation
        optimizer.step()  # updates the weights

        # update loss values
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(loader)  # number of batches in loader
    return (
        total_loss / n,
        total_recon / n,
        total_kl / n,
    )  # return average loss for this epoch


def val_one_epoch(model, loader, beta, device):  # per epoch validation process
    model.eval()  # set model to val mode
    total_loss, total_recon, total_kl = 0, 0, 0  # initialize the losses

    with torch.no_grad():  # with no gradient back propogation on the weights
        for batch, _ in loader:  # for each batch of images in the data loader
            batch = batch.to(device)  # loading batches to GPU

            recon, mu, log_var = model(
                batch
            )  # run inference and retrieve recon, mu and log_var
            loss, recon_loss, kl_loss = elbo_loss(
                recon, batch, mu, log_var, beta=beta
            )  # Calculate loss values

            # Add up total losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    n = len(loader)
    return (
        total_loss / n,
        total_recon / n,
        total_kl / n,
    )  # return average values of losses


if __name__ == "__main__":
    # When starting training
    DEVICE = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # try to initialize cuda as base device for training
    LEARNING_RATE = 1e-4  # learning rate of optimizer
    WARMUP_EPOCHS = 50  # number of warmup epochs
    
    # Setup directories
    os.makedirs("./models/saved", exist_ok=True)
    CHECKPOINT_DIR = "./models/saved"
    SAVE_EVERY = 10  # Save a checkpoint every 10 epochs

    print(f"Training on {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_PATH, batch_size=8, num_workers=4
    )  # set up the data loaders for train, test and validation

    model = VAE().to(DEVICE)  # load model to GPU memory
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # initialize optimizer

    start_epoch = 1

    # --- NEW: Dynamic Checkpoint Loading ---
    resume_input = input("Enter epoch to resume from (Press Enter or 0 to start fresh): ").strip()
    
    if resume_input and resume_input.isdigit() and int(resume_input) > 0:
        resume_epoch = int(resume_input)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vae_checkpoint_epoch{resume_epoch}.pth")
        
        # Fallback to check if they are trying to load the final checkpoint directly
        if not os.path.exists(checkpoint_path):
            fallback_path = os.path.join(CHECKPOINT_DIR, "vae_full_checkpoint.pth")
            if os.path.exists(fallback_path):
                checkpoint_path = fallback_path
                print(f"Epoch-specific checkpoint not found. Falling back to {fallback_path}")

        if os.path.exists(checkpoint_path):
            print(f"Loading VAE Checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            
            # 1. Check if it's the NEW format (Dictionary containing states)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                print("Successfully restored VAE model and optimizer momentum.")
            
            # 2. Check if it's the OLD format (Raw state_dict)
            else:
                model.load_state_dict(checkpoint)
                start_epoch = resume_epoch + 1
                print("[!] Loaded older checkpoint format. Optimizer momentum could not be restored and will start fresh.")
        else:
            print(f"Error: No checkpoint found. Starting fresh.")
            start_epoch = 1
            
    # Ask for total epochs after resolving the start epoch
    target_epochs = int(input(f"Enter total number of epochs to reach (currently at epoch {start_epoch - 1}): "))

    # per epoch training
    for epoch in range(start_epoch, target_epochs + 1):
        beta = kl_annealing(
            epoch, warmup_epochs=WARMUP_EPOCHS, max_beta=1.0
        )  # get annealed beta values

        train_loss, train_recon, train_kl = (
            train_one_epoch(  # start training for one epoch
                model, train_loader, optimizer, beta, DEVICE
            )
        )
        val_loss, val_recon, val_kl = val_one_epoch(  # start validation for one epoch
            model, val_loader, beta, DEVICE
        )

        print(
            f"Epoch {epoch:>4} | beta={beta:.3f} | "
            f"Train Loss={train_loss:.2f} (Recon={train_recon:.2f}, KL={train_kl:.2f}) | "
            f"Val Loss={val_loss:.2f} (Recon={val_recon:.2f}, KL={val_kl:.2f})"
        )

        # --- NEW: Periodic Checkpoint Saving ---
        if epoch % SAVE_EVERY == 0:
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint_state, os.path.join(CHECKPOINT_DIR, f"vae_checkpoint_epoch{epoch}.pth"))
            print(f"[*] Checkpoint saved at epoch {epoch}")

    # Save final weights into .pth file
    torch.save(model.encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "encoder_pretrained.pth"))
    
    # Save the full final checkpoint including optimizer state
    final_checkpoint = {
        "epoch": target_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(final_checkpoint, os.path.join(CHECKPOINT_DIR, "vae_full_checkpoint.pth"))
    
    print("Training complete. Encoder and Full VAE saved.")