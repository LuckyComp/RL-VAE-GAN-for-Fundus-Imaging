from models.varautoencoder import VAE, kl_annealing, elbo_loss
from dataloader import get_dataloaders
import torch.optim as optim
import torch.cuda


def train_one_epoch(model, loader, optimizer, beta, device): # per epoch training process
    model.train() #set model to train mode
    total_loss, total_recon, total_kl = 0, 0, 0 #initialize loss values

    for batch in loader: #for each batch of images from loader
        batch = batch.to(device)              #move batch to device

        optimizer.zero_grad()                  #gradient backprop gets cleared

        recon, mu, log_var = model(batch) #pass the batch of images to model
        loss, recon_loss, kl_loss = elbo_loss(recon, batch, mu, log_var, beta=beta) #calculate loss values

        loss.backward() #back-propogation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #prevent gradient explosions during back propogation
        optimizer.step() #updates the weights

        #update loss values
        total_loss  += loss.item() 
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()

    n = len(loader) #number of batches in loader
    return total_loss / n, total_recon / n, total_kl / n #return average loss for this epoch

def val_one_epoch(model, loader, beta, device): # per epoch validation process
    model.eval() #set model to val mode
    total_loss, total_recon, total_kl = 0, 0, 0 #initialize the losses
 
    with torch.no_grad():                  
        for batch in loader:
            batch = batch.to(device)

            recon, mu, log_var = model(batch)
            loss, recon_loss, kl_loss = elbo_loss(recon, batch, mu, log_var, beta=beta)

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n

if __name__ == "__main__":
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    WARMUP_EPOCHS = 50
    EPOCHS        = int(input("Enter number of training epochs: "))

    print(f"Training on {DEVICE}")

    train_loader, val_loader = get_dataloaders(
        "./data/fundus",
        batch_size  = 16,
        num_workers = 4
    )

    model     = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        beta = kl_annealing(epoch, warmup_epochs=WARMUP_EPOCHS, max_beta=1.0)

        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, beta, DEVICE
        )
        val_loss, val_recon, val_kl = val_one_epoch(
            model, val_loader, beta, DEVICE
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>4} | beta={beta:.3f} | "
                f"Train Loss={train_loss:.2f} (Recon={train_recon:.2f}, KL={train_kl:.2f}) | "
                f"Val Loss={val_loss:.2f} (Recon={val_recon:.2f}, KL={val_kl:.2f})"
            )

    torch.save(model.encoder.state_dict(), "models/saved/encoder_pretrained.pth")
    print("Training complete. Encoder saved.")