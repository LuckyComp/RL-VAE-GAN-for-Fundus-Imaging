import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class FundusDataset(Dataset):
    def __init__(self, image_dir, image_size=256, mode="train"):
        self.image_dir = image_dir
        self.mode = mode
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        
        # Load the cluster labels for ALL modes (Train, Val, Test)
        # The validation Discriminator needs to know the true label to calculate validation loss.
        self.labels_dict = {}
        labels_file = "cluster_labels.json"
        if os.path.exists(labels_file):
            with open(labels_file, "r") as f:
                self.labels_dict = json.load(f)
        else:
            print(f"Warning: {labels_file} not found. Defaulting all labels to 0.")

        if mode == "train":
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15), 
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        print(f"[{mode.upper()}] {len(self.image_paths)} images loaded from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        
        img = Image.open(img_path).convert("RGB")
        tensor_img = self.transforms(img)

        # ALWAYS return exactly 2 values: (image_tensor, label_tensor)
        # If the filename isn't mapped, default to class 0 safely.
        label = self.labels_dict.get(filename, 0) 
        
        return tensor_img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir, batch_size=64, num_workers=8, image_size=256):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test splits.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # 1. Initialize Datasets
    train_dataset = FundusDataset(train_dir, image_size=image_size, mode="train")
    val_dataset = FundusDataset(val_dir, image_size=image_size, mode="val")
    
    # 2. Initialize DataLoaders
    # pin_memory=True speeds up host-to-device (CPU to GPU) transfers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True # Drops incomplete batches to maintain stable batch sizes for BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # 3. Optional Test Loader Handling
    test_loader = None
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = FundusDataset(test_dir, image_size=image_size, mode="test")
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
    else:
        print(f"[TEST] No test directory found at {test_dir} or directory is empty.")

    return train_loader, val_loader, test_loader