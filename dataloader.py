import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


class SquareCrop:
    def __call__(self, image):
        w, h = image.size
        min_wh = min(w, h)
        return T.functional.center_crop(image, min_wh)


class DRDataset(Dataset):
    def __init__(self, csv_file, image_dir, image_size=256, mode="train"):
        self.image_dir = image_dir
        self.mode = mode
        self.image_size = image_size

        # Load the CSV file containing image_filename and dr_grade
        self.df = pd.read_csv(csv_file)

        # Initialize RAM cache lists
        self.cached_images = []
        self.cached_labels = []

        print(f"[*] Caching {len(self.df)} {mode.upper()} images into System RAM...")

        # 1. Deterministic Transforms: Applied once before caching to save RAM
        cache_transforms = T.Compose([SquareCrop(), T.Resize((image_size, image_size))])

        # Load everything into memory during initialization
        for idx in tqdm(range(len(self.df)), desc=f"Loading {mode} data"):
            row = self.df.iloc[idx]
            img_name = row["image_filename"]
            label = int(row["dr_grade"])

            img_path = os.path.join(self.image_dir, img_name)

            # Read image, convert to RGB, and apply deterministic resizing/cropping
            img = Image.open(img_path).convert("RGB")
            img = cache_transforms(img)

            self.cached_images.append(img)
            self.cached_labels.append(label)

        # 2. Stochastic Transforms: Applied dynamically during __getitem__
        if mode == "train":
            self.transforms = T.Compose(
                [
                    # T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        print(
            f"[{mode.upper()}] Cache complete. {len(self.cached_images)} images ready."
        )

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        # Pull the pre-resized PIL image and label directly from RAM
        img = self.cached_images[idx]
        label = self.cached_labels[idx]

        # Apply stochastic augmentations on the fly
        tensor_img = self.transforms(img)

        # Returns the 0-4 DR grade as a PyTorch long tensor
        return tensor_img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(
    data_dir, batch_size=64, num_workers=8, image_size=256, subset_fraction=1.0
):
    """
    Creates and returns PyTorch DataLoaders using the unified images directory
    and stratified CSV splits.
    """
    # Define paths based on the new unified structure
    image_dir = os.path.join(data_dir, "normalized_images_unified")
    train_csv = os.path.join(data_dir, "train_labels.csv")
    val_csv = os.path.join(data_dir, "val_labels.csv")
    test_csv = os.path.join(data_dir, "test_labels.csv")

    # 1. Initialize Datasets
    train_dataset = DRDataset(train_csv, image_dir, image_size=image_size, mode="train")
    val_dataset = DRDataset(val_csv, image_dir, image_size=image_size, mode="val")

    # 2. Handle Prototype Subset Fraction
    if subset_fraction < 1.0:
        np.random.seed(42)

        train_subset_size = int(len(train_dataset) * subset_fraction)
        train_indices = np.random.choice(
            len(train_dataset), train_subset_size, replace=False
        ).tolist()
        train_dataset = Subset(train_dataset, train_indices)

        val_subset_size = int(len(val_dataset) * subset_fraction)
        val_indices = np.random.choice(
            len(val_dataset), val_subset_size, replace=False
        ).tolist()
        val_dataset = Subset(val_dataset, val_indices)

        print(f"[*] RUNNING IN SUBSET MODE ({subset_fraction * 100}%)")
        print(f"[*] Train Size: {train_subset_size} | Val Size: {val_subset_size}")

    # 3. Initialize DataLoaders
    # Note: If num_workers > 0 with RAM caching, multiprocessing can sometimes duplicate memory.
    # If you see RAM spikes, drop num_workers to 0 since disk I/O is no longer the bottleneck.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Optional Test Loader Handling
    test_loader = None
    if os.path.exists(test_csv):
        test_dataset = DRDataset(
            test_csv, image_dir, image_size=image_size, mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        print(f"[TEST] No test CSV found at {test_csv}.")

    return train_loader, val_loader, test_loader
