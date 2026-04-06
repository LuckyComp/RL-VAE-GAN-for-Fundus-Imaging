import torch
import ast
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from dotenv import load_dotenv

def get_data_stats(image_dir, image_size=256):
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    mean, std = torch.zeros(3), torch.zeros(3)

    for p in paths:
        img = transform(Image.open(p).convert("RGB"))
        mean += img.mean(dim=[1, 2])
        std += img.std(dim=[1,2])
    
    mean /= len(paths)
    std /= len(paths)
    return mean.tolist(), std.tolist()

class FundusDataset(Dataset):
    def __init__(self, image_dir, image_size=256, mode="train"):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        
        if mode == "train":
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
               # T.RandomHorizontalFlip(),
               # T.RandomVerticalFlip(),
               # T.RandomRotation(360),
               # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

        print(f"[{mode}] {len(self.image_paths)} images loaded from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transforms(img)


def get_dataloaders(data_dir, image_size=256, batch_size=16, num_workers=4):
    train_loader = DataLoader(
        FundusDataset(os.path.join(data_dir, "train"), image_size, mode="train"),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        FundusDataset(os.path.join(data_dir, "val"), image_size, mode="val"),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        FundusDataset(os.path.join(data_dir, "test"), image_size, mode="test"),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_dotenv()
    image_dir = os.path.join(os.getenv("DATASET_PATH"), "train")
    mean, std = get_data_stats(image_dir)
    print(f"Mean: {mean}\nStandard Deviation: {std}")
