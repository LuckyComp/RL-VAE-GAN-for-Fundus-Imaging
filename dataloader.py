import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


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
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(360),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        print(f"[{mode}] {len(self.image_paths)} images loaded from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transforms(img)


def get_dataloaders(image_dir, image_size=256, batch_size=16, val_split=0.15, num_workers=4):
    train_set = FundusDataset(image_dir, image_size, mode="train")
    val_set   = FundusDataset(image_dir, image_size, mode="val")

    # Split indices reproducibly
    n_val   = int(len(train_set) * val_split)
    n_train = len(train_set) - n_val
    train_set, _ = random_split(train_set, [n_train, n_val])
    _,  val_set  = random_split(val_set,   [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
