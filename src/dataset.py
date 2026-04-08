import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)

        self.image_dir = os.path.join(self.root_dir, "images")
        self.label_dir = os.path.join(self.root_dir, "labels")
        self.depth_dir = os.path.join(self.root_dir, "depth")

        self.images = sorted(os.listdir(self.image_dir))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)

        # Load
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)  # segmentation mask
        depth = Image.open(depth_path)  # grayscale depth

        # Convert
        # image = T.ToTensor()(image)  # [3,H,W]
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        # label = torch.from_numpy(np.array(label)).long()  # [H,W]
        label = np.array(label)          # [H, W, 3]
        label = label[:, :, 0]           # extract class_id channel
        label = torch.from_numpy(label).long()  # [H, W]
        # depth = torch.from_numpy(np.array(depth)).float()  # [H,W]
        depth = torch.from_numpy(np.array(depth)).float() / 255.0 # [H,W] range from [0,1]

        depth = depth.unsqueeze(0)  # [1,H,W]

        return image, label, depth
    

def get_loaders(root_dir, batch_size=8, num_workers=2):
    train_ds = MultiTaskDataset(root_dir, split="train")
    test_ds = MultiTaskDataset(root_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader