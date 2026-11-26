import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF


class PetSegmentationDataset(Dataset):
    def __init__(self, root: str = "./data", image_size: int = 128, download: bool = True):
        self.base_dataset = OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="segmentation",
            download=download,
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, mask = self.base_dataset[idx]

        # Resize image (bilinear) and mask (nearest-neighbor to preserve class IDs)
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)

        # Convert image to float tensor [0.0, 1.0], shape (3, H, W)
        image_tensor = TF.to_tensor(image)

        # Original mask values: 1 = foreground, 2 = background, 3 = border
        # Convert to binary float mask: 1.0 = foreground/border, 0.0 = background
        mask_np = np.array(mask, dtype=np.int64)
        binary_mask = np.where((mask_np == 1) | (mask_np == 3), 1.0, 0.0)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()  # (1, H, W)

        return image_tensor, mask_tensor


def get_dataloaders(batch_size: int = 32, image_size: int = 128, num_workers: int = 4):
    full_dataset = PetSegmentationDataset(image_size=image_size, download=True)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

