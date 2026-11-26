import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. 3D U-Net Model Definition
# ==============================================================================
class DoubleConv3d(nn.Module):
    """(Conv3d -> BatchNorm3d -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet3D(nn.Module):
    """3D Volumetric U-Net for (B, C, D, H, W) tensors."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.features = features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder (3D Downsampling)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv3d(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        # Decoder (3D Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3d(feature * 2, feature))

        # Final 1x1x1 voxel-wise classification head
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        divisor = 2 ** len(self.features)
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        if (d % divisor != 0) or (h % divisor != 0) or (w % divisor != 0):
            raise ValueError(
                f"Input spatial dims {(d, h, w)} must be divisible by {divisor}."
            )

        skip_connections = []

        # Encoder forward pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder forward pass
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose3d
            skip = skip_connections[idx // 2]

            # Concatenate along channel dimension (B, C, D, H, W)
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # DoubleConv3d

        return self.final_conv(x)


# ==============================================================================
# 2. Sample 3D Volumetric Phantom Dataset
# ==============================================================================
class Synthetic3DVolumeDataset(Dataset):
    """
    Generates synthetic 3D medical-style volumes (e.g., 64x64x64) containing
    1 to 3 ellipsoidal lesions/organs with smooth intensity gradients and noise.
    """

    def __init__(
        self,
        num_samples: int = 80,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.volume_shape = volume_shape
        self.seed = seed

        # Precompute normalized coordinate grid [-1, 1]^3
        d, h, w = volume_shape
        zz = torch.linspace(-1.0, 1.0, d)
        yy = torch.linspace(-1.0, 1.0, h)
        xx = torch.linspace(-1.0, 1.0, w)
        self.grid_z, self.grid_y, self.grid_x = torch.meshgrid(
            zz, yy, xx, indexing="ij"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)

        mask = torch.zeros(self.volume_shape, dtype=torch.float32)
        num_blobs = int(torch.randint(1, 4, (1,), generator=gen).item())

        for _ in range(num_blobs):
            cz = torch.empty(1).uniform_(-0.45, 0.45, generator=gen).item()
            cy = torch.empty(1).uniform_(-0.45, 0.45, generator=gen).item()
            cx = torch.empty(1).uniform_(-0.45, 0.45, generator=gen).item()

            rz = torch.empty(1).uniform_(0.15, 0.35, generator=gen).item()
            ry = torch.empty(1).uniform_(0.15, 0.35, generator=gen).item()
            rx = torch.empty(1).uniform_(0.15, 0.35, generator=gen).item()

            ellipsoid = (
                ((self.grid_z - cz) / rz) ** 2
                + ((self.grid_y - cy) / ry) ** 2
                + ((self.grid_x - cx) / rx) ** 2
            ) <= 1.0
            mask = torch.maximum(mask, ellipsoid.float())

        # Build synthetic scan intensity: background + foreground contrast + bias + noise
        bias_field = 0.15 * self.grid_z + 0.10 * self.grid_x
        noise = torch.randn(self.volume_shape, generator=gen) * 0.22
        image = 0.25 + 0.65 * mask + bias_field + noise

        # Standardize volume (zero mean, unit variance)
        image = (image - image.mean()) / (image.std() + 1e-6)

        # Add channel dimension -> (1, D, H, W)
        return image.unsqueeze(0), mask.unsqueeze(0)


# ==============================================================================
# 3. Volumetric Segmentation Loss & Dice Metric
# ==============================================================================
class BCEDiceLoss(nn.Module):
    """Combines BCEWithLogitsLoss and Soft Dice Loss for 3D segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        # Flatten across spatial dimensions (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()

        return 0.5 * bce_loss + 0.5 * dice_loss


def compute_dice_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5
) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ==============================================================================
# 4. GPU Training & Validation Pipeline
# ==============================================================================
def train_and_validate():
    # 1. Configure CUDA GPU device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # 2. Hyperparameters
    volume_shape = (64, 64, 64)
    batch_size = 4
    epochs = 10
    lr = 1e-3

    # 3. Datasets & DataLoaders (pin_memory=True speeds up CPU->GPU transfers)
    train_ds = Synthetic3DVolumeDataset(num_samples=64, volume_shape=volume_shape, seed=100)
    val_ds = Synthetic3DVolumeDataset(num_samples=16, volume_shape=volume_shape, seed=999)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda
    )

    # 4. Initialize 3D Model, Loss, Optimizer, and AMP Scaler
    model = SimpleUNet3D(in_channels=1, out_channels=1, features=(32, 64, 128)).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_dice = 0.0
    checkpoint_path = "best_3d_unet.pth"

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for volumes, masks in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Automatic Mixed Precision forward pass
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(volumes)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * volumes.size(0)

        train_loss /= len(train_ds)

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for volumes, masks in val_loader:
                volumes = volumes.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(volumes)
                    loss = criterion(logits, masks)

                val_loss += loss.item() * volumes.size(0)
                val_dice += compute_dice_score(logits, masks) * volumes.size(0)

        val_loss /= len(val_ds)
        val_dice /= len(val_ds)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining complete. Best Validation Dice: {best_val_dice:.4f}")
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU VRAM allocated: {peak_mb:.1f} MB")

    # 7. Test 3D Volumetric Inference
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    sample_vol, sample_mask = val_ds[0]
    sample_vol = sample_vol.unsqueeze(0).to(device)  # (1, 1, 64, 64, 64)

    with torch.no_grad():
        pred_logits = model(sample_vol)
        pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()

    print(
        f"Inference check -> Input: {tuple(sample_vol.shape)} | "
        f"Predicted Mask: {tuple(pred_mask.shape)} | "
        f"Foreground Voxels: {int(pred_mask.sum().item())}"
    )


if __name__ == "__main__":
    train_and_validate()

