import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. 3D U-NET ARCHITECTURE
# ==============================================================================
class DoubleConv3D(nn.Module):
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
        features: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv3D(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        # Final 1x1x1 voxel-wise classification head
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

            # Concatenate along channel dimension: (B, C, D, H, W)
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # DoubleConv3D

        return self.final_conv(x)


# ==============================================================================
# 2. SAMPLE 3D VOLUMETRIC DATASET
# ==============================================================================
class Synthetic3DVolumeDataset(Dataset):
    """
    Generates synthetic 3D volumetric scans containing 1 to 3 noisy 3D ellipsoids
    (simulating organs/lesions in MRI/CT) and binary ground-truth masks.
    """
    def __init__(self, num_samples: int = 80, volume_size: tuple[int, int, int] = (64, 64, 64), seed: int = 42):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.seed = seed

        # Pre-compute normalized coordinate grid [-1, 1] for fast ellipsoid generation
        d, h, w = volume_size
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        self.grid = (zz, yy, xx)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        zz, yy, xx = self.grid
        mask = torch.zeros_like(zz)
        volume = torch.zeros_like(zz)

        num_blobs = int(torch.randint(1, 4, (1,), generator=g).item())
        for _ in range(num_blobs):
            cz = torch.empty(1).uniform_(-0.55, 0.55, generator=g).item()
            cy = torch.empty(1).uniform_(-0.55, 0.55, generator=g).item()
            cx = torch.empty(1).uniform_(-0.55, 0.55, generator=g).item()

            rz = torch.empty(1).uniform_(0.15, 0.35, generator=g).item()
            ry = torch.empty(1).uniform_(0.15, 0.35, generator=g).item()
            rx = torch.empty(1).uniform_(0.15, 0.35, generator=g).item()

            ellipsoid = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
            intensity = torch.empty(1).uniform_(0.65, 1.0, generator=g).item()

            mask = torch.maximum(mask, ellipsoid.float())
            volume = torch.maximum(volume, ellipsoid.float() * intensity)

        # Add smooth spatial bias field + Gaussian noise to simulate real scanner artifacts
        bias_field = 0.25 * (zz + yy)
        noise = torch.randn(self.volume_size, generator=g) * 0.22
        volume = volume + bias_field + noise

        # Normalize per volume (z-score normalization)
        volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        # Add channel dimension -> (1, D, H, W)
        return volume.unsqueeze(0), mask.unsqueeze(0)


# ==============================================================================
# 3. 3D SEGMENTATION LOSS & METRIC (BCE + SOFT DICE)
# ==============================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        # Flatten spatial and channel dims per batch item: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()
        return 0.5 * bce_loss + 0.5 * dice_loss


@torch.no_grad()
def compute_dice_metric(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    if union == 0:
        return 1.0
    return ((2.0 * intersection) / (union + 1e-6)).item()


# ==============================================================================
# 4. GPU TRAINING & VALIDATION LOOP
# ==============================================================================
def train():
    # Hyperparameters
    volume_size = (64, 64, 64)  # Must be divisible by 8 (2^3 pooling layers)
    batch_size = 4
    epochs = 12
    learning_rate = 1e-3
    checkpoint_path = "unet3d_best.pth"

    # Configure GPU device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected! Verify NVIDIA drivers and CUDA-enabled PyTorch.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimizes 3D convolution kernels for fixed input sizes
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Datasets and DataLoaders
    train_ds = Synthetic3DVolumeDataset(num_samples=80, volume_size=volume_size, seed=100)
    val_ds = Synthetic3DVolumeDataset(num_samples=20, volume_size=volume_size, seed=999)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # Speeds up CPU -> GPU host-to-device memory transfer
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize 3D Model, Loss, Optimizer, and Mixed Precision Scaler
    model = SimpleUNet3D(in_channels=1, out_channels=1, features=(32, 64, 128)).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    # Quick shape sanity check
    sample_x = torch.randn(2, 1, *volume_size, device=device)
    with torch.amp.autocast("cuda"):
        sample_y = model(sample_x)
    print(f"Shape Check: Input {tuple(sample_x.shape)} -> Output {tuple(sample_y.shape)}\n")

    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for volumes, masks in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass in FP16/BF16 mixed precision to halve 3D activation VRAM
            with torch.amp.autocast("cuda"):
                logits = model(volumes)
                loss = criterion(logits, masks)

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * volumes.size(0)

        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for volumes, masks in val_loader:
                volumes = volumes.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(volumes)
                    loss = criterion(logits, masks)

                val_loss += loss.item() * volumes.size(0)
                val_dice += compute_dice_metric(logits, masks) * volumes.size(0)

        val_loss /= len(val_ds)
        val_dice /= len(val_ds)
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"\nTraining complete! Best Val Dice: {best_val_dice:.4f}")
    print(f"Peak GPU VRAM Used: {peak_vram_mb:.1f} MB | Saved weights to '{checkpoint_path}'")


if __name__ == "__main__":
    train()

