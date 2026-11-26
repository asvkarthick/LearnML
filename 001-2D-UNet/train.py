import torch
import torch.nn as nn
from tqdm import tqdm

from model import SimpleUNet
from dataset import get_dataloaders


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Computes batch Dice Coefficient and Intersection-over-Union (IoU)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union - intersection + 1e-6)).mean().item()
    return dice, iou


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass on NVIDIA GPU
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        dice, iou = compute_metrics(logits, masks)
        running_loss += loss.item() * images.size(0)
        total_dice += dice * images.size(0)
        total_iou += iou * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def main():
    # Hyperparameters
    epochs = 15
    batch_size = 32
    learning_rate = 1e-3
    image_size = 128

    # Configure NVIDIA GPU
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU not detected. Verify driver and PyTorch CUDA installation.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # Data, Model, Loss, Optimizer
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, image_size=image_size)
    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        # Save best model checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_unet_pet.pth")
            print(f"  --> Saved new best model (Dice: {best_dice:.4f})")


if __name__ == "__main__":
    main()

