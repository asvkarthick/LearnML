import torch
import matplotlib.pyplot as plt
from model import SimpleUNet
from dataset import get_dataloaders


@torch.no_grad()
def visualize_predictions(checkpoint_path: str = "best_unet_pet.pth", num_samples: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=num_samples, image_size=128)

    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    images, masks = next(iter(val_loader))
    logits = model(images.to(device))
    preds = (torch.sigmoid(logits) > 0.5).float().cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("unet_predictions.png", dpi=150)
    print("Saved visualization to unet_predictions.png")


if __name__ == "__main__":
    visualize_predictions()

