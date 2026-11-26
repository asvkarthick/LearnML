import torch
import torch.nn.functional as F

# 1. Define constants (e.g., 4 classes: 0=Background, 1=Class A, 2=Class B, 3=Class C)
NUM_CLASSES = 4


def dice_loss(logits, target, num_classes=NUM_CLASSES, eps=1e-6):
  # Convert raw model logits to class probabilities along the channel dimension (dim=1)
  probs = F.softmax(logits, dim=1)

  # Convert integer target (B, D, H, W) to one-hot (B, D, H, W, C)
  # Then permute to match probs shape: (B, C, D, H, W)
  target_oh = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

  # Sum over Batch (0), Depth (2), Height (3), and Width (4)
  dims = (0, 2, 3, 4)
  inter = (probs * target_oh).sum(dim=dims)
  union = probs.sum(dim=dims) + target_oh.sum(dim=dims)

  # Compute per-class Dice score
  dice = (2.0 * inter + eps) / (union + eps)

  # Exclude background class (index 0) and return loss
  return 1.0 - dice[1:].mean()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
  torch.manual_seed(42)

  # Dummy 3D volume dimensions: Batch=2, Depth=16, Height=32, Width=32
  B, D, H, W = 2, 16, 32, 32

  # 1. Simulated raw output from a 3D segmentation model (requires_grad=True for backprop)
  logits = torch.randn(B, NUM_CLASSES, D, H, W, requires_grad=True)

  # 2. Simulated ground truth mask with class indices in range [0, NUM_CLASSES - 1]
  # Note: target must be torch.long (int64) for F.one_hot()
  target = torch.randint(0, NUM_CLASSES, (B, D, H, W), dtype=torch.long)

  # 3. Calculate loss
  loss = dice_loss(logits, target, num_classes=NUM_CLASSES)
  print(f"Random Prediction Dice Loss: {loss.item():.4f}")

  # 4. Verify backpropagation works
  loss.backward()
  print(f"Gradients computed successfully: {logits.grad is not None}")

  # 5. Sanity check: Near-perfect prediction should yield loss close to 0.0
  # Create logits heavily favoring the ground truth classes
  perfect_logits = (
      F.one_hot(target, NUM_CLASSES).permute(0, 4, 1, 2, 3).float() * 20.0
  )
  perfect_loss = dice_loss(perfect_logits, target, num_classes=NUM_CLASSES)
  print(f"Perfect Prediction Dice Loss: {perfect_loss.item():.6f}")


