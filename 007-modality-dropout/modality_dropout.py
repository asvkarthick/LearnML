import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Previous building blocks ---
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# --- Your new blocks ---
class ModalityDropout(nn.Module):
    """Randomly drop entire modalities during training (arXiv:1908.06683 Sec 3.1)."""
    def __init__(self, p=0.15, max_drop=2):
        super().__init__()
        self.p = p
        self.max_drop = max_drop

    def forward(self, mods: list[torch.Tensor], training=True):
        if not training or torch.rand(1).item() > self.p:
            return mods

        num_mods = len(mods)
        k = torch.randint(1, min(self.max_drop, num_mods - 1) + 1, (1,)).item()
        drop_idx = set(torch.randperm(num_mods)[:k].tolist())

        return [
            torch.zeros_like(m) if i in drop_idx else m
            for i, m in enumerate(mods)
        ]

class ModalityEncoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = EncoderBlock(1, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.enc4 = EncoderBlock(base * 4, base * 8)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        return x, (s1, s2, s3, s4)

# --- Step-by-step execution ---
torch.manual_seed(42)

# 1. Create 4 separate MRI modalities: each is (Batch=1, Channels=1, 32x32x32)
t1    = torch.ones(1, 1, 32, 32, 32)
t1ce  = torch.ones(1, 1, 32, 32, 32) * 2
t2    = torch.ones(1, 1, 32, 32, 32) * 3
flair = torch.ones(1, 1, 32, 32, 32) * 4
modalities = [t1, t1ce, t2, flair]

# 2. Apply ModalityDropout (using p=1.0 here just to guarantee a drop in this demo)
dropout = ModalityDropout(p=1.0, max_drop=2)
dropped_mods = dropout(modalities, training=True)

print("--- Step 1: Modality Dropout ---")
names = ["T1", "T1ce", "T2", "FLAIR"]
for name, m in zip(names, dropped_mods):
    status = "DROPPED (all zeros)" if m.sum() == 0 else "KEPT"
    print(f"{name:5s}: {status}")

# 3. Encode each modality through its own ModalityEncoder (base=16 to keep memory small)
encoders = nn.ModuleList([ModalityEncoder(base=16) for _ in range(4)])

bottlenecks = []
skips_per_mod = []

for enc, mod in zip(encoders, dropped_mods):
    b_feat, skips = enc(mod)
    bottlenecks.append(b_feat)
    skips_per_mod.append(skips)

print("\n--- Step 2: Single ModalityEncoder Shapes (base=16) ---")
s1, s2, s3, s4 = skips_per_mod[0]
print(f"s1 (enc1 skip): {s1.shape} -> 16 ch, 32x32x32")
print(f"s2 (enc2 skip): {s2.shape} -> 32 ch, 16x16x16")
print(f"s3 (enc3 skip): {s3.shape} -> 64 ch, 8x8x8")
print(f"s4 (enc4 skip): {s4.shape} -> 128 ch, 4x4x4")
print(f"Bottleneck x:   {bottlenecks[0].shape} -> 128 ch, 2x2x2")

# 4. Fuse features across modalities (e.g., via mean/sum) and pass to DecoderBlock
fused_bottleneck = torch.stack(bottlenecks).mean(dim=0) # (1, 128, 2, 2, 2)
fused_s4 = torch.stack([s[3] for s in skips_per_mod]).mean(dim=0) # (1, 128, 4, 4, 4)

# Note: In your DecoderBlock, in_ch must equal 2 * out_ch because of torch.cat([skip, x])
# Since fused_s4 has 128 channels, we project fused_bottleneck to 256 channels first:
center = DoubleConv3D(128, 256)(fused_bottleneck) # (1, 256, 2, 2, 2)
dec4 = DecoderBlock(in_ch=256, out_ch=128)
out_dec4 = dec4(center, fused_s4)

print("\n--- Step 3: Fusion & Decoding ---")
print(f"Fused bottleneck: {fused_bottleneck.shape}")
print(f"Center block:     {center.shape}")
print(f"Decoder 4 output: {out_dec4.shape} (matches s4 resolution!)")


