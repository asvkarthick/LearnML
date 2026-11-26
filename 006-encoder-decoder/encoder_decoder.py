import torch
import torch.nn as nn
import torch.nn.functional as F

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
        skip = self.conv(x)   # Extract features at current resolution
        down = self.pool(skip) # Shrink spatial size by half
        return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsamples and cuts channels in half (e.g., 64 -> 32)
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        # Takes concatenated channels (skip + upsampled = out_ch + out_ch = in_ch)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([skip, x], dim=1) # Combine along channel dimension
        return self.conv(x)

# --- Run Example 1 ---
# Input: Batch=1, Channels=4 (e.g., 4 MRI modalities), Size=16x16x16
x = torch.randn(1, 4, 16, 16, 16)

encoder = EncoderBlock(in_ch=4, out_ch=32)
decoder = DecoderBlock(in_ch=64, out_ch=32) # Expects 64 channels from a deeper bottleneck

# 1. Encode
skip, down = encoder(x)
print(f"Input shape:      {x.shape}")
print(f"Skip shape:       {skip.shape}  <-- Saved for decoder")
print(f"Downsampled shape:{down.shape}   <-- Passed deeper into network")

# Simulate a bottleneck layer doubling channels from 32 -> 64 at 8x8x8 size
bottleneck = DoubleConv3D(32, 64)(down)
print(f"Bottleneck shape: {bottleneck.shape}")

# 2. Decode
out = decoder(bottleneck, skip)
print(f"Decoder output:   {out.shape}  <-- Restored to original size!")


