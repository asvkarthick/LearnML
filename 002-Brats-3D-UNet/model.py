import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path

# BraTS 2024 Post-Treatment Glioma (BraTS-GLI) constants
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]  # Native T1, T1c, T2w, T2-FLAIR
# 0: Background, 1: NETC, 2: SNFH (Edema), 3: ET, 4: RC (Resection Cavity)
NUM_CLASSES = 5


class DoubleConv3D(nn.Module):
    """Two 3x3x3 conv -> GroupNorm -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, out_ch)
        g1 = out_ch // 2 if out_ch % g1 != 0 else g1
        g1 = max(1, g1)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


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
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


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


class UnifiedRepresentationUNet(nn.Module):
    """URN variant: 4 encoders -> mean fusion -> shared decoder."""
    def __init__(self, base=32, num_modalities=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_mods = num_modalities
        self.mod_encoders = nn.ModuleList([
            ModalityEncoder(base=base) for _ in range(num_modalities)
        ])
        self.mod_dropout = ModalityDropout(p=0.15, max_drop=2)

        self.fusion = nn.Sequential(
            nn.Conv3d(base * 8, base * 8, 1, bias=False),
            nn.GroupNorm(8, base * 8),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = DoubleConv3D(base * 8, base * 16)

        self.dec4 = DecoderBlock(base * 16, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2)
        self.dec1 = DecoderBlock(base * 2, base)

        self.final = nn.Conv3d(base, num_classes, 1)

    def _mean_fuse(self, tensor_list, active_mask):
        stacked = torch.stack(tensor_list, dim=0)  # [M, B, C, D, H, W]
        denom = active_mask.sum(dim=0).clamp(min=1).view(-1, 1, 1, 1, 1)
        return (stacked * active_mask.view(self.num_mods, -1, 1, 1, 1, 1)).sum(dim=0) / denom

    def forward(self, x, training=True, present_mask=None):
        mods = [x[:, i:i + 1] for i in range(self.num_mods)]

        if present_mask is not None:
            mods = [
                m * pm.view(-1, 1, 1, 1, 1)
                for m, pm in zip(mods, present_mask.unbind(dim=1))
            ]

        mods = self.mod_dropout(mods, training=training)

        active_mask = torch.stack(
            [(m.abs().sum(dim=(1, 2, 3, 4)) > 0).float() for m in mods], dim=0
        )

        feats, skips_per_mod = [], []
        for enc, m in zip(self.mod_encoders, mods):
            f, skips = enc(m)
            feats.append(f)
            skips_per_mod.append(skips)

        fused = self._mean_fuse(feats, active_mask)
        s1 = self._mean_fuse([s[0] for s in skips_per_mod], active_mask)
        s2 = self._mean_fuse([s[1] for s in skips_per_mod], active_mask)
        s3 = self._mean_fuse([s[2] for s in skips_per_mod], active_mask)
        s4 = self._mean_fuse([s[3] for s in skips_per_mod], active_mask)

        fused = self.fusion(fused)
        x = self.bottleneck(fused)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.final(x)


class BraTS2024Dataset(Dataset):
    def __init__(self, root: Path, split="train", patch_size=128):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.cases = sorted([p for p in self.root.glob("BraTS-*") if p.is_dir()])
        if not self.cases:
            self.cases = sorted([p for p in self.root.glob("*") if p.is_dir()])

        n = len(self.cases)
        if split == "train":
            self.cases = self.cases[:int(n * 0.85)]
        else:
            self.cases = self.cases[int(n * 0.85):]

    @staticmethod
    def _resolve_file(case_dir: Path, suffix: str) -> Path | None:
        prefixed = case_dir / f"{case_dir.name}-{suffix}.nii.gz"
        if prefixed.exists():
            return prefixed
        plain = case_dir / f"{suffix}.nii.gz"
        if plain.exists():
            return plain
        return None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        raw_vols = []
        present_mask = []
        ref_shape = None

        for m in MODALITIES:
            p = self._resolve_file(case, m)
            if p is not None:
                img = nib.load(str(p)).get_fdata(dtype=np.float32)
                ref_shape = img.shape
                present = 1.0
            else:
                img = None
                present = 0.0
            raw_vols.append(img)
            present_mask.append(present)

        if ref_shape is None:
            raise FileNotFoundError(f"No MRI modality files found in {case}")

        # Replace any missing modality with zeros matching actual scan dimensions
        raw_vols = [
            v if v is not None else np.zeros(ref_shape, dtype=np.float32)
            for v in raw_vols
        ]

        seg_p = self._resolve_file(case, "seg")
        if seg_p is None:
            raise FileNotFoundError(f"Segmentation mask missing in {case}")
        seg = nib.load(str(seg_p)).get_fdata(dtype=np.float32).astype(np.int64)
        seg = np.clip(seg, 0, NUM_CLASSES - 1)

        # 1. Z-score normalize each present modality inside non-zero brain mask
        vols = []
        for vol, pres in zip(raw_vols, present_mask):
            if pres > 0:
                mask = vol > 0
                if mask.sum() > 0:
                    mean = vol[mask].mean()
                    std = vol[mask].std() + 1e-8
                    vol = (vol - mean) / std
                    vol[~mask] = 0.0
            vols.append(vol)

        # 2. Crop 128^3 patch (foreground-centered during train, tumor-centroid during val)
        ps = self.patch_size
        sx, sy, sz = ref_shape
        fg_coords = np.argwhere(seg > 0)

        if len(fg_coords) > 0 and (self.split == "train" and np.random.rand() < 0.8):
            center = fg_coords[np.random.randint(len(fg_coords))]
            cx, cy, cz = center
        elif len(fg_coords) > 0 and self.split != "train":
            cx, cy, cz = fg_coords.mean(axis=0).astype(int)
        else:
            cx, cy, cz = sx // 2, sy // 2, sz // 2

        x0 = int(np.clip(cx - ps // 2, 0, max(0, sx - ps)))
        y0 = int(np.clip(cy - ps // 2, 0, max(0, sy - ps)))
        z0 = int(np.clip(cz - ps // 2, 0, max(0, sz - ps)))

        def crop_or_pad(arr):
            sub = arr[x0:x0 + ps, y0:y0 + ps, z0:z0 + ps]
            pad_x = max(0, ps - sub.shape[0])
            pad_y = max(0, ps - sub.shape[1])
            pad_z = max(0, ps - sub.shape[2])
            if pad_x > 0 or pad_y > 0 or pad_z > 0:
                sub = np.pad(sub, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")
            return sub

        vols = np.stack([crop_or_pad(v) for v in vols], axis=0)
        seg = crop_or_pad(seg)

        return {
            "image": torch.from_numpy(vols).float(),
            "label": torch.from_numpy(seg).long(),
            "present": torch.tensor(present_mask, dtype=torch.float32),
            "case_id": case.name,
        }


def dice_loss(logits, target, num_classes=NUM_CLASSES, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * target_oh).sum(dim=dims)
    union = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice[1:].mean()


def train_epoch(model, loader, optim, device, scaler):
    model.train()
    tot = 0.0
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        present = batch["present"].to(device)

        optim.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(img, training=True, present_mask=present)
            ce = F.cross_entropy(logits, lbl)
            d = dice_loss(logits, lbl)
            loss = 0.5 * ce + 0.5 * d

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        tot += loss.item()
    return tot / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dices = {"WT": [], "TC": [], "ET": [], "RC": []}
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        present = batch["present"].to(device)

        logits = model(img, training=False, present_mask=present)
        pred = logits.argmax(dim=1)

        for b in range(pred.shape[0]):
            p, g = pred[b].cpu().numpy(), lbl[b].cpu().numpy()

            def dice_score(mask_p, mask_g):
                sp, sg = mask_p.sum(), mask_g.sum()
                if sp == 0 and sg == 0:
                    return 1.0
                inter = (mask_p & mask_g).sum()
                return float((2.0 * inter) / (sp + sg + 1e-6))

            # BraTS 2024 Sub-regions:
            # WT = NETC(1) + SNFH(2) + ET(3) | TC = NETC(1) + ET(3) | ET = 3 | RC = 4
            wt_p, wt_g = np.isin(p, [1, 2, 3]), np.isin(g, [1, 2, 3])
            tc_p, tc_g = np.isin(p, [1, 3]), np.isin(g, [1, 3])
            et_p, et_g = (p == 3), (g == 3)
            rc_p, rc_g = (p == 4), (g == 4)

            dices["WT"].append(dice_score(wt_p, wt_g))
            dices["TC"].append(dice_score(tc_p, tc_g))
            dices["ET"].append(dice_score(et_p, et_g))
            dices["RC"].append(dice_score(rc_p, rc_g))

    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in dices.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "/home/ka2642/workspace/IMGS740/data/2024_orig/BraTS-GLI/train"

    train_ds = BraTS2024Dataset(data_root, "train")
    val_ds = BraTS2024Dataset(data_root, "val")

    if len(train_ds) == 0:
        raise RuntimeError(f"No training cases found in {data_root}.")

    # Startup sanity check: ensure sample 0 actually loads non-zero MRI & labels
    sample = train_ds[0]
    assert sample["present"].sum() > 0, f"All modalities missing for {sample['case_id']}"
    print(f"Sanity check passed | Case: {sample['case_id']} | Labels present: {torch.unique(sample['label']).tolist()}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = UnifiedRepresentationUNet(base=32, num_classes=NUM_CLASSES).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=300)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best = 0.0
    for epoch in range(300):
        loss = train_epoch(model, train_loader, optim, device, scaler)
        metrics = validate(model, val_loader, device)
        scheduler.step()

        mean_dice = (metrics["WT"] + metrics["TC"] + metrics["ET"] + metrics["RC"]) / 4.0
        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), "urn_brats2024_best.pth")

        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | "
            f"Dice WT {metrics['WT']:.3f} TC {metrics['TC']:.3f} ET {metrics['ET']:.3f} RC {metrics['RC']:.3f}"
        )


if __name__ == "__main__":
    main()

