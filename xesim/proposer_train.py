"""Stage E-prime: train a small UNet proposer from the ablation-pair corpus.

Predicts per-pixel:
- ``present`` ∈ [0, 1]  (sigmoid)
- ``offsets``  Δy, Δx in pixels (linear)
- ``type_logits`` over n_types (softmax; logits[0] = "no cell of any type" / unknown)

Loss = BCE(present) + λ_off * Smooth-L1(offsets, masked) + λ_type * CE(type, masked)
where the masking restricts offset and type losses to pixels inside ablated
cells (present == 1) so that easy negative pixels don't dominate.

Training uses a small two-level UNet (~400k params) and AdamW. Decoding to
discrete cell candidates is done by NMS on a peakiness map = present *
(centroid-vote density).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class _Conv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
        )
    def forward(self, x): return self.body(x)


class TinyProposerUNet(nn.Module):
    """Small two-level UNet with three heads."""

    def __init__(self, in_channels: int = 4, hidden: int = 32, n_types_plus_bg: int = 8):
        super().__init__()
        self.enc1 = _Conv(in_channels, hidden)
        self.enc2 = _Conv(hidden, hidden * 2)
        self.bot = _Conv(hidden * 2, hidden * 4)
        self.up2 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 2, stride=2)
        self.dec2 = _Conv(hidden * 4, hidden * 2)
        self.up1 = nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2)
        self.dec1 = _Conv(hidden * 2, hidden)
        self.head_present = nn.Conv2d(hidden, 1, 1)
        self.head_offsets = nn.Conv2d(hidden, 2, 1)
        self.head_type = nn.Conv2d(hidden, n_types_plus_bg, 1)

    def forward(self, x):
        # Pad to multiple of 4 so two 2x downsamples + 2x upsamples line up exactly.
        H, W = x.shape[-2:]
        pad_h = (-H) % 4
        pad_w = (-W) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        b = self.bot(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = {
            "present_logits": self.head_present(d1)[..., :H, :W],
            "offsets": self.head_offsets(d1)[..., :H, :W],
            "type_logits": self.head_type(d1)[..., :H, :W],
        }
        return out


@dataclass
class _CorpusSplit(Dataset):
    inputs: np.ndarray  # (N, 4, H, W) float32 (ch0-1 continuous intensities)
    present: np.ndarray  # (N, H, W) uint8
    offsets: np.ndarray  # (N, 2, H, W) float16
    type_idx: np.ndarray  # (N, H, W) uint8
    augment: bool = False

    def __len__(self): return self.inputs.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.inputs[i].astype(np.float32))
        p = torch.from_numpy(self.present[i].astype(np.float32)).unsqueeze(0)
        o = torch.from_numpy(self.offsets[i].astype(np.float32))
        t = torch.from_numpy(self.type_idx[i].astype(np.int64))
        if self.augment:
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2]); p = torch.flip(p, dims=[2])
                o = torch.flip(o, dims=[2]); o[1] = -o[1]
                t = torch.flip(t, dims=[1])
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[1]); p = torch.flip(p, dims=[1])
                o = torch.flip(o, dims=[1]); o[0] = -o[0]
                t = torch.flip(t, dims=[0])
        return x, p, o, t


def _focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE: down-weights easy examples (large gap between logit and label)
    and upweights hard ones via (1 - p_t)^gamma. ``pos_weight`` re-weights the
    positive class for class imbalance (passed through to the underlying BCE).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    return ((1 - p_t).clamp(min=1e-6).pow(gamma) * bce).mean()


def train_proposer(
    corpus_path: Path,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 2e-3,
    hidden: int = 32,
    lambda_offsets: float = 0.05,
    lambda_type: float = 0.5,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    seed: int = 0,
) -> dict[str, Any]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    arr = np.load(corpus_path, allow_pickle=True)
    splits = arr["splits"].astype(str)
    type_names = arr["type_names"].astype(str).tolist()
    n_types_plus_bg = len(type_names)  # 0 = unknown/no-cell
    in_channels = arr["inputs"].shape[1]

    train_mask = splits == "train"
    val_mask = splits == "val"
    train_set = _CorpusSplit(
        inputs=arr["inputs"][train_mask],
        present=arr["target_present"][train_mask],
        offsets=arr["target_offsets"][train_mask],
        type_idx=arr["target_type_index"][train_mask],
        augment=True,
    )
    val_set = _CorpusSplit(
        inputs=arr["inputs"][val_mask],
        present=arr["target_present"][val_mask],
        offsets=arr["target_offsets"][val_mask],
        type_idx=arr["target_type_index"][val_mask],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model = TinyProposerUNet(in_channels=in_channels, hidden=hidden, n_types_plus_bg=n_types_plus_bg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class imbalance: positive (cell_present) pixels are rare → weighted BCE.
    pos_weight = torch.tensor([(1 - arr["target_present"][train_mask].mean()) / max(arr["target_present"][train_mask].mean(), 1e-6)], device=device)

    history = []
    best_val = float("inf")
    best_state = None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0; n_batches = 0
        for x, p, o, t in train_loader:
            x = x.to(device); p = p.to(device); o = o.to(device); t = t.to(device)
            out = model(x)
            loss_present = (
                _focal_bce_with_logits(out["present_logits"], p, pos_weight=pos_weight, gamma=focal_gamma)
                if use_focal
                else F.binary_cross_entropy_with_logits(out["present_logits"], p, pos_weight=pos_weight)
            )
            mask = (p > 0.5).float()
            denom = mask.sum().clamp(min=1.0)
            loss_off = (F.smooth_l1_loss(out["offsets"], o, reduction="none") * mask).sum() / denom
            loss_type_full = F.cross_entropy(out["type_logits"], t, reduction="none")
            loss_type = (loss_type_full * mask.squeeze(1)).sum() / denom
            loss = loss_present + lambda_offsets * loss_off + lambda_type * loss_type
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += float(loss.item()); n_batches += 1
        sched.step()

        model.eval()
        val_loss = 0.0; vb = 0
        with torch.no_grad():
            for x, p, o, t in val_loader:
                x = x.to(device); p = p.to(device); o = o.to(device); t = t.to(device)
                out = model(x)
                lp = F.binary_cross_entropy_with_logits(out["present_logits"], p, pos_weight=pos_weight)
                mask = (p > 0.5).float()
                denom = mask.sum().clamp(min=1.0)
                lo = (F.smooth_l1_loss(out["offsets"], o, reduction="none") * mask).sum() / denom
                lt = (F.cross_entropy(out["type_logits"], t, reduction="none") * mask.squeeze(1)).sum() / denom
                val_loss += float((lp + lambda_offsets*lo + lambda_type*lt).item()); vb += 1
        val_avg = val_loss / max(vb, 1)
        train_avg = train_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_avg, "val_loss": val_avg})
        print(f"epoch {epoch:3d}  train_loss={train_avg:.4f}  val_loss={val_avg:.4f}")
        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt = {
        "model": {"in_channels": in_channels, "hidden": hidden, "n_types_plus_bg": n_types_plus_bg},
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "type_names": type_names,
        "history": history,
        "best_val_loss": best_val,
    }
    out_path = output_dir / "proposer.pt"
    torch.save(ckpt, out_path)
    with (output_dir / "training_summary.json").open("w") as fh:
        json.dump({
            "best_val_loss": best_val,
            "epochs": epochs,
            "history": history,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        }, fh, indent=2)
    return {"best_val_loss": best_val, "checkpoint": str(out_path)}
