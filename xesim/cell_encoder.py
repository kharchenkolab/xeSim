"""Per-cell VAE appearance encoder.

A small CNN that encodes a 48×48 RGB crop of a single cell (masked to its
own footprint) into a per-cell latent vector. Used at training time to
provide per-cell appearance information to the renderer; at inference
time in explain-mode (encoding real cells) and replaced by N(0, I) in
generate-mode (forward synthesis).

Architecture: 4-channel input (RGB + binary self-mask) + per-type
embedding broadcast to spatial map; 5-layer conv encoder with 2 stride-2
downsamples; AdaptiveAvgPool to 1×1; two heads emit mu and log_sigma of
size `latent_dim`. The same crop size (48 px) is used at training and
inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CROP_SIZE = 48


class CellEncoder(nn.Module):
    def __init__(self, n_types: int = 7, latent_dim: int = 4, hidden: int = 32,
                  in_channels: int = 4):
        """in_channels = N_image_channels + 1 (mask). Default 4 = 3 RGB + mask
        for backward compat with v16-style 3-channel inputs.
        """
        super().__init__()
        self.type_embed = nn.Embedding(n_types + 1, 8)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels + 8, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 4, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mu = nn.Linear(hidden * 4, latent_dim)
        self.logsig = nn.Linear(hidden * 4, latent_dim)

    def forward(self, x: torch.Tensor, type_idx: torch.Tensor):
        emb = self.type_embed(type_idx)
        emb_map = emb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        h = self.enc(torch.cat([x, emb_map], dim=1)).flatten(1)
        mu = self.mu(h)
        logsig = self.logsig(h).clamp(-4, 2)
        return mu, logsig


def extract_cell_crops(real, cell_label, crop_size: int = CROP_SIZE):
    """Extract a per-cell (3, crop_size, crop_size) RGB crop centered on the
    cell centroid + a (crop_size, crop_size) binary self-mask for each cell.

    Returns (crops, masks) with shapes (N, 3, K, K) and (N, K, K).

    Vectorized with scipy.ndimage.find_objects + scipy.ndimage.center_of_mass:
    one pass over the label image instead of N. On a 2353×2353 image with
    1700+ cells this cut ~31 s → ~3 s in profiling.
    """
    import numpy as np
    from scipy.ndimage import find_objects, center_of_mass

    h, w = cell_label.shape
    nz = np.unique(cell_label); nz = nz[nz > 0]
    half = crop_size // 2
    n_ch = real.shape[0]

    # Cellpose / find_objects requires contiguous labels 1..max. If the
    # input cell_label has gaps, compact-relabel for the bbox lookup, then
    # walk the original labels in their natural order.
    max_lbl = int(nz.max())
    bboxes = find_objects(cell_label, max_label=max_lbl)
    centroids = center_of_mass(cell_label > 0, labels=cell_label,
                                  index=nz.tolist() if len(nz) else [])

    crops = np.zeros((len(nz), n_ch, crop_size, crop_size), dtype=np.float32)
    masks = np.zeros((len(nz), crop_size, crop_size), dtype=np.float32)
    for i, lbl in enumerate(nz):
        bb = bboxes[int(lbl) - 1] if int(lbl) - 1 < len(bboxes) else None
        if bb is None:
            continue
        cy_, cx_ = centroids[i]
        cy = int(round(cy_)); cx = int(round(cx_))
        y0 = max(0, cy - half); y1 = min(h, cy + half)
        x0 = max(0, cx - half); x1 = min(w, cx + half)
        # In-bbox label mask, then place into the crop window centred on
        # centroid.
        ys0, ys1 = max(bb[0].start, y0), min(bb[0].stop, y1)
        xs0, xs1 = max(bb[1].start, x0), min(bb[1].stop, x1)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        sub = cell_label[ys0:ys1, xs0:xs1] == int(lbl)
        if not sub.any():
            continue
        py0 = half - (cy - ys0); px0 = half - (cx - xs0)
        py1 = py0 + (ys1 - ys0); px1 = px0 + (xs1 - xs0)
        own = sub.astype(np.float32)
        crops[i, :, py0:py1, px0:px1] = real[:, ys0:ys1, xs0:xs1] * own[None, :, :]
        masks[i, py0:py1, px0:px1] = own
    return crops, masks
