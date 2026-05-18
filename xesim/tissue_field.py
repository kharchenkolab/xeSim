"""Tissue-type field at coarse resolution (multi-scale tissue composition).

The Gibbs neighbor model captures local same-type clustering but not
larger-order structure (islet vs acinar vs stromal regions). This module
learns a generative model over the COARSE tissue-type composition of a
tile — what fraction of each cell type is present in each ~5-10-cell
region — and uses sampled coarse fields as a per-position type prior
when synthesizing new scenes.

Coarse field representation: (T, K, K) where T is number of cell types
and K is the coarse resolution (default 16). Each cell f[t, i, j] is the
fraction of pixels in the corresponding tile region (of size H/K × W/K)
that belong to a cell of type t. Fractions sum to ≤1; the remainder is
extracellular.

Pipeline:
  - `build_coarse_field`: real cell_label + per-cell types -> coarse field
  - `TissueFieldVAE`: tiny convolutional VAE on coarse fields
  - `sample_coarse_field`: draw a sample from the trained VAE
  - `coarse_field_to_position_prior`: upsample to per-pixel type prior
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_coarse_field(cell_label: np.ndarray,
                        cell_to_type: dict[int, int],
                        n_types: int,
                        coarse_size: int = 16,
                        include_background: bool = False) -> np.ndarray:
    """Build a (T, K, K) coarse type-fraction field from a real tile.

    Args:
      cell_label: (H, W) integer cell label map.
      cell_to_type: {cell_label: type_index_in_0..n_types-1}.
      n_types: number of cell types (including background as type 0 if used).
      coarse_size: K, the coarse resolution per side.
      include_background: if True, channel 0 = extracellular fraction.
        If False, channels are types 1..n_types-1 only and the field doesn't
        sum to 1 (sum ≤ 1, remainder is background).

    Returns:
      (T, K, K) float32 array. If include_background=True, T = n_types;
      else T = n_types - 1.
    """
    H, W = cell_label.shape
    assert H % coarse_size == 0 and W % coarse_size == 0, "tile shape must divide coarse_size"
    bs_h = H // coarse_size
    bs_w = W // coarse_size

    # Per-pixel type field
    ty_field = np.zeros((H, W), dtype=np.int32)
    for L, t in cell_to_type.items():
        ty_field[cell_label == int(L)] = int(t)

    start_t = 0 if include_background else 1
    T_out = n_types - start_t
    out = np.zeros((T_out, coarse_size, coarse_size), dtype=np.float32)
    # Vectorized: reshape into blocks
    blocks = ty_field.reshape(coarse_size, bs_h, coarse_size, bs_w).transpose(0, 2, 1, 3)
    # blocks[i, j] is a bs_h x bs_w patch
    block_size = bs_h * bs_w
    for t in range(start_t, n_types):
        out[t - start_t] = (blocks == t).sum(axis=(-2, -1)) / block_size
    return out


class TissueFieldVAE(nn.Module):
    """Tiny convolutional VAE over (T, K, K) coarse type fields.

    Default K=16, T=7 (cell types, no background channel).
    Latent dim 8 by default.
    """

    def __init__(self, n_types: int = 7, coarse_size: int = 16,
                 latent_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.n_types = n_types
        self.coarse_size = coarse_size
        self.latent_dim = latent_dim
        # Encoder: T x 16 x 16 -> latent
        self.enc = nn.Sequential(
            nn.Conv2d(n_types, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU(),  # 8x8
            nn.Conv2d(hidden, hidden*2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden*2, hidden*2, 3, stride=2, padding=1), nn.GELU(),  # 4x4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(hidden*2, latent_dim)
        self.logsig = nn.Linear(hidden*2, latent_dim)
        # Decoder: latent -> T x 16 x 16
        self.dec_fc = nn.Linear(latent_dim, hidden*2 * 4 * 4)
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(hidden*2, hidden*2, 4, stride=2, padding=1), nn.GELU(),  # 8x8
            nn.Conv2d(hidden*2, hidden, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(),  # 16x16
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, n_types, 1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logsig(h).clamp(-4, 2)

    def decode(self, z):
        h = self.dec_fc(z).view(z.shape[0], -1, 4, 4)
        logits = self.dec(h)
        # Each cell's prediction is a 7-vector of fractions, with implicit
        # background. We use sigmoid so each fraction is in [0, 1] and
        # they may not sum to 1 (remaining mass = extracellular).
        return torch.sigmoid(logits)

    def forward(self, x):
        mu, logsig = self.encode(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logsig) * eps
        recon = self.decode(z)
        return recon, mu, logsig

    @torch.no_grad()
    def sample(self, n: int = 1, device=None) -> torch.Tensor:
        """Sample n coarse fields from the prior. Returns (n, T, K, K)."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def coarse_field_to_position_prior(coarse_field: np.ndarray,
                                     target_h: int = 256,
                                     target_w: int = 256,
                                     include_background_channel: bool = False,
                                     smoothing: float = 0.05) -> np.ndarray:
    """Upsample a (T, K, K) coarse field to (T, H, W) per-pixel type prior.

    Bilinear upsample, then optionally add smoothing toward uniform.
    Returns probabilities per type at each pixel (without normalization;
    caller normalizes as needed for Gibbs).
    """
    import torch
    cf = torch.from_numpy(coarse_field).unsqueeze(0).float()
    up = F.interpolate(cf, size=(target_h, target_w), mode="bilinear", align_corners=False)
    out = up.squeeze(0).numpy()
    if smoothing > 0:
        out = out * (1.0 - smoothing) + smoothing / out.shape[0]
    return out
