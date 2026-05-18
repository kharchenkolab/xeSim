"""Slice-guided synthesis: extract a multi-scale guide from a real tile.

The guide captures large-scale tissue structure (composition at multiple
resolutions, cell density, type marginals) so a synth tile can be
generated that matches the guide on slide/domain level but resamples
all fine details (exact cell positions, individual type assignments,
shapes, appearance latents).

Pipeline:
  guide = extract_guide(real_scene, scales=(4, 8, 16))
  synth_scene = sample_slice_guided_scene(guide, seed=...)

The guide intentionally drops cell-level identity (no individual cell
labels are carried through) so the synth is forced to invent new local
structure within the coarse constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tissue_field import build_coarse_field


@dataclass
class SliceGuide:
    """Multi-scale guide extracted from a real tile."""
    coarse_fields: dict[int, np.ndarray]   # {scale: (T, s, s) fractional type composition}
    density_field: np.ndarray              # (D, D) cell-pixel fraction per block
    type_marginal: np.ndarray              # (T,) global type proportions (excluding background)
    expected_n_cells: int
    type_names: list[str]                  # excluding "unknown" type 0
    tile_shape: tuple[int, int]
    density_scale: int                     # D
    has_nucleus_fraction: float = 0.85     # fraction of cells with a visible nucleus in the guide
    source_id: str | None = None           # for traceability


def extract_guide(scene: Any,
                   type_names: list[str],
                   scales: tuple[int, ...] = (4, 8, 16),
                   density_scale: int = 32,
                   source_id: str | None = None) -> SliceGuide:
    """Build a SliceGuide from a real MechanisticScene.

    Args:
      scene: MechanisticScene with cell_label and per-cell types.
      type_names: full list including 'unknown' at index 0.
      scales: list of coarse resolutions; each gets a (T, s, s) field.
      density_scale: resolution of the density field (D × D blocks).

    Returns: SliceGuide.
    """
    cell_label = scene.cell_label
    H, W = cell_label.shape
    n_types = len(type_names)
    name_to_idx = {n: i for i, n in enumerate(type_names)}
    cell_to_type = {int(c.label): name_to_idx.get(c.cell_type or 'unknown', 0)
                    for c in scene.cells}

    coarse_fields: dict[int, np.ndarray] = {}
    for s in scales:
        cf = build_coarse_field(cell_label, cell_to_type, n_types,
                                  coarse_size=s, include_background=False)
        coarse_fields[int(s)] = cf

    # Density field: per-block fraction of cell-foreground pixels.
    if H % density_scale != 0 or W % density_scale != 0:
        raise ValueError(f"tile shape {(H, W)} must divide density_scale {density_scale}")
    bh = H // density_scale; bw = W // density_scale
    fg = (cell_label > 0).astype(np.float32)
    density = fg.reshape(density_scale, bh, density_scale, bw).transpose(0, 2, 1, 3)
    density_field = density.sum(axis=(-2, -1)) / float(bh * bw)

    # Type marginal: total per-type cell counts, normalized over non-bg cells.
    counts = np.zeros(n_types, dtype=np.float64)
    for c in scene.cells:
        t = name_to_idx.get(c.cell_type or 'unknown', 0)
        counts[t] += 1
    s = counts[1:].sum()
    type_marginal = (counts[1:] / max(s, 1.0)).astype(np.float32)

    n_cells_total = sum(1 for _ in scene.cells)
    n_with_nuc = sum(1 for c in scene.cells if c.nucleus_label is not None)
    has_nuc_frac = n_with_nuc / max(n_cells_total, 1)
    return SliceGuide(
        coarse_fields=coarse_fields,
        density_field=density_field.astype(np.float32),
        type_marginal=type_marginal,
        expected_n_cells=n_cells_total,
        type_names=type_names[1:],
        tile_shape=(H, W),
        density_scale=density_scale,
        has_nucleus_fraction=float(has_nuc_frac),
        source_id=source_id,
    )


def multiscale_log_prior_map(guide: SliceGuide,
                                tile_h: int,
                                tile_w: int,
                                guide_weights: dict[int, float] | None = None,
                                include_type_marginal: bool = True,
                                type_marginal_weight: float = 0.5,
                                smoothing: float = 0.05) -> np.ndarray:
    """Build a (T, H, W) per-pixel log-prior MAP by stacking multi-scale
    coarse-field priors.

    Returned as a full-resolution log-probability map so the downstream
    sampler can sample at any cell centroid without re-doing this work.
    Index 0 of T = "unknown" / background (residual mass).
    """
    import torch
    import torch.nn.functional as F

    T_field = guide.coarse_fields[next(iter(guide.coarse_fields))].shape[0]
    n_types = T_field + 1
    if guide_weights is None:
        guide_weights = {s: 1.0 for s in guide.coarse_fields}

    accum = np.zeros((n_types, tile_h, tile_w), dtype=np.float64)
    for scale, cf in guide.coarse_fields.items():
        w = float(guide_weights.get(scale, 0.0))
        if w == 0.0:
            continue
        cf_t = torch.from_numpy(cf).unsqueeze(0).float()
        up = F.interpolate(cf_t, size=(tile_h, tile_w),
                            mode="bilinear", align_corners=False).squeeze(0).numpy()
        # Per-pixel: type 0 = residual (1 - sum_types), types 1.. = up[t-1]
        row = np.zeros((n_types, tile_h, tile_w), dtype=np.float64)
        row[0] = np.clip(1.0 - up.sum(axis=0), 0.0, None)
        row[1:] = up
        # Normalize per pixel
        s = row.sum(axis=0, keepdims=True).clip(1e-9)
        row = row / s
        row = np.clip(row, 1e-6, None)
        accum += w * np.log(row)

    if include_type_marginal:
        marg = np.zeros(n_types, dtype=np.float64)
        marg[0] = 1e-6
        marg[1:] = np.clip(guide.type_marginal.astype(np.float64), 1e-6, None)
        marg = marg / marg.sum()
        accum += float(type_marginal_weight) * np.log(np.clip(marg, 1e-6, None))[:, None, None]
    if smoothing > 0:
        u = np.log(np.full(n_types, 1.0 / n_types))
        # Add a small uniform offset to every pixel
        accum = (1.0 - smoothing) * accum + smoothing * u[:, None, None] * sum(guide_weights.values())
    return accum
