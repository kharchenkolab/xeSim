"""Bridge-tile rendering: render a coherent scene through TWO overlapping
tile grids offset by half a tile, weight each tile's contribution by
distance-to-tile-center, accumulate.

Why: pure feather-stitching averages two equally-bad edge samples at
each seam. With a bridge grid, every output pixel is sourced primarily
from a tile where it's INTERIOR (its convolutions saw full real
context). The conv-padding edge artifact disappears.

Cost: 2× the renderer forward calls vs single-grid stitching. Memory:
bounded by a single tile's render cost; the accumulator buffer is
O(scene_area).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..mechanistic_scene import MechanisticScene


def _cos2_weight(tile_px: int) -> np.ndarray:
    """Per-pixel weight: cos²(π/2 · r/half) where r = max(|Δy|, |Δx|)
    from tile center. 1 at center, 0 at edges (Chebyshev distance)."""
    half = tile_px / 2.0
    coords = np.abs(np.arange(tile_px, dtype=np.float32) - (half - 0.5)) / half
    coords = np.clip(coords, 0.0, 1.0)
    # Use cosine taper so the weight smoothly hits 0 at the boundary.
    w1d = np.cos(coords * (np.pi / 2.0)) ** 2
    # Chebyshev (max of x/y) weight = outer min of 1D weights
    return np.minimum.outer(w1d, w1d).astype(np.float32)


def _tile_origins(extent_px: int, tile_px: int, offset: int) -> list[int]:
    """Origin coordinates for a tile grid starting at `offset` and spaced
    by `tile_px / 2` (so adjacent tiles overlap by half).

    Bumps the last origin to `extent_px - tile_px` so the final tile
    fits inside the region.
    """
    if tile_px >= extent_px:
        return [0]
    step = tile_px // 2
    origins = []
    o = offset
    while o + tile_px <= extent_px:
        origins.append(o)
        o += step
    if not origins or origins[-1] + tile_px < extent_px:
        last = extent_px - tile_px
        if not origins or last != origins[-1]:
            origins.append(last)
    return origins


def render_with_bridge_tiles(
    model,
    scene: MechanisticScene,
    *,
    tile_size_px: int,
    cell_latents: dict[int, np.ndarray] | None = None,
    seed: int = 0,
    latent_scale: float = 1.0,
    background_mask_sigma: float | None = 3.0,
    use_bridge_grid: bool = True,
    progress: bool = False,
    batch_size: int = 8,
) -> np.ndarray:
    """Render `scene` by tiling the UNet forward pass with bridge-grid
    weighting, returning a `(C, H, W)` float32 image.

    Parameters
    ----------
    model
        Loaded XesimModel.
    scene
        A whole-region MechanisticScene. Must have `cell_label.shape ==
        nucleus_label.shape`.
    tile_size_px
        Per-render tile size in pixels. The renderer is fully
        convolutional and has no norm layers, so this can exceed the
        model's training tile size (256 px for v21) — interior pixels
        get full receptive field context. Recommended: 512 for
        good seam reduction with reasonable memory.
    cell_latents
        Per-cell encoded latents (dict of original_label -> ndarray).
        Held CONSTANT across all tile renders so the same cell looks
        identical in every tile that contains it. This is the key to
        killing the per-tile latent-sampling disconnect.
    seed
        RNG seed for any cells without a latent in `cell_latents`. Held
        constant across tiles — combined with the constant cell set per
        tile (the same cells appear in the same sub-tiles), the random
        latents are stable too (but pass cell_latents for full anchors
        to be safe).
    use_bridge_grid
        If True (default), render the second offset grid for bridge
        coverage of seams. If False, single grid (≈ feather-stitching,
        baseline).
    background_mask_sigma
        Forwarded to model.render.
    progress
        Print per-tile progress.

    Returns
    -------
    image : np.ndarray, shape (C, H, W), float32
    """
    H, W = scene.cell_label.shape
    n_ch = int(model.manifest.get("n_channels", 3))

    # Phase 6: precompute global struct channels + latent LUT once over
    # the whole scene. Per-tile renders just slice instead of recomputing
    # distance transforms / one-hot / etc.
    precomputed = model.precompute_render_inputs(
        scene, cell_latents=cell_latents, seed=seed, latent_scale=latent_scale)

    accum = np.zeros((n_ch, H, W), dtype=np.float32)
    wts = np.zeros((H, W), dtype=np.float32)
    weight = _cos2_weight(tile_size_px)

    grids = [(0, 0)]
    if use_bridge_grid:
        grids.append((tile_size_px // 2, tile_size_px // 2))

    # Collect non-empty tile coords; render in GPU batches.
    tiles_to_render: list[tuple[int, int, MechanisticScene]] = []
    empty_count = 0
    for off_y, off_x in grids:
        y_origins = _tile_origins(H, tile_size_px, off_y)
        x_origins = _tile_origins(W, tile_size_px, off_x)
        for y0 in y_origins:
            y1 = y0 + tile_size_px
            for x0 in x_origins:
                x1 = x0 + tile_size_px
                sub_cell = scene.cell_label[y0:y1, x0:x1].astype(np.int32, copy=False)
                sub_nuc = scene.nucleus_label[y0:y1, x0:x1].astype(np.int32, copy=False)
                if not (sub_cell > 0).any():
                    empty_count += 1
                    continue
                sub_scene = MechanisticScene(
                    image_shape=(tile_size_px, tile_size_px),
                    pixel_size=scene.pixel_size,
                    cell_label=sub_cell,
                    nucleus_label=sub_nuc,
                    cells=scene.cells,
                    scene_id=f"bridge_{y0}_{x0}",
                    provenance=dict(scene.provenance or {}),
                )
                tiles_to_render.append((y0, x0, sub_scene))

    if progress:
        print(f"  bridge_render: {len(tiles_to_render)} non-empty tiles "
              f"({empty_count} empty, skipped) across {len(grids)} grid(s); "
              f"batch_size={batch_size}")

    # Render each tile by slicing the precomputed global inputs.
    for (y0, x0, sub_scene) in tiles_to_render:
        y1 = min(y0 + tile_size_px, H)
        x1 = min(x0 + tile_size_px, W)
        h_eff = y1 - y0
        w_eff = x1 - x0
        tile_out = model.render(
            sub_scene, cell_latents=None, seed=seed,
            latent_scale=latent_scale,
            background_mask_sigma=background_mask_sigma,
            precomputed=precomputed,
            precomputed_slice=(y0, x0, tile_size_px, tile_size_px),
        )
        # tile_out is (C, tile_size_px, tile_size_px); clip to effective region
        accum[:, y0:y1, x0:x1] += tile_out[:, :h_eff, :w_eff] * weight[None, :h_eff, :w_eff]
        wts[y0:y1, x0:x1] += weight[:h_eff, :w_eff]

    return accum / np.maximum(wts[None, :, :], 1e-6)


__all__ = ["render_with_bridge_tiles"]
