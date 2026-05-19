"""Per-z 2D signed-distance-function tessellation.

For each z slice, each cell defines a "preferred mask" = its 2D shape
template, scaled by ellipsoidal taper τ(z) and shifted by (z-z_center)·t_xy/t_z.
A voxel is assigned to the cell with the smallest SDF to its preferred
mask (most-interior point wins; pixels outside every preferred mask go
to the cell whose mask boundary they are closest to).

Computed per-tile to bound memory. Per-cell SDF is restricted to an
extended bounding box; cells far from a pixel don't compete for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CellRecord:
    """All info needed to render a cell at any z slice."""
    cell_idx: int                # unique non-negative integer for cell_label
    cell_id: str
    cell_type: str
    xy_seed: tuple[float, float]
    z_center: float
    z_extent: float
    t_x: float
    t_y: float
    t_z: float
    template_xs: np.ndarray      # relative-to-origin
    template_ys: np.ndarray
    # Optional cell-type-resolver provenance (see xesim.cell_type_resolver).
    # Stays as a dict (or None) here; the 2.5D bundle writer materialises it
    # into cell_type_source / cell_type_confidence / cell_type_evidence
    # columns in ground_truth/cells_synth.parquet. None for legacy code
    # paths that don't run through explain_region's resolver.
    type_resolution: dict | None = None


def _rasterize_polygon(
    xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize a polygon (in pixel coords) into a binary mask of `shape`."""
    from skimage.draw import polygon
    rr, cc = polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Compute signed distance: negative inside, positive outside.

    Uses scipy.ndimage.distance_transform_edt twice (inside and outside).
    """
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if mask.all():
        return -np.ones(mask.shape, dtype=np.float32) * 1e3
    inside = distance_transform_edt(mask).astype(np.float32)
    outside = distance_transform_edt(~mask).astype(np.float32)
    return outside - inside


def preferred_mask_at_z(
    cell: CellRecord, z: float, *,
    tile_origin_um: tuple[float, float],
    tile_size_px: tuple[int, int],
    pixel_size_um: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Build the cell's preferred mask at slice z (rasterized into a
    bounding box within the tile). Returns (mask, (y0, x0, y1, x1)) in
    tile-local pixel coords. Returns None if the cell is "outside" its
    z-extent at this z (τ=0) or its bbox is fully outside the tile.
    """
    dz = z - cell.z_center
    half_ext = cell.z_extent * 0.5
    if half_ext <= 0 or abs(dz) >= half_ext:
        return None
    tau = float(np.sqrt(1.0 - (dz / half_ext) ** 2))
    if tau < 1e-3:
        return None
    # Lateral shift
    shift_x = dz * (cell.t_x / max(cell.t_z, 1e-3))
    shift_y = dz * (cell.t_y / max(cell.t_z, 1e-3))
    cx_um = cell.xy_seed[0] + shift_x
    cy_um = cell.xy_seed[1] + shift_y

    # Scale template by tau and translate to (cx_um, cy_um)
    sx = cell.template_xs * tau + cx_um
    sy = cell.template_ys * tau + cy_um
    # Convert to tile-local pixel coords
    tx0, ty0 = tile_origin_um
    px = (sx - tx0) / pixel_size_um
    py = (sy - ty0) / pixel_size_um
    # Bounding box in tile pixel coords (clamped to tile)
    h_tile, w_tile = tile_size_px
    x0 = int(max(0, np.floor(px.min())))
    x1 = int(min(w_tile, np.ceil(px.max()) + 1))
    y0 = int(max(0, np.floor(py.min())))
    y1 = int(min(h_tile, np.ceil(py.max()) + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    # Rasterize polygon in bbox-local coordinates
    bbox_shape = (y1 - y0, x1 - x0)
    sx_box = px - x0
    sy_box = py - y0
    mask = _rasterize_polygon(sx_box, sy_box, bbox_shape)
    if not mask.any():
        return None
    return mask, (y0, x0, y1, x1)


def tessellate_z_slice(
    cells: Sequence[CellRecord],
    z: float,
    *,
    tile_origin_um: tuple[float, float],
    tile_size_px: tuple[int, int],
    pixel_size_um: float,
    sdf_margin_px: int = 6,
    device: str = "auto",
) -> np.ndarray:
    """Per-z cell_label tessellation, GPU-accelerated.

    Pass 1 (CPU): rasterize each cell's preferred mask polygon into a
        single (H, W) label image. Larger polygons drawn first so
        smaller cells overwrite within their footprint.
    Pass 2 (GPU): propagate labels into the surrounding `sdf_margin_px`
        ring via iterative max-pool dilation. This replaces the prior
        N_cells × `distance_transform_edt` calls with one O(margin)
        GPU loop — for sdf_margin_px=6, just 6 dilation passes total.

    Returns (h_tile, w_tile) int32 array. 0 = background.
    """
    import torch
    h_tile, w_tile = tile_size_px
    label_img = np.zeros((h_tile, w_tile), dtype=np.int32)

    # 1. Rasterize all cell polygons. Order by area descending so
    # smaller cells overwrite larger when they overlap (matches biology
    # — small cells get priority in their own footprint).
    cells_with_masks = []
    for cell in cells:
        result = preferred_mask_at_z(
            cell, z,
            tile_origin_um=tile_origin_um,
            tile_size_px=tile_size_px,
            pixel_size_um=pixel_size_um,
        )
        if result is None:
            continue
        mask, (y0, x0, y1, x1) = result
        area = int(mask.sum())
        cells_with_masks.append((area, cell.cell_idx, mask, (y0, x0, y1, x1)))
    # Sort largest first so smaller cells overwrite their interiors
    cells_with_masks.sort(key=lambda t: -t[0])
    for _, cell_idx, mask, (y0, x0, y1, x1) in cells_with_masks:
        slab = label_img[y0:y1, x0:x1]
        slab[mask] = cell_idx

    if sdf_margin_px <= 0 or not cells_with_masks:
        return label_img

    # 2. GPU distance-based label propagation (max_pool2d dilation).
    # Each iteration extends the labeled region by 1 pixel outward,
    # preserving the label of the nearest origin pixel. After
    # sdf_margin_px iterations, all pixels within `sdf_margin_px` of
    # some cell have inherited that cell's label.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lbl_t = torch.from_numpy(label_img).to(device).float().unsqueeze(0).unsqueeze(0)
    for _ in range(int(sdf_margin_px)):
        dilated = torch.nn.functional.max_pool2d(
            lbl_t, kernel_size=3, stride=1, padding=1)
        new_pixels = (lbl_t == 0) & (dilated > 0)
        lbl_t = torch.where(new_pixels, dilated, lbl_t)
    return lbl_t.squeeze().cpu().numpy().astype(np.int32)


__all__ = ["CellRecord", "preferred_mask_at_z", "tessellate_z_slice"]
