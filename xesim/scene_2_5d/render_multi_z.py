"""Multi-z DAPI rendering via v21 renderer on per-z cell_label.

For each z slice's cell_label stack, build a `MechanisticScene` and
run `model.render`, keeping only the DAPI channel (index 0). Stack
results into a (n_z, H, W) DAPI volume.

Membrane / 18S / αSMA channels stay 2D: they're rendered ONCE at the
focal-plane cell_label (or the z slice closest to most cells'
z_center) and the same 2D image is used across z. This matches real
Xenium where these stains are autofocus projections.

Output structure:
    morphology.ome.tif    — (n_z, H, W) DAPI z-stack, BigTIFF-aware
    morphology_focus/     — per-channel 2D files for non-DAPI stains
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# Plan3d per-type nucleus volumes (µm³). Used to compute per-cell nucleus
# radius for non-tiling small-ellipse nucleus_label.
PLAN3D_NUCLEUS_VOL = {
    "Ductal/tumor epithelial": 229.0,
    "Exocrine epithelial":     161.0,
    "Endothelial":             148.0,
    "Fibroblast / CAF":        123.0,
    "Immune":                  105.0,
    "Endocrine":               150.0,
    "Mural / pericyte":        100.0,
    "unknown":                 130.0,
}


def _nucleus_radius_um(cell_type: str) -> float:
    """Equivalent-sphere nucleus radius (µm) from plan3d per-type volume."""
    V = PLAN3D_NUCLEUS_VOL.get(cell_type, 130.0)
    return float((3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0))


def _stamp_nucleus_label(
    cells_records: Sequence,
    z_um: float,
    *,
    tile_size_px: tuple[int, int],
    tile_origin_um: tuple[float, float],
    pixel_size_um: float,
) -> np.ndarray:
    """Build a non-tiling nucleus_label at z by stamping a small ellipse
    per cell at its (tilt-shifted) xy_seed. Nucleus has its own
    ellipsoidal taper τ_nuc keyed to a per-type nucleus z-extent
    (defaulted to 0.7 × cell z_extent, roughly matching plan3d N:C
    ratios). Returns int32 (H, W) with cell_idx at nucleus pixels.
    """
    h, w = tile_size_px
    tx0, ty0 = tile_origin_um
    psz = pixel_size_um
    out = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.indices((h, w), dtype=np.float32)

    for c in cells_records:
        # Nucleus z-extent ~ 0.7 × cell z_extent (per plan3d N:C ratio)
        z_ext_nuc = c.z_extent * 0.7
        dz = z_um - c.z_center
        if abs(dz) >= z_ext_nuc / 2.0:
            continue
        tau_nuc = np.sqrt(1.0 - (dz / (z_ext_nuc / 2.0)) ** 2)
        if tau_nuc < 1e-3:
            continue
        # Lateral shift (same tilt as the cell)
        shift_x = dz * (c.t_x / max(c.t_z, 1e-3))
        shift_y = dz * (c.t_y / max(c.t_z, 1e-3))
        cx_um = c.xy_seed[0] + shift_x
        cy_um = c.xy_seed[1] + shift_y
        # Per-type nucleus radius (µm) × τ_nuc
        r_um = _nucleus_radius_um(c.cell_type) * tau_nuc
        r_px = r_um / psz
        # xy in tile-local pixels
        cx_px = (cx_um - tx0) / psz
        cy_px = (cy_um - ty0) / psz
        # Compute (x-cx)² + (y-cy)² < r² with bbox optimization
        x0 = max(0, int(cx_px - r_px - 1))
        x1 = min(w, int(cx_px + r_px + 2))
        y0 = max(0, int(cy_px - r_px - 1))
        y1 = min(h, int(cy_px + r_px + 2))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dxx = (xx[y0:y1, x0:x1] - cx_px) ** 2
        dyy = (yy[y0:y1, x0:x1] - cy_px) ** 2
        mask = (dxx + dyy) < (r_px ** 2)
        # Stamp only where current pixel is empty (no nucleus already there
        # — nuclei rarely overlap in 3D for non-touching cells; if they do,
        # first stamp wins)
        target = out[y0:y1, x0:x1]
        target[mask & (target == 0)] = c.cell_idx
        out[y0:y1, x0:x1] = target
    return out


def render_multi_z_dapi(
    model,
    cell_label_3d: np.ndarray,           # (n_z, H, W) int32
    *,
    cells_records: Sequence,             # CellRecord list with cell_idx → cell_type
    pixel_size_um: float,
    background_mask_sigma: float | None = 3.0,
    nucleus_erode_um: float = 0.7,       # legacy fallback only
    nucleus_templates: dict | None = None,  # cell_id → (xs_rel, ys_rel) nucleus polygon
    z_slices_um: np.ndarray | None = None,  # for τ_nuc taper
    tile_origin_um: tuple[float, float] = (0.0, 0.0),
    cell_latents: dict[int, np.ndarray] | None = None,
    target_p99: float | None = 1.0,  # rescale so p99=target; None disables
    use_bfloat16: bool = True,      # cuts activation memory ~30-40%
    clear_cache_per_z: bool = False,  # not needed when z-batched
    inner_tile_px: int = 256,       # inner-tile size; smaller because we batch z
    pre_stamped: tuple[np.ndarray, np.ndarray] | None = None,
    # ^ (cl_3d, nl_3d) if nucleus stamping was already done by the producer
) -> np.ndarray:
    """Render the DAPI channel at each z slice via z-batched bridge tiling.

    For each inner sub-tile (default 256×256 px), all `n_z` z-slices are
    packed into the batch dim and rendered in a single `model.render_batch`
    forward pass. Cuts 12× the kernel-launch overhead vs the prior
    per-z bridge_render approach. Memory is bounded by inner_tile_px ·
    n_z (set inner_tile_px=256 for 12 z slices ≈ 4-5 GB activation).
    """
    from ..mechanistic_scene import MechanisticScene, MechanisticCell

    n_z, h, w = cell_label_3d.shape
    out = np.zeros((n_z, h, w), dtype=np.float32)

    cells_by_idx = {c.cell_idx: c for c in cells_records}
    mech_cells_full = tuple(MechanisticCell(
        cell_id=c.cell_id, label=cell_idx,
        source="observed_anchor", cell_type=c.cell_type,
        nucleus_label=cell_idx,
        provenance={"is_ghost": False, "source_tag": "2.5d_sdf"},
    ) for cell_idx, c in sorted(cells_by_idx.items()))

    # 1. Pre-stamp nucleus polygons across all z. Also mutates a copy of
    # cell_label_3d so the renderer's cell ⊇ nucleus invariant holds.
    # If the producer already did the stamping (producer-consumer path),
    # use those arrays directly to avoid duplicate CPU work.
    if pre_stamped is not None:
        cl_3d, nl_3d = pre_stamped
    else:
        cl_3d = cell_label_3d.astype(np.int32, copy=True)
        nl_3d = np.zeros_like(cl_3d)
        if nucleus_templates is not None and z_slices_um is not None:
            for zi in range(n_z):
                nl_3d[zi] = _stamp_nucleus_polygons(
                    cells_by_idx, float(z_slices_um[zi]), nucleus_templates,
                    tile_size_px=(h, w), tile_origin_um=tile_origin_um,
                    pixel_size_um=pixel_size_um,
                    cell_label_at_z=cl_3d[zi],
                )
        else:
            for zi in range(n_z):
                nl_3d[zi] = _stamp_nucleus_from_cell_label(
                    cl_3d[zi], cells_by_idx, pixel_size_um)

    # 2. Inner-tile loop with z-batching
    inner = int(inner_tile_px)
    autocast_dtype = None
    if use_bfloat16:
        import torch
        if torch.cuda.is_available():
            autocast_dtype = torch.bfloat16

    accum_wt = np.zeros((h, w), dtype=np.float32)
    # Simple 1.0 weighting — the model.render output already handles
    # boundary masking, and we'll average at the inner-tile overlap (none here).
    # Inner tiles are non-overlapping here, so no feather needed.
    for y0 in range(0, h, inner):
        y1 = min(h, y0 + inner)
        h_eff = y1 - y0
        for x0 in range(0, w, inner):
            x1 = min(w, x0 + inner)
            w_eff = x1 - x0
            # Pad to (inner, inner) so all scenes in the batch share shape.
            # Pad with 0 (background); padded regions are masked out anyway.
            inner_cls = np.zeros((n_z, inner, inner), dtype=np.int32)
            inner_nls = np.zeros((n_z, inner, inner), dtype=np.int32)
            inner_cls[:, :h_eff, :w_eff] = cl_3d[:, y0:y1, x0:x1]
            inner_nls[:, :h_eff, :w_eff] = nl_3d[:, y0:y1, x0:x1]
            # Skip fully-empty inner tiles
            if not (inner_cls > 0).any():
                continue
            # Build n_z scenes (one per z); they share image_shape so render_batch works.
            scenes = [
                MechanisticScene(
                    image_shape=(inner, inner), pixel_size=pixel_size_um,
                    cell_label=inner_cls[zi], nucleus_label=inner_nls[zi],
                    cells=mech_cells_full, scene_id=f"z{zi}",
                    provenance={"scene_mode": "2.5d", "z_slice": int(zi)},
                )
                for zi in range(n_z)
            ]
            outs = model.render_batch(
                scenes, cell_latents=cell_latents, seed=0,
                background_mask_sigma=background_mask_sigma,
                autocast_dtype=autocast_dtype,
            )
            for zi, o in enumerate(outs):
                out[zi, y0:y1, x0:x1] = o[0, :h_eff, :w_eff].astype(np.float32, copy=False)

    if target_p99 is not None and out.size > 0:
        nz_pixels = out[out > 0]
        if nz_pixels.size > 0:
            p99 = float(np.percentile(nz_pixels, 99))
            if p99 > 1e-6:
                scale = target_p99 / p99
                out = np.clip(out * scale, 0.0, target_p99)
    return out


def _stamp_nucleus_polygons(
    cells_by_idx: dict,
    z_um: float,
    nucleus_templates: dict,         # cell_id → (xs_rel, ys_rel) µm
    *,
    tile_size_px: tuple[int, int],
    tile_origin_um: tuple[float, float],
    pixel_size_um: float,
    cell_label_at_z: np.ndarray,    # mutated to include nucleus regions
) -> np.ndarray:
    """Stamp per-cell nucleus polygons tilted+tapered at slice z.

    Does NOT clip to cell_label — nuclei get their full real polygon
    extent. The caller's cell_label_at_z gets adjusted to include
    nucleus regions (cell_label = nucleus_label where conflicting,
    so the renderer's cell vs nucleus channels are consistent)."""
    from skimage.draw import polygon as sk_polygon

    h, w = tile_size_px
    tx0, ty0 = tile_origin_um
    psz = pixel_size_um
    out = np.zeros((h, w), dtype=np.int32)
    for cell_idx, c in cells_by_idx.items():
        z_ext_nuc = c.z_extent * 0.7
        dz = z_um - c.z_center
        if z_ext_nuc <= 0 or abs(dz) >= z_ext_nuc / 2.0:
            continue
        tau_nuc = float(np.sqrt(1.0 - (dz / (z_ext_nuc / 2.0)) ** 2))
        if tau_nuc < 1e-3:
            continue
        shift_x = dz * (c.t_x / max(c.t_z, 1e-3))
        shift_y = dz * (c.t_y / max(c.t_z, 1e-3))
        cx_um = c.xy_seed[0] + shift_x
        cy_um = c.xy_seed[1] + shift_y
        tmpl = nucleus_templates.get(c.cell_id)
        if tmpl is None:
            r_um = _nucleus_radius_um(c.cell_type) * tau_nuc
            theta = np.linspace(0, 2 * np.pi, 24)
            xs_rel = r_um * np.cos(theta); ys_rel = r_um * np.sin(theta)
        else:
            xs_rel = np.asarray(tmpl[0], dtype=np.float32) * tau_nuc
            ys_rel = np.asarray(tmpl[1], dtype=np.float32) * tau_nuc
        px = (xs_rel + cx_um - tx0) / psz
        py = (ys_rel + cy_um - ty0) / psz
        rr, cc = sk_polygon(py, px, shape=(h, w))
        if len(rr) == 0:
            continue
        # Stamp: nucleus takes precedence over previous nucleus (first
        # cell wins overlap, but with biological cells nuclei rarely
        # overlap in 3D)
        empty = out[rr, cc] == 0
        rr_e = rr[empty]; cc_e = cc[empty]
        out[rr_e, cc_e] = int(cell_idx)
        # Adjust cell_label_at_z to include nucleus regions for this
        # cell — ensures renderer sees consistent cell⊇nucleus
        cell_label_at_z[rr_e, cc_e] = int(cell_idx)
    return out


def _stamp_nucleus_from_cell_label(
    cl: np.ndarray, cells_by_idx: dict, pixel_size_um: float,
) -> np.ndarray:
    """For each cell present in `cl`, stamp a non-tiling round nucleus at
    the cell's centroid in tile-local pixel coords. The nucleus radius
    is the per-type plan3d nucleus radius; ignores the cell's xy template
    shape (so the nucleus is round and small, not polygon-shaped)."""
    h, w = cl.shape
    out = np.zeros_like(cl)
    unique_labs = np.unique(cl)
    unique_labs = unique_labs[unique_labs > 0]
    yy, xx = np.indices((h, w), dtype=np.float32)
    from scipy.ndimage import center_of_mass as ndi_com
    centroids = ndi_com(cl > 0, labels=cl, index=unique_labs.tolist())
    for lab, (cy, cx) in zip(unique_labs, centroids):
        c = cells_by_idx.get(int(lab))
        if c is None or not (np.isfinite(cy) and np.isfinite(cx)):
            continue
        r_um = _nucleus_radius_um(c.cell_type)
        r_px = r_um / pixel_size_um
        x0 = max(0, int(cx - r_px - 1)); x1 = min(w, int(cx + r_px + 2))
        y0 = max(0, int(cy - r_px - 1)); y1 = min(h, int(cy + r_px + 2))
        if x1 - x0 < 1 or y1 - y0 < 1: continue
        dxx = (xx[y0:y1, x0:x1] - cx) ** 2
        dyy = (yy[y0:y1, x0:x1] - cy) ** 2
        mask = (dxx + dyy) < (r_px ** 2)
        # Restrict to interior of the cell's tile-z mask (don't stamp
        # nucleus outside the cell at this z)
        cell_mask_here = cl[y0:y1, x0:x1] == int(lab)
        mask = mask & cell_mask_here
        target = out[y0:y1, x0:x1]
        target[mask & (target == 0)] = int(lab)
        out[y0:y1, x0:x1] = target
    return out


__all__ = ["render_multi_z_dapi"]
