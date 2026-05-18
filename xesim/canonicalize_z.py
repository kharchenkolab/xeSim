"""Per-cell nucleus z-level fitting from multi-z DAPI stacks.

Phase 0 step 2: for each cell with a nucleus polygon (or nucleus_label
mask), compute z_center (intensity-weighted mean z), z_extent (FWHM of
DAPI z-profile), and z_confidence (signal range / noise floor) from the
bundle's morphology.ome.tif z-stack.

Simple intensity-weighted approach — not the full plan3d HMC posterior.
That's a future v2 refinement (see misc/2.5D.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def fit_per_cell_z(
    cell_label: np.ndarray,            # (H, W) int label image — used to identify cell pixels
    nucleus_label: np.ndarray,         # (H, W) int label image — for masking to nucleus only
    cell_ids_order: Sequence[str],     # cell_ids ordered as np.unique(cell_label)[1:] expects
    dapi_zstack: np.ndarray,           # (nz, H, W) float
    z_axis_um: np.ndarray,             # (nz,) z-positions in µm
    *,
    min_pixels: int = 4,
    snr_floor: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Fit z_center, z_extent, z_confidence per cell from DAPI z-profile.

    For each cell (positive label value in cell_label), takes the cell's
    pixels (optionally restricted to nucleus pixels via `nucleus_label > 0`)
    and computes per-z mean DAPI intensity → z-profile. Then:
      z_center = intensity-weighted mean of z
      z_extent = FWHM of z-profile
      z_confidence = signal range / baseline (proxy for SNR)

    Returns three (N_cells,) float arrays aligned with `cell_ids_order`
    (which corresponds to sorted unique label values in cell_label, as
    canonicalize stores them).
    """
    n_cells = len(cell_ids_order)
    z_center = np.full(n_cells, np.nan, dtype=np.float32)
    z_extent = np.full(n_cells, np.nan, dtype=np.float32)
    z_confidence = np.full(n_cells, np.nan, dtype=np.float32)

    if cell_label.size == 0 or dapi_zstack.size == 0:
        return {
            "z_center_um": z_center,
            "z_extent_um": z_extent,
            "z_confidence": z_confidence,
        }

    from scipy.ndimage import mean as ndimage_mean, sum as ndimage_sum

    unique_labels = np.unique(cell_label)
    # Exclude background (0); the remaining values correspond positionally
    # to entries in cell_ids_order (canonicalize uses np.unique sorted order).
    label_vals = unique_labels[unique_labels != 0]
    nz = dapi_zstack.shape[0]
    if len(label_vals) == 0:
        return {
            "z_center_um": z_center,
            "z_extent_um": z_extent,
            "z_confidence": z_confidence,
        }
    label_vals = label_vals[: n_cells]   # paranoia
    n_label = len(label_vals)
    z_spacing = float(z_axis_um[1] - z_axis_um[0]) if nz > 1 else 1.0

    # Vectorized per-cell, per-z mean DAPI in one C call per z slice.
    # Two label images: (a) nucleus-only labels (preferred), (b) cell labels.
    # For each z, ndimage.mean returns one value per label.
    nuc_label_image = np.where(nucleus_label > 0, cell_label, 0).astype(cell_label.dtype)
    has_nuc_mask = nucleus_label > 0

    # Pixel counts per cell, used to pick nucleus-vs-cell fallback per cell.
    cell_pixel_count = ndimage_sum(
        np.ones_like(cell_label, dtype=np.float32),
        labels=cell_label, index=label_vals)
    nuc_pixel_count = ndimage_sum(
        has_nuc_mask.astype(np.float32),
        labels=cell_label, index=label_vals)
    use_nuc = (nuc_pixel_count >= min_pixels)
    enough_pixels = (use_nuc | (cell_pixel_count >= min_pixels))

    # Two profile matrices: (n_label, nz). Per cell pick which to use later.
    nuc_profile = np.zeros((n_label, nz), dtype=np.float32)
    cell_profile = np.zeros((n_label, nz), dtype=np.float32)
    for zi in range(nz):
        slice_data = dapi_zstack[zi]
        nuc_profile[:, zi] = ndimage_mean(
            slice_data, labels=nuc_label_image, index=label_vals)
        cell_profile[:, zi] = ndimage_mean(
            slice_data, labels=cell_label, index=label_vals)
    # Replace NaN (cells with no nucleus pixels in nuc_label_image) with 0
    nuc_profile = np.nan_to_num(nuc_profile, nan=0.0)

    # Choose per-cell profile
    profiles = np.where(use_nuc[:, None], nuc_profile, cell_profile)

    # Vectorize z_center (argmax + quadratic refinement) + FWHM + confidence
    prof_min = profiles.min(axis=1, keepdims=True)
    prof_b = profiles - prof_min
    rng = prof_b.max(axis=1)
    valid = enough_pixels & (rng > snr_floor)

    # argmax with quadratic refinement
    peak_idx = np.argmax(prof_b, axis=1)
    zc_array = np.full(n_label, np.nan, dtype=np.float32)
    ze_array = np.zeros(n_label, dtype=np.float32)
    conf_array = np.zeros(n_label, dtype=np.float32)

    if valid.any():
        valid_idx = np.where(valid)[0]
        for pos in valid_idx:
            peak = int(peak_idx[pos])
            if 0 < peak < nz - 1:
                y0 = prof_b[pos, peak - 1]; y1 = prof_b[pos, peak]; y2 = prof_b[pos, peak + 1]
                denom = (y0 - 2 * y1 + y2)
                if abs(denom) > 1e-6:
                    delta = 0.5 * (y0 - y2) / denom
                    delta = max(-1.0, min(1.0, float(delta)))
                else:
                    delta = 0.0
                zc_array[pos] = float((peak + delta) * z_spacing + z_axis_um[0])
            else:
                zc_array[pos] = float(z_axis_um[peak])

            # FWHM
            half = prof_b[pos, peak] / 2.0
            above = prof_b[pos] > half
            if above[peak]:
                lo = peak
                while lo > 0 and above[lo - 1]:
                    lo -= 1
                hi = peak
                while hi < nz - 1 and above[hi + 1]:
                    hi += 1
                ze_array[pos] = float(z_axis_um[hi] - z_axis_um[lo]) if hi > lo else 0.0
            baseline = max(float(profiles[pos].min()), 1.0)
            conf_array[pos] = float(rng[pos] / baseline)

    # Copy into pre-allocated output arrays
    n_copy = min(n_label, n_cells)
    z_center[:n_copy] = zc_array[:n_copy]
    z_extent[:n_copy] = ze_array[:n_copy]
    z_confidence[:n_copy] = conf_array[:n_copy]

    return {
        "z_center_um": z_center,
        "z_extent_um": z_extent,
        "z_confidence": z_confidence,
    }


def read_dapi_zstack_crop(
    morphology_path: Path,
    crop_bounds_um: tuple[float, float, float, float],  # xmin, xmax, ymin, ymax
    pixel_size_um: float,
    target_shape: tuple[int, int],     # (H, W) of the crop's rasterized images
) -> np.ndarray:
    """Read the multi-z DAPI slab for one canonical crop.

    Returns (nz, H, W) float32. The xy crop matches what canonicalize
    rasterizes for cell_label/nucleus_label (origin top-left, pixel_size
    matches the bundle).
    """
    import tifffile
    import zarr

    xmin, xmax, ymin, ymax = crop_bounds_um
    H_target, W_target = target_shape

    store = tifffile.imread(str(morphology_path), aszarr=True)
    try:
        arr = zarr.open(store, mode="r")["0"]
        # arr shape: (nz, H_full, W_full)
        nz, H_full, W_full = arr.shape
        # xy bounds in pixels
        x0 = max(0, int(np.floor(xmin / pixel_size_um)))
        x1 = min(W_full, int(np.ceil(xmax / pixel_size_um)))
        y0 = max(0, int(np.floor(ymin / pixel_size_um)))
        y1 = min(H_full, int(np.ceil(ymax / pixel_size_um)))
        slab = np.asarray(arr[:, y0:y1, x0:x1]).astype(np.float32)
    finally:
        try:
            store.close()
        except Exception:
            pass

    # Pad / crop to target_shape if rasterization gave a slightly different size
    h, w = slab.shape[1], slab.shape[2]
    if h != H_target or w != W_target:
        pad_h = max(0, H_target - h); pad_w = max(0, W_target - w)
        if pad_h or pad_w:
            slab = np.pad(slab, ((0, 0), (0, pad_h), (0, pad_w)),
                            mode="constant", constant_values=0)
        slab = slab[:, :H_target, :W_target]

    return slab


def fit_per_cell_z_for_crop(
    morphology_path: Path | None,
    crop_bounds_um: tuple[float, float, float, float],
    pixel_size_um: float,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_ids: list[str],
    z_spacing_um: float = 3.0,
) -> dict[str, np.ndarray]:
    """High-level helper that reads the z-slab and runs the per-cell fit.

    If morphology_path is None (e.g., single-z bundle), returns NaN arrays.
    """
    n_cells = len(cell_ids)
    if morphology_path is None or not morphology_path.exists():
        return {
            "z_center_um": np.full(n_cells, np.nan, dtype=np.float32),
            "z_extent_um": np.full(n_cells, np.nan, dtype=np.float32),
            "z_confidence": np.full(n_cells, np.nan, dtype=np.float32),
        }

    target_shape = (cell_label.shape[0], cell_label.shape[1])
    zslab = read_dapi_zstack_crop(
        morphology_path, crop_bounds_um, pixel_size_um, target_shape,
    )
    nz = zslab.shape[0]
    z_axis = np.arange(nz, dtype=np.float32) * z_spacing_um

    return fit_per_cell_z(
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        cell_ids_order=cell_ids,
        dapi_zstack=zslab,
        z_axis_um=z_axis,
    )
