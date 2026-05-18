"""3D cell bodies + 3D molecule emission for 2.5D scenes.

Given a stack of per-z `cell_label` arrays from `sdf_tess`, each cell's
3D body = the union of its voxels across z. Molecules are emitted
uniformly within each cell's 3D body, with z coordinates resolved at
voxel-center precision.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def stack_z_labels(
    cells: Sequence,                # list of CellRecord (from sdf_tess)
    z_slices: Sequence[float],
    *,
    tile_origin_um: tuple[float, float],
    tile_size_px: tuple[int, int],
    pixel_size_um: float,
) -> np.ndarray:
    """Build a (n_z, H, W) int32 stack of cell_label across z slices.

    Cells are passed once; each z slice runs its own SDF tessellation.
    """
    from .sdf_tess import tessellate_z_slice
    n_z = len(z_slices)
    h, w = tile_size_px
    stack = np.zeros((n_z, h, w), dtype=np.int32)
    for zi, z in enumerate(z_slices):
        stack[zi] = tessellate_z_slice(
            cells, z=z,
            tile_origin_um=tile_origin_um,
            tile_size_px=tile_size_px,
            pixel_size_um=pixel_size_um,
        )
    return stack


def emit_molecules_3d(
    cells: Sequence,                # CellRecord list (with cell_idx, cell_id, cell_type)
    cell_label_3d: np.ndarray,      # (n_z, H, W) int32 from stack_z_labels
    *,
    z_slices: Sequence[float],
    tile_origin_um: tuple[float, float],
    pixel_size_um: float,
    per_cell_molecule_counts: dict[int, int] | None = None,
    default_count_per_cell: int = 20,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample molecules uniformly within each cell's 3D body.

    `per_cell_molecule_counts`: maps `cell_idx` to target molecule count.
    Cells not in the dict get `default_count_per_cell`.

    Returns a DataFrame with columns:
        x_true, y_true, z_true (µm)
        true_cell_idx, true_cell_id, true_cell_type
        source (always 'body' in this minimal version)
    """
    rng = rng or np.random.default_rng(0)
    n_z, h, w = cell_label_3d.shape
    z_arr = np.asarray(z_slices, dtype=np.float32)
    if len(z_arr) > 1:
        z_spacing = float(z_arr[1] - z_arr[0])
    else:
        z_spacing = 1.0
    tx0, ty0 = tile_origin_um
    psz = float(pixel_size_um)

    cells_by_idx = {c.cell_idx: c for c in cells}

    # Single pass over the volume: pull all non-zero voxels, sort by label,
    # then sample N per label from each contiguous run. Avoids the N_cells ×
    # full-volume comparisons of the old loop.
    nz_mask = cell_label_3d > 0
    if not nz_mask.any():
        return pd.DataFrame(columns=["x_true", "y_true", "z_true",
                                       "true_cell_idx", "true_cell_id",
                                       "true_cell_type", "source"])
    zi_all, py_all, px_all = np.nonzero(nz_mask)
    lbl_all = cell_label_3d[zi_all, py_all, px_all].astype(np.int64, copy=False)

    sort_idx = np.argsort(lbl_all, kind='stable')
    lbl_sorted = lbl_all[sort_idx]
    zi_sorted = zi_all[sort_idx]
    py_sorted = py_all[sort_idx]
    px_sorted = px_all[sort_idx]

    uniq, run_starts = np.unique(lbl_sorted, return_index=True)
    run_ends = np.append(run_starts[1:], len(lbl_sorted))

    # Resolve per-cell counts upfront
    counts = np.empty(len(uniq), dtype=np.int64)
    valid = np.ones(len(uniq), dtype=bool)
    for k, label in enumerate(uniq):
        c = cells_by_idx.get(int(label))
        if c is None:
            valid[k] = False
            counts[k] = 0
            continue
        if per_cell_molecule_counts is not None:
            counts[k] = max(0, int(per_cell_molecule_counts.get(int(label),
                                                                       default_count_per_cell)))
        else:
            counts[k] = max(0, int(default_count_per_cell))
    valid &= counts > 0
    if not valid.any():
        return pd.DataFrame(columns=["x_true", "y_true", "z_true",
                                       "true_cell_idx", "true_cell_id",
                                       "true_cell_type", "source"])

    # For each valid cell, draw `counts[k]` indices uniformly from its run.
    n_total = int(counts[valid].sum())
    pick_global_idx = np.empty(n_total, dtype=np.int64)
    pick_label = np.empty(n_total, dtype=np.int64)
    write = 0
    for k in np.where(valid)[0]:
        s, e = int(run_starts[k]), int(run_ends[k])
        n_vox = e - s
        if n_vox <= 0:
            continue
        N = int(counts[k])
        offsets = rng.integers(0, n_vox, size=N)
        pick_global_idx[write:write + N] = s + offsets
        pick_label[write:write + N] = int(uniq[k])
        write += N
    pick_global_idx = pick_global_idx[:write]
    pick_label = pick_label[:write]

    zi = zi_sorted[pick_global_idx]
    py = py_sorted[pick_global_idx]
    px = px_sorted[pick_global_idx]

    N_total = write
    jit_x = rng.uniform(-0.5, 0.5, size=N_total).astype(np.float32)
    jit_y = rng.uniform(-0.5, 0.5, size=N_total).astype(np.float32)
    jit_z = rng.uniform(-z_spacing / 2.0, z_spacing / 2.0, size=N_total).astype(np.float32)
    x_um = (tx0 + (px.astype(np.float32) + jit_x) * psz).astype(np.float32)
    y_um = (ty0 + (py.astype(np.float32) + jit_y) * psz).astype(np.float32)
    z_um = (z_arr[zi] + jit_z).astype(np.float32)

    # Resolve cell_id / cell_type per row via label lookup
    cell_ids = np.empty(N_total, dtype=object)
    cell_types = np.empty(N_total, dtype=object)
    for i in range(N_total):
        c = cells_by_idx[int(pick_label[i])]
        cell_ids[i] = c.cell_id
        cell_types[i] = c.cell_type

    return pd.DataFrame({
        "x_true": x_um, "y_true": y_um, "z_true": z_um,
        "true_cell_idx": pick_label.astype(np.int32),
        "true_cell_id": cell_ids,
        "true_cell_type": cell_types,
        "source": np.full(N_total, "body", dtype=object),
    })


__all__ = ["stack_z_labels", "emit_molecules_3d"]
