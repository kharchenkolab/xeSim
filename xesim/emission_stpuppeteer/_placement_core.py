"""Dimension-agnostic per-cell SDF placement helper.

This is the shared core of ``placement_2d`` / ``placement_3d``. It works
on any-dimensional ``cell_label`` arrays — the ``sampling`` tuple passed
to ``scipy.ndimage.distance_transform_edt`` makes distances anisotropic
in µm, and the rest of the algorithm doesn't care whether arrays are
``(H, W)`` or ``(Z, H, W)``.

See §4 of tmp/cellAdmix-integration.md for the model. Brief summary:

- Non-leaked transcripts: uniform sample from voxels where
  ``cell_label == c``.
- Leaked transcripts: per-cell bounded SDF. For cell ``c`` we run
  ``distance_transform_edt(~(cell_label == c), sampling=...)`` on a
  bbox around the cell (expanded by ``max_dist_um``). The result is the
  distance from every voxel in the bbox to cell c's surface — including
  voxels inside other cells. Voxels within ``max_dist_um`` are weighted
  by ``exp(-d / λ)`` and one is sampled per leaked transcript.

The per-cell-independent halos let cells' density contributions ADD in
overlapping regions, and a transcript from A can land inside cell B
(``landed_in_cell_id`` records that for ground truth).
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, find_objects

logger = logging.getLogger(__name__)


def place_via_per_cell_sdf(
    trs_df: pd.DataFrame,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    sampling: Sequence[float],
    voxel_to_um: Callable[[np.ndarray], np.ndarray],
    leak_lam_per_cell: dict,
    max_dist_per_cell: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Place transcripts in space using per-cell bounded SDF for halos.

    Parameters
    ----------
    trs_df : pd.DataFrame
        Long transcript table with columns ``cell_id`` (str),
        ``feature_name`` (str), ``is_leaked`` (bool), ``compartment`` (str).
        Other columns are passed through.
    cell_label : np.ndarray of int (any-D)
        Rasterised cell label array. 0 = background.
    nucleus_label : np.ndarray of int (any-D, same shape as cell_label)
        Nucleus label array, used to flag ``overlaps_nucleus``.
    sampling : sequence of float
        ``(z_step_um, psz_um, psz_um)`` for 3D, ``(psz_um, psz_um)`` for 2D.
        Passed to ``distance_transform_edt`` so distances are in true µm.
    voxel_to_um : callable
        Maps an array of voxel indices of shape ``(n, ndim)`` to an array
        of µm coordinates of shape ``(n, ndim)``. Half-voxel jitter is
        applied inside this function before the call.
    leak_lam_per_cell : dict[label -> float]
        Per-cell decay constant λ in µm. ``exp(-d/λ)`` weights inside the
        halo. STpuppeteer's existing parameterisation:
        ``lam = -max_dist / log(1 - coverage)`` with coverage = 0.97.
    max_dist_per_cell : dict[label -> float]
        Per-cell hard cutoff for leakage in µm. Voxels farther than this
        from the cell's surface are excluded from the halo sampler.
    rng : np.random.Generator
        NumPy RNG.

    Returns
    -------
    pd.DataFrame
        ``trs_df`` extended with µm coordinates and provenance flags:

        - ``x_location`` / ``y_location`` / ``z_location`` (z only in 3D)
        - ``overlaps_nucleus`` (uint8)
        - ``landed_in_cell_id`` (str or "")  — for leaked transcripts;
          empty string for non-leaked (cell_id is already the source cell).
    """
    ndim = cell_label.ndim
    n_tr = len(trs_df)

    if n_tr == 0:
        return _empty_with_coords(trs_df, ndim)

    # Map cell_id (string) → integer label (used to index cell_label).
    # The xeSim scene uses cell.label as the integer key; we don't have
    # direct access to MechanisticCell here, but trs_df["cell_id"] strings
    # were emitted by STpuppeteer's counts_to_transcript_df from the same
    # source as cell_label values. We need a mapping; the caller (emit.py)
    # passes it via the leak_lam_per_cell / max_dist_per_cell dicts which
    # are keyed by integer label. We rebuild cell_id → label here from a
    # column the caller is expected to populate, OR fall through to a
    # built-in scan if the column is absent.
    if "_label" in trs_df.columns:
        labels = trs_df["_label"].to_numpy()
    else:
        # Fall through: assume cell_id ↔ label is a simple "cell_<N>" or
        # an integer-ish string; otherwise the caller MUST set _label.
        # In practice emit.py always sets it, so this is defensive only.
        raise ValueError(
            "place_via_per_cell_sdf requires trs_df['_label'] (int per row). "
            "Caller (emit.py) is expected to populate it from the scene."
        )

    # Pre-compute per-label bboxes once. scipy.ndimage.find_objects returns
    # a list indexed by (label-1); None for labels that aren't present.
    max_label = int(cell_label.max()) if cell_label.size else 0
    bboxes = find_objects(cell_label, max_label=max_label) if max_label > 0 else []

    # Output buffers — we fill in tile-local µm at the end via voxel_to_um.
    voxel_coords = np.empty((n_tr, ndim), dtype=np.int64)
    overlaps_nuc = np.zeros(n_tr, dtype=np.uint8)
    landed_in_label = np.zeros(n_tr, dtype=np.int64)  # 0 = true exterior

    # Group transcripts by source cell label so we can run the per-cell
    # SDF once per cell rather than per transcript.
    groups: dict[int, np.ndarray] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    for lab in list(groups.keys()):
        groups[lab] = np.asarray(groups[lab], dtype=np.int64)

    psz_x = float(sampling[-1])  # for sub-voxel jitter scale on (x,y) axes

    for lab, row_idx in groups.items():
        if lab <= 0 or lab > len(bboxes):
            continue
        bbox = bboxes[lab - 1]
        if bbox is None:
            continue

        sub_trs = trs_df.iloc[row_idx]
        is_leaked = sub_trs["is_leaked"].to_numpy(dtype=bool)
        n_leak = int(is_leaked.sum())
        n_inside = int((~is_leaked).sum())

        # ---- Inside placement: uniform from cell_label == lab ----
        if n_inside > 0:
            inside_local = np.argwhere(cell_label[bbox] == lab)
            if inside_local.shape[0] == 0:
                # Defensive: bbox exists but no voxels match. Shouldn't
                # happen post-find_objects.
                continue
            offset = np.array([s.start for s in bbox], dtype=np.int64)
            picks = rng.integers(0, inside_local.shape[0], size=n_inside)
            voxels_inside = inside_local[picks] + offset
            inside_row_idx = row_idx[~is_leaked]
            voxel_coords[inside_row_idx] = voxels_inside
            # Nucleus overlap check by looking up nucleus_label at each voxel.
            # Multi-dim indexing via tuple of per-axis arrays.
            nuc_vals = nucleus_label[tuple(voxels_inside.T)]
            overlaps_nuc[inside_row_idx] = (nuc_vals == lab).astype(np.uint8)
            # Non-leaked: landed_in_cell_id stays 0 (interpreted as
            # "transcript is inside its own source cell").
            landed_in_label[inside_row_idx] = lab

        # ---- Leaked placement: per-cell bounded SDF ----
        if n_leak > 0:
            lam = float(leak_lam_per_cell.get(lab, 0.0))
            max_d = float(max_dist_per_cell.get(lab, 0.0))
            if lam <= 0.0 or max_d <= 0.0:
                # No leakage configured for this cell type. Should not
                # happen because is_leaked is True only when p_eff > 0,
                # but be defensive: treat the leaked transcripts as
                # interior fallback (still inside the cell).
                logger.debug(
                    "Cell %d has is_leaked transcripts but λ or max_dist <= 0; "
                    "falling through to interior placement.", lab
                )
                inside_local = np.argwhere(cell_label[bbox] == lab)
                if inside_local.shape[0] == 0:
                    continue
                offset = np.array([s.start for s in bbox], dtype=np.int64)
                picks = rng.integers(0, inside_local.shape[0], size=n_leak)
                voxels_leak = inside_local[picks] + offset
                leak_row_idx = row_idx[is_leaked]
                voxel_coords[leak_row_idx] = voxels_leak
                landed_in_label[leak_row_idx] = lab
                continue

            # Expand bbox by ⌈max_d / sampling⌉ voxels on every axis to
            # capture the full halo support around this cell.
            expanded = _expand_bbox(bbox, max_d, sampling, cell_label.shape)
            mask_c_local = cell_label[expanded] == lab
            # Distance from every voxel in the expanded bbox to cell c's
            # surface (interior of c is 0, everywhere else is positive).
            # NOTE: ~mask_c_local means "outside cell c" — we measure
            # distance from voxels-outside-c to the nearest c-voxel. Other
            # cells' bodies are not excluded; their voxels just have a
            # positive distance to c, same as true exterior voxels.
            d_to_c = distance_transform_edt(~mask_c_local, sampling=sampling)
            # In-range halo mask: outside c AND within max_dist.
            in_range = (d_to_c > 0) & (d_to_c < max_d)
            local_coords = np.argwhere(in_range)
            if local_coords.shape[0] == 0:
                # Halo is empty (cell is isolated and tile is too small),
                # or pathological. Fall back to interior placement so we
                # don't lose the transcripts entirely.
                logger.debug(
                    "Cell %d: empty halo within max_dist=%.2f µm; "
                    "falling back to interior placement.", lab, max_d
                )
                inside_local = np.argwhere(mask_c_local)
                if inside_local.shape[0] == 0:
                    continue
                picks = rng.integers(0, inside_local.shape[0], size=n_leak)
                voxels_leak_local = inside_local[picks]
            else:
                # Weight voxels by exp(-d/λ) and sample with replacement.
                d_in_range = d_to_c[in_range]
                w = np.exp(-d_in_range / lam)
                w_sum = w.sum()
                if w_sum <= 0.0:
                    # All weights underflowed — shouldn't happen with
                    # reasonable λ but be defensive.
                    picks = rng.integers(0, local_coords.shape[0], size=n_leak)
                else:
                    w /= w_sum
                    picks = rng.choice(local_coords.shape[0], size=n_leak, p=w)
                voxels_leak_local = local_coords[picks]
            offset = np.array([s.start for s in expanded], dtype=np.int64)
            voxels_leak = voxels_leak_local + offset
            leak_row_idx = row_idx[is_leaked]
            voxel_coords[leak_row_idx] = voxels_leak
            # Record which cell's volume each leaked transcript landed in
            # (0 = true exterior, lab' = inside neighbour cell lab').
            landed_in_label[leak_row_idx] = cell_label[tuple(voxels_leak.T)]
            # Leaked transcripts don't get the overlaps_nucleus flag —
            # by definition they're outside their source cell.

    # ---- Convert voxel indices → µm with sub-voxel jitter ----
    # Jitter is uniform in ±half_voxel along each axis (matches xeSim's
    # legacy 2.5D emitter; gives continuous output coordinates).
    jitter = np.zeros((n_tr, ndim), dtype=np.float64)
    for axis in range(ndim):
        half = 0.5 * float(sampling[axis])
        jitter[:, axis] = rng.uniform(-half, +half, size=n_tr)
    um_coords = voxel_to_um(voxel_coords) + jitter

    # Map landed_in_label → cell_id strings using trs_df's own cell_id +
    # label columns as the reverse map.
    label_to_cell_id = dict(zip(trs_df["_label"].to_numpy(),
                                  trs_df["cell_id"].to_numpy()))
    landed_in_cell_id = np.array(
        [label_to_cell_id.get(int(l), "") for l in landed_in_label],
        dtype=object,
    )

    # Assemble output. We preserve all of trs_df's input columns.
    out = trs_df.copy()
    if ndim == 2:
        out["y_location"] = um_coords[:, 0]
        out["x_location"] = um_coords[:, 1]
    else:
        out["z_location"] = um_coords[:, 0]
        out["y_location"] = um_coords[:, 1]
        out["x_location"] = um_coords[:, 2]
    out["overlaps_nucleus"] = overlaps_nuc
    out["landed_in_cell_id"] = landed_in_cell_id
    return out


def _empty_with_coords(trs_df: pd.DataFrame, ndim: int) -> pd.DataFrame:
    """Return an empty DataFrame with the expected output columns."""
    out = trs_df.copy()
    out["x_location"] = pd.array([], dtype="float64")
    out["y_location"] = pd.array([], dtype="float64")
    if ndim == 3:
        out["z_location"] = pd.array([], dtype="float64")
    out["overlaps_nucleus"] = pd.array([], dtype="uint8")
    out["landed_in_cell_id"] = pd.array([], dtype="object")
    return out


def _expand_bbox(
    bbox: tuple,
    margin_um: float,
    sampling: Sequence[float],
    full_shape: tuple,
) -> tuple:
    """Expand a tuple-of-slices bbox by ``margin_um`` in each axis, clipped.

    Margin is converted from µm to voxels per axis via the sampling tuple
    so that 2D (psz, psz) and 3D (z_step, psz, psz) work uniformly.
    """
    expanded = []
    for axis, (s, samp, n) in enumerate(zip(bbox, sampling, full_shape)):
        m = int(np.ceil(margin_um / float(samp))) + 1
        new_start = max(0, s.start - m)
        new_stop = min(n, s.stop + m)
        expanded.append(slice(new_start, new_stop))
    return tuple(expanded)
