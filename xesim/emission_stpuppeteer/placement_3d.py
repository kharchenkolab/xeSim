"""2.5D placement: thin wrapper around the shared per-cell SDF helper.

Mirrors ``placement_2d.py``; differs only in array dimensionality and
the voxel→µm conversion (z is read from the discrete ``z_slices_um``
list rather than computed from a uniform formula, matching xeSim's
legacy 2.5D emitter convention so non-uniform z-stacks work too).

Anisotropy is handled by the ``sampling`` tuple
``(z_step_um, psz_um, psz_um)``: ``scipy.ndimage.distance_transform_edt``
weights z displacements by ``z_step_um`` and xy by ``psz_um``, producing
true-µm distances even when z spacing is ~14× the xy pixel size.

See ``_placement_core`` for the algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._placement_core import place_via_per_cell_sdf


def place_3d(
    trs_df: pd.DataFrame,
    cell_label_3d: np.ndarray,
    nucleus_label_3d: np.ndarray,
    *,
    psz_um: float,
    z_slices_um: list,
    tile_origin_um: tuple[float, float],
    leak_lam_per_cell: dict,
    max_dist_per_cell: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Place 2.5D transcripts via per-cell bounded SDF.

    Parameters
    ----------
    trs_df : pd.DataFrame
        Per-transcript decisions (must have ``cell_id``, ``feature_name``,
        ``is_leaked``, ``_label``).
    cell_label_3d, nucleus_label_3d : ndarray of shape (n_z, H, W)
        Rasterised 3D label arrays.
    psz_um : float
        Xy pixel size in µm.
    z_slices_um : sequence of float
        Absolute z position of each z-plane in scene µm. Length n_z.
        Need not be uniformly spaced, but the EDT's anisotropy assumes
        a single representative ``z_step`` — we use the median spacing.
    tile_origin_um : (xmin, ymin)
        Tile origin in scene µm (compose.py convention).
    leak_lam_per_cell, max_dist_per_cell : dict[int, float]
        Per-cell-label decay constant and hard cutoff.
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame
        ``trs_df`` extended with ``x_location``, ``y_location``,
        ``z_location``, ``overlaps_nucleus``, ``landed_in_cell_id``.
    """
    z_arr = np.asarray(z_slices_um, dtype=np.float64)
    # Sampling tuple for EDT. Use median z-step so a non-uniform stack
    # still gives a single anisotropic distance metric. Uniform-stack
    # case (the usual one) is exact; non-uniform is approximate.
    if len(z_arr) >= 2:
        z_step_um = float(np.median(np.diff(z_arr)))
    else:
        z_step_um = 1.0  # degenerate single-plane stack
    sampling = (z_step_um, psz_um, psz_um)

    xmin, ymin = float(tile_origin_um[0]), float(tile_origin_um[1])

    def voxel_to_um(coords: np.ndarray) -> np.ndarray:
        # coords shape (n, 3): (z_idx, y_idx, x_idx). Convert each axis
        # independently. The core adds ±half_voxel jitter per axis on
        # top so the output is uniformly distributed within each voxel.
        out = np.empty_like(coords, dtype=np.float64)
        # z: discrete lookup into the slice list. Clip indices to
        # the valid range defensively (placement shouldn't go OOB but
        # be safe in case the core picked an edge voxel).
        zi = np.clip(coords[:, 0].astype(np.int64), 0, len(z_arr) - 1)
        out[:, 0] = z_arr[zi]
        # y, x: same convention as 2D (voxel center → µm).
        out[:, 1] = (coords[:, 1].astype(np.float64) + 0.5) * psz_um + ymin
        out[:, 2] = (coords[:, 2].astype(np.float64) + 0.5) * psz_um + xmin
        return out

    return place_via_per_cell_sdf(
        trs_df=trs_df,
        cell_label=cell_label_3d,
        nucleus_label=nucleus_label_3d,
        sampling=sampling,
        voxel_to_um=voxel_to_um,
        leak_lam_per_cell=leak_lam_per_cell,
        max_dist_per_cell=max_dist_per_cell,
        rng=rng,
    )
