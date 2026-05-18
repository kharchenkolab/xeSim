"""2D placement: thin wrapper around the shared per-cell SDF helper.

Mirrors ``placement_3d``; differs only in array dimensionality and
the voxel→µm conversion. See ``_placement_core`` for the algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._placement_core import place_via_per_cell_sdf


def place_2d(
    trs_df: pd.DataFrame,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    *,
    psz_um: float,
    tile_origin_um: tuple[float, float] = (0.0, 0.0),
    leak_lam_per_cell: dict,
    max_dist_per_cell: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Place 2D transcripts via per-cell bounded SDF.

    Parameters
    ----------
    trs_df : pd.DataFrame
        Per-transcript decisions (must have ``cell_id``, ``feature_name``,
        ``is_leaked``, ``_label``).
    cell_label, nucleus_label : ndarray of shape (H, W)
        Rasterised label arrays at the tile's xy grid.
    psz_um : float
        Xy pixel size in µm.
    tile_origin_um : (y0, x0)
        Tile origin in scene µm coordinates. Output xy is in scene µm.
    leak_lam_per_cell, max_dist_per_cell : dict[int, float]
        Per-cell-label decay constant and hard cutoff for the halo. See
        the design doc §5.3 for the parameterisation.
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame
        ``trs_df`` extended with ``x_location``, ``y_location``,
        ``overlaps_nucleus``, ``landed_in_cell_id``.
    """
    y0, x0 = float(tile_origin_um[0]), float(tile_origin_um[1])

    def voxel_to_um(coords: np.ndarray) -> np.ndarray:
        # coords shape (n, 2): (y_idx, x_idx). Voxel center =
        # origin + (idx + 0.5) * psz; the core adds ±half_voxel jitter
        # on top so the output is uniformly distributed within the voxel.
        out = (coords.astype(np.float64) + 0.5) * psz_um
        out[:, 0] += y0
        out[:, 1] += x0
        return out

    return place_via_per_cell_sdf(
        trs_df=trs_df,
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        sampling=(psz_um, psz_um),
        voxel_to_um=voxel_to_um,
        leak_lam_per_cell=leak_lam_per_cell,
        max_dist_per_cell=max_dist_per_cell,
        rng=rng,
    )
