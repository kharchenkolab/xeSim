"""Tissue-context-aware ghost cell sampler.

Ghost cells absorb all non-anchor signal in the 2D scene (out-of-plane
ghosts, process-like spillover, isolated background detections).

Algorithm:
- Sample ghost xy uniformly in tile (with optional low-density bias)
- For each ghost: find anchors within R_context (default 50 µm)
- Sample ghost type from local empirical distribution (Laplace-smoothed,
  global fallback if too few neighbors)
- Borrow contour from a same-type anchor, scaled 0.4–0.7
- Per-ghost molecule count ~ NegBinomial(μ, φ) → long-tailed

Calibration: ``ghost_count`` and ``mol_count_mu`` should be tuned so
total ghost molecules / total molecules ≈ ``noise_fraction``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class GhostCellRecord:
    """A single ghost cell — type, position, contour vertices."""

    cell_id: str
    cell_type: str
    centroid_x: float
    centroid_y: float
    contour_x: np.ndarray         # (n_vertices,) µm
    contour_y: np.ndarray
    target_mol_count: int
    source_anchor_id: str          # which anchor's contour we borrowed


def sample_ghost_cells(
    anchor_records: pd.DataFrame,
    tile_bounds_um: tuple[float, float, float, float],
    *,
    n_ghosts: int,
    rng: np.random.Generator | None = None,
    r_context_um: float = 50.0,
    min_neighbors: int = 3,
    laplace_smoothing: float = 1.0,
    f_size_min: float = 0.4,
    f_size_max: float = 0.7,
    mol_count_mu: float = 5.0,
    mol_count_dispersion: float = 1.0,
    global_type_distribution: dict[str, float] | None = None,
) -> list[GhostCellRecord]:
    """Sample ghost cells given anchor cells in the tile and surrounds.

    Parameters
    ----------
    anchor_records
        DataFrame with columns ``cell_id, cell_type, centroid_x, centroid_y,
        contour_x, contour_y``. ``contour_x``/``contour_y`` are np.ndarray
        of polygon vertices in µm. Should include anchors near the tile
        (within ``r_context_um``) — not just inside the tile, so ghosts
        near tile edges have context.
    tile_bounds_um
        ``(xmin, xmax, ymin, ymax)`` µm. Ghost cell centroids sampled
        inside this rectangle.
    n_ghosts
        Number of ghost cells to sample. Caller chooses based on
        noise-fraction calibration.
    r_context_um
        Radius for computing local anchor type distribution.
    min_neighbors
        Below this neighbor count, fall back to ``global_type_distribution``.
    laplace_smoothing
        Add this pseudo-count to each type before sampling.
    f_size_min, f_size_max
        Range of contour scale factor (relative to borrowed anchor).
    mol_count_mu, mol_count_dispersion
        NegBin(mean=μ, dispersion=φ) parameters for per-ghost molecule count.
        Geometric ≈ NegBin(μ=μ, φ=1.0).
    global_type_distribution
        Bundle-wide fallback distribution. If None, defaults to uniform
        across observed anchor types.

    Returns
    -------
    list of GhostCellRecord
    """
    rng = rng or np.random.default_rng(0)
    if n_ghosts <= 0:
        return []
    xmin, xmax, ymin, ymax = tile_bounds_um

    if len(anchor_records) == 0:
        return []

    # Build per-type contour pool: for each type, anchors whose contour we can borrow
    anchor_records = anchor_records.reset_index(drop=True)
    types_observed = list(sorted(set(anchor_records["cell_type"].dropna().astype(str))))
    if not types_observed:
        return []

    # Global type distribution fallback
    if global_type_distribution is None:
        counts = anchor_records["cell_type"].value_counts().to_dict()
        total = float(sum(counts.values())) or 1.0
        global_type_distribution = {t: counts.get(t, 0) / total for t in types_observed}

    # Per-type anchor index pool (rows with that type)
    type_to_indices: dict[str, np.ndarray] = {}
    for t in types_observed:
        m = anchor_records["cell_type"].astype(str) == t
        type_to_indices[t] = np.where(m.to_numpy())[0]

    # KDTree over anchor centroids for fast local lookup
    coords = anchor_records[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    tree = cKDTree(coords)

    # Sample n_ghosts uniform random xy positions
    xs = rng.uniform(xmin, xmax, size=n_ghosts)
    ys = rng.uniform(ymin, ymax, size=n_ghosts)

    ghosts: list[GhostCellRecord] = []
    for i in range(n_ghosts):
        cx, cy = float(xs[i]), float(ys[i])

        # Local anchor type distribution (Laplace-smoothed)
        nbr_idx = tree.query_ball_point([cx, cy], r_context_um)
        if len(nbr_idx) < min_neighbors:
            # Fallback to global
            type_dist = global_type_distribution
        else:
            local_types = anchor_records["cell_type"].iloc[nbr_idx].astype(str)
            local_counts = local_types.value_counts().to_dict()
            # Laplace smoothing: every observed type gets +laplace_smoothing
            smoothed = {t: local_counts.get(t, 0.0) + laplace_smoothing
                          for t in types_observed}
            total = sum(smoothed.values())
            type_dist = {t: v / total for t, v in smoothed.items()}

        # Sample type
        ts = list(type_dist.keys())
        ps = np.asarray([type_dist[t] for t in ts], dtype=np.float64)
        ps = ps / ps.sum()
        ghost_type = ts[int(rng.choice(len(ts), p=ps))]

        # Pick a same-type anchor to borrow a contour from
        candidates = type_to_indices.get(ghost_type, np.array([], dtype=int))
        if len(candidates) == 0:
            # type with no anchor in this tile's record list - skip
            continue
        src_idx = int(rng.choice(candidates))
        src = anchor_records.iloc[src_idx]
        src_contour_x = np.asarray(src["contour_x"], dtype=np.float64)
        src_contour_y = np.asarray(src["contour_y"], dtype=np.float64)
        src_cx = float(src["centroid_x"]); src_cy = float(src["centroid_y"])

        # Translate to ghost position, scale by random factor
        scale = float(rng.uniform(f_size_min, f_size_max))
        ghost_contour_x = cx + (src_contour_x - src_cx) * scale
        ghost_contour_y = cy + (src_contour_y - src_cy) * scale

        # Molecule count via NB (parameterized as gamma-poisson)
        # NB(mean=μ, dispersion=φ): variance = μ + μ²/φ
        # When φ→∞ it's Poisson; when φ=1 it's geometric.
        # NB sample: gamma(α=φ, β=φ/μ) → Poisson(λ=gamma).
        lam = rng.gamma(shape=mol_count_dispersion,
                         scale=mol_count_mu / mol_count_dispersion)
        n_mol = int(rng.poisson(lam=lam))

        ghosts.append(GhostCellRecord(
            cell_id=f"ghost_{i:06d}",
            cell_type=ghost_type,
            centroid_x=cx, centroid_y=cy,
            contour_x=ghost_contour_x,
            contour_y=ghost_contour_y,
            target_mol_count=n_mol,
            source_anchor_id=str(src["cell_id"]),
        ))

    return ghosts


def calibrate_n_ghosts(
    target_noise_fraction: float,
    n_anchor_molecules: int,
    mol_count_mu: float,
) -> int:
    """Pick n_ghosts so total ghost molecules / total ≈ target_noise_fraction.

    Total = anchor + ghost. ghost = n_ghosts × E[mol_count] = n_ghosts × μ.
    noise_fraction = ghost / (anchor + ghost)
    → n_ghosts × μ = noise_fraction × (anchor + n_ghosts × μ)
    → n_ghosts × μ × (1 − noise_fraction) = noise_fraction × anchor
    → n_ghosts = noise_fraction × anchor / (μ × (1 − noise_fraction))
    """
    if target_noise_fraction <= 0 or target_noise_fraction >= 1:
        raise ValueError("target_noise_fraction must be in (0, 1)")
    if mol_count_mu <= 0:
        raise ValueError("mol_count_mu must be > 0")
    n = target_noise_fraction * n_anchor_molecules / (mol_count_mu * (1 - target_noise_fraction))
    return max(0, int(round(n)))


__all__ = ["GhostCellRecord", "sample_ghost_cells", "calibrate_n_ghosts"]
