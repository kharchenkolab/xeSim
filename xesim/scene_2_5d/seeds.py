"""Sample unobserved cell seeds (cells 10x missed in 2D segmentation).

Strategy: for a given region, add a fraction of synthetic cells beyond
the observed Xenium cells. Per-type under-segmentation rates come from
plan3d's per-type undersegmentation probabilities (~0.30 for Ductal,
~0.50 for Endothelial/Fibroblast, etc.).

Each new cell gets:
  - xy_seed: from the local zone density (so islet cells go in islet
    regions, exocrine cells in exocrine regions)
  - cell_type: sampled per-type by ratio
  - z_center: uniform within the imaged depth, biased to away-from-focal
    (where 10x would have detected it)
  - z_extent: per-type empirical distribution from observed z_attrs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Plan3d per-type undersegmentation probabilities (from
# `misc/plan3d.findings.md` Phase 1 population summary). Used as
# the fraction of observed cells that have a 10x-missed "sibling"
# nearby in z.
PLAN3D_UNDERSEG_PROBS = {
    "Ductal/tumor epithelial": 0.32,
    "Exocrine epithelial":     0.29,
    "Endothelial":             0.53,
    "Fibroblast / CAF":        0.49,
    "Immune":                  0.33,
    "Endocrine":               0.30,   # not in plan3d table; use middle
    "Mural / pericyte":        0.40,   # not in plan3d table; use middle
    "unknown":                 0.30,
}

DEFAULT_IMAGED_DEPTH_UM = 33.0


@dataclass(frozen=True)
class UnobservedSeed:
    xy_seed: tuple[float, float]
    cell_type: str
    z_center: float
    z_extent: float
    zone: int

    @property
    def cell_id(self) -> str:
        # Synthetic id, distinct from real Xenium ids
        return f"__unobs_{id(self):x}"


def _per_type_z_extent_dist(z_attrs: pd.DataFrame,
                              cell_annotation_df: pd.DataFrame,
                              ) -> dict[str, np.ndarray]:
    """For each cell type, an empirical sample of observed z_extents
    (to draw from for unobserved cells)."""
    ann_col = ("merged_annotation" if "merged_annotation" in cell_annotation_df.columns
                else "_cell_type")
    m = z_attrs.merge(cell_annotation_df[["cell_id", ann_col]].rename(
        columns={ann_col: "cell_type"}), on="cell_id", how="left")
    m["cell_type"] = m["cell_type"].fillna("unknown")
    m = m[np.isfinite(m["z_extent_um"]) & (m["z_extent_um"] > 0)]
    out = {}
    for t, g in m.groupby("cell_type"):
        out[t] = g["z_extent_um"].to_numpy(dtype=np.float32)
    return out


def _per_type_z_center_dist(z_attrs: pd.DataFrame,
                              cell_annotation_df: pd.DataFrame,
                              ) -> dict[str, np.ndarray]:
    """For each cell type, an empirical sample of observed z_centers — the
    tissue section's z-position/thickness.

    Real nuclei concentrate in the mid tissue-section (e.g. ~12-21µm of a 33µm
    z-stack), NOT uniformly across the acquisition depth. Unobserved cells are
    drawn from this measured distribution (they sit in the same section as
    observed cells), instead of the old edge-biased heuristic that placed them
    at the z-extremes where there is no tissue. Includes a ``__global__`` pool
    for sparse types.
    """
    ann_col = ("merged_annotation" if "merged_annotation" in cell_annotation_df.columns
                else "_cell_type")
    m = z_attrs.merge(cell_annotation_df[["cell_id", ann_col]].rename(
        columns={ann_col: "cell_type"}), on="cell_id", how="left")
    m["cell_type"] = m["cell_type"].fillna("unknown")
    m = m[np.isfinite(m["z_center_um"])]
    out = {}
    for t, g in m.groupby("cell_type"):
        out[t] = g["z_center_um"].to_numpy(dtype=np.float32)
    out["__global__"] = m["z_center_um"].to_numpy(dtype=np.float32)
    return out


def sample_unobserved_seeds(
    region_bounds_um: tuple[float, float, float, float],
    observed_cells: pd.DataFrame,
    *,
    z_attrs: pd.DataFrame,
    cell_annotation_df: pd.DataFrame,
    imaged_depth_um: float = DEFAULT_IMAGED_DEPTH_UM,
    underseg_probs: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[UnobservedSeed]:
    """Sample unobserved-cell seeds for a region.

    Parameters
    ----------
    region_bounds_um
        (xmin, ymin, xmax, ymax) in world µm.
    observed_cells
        DataFrame of observed cells in (or near) the region, with columns
        `cell_id, centroid_x, centroid_y, cell_type, zone`. xy used as
        the local density estimator; cell_type, zone used for per-type
        sampling.
    z_attrs
        Bundle cells_z.parquet contents (cell_id, z_center_um, z_extent_um).
    cell_annotation_df
        Bundle annotation table (cell_id, merged_annotation).
    imaged_depth_um
        Depth of the bundle's z-stack imaged volume.
    underseg_probs
        Per-type fraction of new cells to add. Defaults to
        `PLAN3D_UNDERSEG_PROBS`.
    rng
        Random generator.
    """
    rng = rng or np.random.default_rng(0)
    p = underseg_probs or PLAN3D_UNDERSEG_PROBS
    xmin, ymin, xmax, ymax = region_bounds_um

    # Per-type empirical z_extent + z_center distributions from observed cells
    z_ext_dist = _per_type_z_extent_dist(z_attrs, cell_annotation_df)
    z_ctr_dist = _per_type_z_center_dist(z_attrs, cell_annotation_df)

    seeds: list[UnobservedSeed] = []
    if len(observed_cells) == 0:
        return seeds

    # For each (cell_type, zone) group: take observed xy as a density
    # source; sample new cells at jittered locations near observed.
    by_type_zone = observed_cells.groupby(["cell_type", "zone"], dropna=False)
    for (ct, zone), g in by_type_zone:
        ct = str(ct); zone = int(zone) if pd.notnull(zone) else 0
        n_observed = len(g)
        frac = p.get(ct, 0.30)
        n_new = int(round(n_observed * frac))
        if n_new == 0:
            continue
        # Pick xy from observed centroids + add per-cell jitter
        idx_pool = g.index.to_numpy()
        chosen = rng.choice(idx_pool, size=n_new, replace=True)
        jitter_xy = rng.normal(scale=5.0, size=(n_new, 2)).astype(np.float32)
        for i, j in enumerate(chosen):
            row = g.loc[j]
            cx = float(row["centroid_x"]) + float(jitter_xy[i, 0])
            cy = float(row["centroid_y"]) + float(jitter_xy[i, 1])
            if not (xmin <= cx < xmax and ymin <= cy < ymax):
                continue
            # z_center from the observed section z-distribution: unobserved
            # cells sit within the same tissue section as observed cells
            # (mid-concentrated), NOT at the acquisition z-edges. (The old
            # edge-biased uniform contradicted the measured z_attrs
            # distribution and put in-focus nuclei on empty edge planes.)
            zc_pool = z_ctr_dist.get(ct)
            if zc_pool is None or len(zc_pool) < 20:
                zc_pool = z_ctr_dist.get("__global__")
            if zc_pool is not None and len(zc_pool) > 0:
                zc = float(rng.choice(zc_pool) + rng.normal(scale=1.5))
            else:
                zc = imaged_depth_um / 2.0
            zc = float(np.clip(zc, 0.0, imaged_depth_um))
            # z_extent from per-type empirical
            ze_pool = z_ext_dist.get(ct, np.array([15.0], dtype=np.float32))
            ze = float(rng.choice(ze_pool))
            seeds.append(UnobservedSeed(
                xy_seed=(cx, cy), cell_type=ct,
                z_center=zc, z_extent=ze, zone=zone,
            ))
    return seeds


__all__ = ["UnobservedSeed", "PLAN3D_UNDERSEG_PROBS", "sample_unobserved_seeds"]
