"""Per-cell 2D shape template bank.

Observed cells return their own Xenium 2D contour. Unobserved cells
(synthetic, placed in off-plane z-bands) draw a contour from a per-type
+ per-zone bank of real Xenium contours from the bundle — biology comes
from the data, not from parametric shape models.

"Zone" = cellAdmix dominant factor: cells in the same tissue context
(islet, exocrine, stromal, vascular, etc) share shape statistics.

Cached at `<bundle_parent>/_xesim_z_attrs/templates_bank.parquet` so
the bank build is one-time per bundle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def templates_bank_path(bundle_path: str | Path) -> Path:
    return Path(bundle_path).resolve().parent / "_xesim_z_attrs" / "templates_bank.parquet"


def nucleus_bank_path(bundle_path: str | Path) -> Path:
    return Path(bundle_path).resolve().parent / "_xesim_z_attrs" / "nucleus_bank.parquet"


def build_template_bank(
    bundle_path: str | Path,
    *,
    cell_factor_fractions: pd.DataFrame,
    cell_annotation_df: pd.DataFrame,
    overwrite: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Build a per-cell (cell_id, type, zone, polygon_vertices_centered).

    Each row is one cell from the bundle's `cell_boundaries.parquet`,
    with the polygon **centered at origin** (xy_centroid subtracted)
    so it can be used as a portable template — drop it at any
    (x_seed, y_seed) by adding back.

    Returns columns: `cell_id, cell_type, zone, area_um2, vertex_x_rel,
    vertex_y_rel, n_vertices`. `vertex_x_rel` and `vertex_y_rel` are
    list[float] (vertex coords relative to centroid).

    `cell_factor_fractions` should have columns `cell_id`,
    `dominant_factor` (used as zone). `cell_annotation_df` should have
    `cell_id` and `merged_annotation` (used as cell_type).
    """
    out_path = templates_bank_path(bundle_path)
    if out_path.exists() and not overwrite:
        if progress:
            print(f"[templates] loading cached bank {out_path}")
        return pd.read_parquet(out_path)

    cb_path = Path(bundle_path) / "cell_boundaries.parquet"
    cb = pd.read_parquet(cb_path)
    if progress:
        print(f"[templates] reading polygons from {cb_path}: {len(cb)} vertices")

    cf = cell_factor_fractions[["cell_id", "dominant_factor"]].drop_duplicates("cell_id")
    cf = cf.rename(columns={"dominant_factor": "zone"})
    ann_col = "merged_annotation" if "merged_annotation" in cell_annotation_df.columns else "_cell_type"
    ann = cell_annotation_df[["cell_id", ann_col]].drop_duplicates("cell_id")
    ann = ann.rename(columns={ann_col: "cell_type"})

    # Compute per-cell centroid + relative vertices via groupby
    rows = []
    n_cells_total = cb["cell_id"].nunique()
    n_progress = max(1, n_cells_total // 20)
    n_done = 0
    for cid, g in cb.groupby("cell_id", sort=False):
        xs = g["vertex_x"].to_numpy(dtype=np.float32)
        ys = g["vertex_y"].to_numpy(dtype=np.float32)
        if len(xs) < 3:
            continue
        cx = float(xs.mean()); cy = float(ys.mean())
        xs_rel = xs - cx
        ys_rel = ys - cy
        # Shoelace area
        area = 0.5 * float(abs(np.dot(xs_rel, np.roll(ys_rel, 1)) -
                                np.dot(ys_rel, np.roll(xs_rel, 1))))
        rows.append({
            "cell_id": str(cid),
            "centroid_x": cx, "centroid_y": cy,
            "area_um2": area,
            "vertex_x_rel": xs_rel.tolist(),
            "vertex_y_rel": ys_rel.tolist(),
            "n_vertices": int(len(xs)),
        })
        n_done += 1
        if progress and n_done % n_progress == 0:
            print(f"  [{n_done}/{n_cells_total}]")
    df = pd.DataFrame(rows)
    # Merge type + zone
    df = df.merge(ann, on="cell_id", how="left")
    df = df.merge(cf, on="cell_id", how="left")
    df["cell_type"] = df["cell_type"].fillna("unknown")
    df["zone"] = df["zone"].fillna(0).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if progress:
        print(f"[templates] wrote {out_path}: {len(df)} cells, "
              f"types: {df['cell_type'].nunique()}, zones: {df['zone'].nunique()}")
    return df


class TemplateBank:
    """Wraps the per-cell template DataFrame to give a fast sampler API.

    Usage:
        bank = TemplateBank.from_bundle(bundle_path, ...)
        # Observed cells: get their own polygon
        polygon = bank.get_observed_template(cell_id)
        # Unobserved cells: sample a polygon for a given type+zone
        polygon = bank.sample_template(cell_type, zone, rng)
    """
    def __init__(self, df: pd.DataFrame, rng: np.random.Generator | None = None):
        self.df = df.reset_index(drop=True)
        self.rng = rng or np.random.default_rng(0)
        # Build index for fast (type, zone) lookup
        self._by_type_zone: dict[tuple[str, int], np.ndarray] = {}
        for (t, z), g in self.df.groupby(["cell_type", "zone"]):
            self._by_type_zone[(t, int(z))] = g.index.to_numpy()
        self._by_type: dict[str, np.ndarray] = {}
        for t, g in self.df.groupby("cell_type"):
            self._by_type[t] = g.index.to_numpy()
        self._by_id: dict[str, int] = {cid: i for i, cid in enumerate(self.df["cell_id"])}

    @classmethod
    def from_bundle(cls, bundle_path: str | Path, *,
                      cell_factor_fractions: pd.DataFrame,
                      cell_annotation_df: pd.DataFrame,
                      rng: np.random.Generator | None = None,
                      ) -> "TemplateBank":
        df = build_template_bank(
            bundle_path,
            cell_factor_fractions=cell_factor_fractions,
            cell_annotation_df=cell_annotation_df,
        )
        return cls(df, rng=rng)

    def get_observed_template(self, cell_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (xs_rel, ys_rel) for the observed cell. Centered at origin.

        Raises KeyError if cell_id not in bank.
        """
        i = self._by_id[str(cell_id)]
        row = self.df.iloc[i]
        return (np.asarray(row["vertex_x_rel"], dtype=np.float32),
                np.asarray(row["vertex_y_rel"], dtype=np.float32))

    def sample_template(self, cell_type: str, zone: int | None = None,
                          rng: np.random.Generator | None = None,
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a template polygon for (cell_type, zone). Returns
        (xs_rel, ys_rel) centered at origin. Falls back to type-only
        sample if (type, zone) is empty; falls back to global random
        if type is empty too."""
        rng = rng or self.rng
        idx_pool = self._by_type_zone.get((cell_type, int(zone) if zone is not None else -1))
        if idx_pool is None or len(idx_pool) == 0:
            idx_pool = self._by_type.get(cell_type)
        if idx_pool is None or len(idx_pool) == 0:
            idx_pool = np.arange(len(self.df))
        choice = int(rng.choice(idx_pool))
        row = self.df.iloc[choice]
        return (np.asarray(row["vertex_x_rel"], dtype=np.float32),
                np.asarray(row["vertex_y_rel"], dtype=np.float32))


def build_nucleus_bank(
    bundle_path: str | Path,
    *,
    cell_factor_fractions: pd.DataFrame,
    cell_annotation_df: pd.DataFrame,
    overwrite: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Build per-cell nucleus template bank, same shape as cell template
    bank but from nucleus_boundaries.parquet.

    Cells without a nucleus in the bundle get no row in this bank.
    """
    out_path = nucleus_bank_path(bundle_path)
    if out_path.exists() and not overwrite:
        if progress:
            print(f"[nucleus_bank] loading cached {out_path}")
        return pd.read_parquet(out_path)
    nb_path = Path(bundle_path) / "nucleus_boundaries.parquet"
    if not nb_path.exists():
        if progress:
            print(f"[nucleus_bank] {nb_path} not found; returning empty bank")
        return pd.DataFrame()
    nb = pd.read_parquet(nb_path)
    if progress:
        print(f"[nucleus_bank] reading {len(nb)} vertices from {nb_path}")

    cf = cell_factor_fractions[["cell_id", "dominant_factor"]].drop_duplicates("cell_id")
    cf = cf.rename(columns={"dominant_factor": "zone"})
    ann_col = "merged_annotation" if "merged_annotation" in cell_annotation_df.columns else "_cell_type"
    ann = cell_annotation_df[["cell_id", ann_col]].drop_duplicates("cell_id")
    ann = ann.rename(columns={ann_col: "cell_type"})

    rows = []
    for cid, g in nb.groupby("cell_id", sort=False):
        xs = g["vertex_x"].to_numpy(dtype=np.float32)
        ys = g["vertex_y"].to_numpy(dtype=np.float32)
        if len(xs) < 3:
            continue
        cx = float(xs.mean()); cy = float(ys.mean())
        xs_rel = xs - cx; ys_rel = ys - cy
        area = 0.5 * float(abs(np.dot(xs_rel, np.roll(ys_rel, 1)) -
                                np.dot(ys_rel, np.roll(xs_rel, 1))))
        rows.append({
            "cell_id": str(cid),
            "centroid_x": cx, "centroid_y": cy,
            "area_um2": area,
            "vertex_x_rel": xs_rel.tolist(),
            "vertex_y_rel": ys_rel.tolist(),
            "n_vertices": int(len(xs)),
        })
    df = pd.DataFrame(rows)
    df = df.merge(ann, on="cell_id", how="left")
    df = df.merge(cf, on="cell_id", how="left")
    df["cell_type"] = df["cell_type"].fillna("unknown")
    df["zone"] = df["zone"].fillna(0).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if progress:
        print(f"[nucleus_bank] wrote {out_path}: {len(df)} cells")
    return df


class NucleusBank(TemplateBank):
    """Same API as TemplateBank, but built from nucleus_boundaries.parquet."""
    @classmethod
    def from_bundle(cls, bundle_path: str | Path, *,
                      cell_factor_fractions: pd.DataFrame,
                      cell_annotation_df: pd.DataFrame,
                      rng: np.random.Generator | None = None,
                      ) -> "NucleusBank":
        df = build_nucleus_bank(
            bundle_path, cell_factor_fractions=cell_factor_fractions,
            cell_annotation_df=cell_annotation_df,
        )
        return cls(df, rng=rng)


__all__ = ["build_template_bank", "templates_bank_path", "TemplateBank",
           "build_nucleus_bank", "nucleus_bank_path", "NucleusBank"]
