"""Per-cell z-attribute fitting + bundle-level persistence.

Bundle-level layout (matches Phase 2.G cellAdmix pattern):

    <bundle_parent>/_xesim_z_attrs/cells_z.parquet
        columns: cell_id, z_center_um, z_extent_um, z_confidence

The fitter is `xesim.canonicalize_z.fit_per_cell_z` (already exists).
This module just runs it across the whole bundle in tiles and stitches
the per-tile fits into one global table, then provides a loader that
caches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def bundle_z_attrs_dir(bundle_path: str | Path) -> Path:
    return Path(bundle_path).resolve().parent / "_xesim_z_attrs"


def cells_z_path(bundle_path: str | Path) -> Path:
    return bundle_z_attrs_dir(bundle_path) / "cells_z.parquet"


def _state_path(bundle_path: str | Path) -> Path:
    return bundle_z_attrs_dir(bundle_path) / "cells_z.state"


def _cache_state(bundle_path: str | Path) -> str | None:
    """'complete' | 'partial' | None. A legacy cache (parquet present, no
    state file) predates region-scoped fitting and is treated as complete —
    it could only have come from the old whole-bundle fit."""
    sp = _state_path(bundle_path)
    if sp.exists():
        s = sp.read_text().strip()
        return s if s in ("complete", "partial") else None
    return "complete" if cells_z_path(bundle_path).exists() else None


def _set_state(bundle_path: str | Path, state: str) -> None:
    sp = _state_path(bundle_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(state)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _merge_into_cache(out_path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """Union a region fit into the existing cache (new fits win for shared
    cell_ids), atomically. Returns the merged frame."""
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        merged = pd.concat(
            [existing[~existing["cell_id"].isin(df_new["cell_id"])], df_new],
            ignore_index=True)
    else:
        merged = df_new
    merged = merged.sort_values("cell_id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(merged, out_path)
    return merged


def load_cells_z(bundle_path: str | Path) -> pd.DataFrame | None:
    """Load cached per-cell z attrs if present, else None."""
    p = cells_z_path(bundle_path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def fit_bundle_cells_z(
    bundle_path: str | Path,
    *,
    tile_um: float = 1000.0,
    overlap_um: float = 40.0,
    overwrite: bool = False,
    progress: bool = True,
    num_workers: int = 8,
    bounds_um: tuple[float, float, float, float] | None = None,
    halo_um: float = 25.0,
) -> pd.DataFrame:
    """Fit per-cell z attributes over the whole bundle in tiles and
    write to ``<bundle_parent>/_xesim_z_attrs/cells_z.parquet``.

    Each cell appears in one or more tiles; the fit from the tile whose
    xy center is closest to the cell's centroid is kept. Cells with no
    successful fit (NaN result) get NaN in the output.

    Tile size 256 µm × 256 µm with 30 µm overlap ensures most cells'
    nuclei are fully within at least one tile. The overlap is important
    near tile boundaries.

    Returns the resulting DataFrame.
    """
    from ..canonicalize_z import fit_per_cell_z_for_crop
    from ..xenium import resolve_bundle
    from ..images import ome_image_shape
    from ..scene_2d.load_geometry import load_geometry
    from ..models import CropBox

    out_path = cells_z_path(bundle_path)
    # Whole-bundle short-circuit only: a region fit always (re-)fits its
    # region and merges, so it must not early-return on cache presence.
    if bounds_um is None and out_path.exists() and not overwrite:
        if progress:
            print(f"[fit_bundle_cells_z] {out_path} exists; loading.")
        return pd.read_parquet(out_path)

    bundle = resolve_bundle(bundle_path)
    morph_path = Path(bundle_path) / "morphology.ome.tif"
    if not morph_path.exists():
        raise FileNotFoundError(
            f"Bundle morphology.ome.tif not at {morph_path} — z fitting "
            f"requires the multi-z DAPI stack.")
    h_full, w_full = ome_image_shape(bundle.morphology_focus_paths[0])
    psz = float(bundle.pixel_size)
    x_max = w_full * psz
    y_max = h_full * psz

    # Grid extent: whole FOV, or just the requested region expanded by a halo
    # (so a small --tile/--region preview doesn't pay to fit the whole bundle;
    # the per-cell z-fit has no cross-cell coupling, so a region fit gives the
    # same per-cell results — the halo ensures edge cells get full footprints).
    if bounds_um is None:
        gx0, gy0, gx1, gy1 = 0.0, 0.0, x_max, y_max
    else:
        bx0, by0, bx1, by1 = bounds_um
        gx0, gy0 = max(0.0, bx0 - halo_um), max(0.0, by0 - halo_um)
        gx1, gy1 = min(x_max, bx1 + halo_um), min(y_max, by1 + halo_um)
    # A region that spans the whole FOV is a complete fit (e.g. a whole-bundle
    # request passed as explicit bounds), so it still earns the 'complete' mark.
    covers_full = (gx0 <= 0.0 and gy0 <= 0.0 and gx1 >= x_max and gy1 >= y_max)
    step = tile_um - overlap_um
    xs = list(np.arange(gx0, gx1, step)) or [gx0]
    ys = list(np.arange(gy0, gy1, step)) or [gy0]
    grid = [(x, y) for y in ys for x in xs]
    if progress:
        scope = "whole bundle" if bounds_um is None else "region"
        print(f"[fit_bundle_cells_z] {scope} {gx0:.0f}-{gx1:.0f}×{gy0:.0f}-{gy1:.0f} µm, "
              f"{len(grid)} tiles of {tile_um:.0f} µm")

    def _fit_one_tile(k_xy):
        k, (x0, y0) = k_xy
        x1 = min(x0 + tile_um, gx1)
        y1 = min(y0 + tile_um, gy1)
        if x1 - x0 < 2.0 or y1 - y0 < 2.0:
            return []
        crop = CropBox(xmin=x0, xmax=x1, ymin=y0, ymax=y1, crop_id=f"zfit_{k}")
        h_px = int(round((y1 - y0) / psz))
        w_px = int(round((x1 - x0) / psz))
        try:
            cl, nl, cell_ids, _, _ = load_geometry(bundle, crop, shape=(h_px, w_px))
        except Exception as e:
            return [("__error__", k, str(e))]
        if len(cell_ids) == 0:
            return []
        out = fit_per_cell_z_for_crop(
            morph_path, (x0, x1, y0, y1), psz, cl, nl, list(cell_ids),
            z_spacing_um=3.0,
        )
        tile_center_x = (x0 + x1) / 2.0
        tile_center_y = (y0 + y1) / 2.0
        rows = []
        # Per-cell xy centroid via scipy.ndimage.center_of_mass. cl values
        # are bundle-global labels (sparse big integers), so query with
        # np.unique(cl)[1:] which IS the actual label set in this tile and
        # IS positionally aligned with cell_ids (load_geometry guarantees).
        from scipy.ndimage import center_of_mass as ndi_center_of_mass
        actual_labels = np.unique(cl)
        actual_labels = actual_labels[actual_labels != 0]
        centroids = ndi_center_of_mass(cl > 0, labels=cl, index=actual_labels.tolist())
        for i, cid in enumerate(cell_ids):
            zc = float(out["z_center_um"][i])
            ze = float(out["z_extent_um"][i])
            zconf = float(out["z_confidence"][i])
            if not np.isfinite(zc):
                continue
            cy_px, cx_px = centroids[i]
            if not (np.isfinite(cy_px) and np.isfinite(cx_px)):
                continue
            cy_um = y0 + cy_px * psz
            cx_um = x0 + cx_px * psz
            dist_to_center = np.hypot(cx_um - tile_center_x, cy_um - tile_center_y)
            rows.append((str(cid), zc, ze, zconf, dist_to_center))
        return rows

    # Aggregate per-cell fits across tiles. Per cell, prefer the fit from
    # whichever tile's center is closest to the cell's xy centroid.
    by_cell: dict[str, tuple[float, float, float, float]] = {}
    completed = 0
    n_tile_progress = max(1, len(grid) // 20)
    tile_args = list(enumerate(grid))
    if num_workers <= 1:
        for ta in tile_args:
            rows = _fit_one_tile(ta)
            for row in rows:
                if isinstance(row, tuple) and row and row[0] == "__error__":
                    if progress:
                        print(f"  tile {row[1]}: load_geometry failed: {row[2]}")
                    continue
                cid, zc, ze, zconf, dist = row
                prev = by_cell.get(cid)
                if prev is None or dist < prev[3]:
                    by_cell[cid] = (zc, ze, zconf, dist)
            completed += 1
            if progress and completed % n_tile_progress == 0:
                print(f"  [tile {completed}/{len(grid)}] {len(by_cell)} cells fit so far")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_fit_one_tile, ta): ta for ta in tile_args}
            for fut in as_completed(futures):
                rows = fut.result()
                for row in rows:
                    if isinstance(row, tuple) and row and row[0] == "__error__":
                        if progress:
                            print(f"  tile {row[1]}: load_geometry failed: {row[2]}")
                        continue
                    cid, zc, ze, zconf, dist = row
                    prev = by_cell.get(cid)
                    if prev is None or dist < prev[3]:
                        by_cell[cid] = (zc, ze, zconf, dist)
                completed += 1
                if progress and completed % n_tile_progress == 0:
                    print(f"  [tile {completed}/{len(grid)}] {len(by_cell)} cells fit so far")

    rows = [{"cell_id": cid, "z_center_um": zc, "z_extent_um": ze,
              "z_confidence": zconf}
             for cid, (zc, ze, zconf, _dist) in by_cell.items()]
    df = pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bounds_um is None or covers_full:
        # Whole-bundle (or full-FOV region) fit → authoritative cache + marker.
        _atomic_write_parquet(df, out_path)
        _set_state(bundle_path, "complete")
        if progress:
            print(f"[fit_bundle_cells_z] wrote {out_path}: {len(df)} cells (complete)")
        return df
    # Region fit → merge into any existing cache; mark partial so a later
    # whole-bundle request knows to (re)fit the rest.
    merged = _merge_into_cache(out_path, df)
    _set_state(bundle_path, "partial")
    if progress:
        print(f"[fit_bundle_cells_z] merged {len(df)} region cells; "
              f"cache now {len(merged)} (partial)")
    return df


def load_or_fit_cells_z(
    bundle_path: str | Path,
    *,
    region_um: tuple[float, float, float, float] | None = None,
    halo_um: float = 25.0,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Load cached cells_z.parquet if present; otherwise fit and write.

    When ``region_um`` is given (a small --tile/--region preview), and no
    COMPLETE cache exists, fit only that region (+halo) and merge — so a
    preview doesn't trigger a whole-bundle fit (10-15 min on first touch).
    A complete cache covers any region, so it's used as-is. ``region_um=None``
    (whole-bundle / scene-first precompute) fits/uses the whole bundle.

    Returns DataFrame with columns [cell_id, z_center_um, z_extent_um,
    z_confidence].
    """
    fit_kwargs = fit_kwargs or {}
    state = _cache_state(bundle_path)
    if state == "complete":
        df = load_cells_z(bundle_path)
        if df is not None:
            return df
    if region_um is None:
        return fit_bundle_cells_z(bundle_path, **fit_kwargs)
    return fit_bundle_cells_z(bundle_path, bounds_um=tuple(region_um),
                              halo_um=halo_um, **fit_kwargs)


__all__ = [
    "bundle_z_attrs_dir",
    "cells_z_path",
    "load_cells_z",
    "fit_bundle_cells_z",
    "load_or_fit_cells_z",
]
