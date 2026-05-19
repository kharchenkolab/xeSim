"""Read an existing xeSim synth bundle and rebuild the inputs `emit_2d` /
`emit_3d` need: cell records + rasterised `cell_label` / `nucleus_label`.

Used by ``re-emit-molecules`` to skip the slow rendering step — we
reuse the bundle's morphology image and cell geometries as-is, and
only re-sample transcripts against a new STpuppeteer config.

Bundle artifacts read (per the synth-bundle layout xeSim writes):

- ``cells.parquet`` — public Xenium schema (cell_id, centroid, areas, counts).
- ``cell_boundaries.parquet`` / ``nucleus_boundaries.parquet`` — polygon vertices.
- ``ground_truth/cells_synth.parquet`` — cell_type, is_ghost, source.
- ``ground_truth/cells_3d.parquet`` (2.5D only) — z_center, z_extent, tilt.
- ``experiment.xenium`` — pixel_size, z_step, tile_bounds, n_z.

The morphology image, gene_panel, and renderer artifacts are NOT read —
re-emit doesn't need them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mode detection + metadata
# ---------------------------------------------------------------------------


@dataclass
class BundleMeta:
    """Parsed ``experiment.xenium`` + scene-mode flag."""
    scene_mode: str                              # "2d" or "2.5d"
    pixel_size_um: float
    z_step_um: float                             # 0.0 in pure-2D bundles
    tile_bounds_um: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    n_z: int                                     # 1 for 2D
    n_cells: int


def read_bundle_meta(bundle_dir: Path) -> BundleMeta:
    """Parse ``experiment.xenium`` and infer 2D vs 2.5D."""
    with open(bundle_dir / "experiment.xenium") as f:
        exp = json.load(f)
    # xeSim writes scene_mode under synth_metadata; fall back to n_z heuristic.
    synth_meta = exp.get("synth_metadata", {}) or {}
    scene_mode = synth_meta.get("scene_mode")
    n_z = int(synth_meta.get("n_z", 1))
    if scene_mode is None:
        # Heuristic for bundles missing synth_metadata: n_z > 1 → 2.5D.
        scene_mode = "2.5d" if n_z > 1 else "2d"
    tb = exp.get("tile_bounds_um")
    if tb is None:
        # Older bundles may lack tile_bounds — leave 0,0 and warn elsewhere.
        tb = (0.0, 0.0, 0.0, 0.0)
    return BundleMeta(
        scene_mode=scene_mode,
        pixel_size_um=float(exp["pixel_size"]),
        z_step_um=float(exp.get("z_step_size", 0.0)),
        tile_bounds_um=tuple(float(v) for v in tb),
        n_z=n_z,
        n_cells=int(exp.get("num_cells", 0)),
    )


# ---------------------------------------------------------------------------
# Cell-record assembly
# ---------------------------------------------------------------------------


def _load_polygons_by_id(parquet_path: Path) -> dict[str, np.ndarray]:
    """Return ``{cell_id: (n_vertices, 2) ndarray of (x, y) µm}``.

    The synth bundle writes cell_boundaries.parquet with columns
    cell_id, vertex_x, vertex_y, label_id; the absolute polygon vertices
    are what we want here.
    """
    df = pd.read_parquet(parquet_path)
    if not {"cell_id", "vertex_x", "vertex_y"}.issubset(df.columns):
        raise ValueError(
            f"{parquet_path} missing expected columns; got {df.columns.tolist()}"
        )
    out: dict[str, np.ndarray] = {}
    for cid, grp in df.groupby("cell_id", sort=False):
        out[str(cid)] = grp[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float32)
    return out


def _load_cells_synth(bundle_dir: Path) -> pd.DataFrame:
    """Read ground_truth/cells_synth.parquet — cell_type + provenance."""
    return pd.read_parquet(bundle_dir / "ground_truth" / "cells_synth.parquet")


# ---------------------------------------------------------------------------
# 2.5D rasterisation: rebuild CellRecord + cell_label_3d / nucleus_label_3d
# ---------------------------------------------------------------------------


def build_cell_records_25d(
    bundle_dir: Path,
    meta: BundleMeta,
) -> tuple[list, np.ndarray]:
    """Build CellRecord-shaped objects + the z_slices_um array.

    CellRecord's required fields (per sdf_tess.py) are: cell_idx, cell_id,
    cell_type, xy_seed, z_center, z_extent, t_x, t_y, t_z, template_xs,
    template_ys. We pull the first 9 from ground_truth/cells_3d.parquet
    and reconstruct templates by subtracting the centroid from absolute
    polygon vertices in cell_boundaries.parquet.
    """
    from ..scene_2_5d.sdf_tess import CellRecord  # noqa: F401 — local import alias

    cells_3d = pd.read_parquet(bundle_dir / "ground_truth" / "cells_3d.parquet")
    poly_xy = _load_polygons_by_id(bundle_dir / "cell_boundaries.parquet")

    records: list = []
    missing_poly = 0
    for row in cells_3d.itertuples(index=False):
        cid = str(row.cell_id)
        if cid not in poly_xy:
            # cells_3d may carry unobserved cells without a polygon; skip those —
            # they didn't emit transcripts in the original render either.
            if getattr(row, "is_unobserved", False):
                missing_poly += 1
                continue
            # Observed cell with no polygon — bundle malformed; skip with note.
            missing_poly += 1
            continue
        xy = poly_xy[cid]
        # Templates are vertex offsets from the cell's xy seed (centroid).
        cx, cy = float(row.centroid_x), float(row.centroid_y)
        records.append(CellRecord(
            cell_idx=int(row.cell_idx),
            cell_id=cid,
            cell_type=str(row.cell_type) if row.cell_type is not None else "",
            xy_seed=(cx, cy),
            z_center=float(row.z_center_um),
            z_extent=float(row.z_extent_um),
            t_x=float(row.t_x),
            t_y=float(row.t_y),
            t_z=float(row.t_z),
            template_xs=(xy[:, 0] - cx).astype(np.float32),
            template_ys=(xy[:, 1] - cy).astype(np.float32),
        ))

    # z_slices_um: regular grid using z_step from experiment.xenium.
    # xeSim's 2.5D writer uses 12 planes × 3 µm by default; we mirror.
    z_slices_um = np.arange(meta.n_z, dtype=np.float64) * meta.z_step_um
    return records, z_slices_um


def rasterize_25d(
    records: list,
    z_slices_um: np.ndarray,
    meta: BundleMeta,
    *,
    use_nuclei: bool = False,
    bundle_dir: Path | None = None,
) -> np.ndarray:
    """Reproduce the (n_z, H, W) cell_label_3d or nucleus_label_3d via the
    same SDF tessellator the original render used.

    When ``use_nuclei`` is False we rasterise the CellRecord polygons
    directly (the cell bodies). When True we substitute each cell's
    template with nucleus polygons from ``nucleus_boundaries.parquet``
    so the output matches the nucleus_label produced by the original
    bundle's nucleus stamping pass.
    """
    from ..scene_2_5d.bodies_3d import stack_z_labels

    xmin, ymin, xmax, ymax = meta.tile_bounds_um
    psz = meta.pixel_size_um
    h_tile = int(round((ymax - ymin) / psz))
    w_tile = int(round((xmax - xmin) / psz))

    cells_for_raster = records
    if use_nuclei:
        if bundle_dir is None:
            raise ValueError("bundle_dir required when use_nuclei=True")
        nuc_poly = _load_polygons_by_id(bundle_dir / "nucleus_boundaries.parquet")
        # Swap each record's template for the corresponding nucleus polygon.
        # Skip cells without a nucleus polygon (they get no nuclear voxels).
        from ..scene_2_5d.sdf_tess import CellRecord
        cells_for_raster = []
        for c in records:
            if c.cell_id not in nuc_poly:
                continue
            xy = nuc_poly[c.cell_id]
            cx, cy = c.xy_seed
            cells_for_raster.append(CellRecord(
                cell_idx=c.cell_idx, cell_id=c.cell_id, cell_type=c.cell_type,
                xy_seed=c.xy_seed,
                z_center=c.z_center, z_extent=c.z_extent,
                t_x=c.t_x, t_y=c.t_y, t_z=c.t_z,
                template_xs=(xy[:, 0] - cx).astype(np.float32),
                template_ys=(xy[:, 1] - cy).astype(np.float32),
            ))

    return stack_z_labels(
        cells_for_raster, z_slices_um.tolist(),
        tile_origin_um=(xmin, ymin),
        tile_size_px=(h_tile, w_tile),
        pixel_size_um=psz,
    )


# ---------------------------------------------------------------------------
# 2D rasterisation: polygons → cell_label / nucleus_label (single plane)
# ---------------------------------------------------------------------------


def build_cell_records_2d(
    bundle_dir: Path,
    meta: BundleMeta,
) -> list:
    """Build MechanisticCell-shaped records from cells_synth + boundaries.

    Reads both anchors and ghosts. Anchors come from cell_boundaries.parquet;
    ghost polygons (when present in the bundle) come from
    ground_truth/ghost_cell_boundaries.parquet. Each record carries
    ``provenance['is_ghost']`` so cell_adapter can filter ghosts out at
    emission time, while the rasteriser below still paints both into
    cell_label so leak placement sees the same geometry the renderer did.

    The 2D rasteriser assigns label = enumeration order starting at 1, so
    we mirror that by walking cells_synth in row order.
    """
    cells_synth = _load_cells_synth(bundle_dir)
    anchor_polys = _load_polygons_by_id(bundle_dir / "cell_boundaries.parquet")
    # Ghost polygons live in a separate file (per scene_2d/bundle_writer.py:20);
    # absent when the original explain ran with --no-ghosts.
    ghost_path = bundle_dir / "ground_truth" / "ghost_cell_boundaries.parquet"
    ghost_polys = (_load_polygons_by_id(ghost_path)
                       if ghost_path.exists() else {})

    records: list = []
    for i, row in enumerate(cells_synth.itertuples(index=False), start=1):
        cid = str(row.cell_id)
        is_ghost = bool(getattr(row, "is_ghost", False))
        # Lookup polygon in the right pool. Skip with a debug note rather
        # than warn — a bundle generated with --no-ghosts still has rows
        # in cells_synth flagged is_ghost=True if any tier-2 retyping
        # left them there, but we don't have geometry to rasterise them.
        pool = ghost_polys if is_ghost else anchor_polys
        if cid not in pool:
            continue
        records.append(SimpleNamespace(
            cell_id=cid,
            label=i,                  # integer key into cell_label
            cell_type=str(row.cell_type) if row.cell_type is not None else "",
            provenance={"is_ghost": is_ghost},
        ))
    return records


def rasterize_2d(
    records: list,
    bundle_dir: Path,
    meta: BundleMeta,
    *,
    use_nuclei: bool = False,
) -> np.ndarray:
    """Rasterise both anchor and ghost polygons into an (H, W) int32 label array.

    Anchors and ghosts share the same int32 image and same label numbering
    (assigned by build_cell_records_2d in row order). cell_adapter will
    filter ghosts out at emission time, but their pixels stay in
    cell_label so per-cell SDF halos can land inside ghost territory and
    the leak placement matches the rendered morphology.

    First-write-wins by record order; in practice 2D anchor cells are
    Voronoi-disjoint and ghosts are placed by the original render to
    potentially overlap anchors. The order of records (anchor-first if
    cells_synth was written that way) determines who "wins" overlap pixels.
    """
    from skimage.draw import polygon as skpoly

    if use_nuclei:
        anchor_polys = _load_polygons_by_id(bundle_dir / "nucleus_boundaries.parquet")
        ghost_nuc_path = bundle_dir / "ground_truth" / "ghost_nucleus_boundaries.parquet"
        ghost_polys = (_load_polygons_by_id(ghost_nuc_path)
                           if ghost_nuc_path.exists() else {})
    else:
        anchor_polys = _load_polygons_by_id(bundle_dir / "cell_boundaries.parquet")
        ghost_path = bundle_dir / "ground_truth" / "ghost_cell_boundaries.parquet"
        ghost_polys = (_load_polygons_by_id(ghost_path)
                           if ghost_path.exists() else {})

    xmin, ymin, xmax, ymax = meta.tile_bounds_um
    psz = meta.pixel_size_um
    h = int(round((ymax - ymin) / psz))
    w = int(round((xmax - xmin) / psz))
    label_img = np.zeros((h, w), dtype=np.int32)

    for c in records:
        is_ghost = bool((c.provenance or {}).get("is_ghost", False))
        pool = ghost_polys if is_ghost else anchor_polys
        if c.cell_id not in pool:
            continue
        xy = pool[c.cell_id]
        # Convert absolute µm vertices → tile-local pixel coords.
        xs_px = (xy[:, 0] - xmin) / psz
        ys_px = (xy[:, 1] - ymin) / psz
        rr, cc = skpoly(ys_px, xs_px, shape=(h, w))
        # First-write-wins: only set pixels that are still 0.
        mask = label_img[rr, cc] == 0
        label_img[rr, cc] = np.where(mask, c.label, label_img[rr, cc])
    return label_img


# ---------------------------------------------------------------------------
# Top-level: load a bundle ready for re-emission
# ---------------------------------------------------------------------------


@dataclass
class LoadedBundle:
    """Everything ``emit_2d`` / ``emit_3d`` need, derived from disk."""
    meta: BundleMeta
    cells: list                      # MechanisticCell (2D) or CellRecord (2.5D) shaped
    cell_label: np.ndarray           # (H, W) for 2D, (n_z, H, W) for 2.5D
    nucleus_label: np.ndarray        # same shape
    z_slices_um: np.ndarray | None   # set for 2.5D only


def load_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """Load + rasterise a synth bundle for re-emission.

    Detects 2D vs 2.5D from ``experiment.xenium``'s synth_metadata and
    runs the appropriate rasteriser. Returns a ``LoadedBundle`` whose
    fields can be passed directly to ``emit_2d`` or ``emit_3d``.
    """
    bundle_dir = Path(bundle_dir)
    meta = read_bundle_meta(bundle_dir)

    if meta.scene_mode == "2.5d":
        records, z_slices = build_cell_records_25d(bundle_dir, meta)
        cell_label_3d = rasterize_25d(records, z_slices, meta,
                                          use_nuclei=False, bundle_dir=bundle_dir)
        nucleus_label_3d = rasterize_25d(records, z_slices, meta,
                                             use_nuclei=True, bundle_dir=bundle_dir)
        return LoadedBundle(
            meta=meta, cells=records,
            cell_label=cell_label_3d,
            nucleus_label=nucleus_label_3d,
            z_slices_um=z_slices,
        )
    elif meta.scene_mode == "2d":
        records = build_cell_records_2d(bundle_dir, meta)
        cell_label = _load_or_rasterize_2d(
            records, bundle_dir, meta, use_nuclei=False)
        nucleus_label = _load_or_rasterize_2d(
            records, bundle_dir, meta, use_nuclei=True)
        return LoadedBundle(
            meta=meta, cells=records,
            cell_label=cell_label,
            nucleus_label=nucleus_label,
            z_slices_um=None,
        )
    else:
        raise ValueError(f"unknown scene_mode {meta.scene_mode!r} "
                         f"in experiment.xenium synth_metadata")


# ---------------------------------------------------------------------------
# Cached rasterisation: skip the per-polygon redraw on subsequent loads
# ---------------------------------------------------------------------------


def _expected_shape_2d(meta) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = meta.tile_bounds_um
    psz = meta.pixel_size_um
    return (int(round((ymax - ymin) / psz)),
            int(round((xmax - xmin) / psz)))


def _load_or_rasterize_2d(records, bundle_dir, meta, *, use_nuclei: bool) -> np.ndarray:
    """Return the cached rasterised label array if present, else rasterise
    once and persist it to ``ground_truth/`` for subsequent loads.

    Polygon rasterisation of a 140k-cell pancreas bundle takes ~5–7 min
    on one core; caching turns that into ~1 s mmap-load on every
    subsequent re-emit. Cache is regenerated whenever the shape disagrees
    with what ``meta`` would imply (defensive against bundle edits) or
    when the file is missing.
    """
    name = "nucleus_label.npy" if use_nuclei else "cell_label.npy"
    cache_path = Path(bundle_dir) / "ground_truth" / name
    expected = _expected_shape_2d(meta)
    # Source polygon files — cache is stale if any is newer than the cache.
    poly_sources = []
    if use_nuclei:
        poly_sources.append(Path(bundle_dir) / "nucleus_boundaries.parquet")
        gp = Path(bundle_dir) / "ground_truth" / "ghost_nucleus_boundaries.parquet"
        if gp.exists(): poly_sources.append(gp)
    else:
        poly_sources.append(Path(bundle_dir) / "cell_boundaries.parquet")
        gp = Path(bundle_dir) / "ground_truth" / "ghost_cell_boundaries.parquet"
        if gp.exists(): poly_sources.append(gp)
    if cache_path.exists():
        try:
            cache_mtime = cache_path.stat().st_mtime
            stale = any(p.stat().st_mtime > cache_mtime
                            for p in poly_sources if p.exists())
            if not stale:
                arr = np.load(cache_path, mmap_mode="r")
                if arr.shape == expected:
                    return arr
        except Exception:
            pass  # corrupt cache; fall through to regeneration
    arr = rasterize_2d(records, bundle_dir, meta, use_nuclei=use_nuclei)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    except OSError:
        # Read-only bundle, etc. — silently skip caching; re-emit still works.
        pass
    return arr
