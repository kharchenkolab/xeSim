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

    The 2D rasteriser (below) assigns the integer label = enumeration order
    starting at 1, so we mirror that by walking cells_synth in row order.
    """
    cells_synth = _load_cells_synth(bundle_dir)
    poly_xy = _load_polygons_by_id(bundle_dir / "cell_boundaries.parquet")

    records: list = []
    for i, row in enumerate(cells_synth.itertuples(index=False), start=1):
        cid = str(row.cell_id)
        if cid not in poly_xy:
            continue
        records.append(SimpleNamespace(
            cell_id=cid,
            label=i,                  # integer key into cell_label
            cell_type=str(row.cell_type) if row.cell_type is not None else "",
            provenance={"is_ghost": bool(getattr(row, "is_ghost", False))},
        ))
    return records


def rasterize_2d(
    records: list,
    bundle_dir: Path,
    meta: BundleMeta,
    *,
    use_nuclei: bool = False,
) -> np.ndarray:
    """Rasterise polygons into an (H, W) int32 label array.

    First-write-wins ordering by cell.label so smaller labels = top of stack;
    in practice 2D anchor cells are Voronoi-disjoint so overlap is rare.
    """
    from skimage.draw import polygon as skpoly

    poly_path = bundle_dir / ("nucleus_boundaries.parquet" if use_nuclei
                                 else "cell_boundaries.parquet")
    poly_xy = _load_polygons_by_id(poly_path)

    xmin, ymin, xmax, ymax = meta.tile_bounds_um
    psz = meta.pixel_size_um
    h = int(round((ymax - ymin) / psz))
    w = int(round((xmax - xmin) / psz))
    label_img = np.zeros((h, w), dtype=np.int32)

    for c in records:
        if c.cell_id not in poly_xy:
            continue
        xy = poly_xy[c.cell_id]
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
        cell_label = rasterize_2d(records, bundle_dir, meta, use_nuclei=False)
        nucleus_label = rasterize_2d(records, bundle_dir, meta, use_nuclei=True)
        return LoadedBundle(
            meta=meta, cells=records,
            cell_label=cell_label,
            nucleus_label=nucleus_label,
            z_slices_um=None,
        )
    else:
        raise ValueError(f"unknown scene_mode {meta.scene_mode!r} "
                         f"in experiment.xenium synth_metadata")
