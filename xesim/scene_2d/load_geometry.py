"""Shared cell + nucleus mask loader.

Used by both `canonicalize-bundle` (training data prep) and `explain_region`
(inference). Single source of truth for: given a bundle + crop bounds + target
shape, produce a (cell_label, nucleus_label, cell_ids, nucleus_ids) tuple,
preferring 10x's pre-computed masks from cells.zarr.zip when available, falling
back to fresh polygon rasterization otherwise.

Before this module existed, `canonicalize.py` used zarr-then-polygon while
`scene_2d/tile_pipeline.py` only did polygon rasterization. The mismatch caused
IoU ≈ 0.72 between the masks produced by the two paths for the same cells —
the renderer (trained on canonical's zarr-derived masks) saw a different
distribution at inference. This module unifies the paths.
"""
from __future__ import annotations
from typing import Any

import numpy as np


def load_geometry(
    bundle: Any,
    crop_box: Any,
    shape: tuple[int, int],
    *,
    geometry_source: str = "auto",
    prefer_parquet: bool = True,
    zarr_reader: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load (cell_label, nucleus_label, cell_ids, nucleus_ids, source_tag).

    Tries the cells.zarr.zip-derived masks first when available (matches 10x's
    own rasterization), then falls back to polygon rasterization. The
    ``source_tag`` records which path produced the masks for downstream
    bookkeeping.

    Parameters
    ----------
    bundle
        Resolved Xenium bundle (from xesim.xenium.resolve_bundle).
    crop_box
        CropBox-like object with xmin/xmax/ymin/ymax in µm + crop_id.
    shape
        Target (H, W) shape in pixels.
    geometry_source
        "auto" (default): try zarr, fall back to polygons.
        "zarr": require zarr; raise if unavailable.
        "polygons": skip zarr, use polygons.
    prefer_parquet
        For the polygon path: read from parquet if available (faster).
    zarr_reader
        Optional pre-opened CellsZarrMaskReader. If None and zarr is available
        in the bundle, one is opened internally.
    """
    from ..raster import rasterize_polygons
    from ..xenium import read_boundary_polygons
    from ..canonicalize import _open_zarr_reader

    if zarr_reader is None and geometry_source in {"auto", "zarr"}:
        zarr_reader = _open_zarr_reader(bundle, geometry_source, [])

    if geometry_source in {"auto", "zarr"} and zarr_reader is not None:
        try:
            cell = zarr_reader.read(crop_box, "cell", shape=shape)
            nucleus = zarr_reader.read(crop_box, "nucleus", shape=shape)
            if (geometry_source == "zarr"
                    or np.any(cell.label > 0) or np.any(nucleus.label > 0)):
                return (cell.label, nucleus.label, cell.ids, nucleus.ids,
                        cell.source)
        except Exception:
            if geometry_source == "zarr":
                raise

    if geometry_source == "zarr":
        raise RuntimeError(
            "geometry_source='zarr' requested but cells.zarr.zip unavailable")

    cell_polys = read_boundary_polygons(
        bundle.cell_boundaries_path, crop_box, prefer_parquet=prefer_parquet)
    nucleus_polys = read_boundary_polygons(
        bundle.nucleus_boundaries_path, crop_box, prefer_parquet=prefer_parquet)
    cell_label, cell_ids = rasterize_polygons(
        cell_polys, crop_box, shape, bundle.pixel_size)
    nucleus_label, nucleus_ids = rasterize_polygons(
        nucleus_polys, crop_box, shape, bundle.pixel_size)
    return cell_label, nucleus_label, cell_ids, nucleus_ids, "polygons:boundaries"


__all__ = ["load_geometry"]
