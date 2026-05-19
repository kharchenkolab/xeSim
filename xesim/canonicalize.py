from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import geometry_feature_arrays, geometry_summary
from .images import ImageStackReader, channel_names, ome_image_shape
from .models import CropBox, CropRecord
from .qc import fit_display_lut, normalize_images_with_lut
from .raster import rasterize_polygons
from .schema import CROP_MANIFEST_TYPE, MANIFEST_SCHEMA_VERSION, SPLIT_MANIFEST_TYPE
from .xenium import (
    choose_crop_boxes,
    read_boundary_polygons,
    read_cells,
    resolve_bundle,
    transcript_counts_by_crop,
    transcript_scan_strategy,
)
from .zarr_masks import CellsZarrMaskReader


def inspect_bundle(path_or_dir: str | Path) -> dict[str, Any]:
    bundle = resolve_bundle(path_or_dir)
    cells = read_cells(bundle)
    summary = bundle.to_summary()
    summary["num_cells_readable"] = len(cells)
    if cells:
        xs = [c.x for c in cells]
        ys = [c.y for c in cells]
        summary["cell_bounds_um"] = {
            "xmin": min(xs),
            "xmax": max(xs),
            "ymin": min(ys),
            "ymax": max(ys),
        }
    summary["dependencies"] = _dependency_status()
    return summary


def _dependency_status() -> dict[str, bool]:
    import importlib.util

    return {
        name: importlib.util.find_spec(name) is not None
        for name in ["pyarrow", "tifffile", "zarr", "imagecodecs", "scipy", "torch", "numpy"]
    }


def canonicalize_bundle(
    path_or_dir: str | Path,
    output_dir: Path,
    num_crops: int = 16,
    crop_size_um: float = 64.0,
    min_qv: float = 20.0,
    normalize_images: bool = True,
    include_images: bool = True,
    prefer_parquet: bool = True,
    geometry_source: str = "auto",
    crop_selection: str = "stratified",
    seed: int = 1,
    crop_boxes: list[CropBox] | None = None,
    split_method: str = "spatial_x_quantile",
    val_regions: list[CropBox] | None = None,
    test_regions: list[CropBox] | None = None,
    val_polygons: list[dict[str, Any]] | None = None,
    test_polygons: list[dict[str, Any]] | None = None,
    annotation_path: str | Path | None = None,
    annotation_col: str = "merged_annotation",
    cell_id_col: str = "cell_id",
    stratified_alpha: float = 0.5,
    stratified_k: int = 20,
    stratified_within_pick: str = "density",
) -> Path:
    """Create canonical crop npz files and a manifest from one Xenium bundle.

    ``crop_selection`` controls which cells anchor the training crops.
    Default ``"density"`` picks the highest-cell-density windows — biased
    toward dense epithelium / islets where membrane localization detail
    matters most. ``"spread"`` samples uniformly across the bundle
    (includes sparse / edge regions) and was the default in v22; trained
    renderers visibly under-rendered membrane peaks on dense regions
    (~90% real → 62%). The v21 default of ``"density"`` recovers the
    membrane fidelity. ``"random"`` and ``"grid"`` are also available.
    """

    if geometry_source not in {"auto", "zarr", "polygons"}:
        raise ValueError("geometry_source must be auto, zarr, or polygons")
    bundle = resolve_bundle(path_or_dir)
    cells = read_cells(bundle, prefer_parquet=prefer_parquet)
    if not cells:
        raise ValueError("Cannot canonicalize bundle without readable cells")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = bundle.morphology_focus_paths
    channels = channel_names(image_paths) if include_images else ()
    image_bounds = None
    if include_images and image_paths:
        h, w = ome_image_shape(image_paths[0])
        image_bounds = (0.0, w * bundle.pixel_size, 0.0, h * bundle.pixel_size)
    if crop_boxes:
        crops = _renumber_crop_boxes(crop_boxes)
        crop_selection_mode = "manual"
    else:
        # For stratified selection, attach cell types to cells from the
        # annotation CSV so the clusterer has compositional features. Falls
        # back to density inside choose_crop_boxes if types aren't populated.
        if crop_selection == "stratified" and annotation_path is not None:
            import pandas as _pd
            ann_df = _pd.read_csv(str(annotation_path))
            if cell_id_col in ann_df.columns and annotation_col in ann_df.columns:
                ann_map = dict(zip(
                    ann_df[cell_id_col].astype(str),
                    ann_df[annotation_col].astype(str)))
                cells = [
                    type(c)(
                        cell_id=c.cell_id, x=c.x, y=c.y, z=c.z,
                        cell_area=c.cell_area, nucleus_area=c.nucleus_area,
                        transcript_count=c.transcript_count,
                        cell_type=ann_map.get(c.cell_id, c.cell_type),
                    ) for c in cells
                ]
        crops = choose_crop_boxes(
            cells,
            num_crops,
            crop_size_um,
            bounds=image_bounds,
            selection=crop_selection,
            seed=seed,
            stratified_k=stratified_k,
            stratified_alpha=stratified_alpha,
            stratified_within_pick=stratified_within_pick,
        )
        crop_selection_mode = crop_selection
    split_by_crop = assign_spatial_splits(
        crops,
        method=split_method,
        val_regions=val_regions,
        test_regions=test_regions,
        val_polygons=val_polygons,
        test_polygons=test_polygons,
    )
    transcript_counts = transcript_counts_by_crop(
        bundle, crops, min_qv=min_qv, prefer_parquet=prefer_parquet
    )
    records: list[CropRecord] = []
    geometry_warnings: list[str] = []
    image_reader = ImageStackReader(image_paths, bundle.pixel_size) if include_images and image_paths else None
    raw_images_by_crop: dict[str, np.ndarray] = {}
    image_normalization: dict[str, Any] = {"mode": "none", "lut": None}
    if image_reader is not None:
        for crop in crops:
            raw_images_by_crop[crop.crop_id] = image_reader.read(crop)
        if normalize_images:
            image_lut = fit_display_lut([images for images in raw_images_by_crop.values() if images.size])
            image_normalization = {"mode": "global_real_crop_lut", "lut": image_lut}
        else:
            image_normalization = {"mode": "raw", "lut": None}
    zarr_reader = _open_zarr_reader(bundle, geometry_source, geometry_warnings)
    try:
        for crop in crops:
            images = (
                raw_images_by_crop.get(crop.crop_id, np.zeros((0, 0, 0), dtype=np.float32))
                if image_reader is not None
                else np.zeros((0, 0, 0), dtype=np.float32)
            )
            if normalize_images and images.size and image_normalization["lut"] is not None:
                images = normalize_images_with_lut(images, image_normalization["lut"])
            shape = _shape_for_crop(images, crop, bundle.pixel_size)
            cell_label, nucleus_label, cell_ids, nucleus_ids, crop_geometry_source = _load_geometry(
                bundle,
                crop,
                shape,
                geometry_source=geometry_source,
                prefer_parquet=prefer_parquet,
                zarr_reader=zarr_reader,
            )
            geometry_features = geometry_feature_arrays(cell_label, nucleus_label, bundle.pixel_size)
            geometry_qc = geometry_summary(
                cell_label,
                nucleus_label,
                bundle.pixel_size,
                features=geometry_features,
            )
            total_tx, assigned_tx = transcript_counts.get(crop.crop_id, (0, 0))
            split = split_by_crop[crop.crop_id]
            npz_rel = Path("crops") / f"{crop.crop_id}.npz"
            npz_path = output_dir / npz_rel
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                images=images,
                cell_label=cell_label,
                nucleus_label=nucleus_label,
                cell_ids=cell_ids,
                nucleus_ids=nucleus_ids,
                crop_bounds_um=np.asarray([crop.xmin, crop.xmax, crop.ymin, crop.ymax], dtype=np.float32),
                pixel_size=np.float32(bundle.pixel_size),
                **geometry_features,
            )
            records.append(
                CropRecord(
                    crop_id=crop.crop_id,
                    crop_box=crop,
                    npz_path=npz_path,
                    split=split,
                    image_channels=channels,
                    transcript_count=total_tx,
                    assigned_transcript_count=assigned_tx,
                    cell_count=len(cell_ids),
                    nucleus_count=len(nucleus_ids),
                    qc={
                        **geometry_qc,
                        "geometry_source": crop_geometry_source,
                    },
                )
            )
    finally:
        if zarr_reader is not None:
            zarr_reader.close()
    split_manifest_path = output_dir / "splits.json"
    _write_splits(split_manifest_path, split_by_crop, method=split_method)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(
            {
                "type": CROP_MANIFEST_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "bundle": bundle.to_summary(),
                "image_channels": list(channels),
                "num_crops": len(records),
                "crop_size_um": crop_size_um,
                "crop_selection": {
                    "mode": crop_selection_mode,
                    "seed": seed if crop_selection_mode == "random" else None,
                    "num_requested": len(crop_boxes) if crop_boxes else num_crops,
                },
                "min_qv": min_qv,
                "image_normalization": image_normalization,
                "transcript_counting": {
                    "min_qv": min_qv,
                    "scan_strategy": transcript_scan_strategy(crops),
                },
                "geometry_source": geometry_source,
                "geometry_warnings": geometry_warnings,
                "split_path": "splits.json",
                "split_method": split_method,
                "split_regions": {
                    "val": {
                        "boxes": _region_payload(val_regions),
                        "polygons": _polygon_payload(val_polygons),
                    },
                    "test": {
                        "boxes": _region_payload(test_regions),
                        "polygons": _polygon_payload(test_polygons),
                    },
                },
                "crops": [record.to_dict(root=output_dir) for record in records],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return manifest_path


def _renumber_crop_boxes(crop_boxes: list[CropBox]) -> list[CropBox]:
    return [
        CropBox(
            crop_id=f"crop_{i + 1:05d}",
            xmin=float(box.xmin),
            xmax=float(box.xmax),
            ymin=float(box.ymin),
            ymax=float(box.ymax),
            zmin=box.zmin,
            zmax=box.zmax,
        )
        for i, box in enumerate(crop_boxes)
    ]


def _region_payload(regions: list[CropBox] | None) -> list[dict[str, Any]]:
    return [] if not regions else [region.to_dict() for region in regions]


def _polygon_payload(polygons: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [] if not polygons else list(polygons)


def _load_geometry(
    bundle: Any,
    crop: Any,
    shape: tuple[int, int],
    geometry_source: str,
    prefer_parquet: bool,
    zarr_reader: CellsZarrMaskReader | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if geometry_source in {"auto", "zarr"} and zarr_reader is not None:
        try:
            cell = zarr_reader.read(crop, "cell", shape=shape)
            nucleus = zarr_reader.read(crop, "nucleus", shape=shape)
            if geometry_source == "zarr" or np.any(cell.label > 0) or np.any(nucleus.label > 0):
                return cell.label, nucleus.label, cell.ids, nucleus.ids, cell.source
        except Exception:
            if geometry_source == "zarr":
                raise
    cell_polys = read_boundary_polygons(
        bundle.cell_boundaries_path, crop, prefer_parquet=prefer_parquet
    )
    nucleus_polys = read_boundary_polygons(
        bundle.nucleus_boundaries_path, crop, prefer_parquet=prefer_parquet
    )
    cell_label, cell_ids = rasterize_polygons(cell_polys, crop, shape, bundle.pixel_size)
    nucleus_label, nucleus_ids = rasterize_polygons(nucleus_polys, crop, shape, bundle.pixel_size)
    return cell_label, nucleus_label, cell_ids, nucleus_ids, "polygons:boundaries"


def _open_zarr_reader(
    bundle: Any,
    geometry_source: str,
    geometry_warnings: list[str],
) -> CellsZarrMaskReader | None:
    if geometry_source not in {"auto", "zarr"} or bundle.cells_zarr_path is None:
        return None
    try:
        return CellsZarrMaskReader(bundle.cells_zarr_path, bundle.pixel_size)
    except Exception as exc:
        if geometry_source == "zarr":
            raise
        geometry_warnings.append(f"cells.zarr.zip unavailable, falling back to polygons: {exc}")
        return None


def _shape_for_crop(images: np.ndarray, crop: Any, pixel_size: float) -> tuple[int, int]:
    if images.size and images.ndim == 3:
        return int(images.shape[1]), int(images.shape[2])
    h = max(1, int(np.ceil((crop.ymax - crop.ymin) / pixel_size)))
    w = max(1, int(np.ceil((crop.xmax - crop.xmin) / pixel_size)))
    return h, w


def assign_spatial_splits(
    crops: list[Any],
    method: str = "spatial_x_quantile",
    val_regions: list[CropBox] | None = None,
    test_regions: list[CropBox] | None = None,
    val_polygons: list[dict[str, Any]] | None = None,
    test_polygons: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Assign deterministic train/val/test splits by spatial policy.

    Small smoke manifests with fewer than three crops are kept entirely in train
    so downstream commands can run on tiny examples without empty-train issues.
    """

    if method not in {"spatial_x_quantile", "spatial_y_quantile", "spatial_checkerboard"}:
        raise ValueError(
            "split method must be spatial_x_quantile, spatial_y_quantile, or spatial_checkerboard"
        )
    if len(crops) < 3:
        split_by_crop = {crop.crop_id: "train" for crop in crops}
        return _apply_region_splits(crops, split_by_crop, val_regions, test_regions, val_polygons, test_polygons)
    if method == "spatial_y_quantile":
        ordered = sorted(crops, key=lambda crop: (crop.center_y, crop.center_x, crop.crop_id))
        return _apply_region_splits(
            crops,
            _assign_ordered_holdouts(ordered),
            val_regions,
            test_regions,
            val_polygons,
            test_polygons,
        )
    if method == "spatial_checkerboard":
        return _apply_region_splits(
            crops,
            _assign_checkerboard_splits(crops),
            val_regions,
            test_regions,
            val_polygons,
            test_polygons,
        )
    ordered = sorted(crops, key=lambda crop: (crop.center_x, crop.center_y, crop.crop_id))
    return _apply_region_splits(
        crops,
        _assign_ordered_holdouts(ordered),
        val_regions,
        test_regions,
        val_polygons,
        test_polygons,
    )


def _apply_region_splits(
    crops: list[Any],
    split_by_crop: dict[str, str],
    val_regions: list[CropBox] | None,
    test_regions: list[CropBox] | None,
    val_polygons: list[dict[str, Any]] | None = None,
    test_polygons: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    if not val_regions and not test_regions and not val_polygons and not test_polygons:
        return split_by_crop
    out = dict(split_by_crop)
    for crop in crops:
        if _center_in_regions(crop, test_regions) or _center_in_polygons(crop, test_polygons):
            out[crop.crop_id] = "test"
        elif _center_in_regions(crop, val_regions) or _center_in_polygons(crop, val_polygons):
            out[crop.crop_id] = "val"
    return out


def _center_in_regions(crop: Any, regions: list[CropBox] | None) -> bool:
    if not regions:
        return False
    return any(region.contains_xy(crop.center_x, crop.center_y) for region in regions)


def _center_in_polygons(crop: Any, polygons: list[dict[str, Any]] | None) -> bool:
    if not polygons:
        return False
    return any(_point_in_polygon(crop.center_x, crop.center_y, polygon.get("points_um", [])) for polygon in polygons)


def _point_in_polygon(x: float, y: float, points: list[list[float]]) -> bool:
    if len(points) < 3:
        return False
    inside = False
    xj, yj = points[-1]
    for xi, yi in points:
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        xj, yj = xi, yi
    return inside


def _assign_ordered_holdouts(ordered: list[Any]) -> dict[str, str]:
    n = len(ordered)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    split_by_crop = {crop.crop_id: "train" for crop in ordered}
    for crop in ordered[-n_test:]:
        split_by_crop[crop.crop_id] = "test"
    for crop in ordered[-(n_test + n_val) : -n_test]:
        split_by_crop[crop.crop_id] = "val"
    return split_by_crop


def _assign_checkerboard_splits(crops: list[Any]) -> dict[str, str]:
    xs = np.asarray([crop.center_x for crop in crops], dtype=np.float32)
    ys = np.asarray([crop.center_y for crop in crops], dtype=np.float32)
    x_mid = float(np.median(xs))
    y_mid = float(np.median(ys))
    split_by_crop = {crop.crop_id: "train" for crop in crops}
    for crop in crops:
        high_x = crop.center_x >= x_mid
        high_y = crop.center_y >= y_mid
        if high_x and high_y:
            split_by_crop[crop.crop_id] = "test"
        elif (not high_x) and high_y:
            split_by_crop[crop.crop_id] = "val"
    if "val" not in split_by_crop.values() or "test" not in split_by_crop.values():
        ordered = sorted(crops, key=lambda crop: (crop.center_x, crop.center_y, crop.crop_id))
        return _assign_ordered_holdouts(ordered)
    return split_by_crop


def _write_splits(path: Path, split_by_crop: dict[str, str], method: str = "spatial_x_quantile") -> None:
    groups = {"train": [], "val": [], "test": []}
    for crop_id, split in sorted(split_by_crop.items()):
        groups.setdefault(split, []).append(crop_id)
    with path.open("w") as handle:
        json.dump(
            {
                "type": SPLIT_MANIFEST_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "method": method,
                "splits": groups,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
