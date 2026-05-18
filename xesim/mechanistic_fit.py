from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .mechanistic_render import _gaussian_blur, _label_boundary
from .mechanistic_ridges import membrane_ridge_evidence
from .mechanistic_scene import MechanisticParams
from .mechanistic_schema import MECHANISTIC_REPORT_TYPE
from .raster import binary_dilation
from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


def fit_mechanistic_priors(
    crop_manifest_path: Path,
    output_dir: Path,
    max_crops: int = 64,
    split: str | None = None,
    cell_types_path: Path | None = None,
) -> Path:
    """Fit first-pass Plan3 renderer parameters from canonical real crops.

    When ``split`` is provided (``train``/``val``/``test``), only crops whose
    per-record ``split`` field matches are used. This is the supported way to
    fit priors on the train split alone, leaving val/test as held-out.

    When ``cell_types_path`` is provided, per-type DAPI/membrane emission
    scales are also fitted by aggregating per-cell intensities by type.
    Resulting params include a ``per_type_efficiencies`` field keyed by
    type name.
    """

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = crop_manifest_path.parent
    all_records = manifest.get("crops", [])
    if split is not None:
        all_records = [r for r in all_records if r.get("split") == split]
    records = all_records[: max(0, max_crops)]
    cell_type_artifact = None
    cell_type_lookup: dict[str, dict[str, int]] = {}
    type_names: list[str] = []
    if cell_types_path is not None:
        from .cell_types import load_cell_type_assignment

        cell_type_artifact = load_cell_type_assignment(cell_types_path)
        type_names = list(cell_type_artifact["type_names"])
        for crop in cell_type_artifact.get("crops", []):
            cell_type_lookup[str(crop["crop_id"])] = {
                str(k): int(v) for k, v in crop.get("cell_id_to_type_index", {}).items()
            }
    per_type_dapi: dict[int, list[float]] = {}
    per_type_membrane: dict[int, list[float]] = {}
    per_type_ridge_widths: dict[int, list[float]] = {}
    per_type_dropout_proxy: dict[int, list[float]] = {}
    per_type_interior_fraction: dict[int, list[float]] = {}
    per_type_dapi_texture: dict[int, list[float]] = {}
    per_type_cell_areas_um2: dict[int, list[float]] = {}
    per_type_aspect_ratios: dict[int, list[float]] = {}
    per_type_nucleus_to_cell: dict[int, list[float]] = {}
    per_type_polya_cytoplasm: dict[int, list[float]] = {}
    polya_global_means: list[float] = []
    # Plan3 v32: per-type extracellular membrane halo samples. For each cell
    # of a given type, we sample real membrane intensity in radial shells
    # outside its footprint (and outside all other cells of any type) at a
    # set of distances, then fit per-type ``halo_amp`` (intensity at d=1px)
    # and ``halo_tau_px`` (exponential decay) from the median radial profile.
    halo_distance_bins = (1.0, 3.0, 6.0, 10.0)
    per_type_halo_samples: dict[int, list[list[float]]] = {}
    crop_stats: list[dict[str, Any]] = []
    for record in records:
        data = np.load(root / record["npz_path"], allow_pickle=True)
        if "images" not in data.files or data["images"].ndim != 3 or data["images"].shape[0] < 2:
            continue
        images = data["images"].astype(np.float32, copy=False)
        cell_label = data["cell_label"].astype(np.int32, copy=False)
        nucleus_label = data["nucleus_label"].astype(np.int32, copy=False)
        cell_ids_arr = [str(v) for v in data["cell_ids"].tolist()] if "cell_ids" in data.files else []
        crop_stats.append(_crop_stats(record, images[:2], cell_label, nucleus_label))
        # PolyA channel (channel 2) statistics for global + per-type fitting.
        if images.shape[0] >= 3:
            polya_channel = images[2]
            cell_mask = cell_label > 0
            nucleus_mask = nucleus_label > 0
            cytoplasm_mask = cell_mask & ~nucleus_mask
            if np.any(cytoplasm_mask):
                polya_global_means.append(float(np.mean(polya_channel[cytoplasm_mask])))
        if cell_type_lookup:
            mapping = cell_type_lookup.get(str(record.get("crop_id"))) or {}
            _accumulate_per_type_intensities(
                images[:2],
                cell_label,
                nucleus_label,
                cell_ids_arr,
                mapping,
                per_type_dapi,
                per_type_membrane,
            )
            if images.shape[0] >= 3:
                _accumulate_per_type_polya(
                    images[2],
                    cell_label,
                    nucleus_label,
                    cell_ids_arr,
                    mapping,
                    per_type_polya_cytoplasm,
                )
            _accumulate_per_type_membrane_physics(
                images[:2],
                cell_label,
                nucleus_label,
                cell_ids_arr,
                mapping,
                per_type_ridge_widths,
                per_type_dropout_proxy,
                per_type_interior_fraction,
                per_type_dapi_texture,
            )
            # pixel_size is stored in each crop npz, not on the manifest record.
            data_pixel = data.get("pixel_size")
            if data_pixel is not None:
                pixel_size = float(np.asarray(data_pixel))
            else:
                pixel_size = float(record.get("pixel_size", manifest.get("pixel_size", 1.0)))
            if not pixel_size or pixel_size <= 0:
                pixel_size = 1.0
            _accumulate_per_type_shape(
                cell_label,
                nucleus_label,
                cell_ids_arr,
                mapping,
                pixel_size,
                per_type_cell_areas_um2,
                per_type_aspect_ratios,
                per_type_nucleus_to_cell,
            )
            _accumulate_per_type_halo_samples(
                images[1],  # membrane channel
                cell_label,
                cell_ids_arr,
                mapping,
                halo_distance_bins,
                per_type_halo_samples,
            )

    params = _params_from_stats(crop_stats)
    params_dict = params.to_dict()
    # Fit the global polyA cytoplasm-mean intensity from real measurements,
    # if a polyA channel was available. This is what the renderer uses as
    # ``polya_base`` when no cell-specific override is set.
    if polya_global_means:
        params_dict["polya_base"] = float(np.clip(np.median(polya_global_means), 0.005, 0.40))
    per_type_efficiencies: dict[str, dict[str, float]] = {}
    if cell_type_artifact is not None:
        per_type_efficiencies = _per_type_efficiencies(
            type_names, per_type_dapi, per_type_membrane,
            global_dapi=params.dapi_base, global_membrane=params.membrane_base,
        )
        # Layer per-type membrane physics + DAPI texture on top.
        _augment_with_membrane_physics(
            per_type_efficiencies,
            type_names,
            per_type_ridge_widths,
            per_type_dropout_proxy,
            per_type_interior_fraction,
            per_type_dapi_texture,
            global_params=params,
        )
        # Per-type shape moments (F3).
        _augment_with_shape_moments(
            per_type_efficiencies,
            type_names,
            per_type_cell_areas_um2,
            per_type_aspect_ratios,
            per_type_nucleus_to_cell,
        )
        # Per-type polyA cytoplasm efficiency.
        global_polya = float(params_dict.get("polya_base", params.polya_base))
        _augment_with_polya(
            per_type_efficiencies,
            type_names,
            per_type_polya_cytoplasm,
            global_polya=global_polya,
        )
        params_dict["per_type_efficiencies"] = per_type_efficiencies
        # Plan3 v32: fit per-type extracellular membrane halo and write to
        # MechanisticParams dicts. Background-subtract using the global
        # membrane_background measurement so the halo represents the
        # cell-attributable extra signal above the tissue floor.
        bg_samples = [
            float(s.get("membrane_background_mean", 0.0))
            for s in crop_stats
            if s.get("membrane_background_mean") is not None
        ]
        bg_membrane = float(np.median(bg_samples)) if bg_samples else 0.0
        amp_dict, width_dict = _fit_per_type_extracellular_ridge(
            type_names,
            per_type_halo_samples,
            halo_distance_bins,
            background_membrane=max(0.0, bg_membrane),
        )
        params_dict["per_type_extracellular_membrane_amp"] = amp_dict
        params_dict["per_type_extracellular_membrane_width_px"] = width_dict
    params_path = output_dir / "mechanistic_params.json"
    with params_path.open("w") as handle:
        json.dump(params_dict, handle, indent=2, sort_keys=True)

    report_path = output_dir / "mechanistic_prior_report.json"
    with report_path.open("w") as handle:
        json.dump(
            {
                "type": MECHANISTIC_REPORT_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "report_kind": "mechanistic_prior_fit.v0",
                "source_manifest": str(crop_manifest_path),
                "num_crops_requested": int(max_crops),
                "num_crops_used": len(crop_stats),
                "split_filter": split,
                "params_path": params_path.name,
                "summary": _summary(crop_stats),
                "params": params.to_dict(),
                "records": crop_stats,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return report_path


def load_mechanistic_params(path: Path) -> MechanisticParams:
    with path.open() as handle:
        obj = json.load(handle)
    return MechanisticParams.from_dict(obj)


def _crop_stats(
    record: dict[str, Any],
    images: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
) -> dict[str, Any]:
    cell = cell_label > 0
    nucleus = nucleus_label > 0
    boundary = _label_boundary(cell_label)
    interior = cell & ~boundary & ~nucleus
    background = ~(cell | nucleus)
    dapi = images[0]
    membrane = images[1]
    ridge = _membrane_ridge_stats(membrane, boundary, cell, background)
    return {
        "crop_id": record.get("crop_id"),
        "nucleus_fraction": float(np.mean(nucleus)) if nucleus.size else 0.0,
        "cell_fraction": float(np.mean(cell)) if cell.size else 0.0,
        "boundary_fraction": float(np.mean(boundary)) if boundary.size else 0.0,
        "dapi_nucleus_mean": _masked_mean(dapi, nucleus),
        "dapi_background_mean": _masked_mean(dapi, background),
        "dapi_background_std": _masked_std(dapi, background),
        "membrane_boundary_mean": _masked_mean(membrane, boundary),
        "membrane_interior_mean": _masked_mean(membrane, interior),
        "membrane_background_mean": _masked_mean(membrane, background),
        "membrane_background_std": _masked_std(membrane, background),
        "membrane_boundary_to_interior_ratio": _safe_ratio(
            _masked_mean(membrane, boundary),
            _masked_mean(membrane, interior),
        ),
        **ridge,
    }


def _params_from_stats(records: list[dict[str, Any]]) -> MechanisticParams:
    if not records:
        return MechanisticParams()
    dapi_base = _clipped_median(records, "dapi_nucleus_mean", 0.15, 0.85)
    membrane_boundary = _clipped_median(records, "membrane_boundary_mean", 0.05, 0.75)
    membrane_interior = _clipped_median(records, "membrane_interior_mean", 0.0, 0.5)
    membrane_background = _clipped_median(records, "membrane_background_mean", 0.0, 0.5)
    ridge_near = _clipped_median(records, "membrane_ridge_near_boundary_mean", 0.0, 0.8)
    ridge_background = _clipped_median(records, "membrane_ridge_background_mean", 0.0, 0.8)
    ridge_enrichment = _clipped_median(records, "membrane_ridge_boundary_enrichment", 0.25, 8.0)
    ridge_coverage = _clipped_median(records, "membrane_ridge_near_boundary_coverage", 0.02, 0.95)
    ridge_width = _clipped_median(records, "membrane_ridge_width_px", 0.7, 5.0)
    background_noise = 0.5 * (
        _clipped_median(records, "dapi_background_std", 0.0, 0.12)
        + _clipped_median(records, "membrane_background_std", 0.0, 0.12)
    )
    interior_ratio = _safe_ratio(membrane_interior, max(membrane_boundary, 1e-6))
    ridge_strength = max(membrane_boundary, ridge_near * 1.8, (ridge_near - ridge_background) * 3.0)
    dropout_keep = float(np.clip(0.35 + 0.45 * ridge_coverage + 0.05 * np.log1p(ridge_enrichment), 0.35, 0.88))
    # Extracellular tissue level: bulk membrane intensity in regions outside
    # cells. Reflects unsegmented tissue + diffuse + out-of-plane membrane
    # contributions and has to be much higher than the bundle-level haze
    # (which is the optical glow component) for the simulator to match real
    # background mean.
    extracellular_level = float(np.clip(membrane_background * 0.85, 0.02, 0.18))
    return MechanisticParams(
        dapi_base=dapi_base,
        dapi_texture_scale=0.10,
        dapi_blur_sigma_px=1.2,
        membrane_base=min(0.75, max(0.05, ridge_strength)),
        membrane_width_px=float(np.clip(ridge_width, 0.8, 3.5)),
        membrane_dropout_keep_prob=dropout_keep,
        membrane_dropout_scale_px=10.0,
        membrane_interior_fraction=min(0.07, max(0.008, interior_ratio * 0.035)),
        membrane_out_of_focus_fraction=float(np.clip(0.10 + 0.12 * (1.0 - dropout_keep), 0.10, 0.24)),
        membrane_out_of_focus_sigma_px=4.0,
        haze_level=min(0.08, max(0.005, membrane_background * 0.25)),
        haze_scale_px=18.0,
        extracellular_tissue_level=extracellular_level,
        extracellular_tissue_scale_px=24.0,
        extracellular_tissue_texture=0.40,
        illumination_scale=0.04,
        noise_sigma=min(0.04, max(0.004, background_noise * 0.5)),
        seed=1,
    )


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "dapi_nucleus_mean",
        "dapi_background_mean",
        "membrane_boundary_mean",
        "membrane_interior_mean",
        "membrane_background_mean",
        "membrane_boundary_to_interior_ratio",
        "membrane_ridge_near_boundary_mean",
        "membrane_ridge_background_mean",
        "membrane_ridge_boundary_enrichment",
        "membrane_ridge_near_boundary_coverage",
        "membrane_ridge_width_px",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if values:
            out[key] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
    return out


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape or not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape or not np.any(mask):
        return 0.0
    return float(np.std(values[mask]))


def _membrane_ridge_stats(
    membrane: np.ndarray,
    boundary: np.ndarray,
    cell: np.ndarray,
    background: np.ndarray,
) -> dict[str, float]:
    evidence = membrane_ridge_evidence(membrane)
    near_boundary = binary_dilation(boundary, iterations=3)
    near_boundary &= binary_dilation(cell, iterations=1)
    far_background = background & ~binary_dilation(boundary, iterations=6)
    if not np.any(far_background):
        far_background = background
    threshold_values = evidence[far_background] if np.any(far_background) else evidence.reshape(-1)
    threshold = float(np.percentile(threshold_values, 90.0)) if threshold_values.size else 0.0
    ridge_mask = (evidence > threshold) & near_boundary
    boundary_count = int(np.sum(boundary))
    width_px = _safe_ratio(float(np.sum(ridge_mask)), float(boundary_count))
    near_mean = _masked_mean(evidence, near_boundary)
    background_mean = _masked_mean(evidence, far_background)
    return {
        "membrane_ridge_near_boundary_mean": near_mean,
        "membrane_ridge_background_mean": background_mean,
        "membrane_ridge_boundary_enrichment": _safe_ratio(near_mean, background_mean),
        "membrane_ridge_near_boundary_coverage": float(np.mean(evidence[near_boundary] > threshold)) if np.any(near_boundary) else 0.0,
        "membrane_ridge_width_px": float(np.clip(width_px, 0.0, 8.0)),
        "membrane_ridge_threshold": threshold,
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 1e-8 else 0.0


def _clipped_median(records: list[dict[str, Any]], key: str, lo: float, hi: float) -> float:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return float((lo + hi) / 2.0)
    return float(np.clip(np.median(values), lo, hi))


def _accumulate_per_type_intensities(
    images: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type: dict[str, int],
    per_type_dapi: dict[int, list[float]],
    per_type_membrane: dict[int, list[float]],
) -> None:
    """Per-cell average intensities, indexed by type."""

    if cell_label.ndim != 2 or images.ndim != 3:
        return
    unique_labels = np.unique(cell_label)
    nonzero = unique_labels[unique_labels > 0].tolist()
    if len(nonzero) != len(cell_ids):
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    boundary = _label_boundary(cell_label)
    nucleus_mask = nucleus_label > 0
    dapi = images[0]
    membrane = images[1]
    for label_value, cell_id in zip(nonzero, cell_ids):
        type_idx = int(cell_id_to_type.get(str(cell_id), 0))
        if type_idx == 0:
            continue
        cell_mask = cell_label == int(label_value)
        nuc_in_cell = cell_mask & nucleus_mask
        if np.any(nuc_in_cell):
            per_type_dapi.setdefault(type_idx, []).append(float(np.mean(dapi[nuc_in_cell])))
        boundary_in_cell = cell_mask & boundary
        if np.any(boundary_in_cell):
            per_type_membrane.setdefault(type_idx, []).append(float(np.mean(membrane[boundary_in_cell])))


def _accumulate_per_type_membrane_physics(
    images: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type: dict[str, int],
    per_type_widths: dict[int, list[float]],
    per_type_dropout: dict[int, list[float]],
    per_type_interior: dict[int, list[float]],
    per_type_dapi_texture: dict[int, list[float]],
) -> None:
    """Per-cell membrane width, dropout-keep proxy, interior fraction, DAPI texture."""

    if cell_label.ndim != 2 or images.ndim != 3:
        return
    unique_labels = np.unique(cell_label)
    nonzero = unique_labels[unique_labels > 0].tolist()
    if len(nonzero) != len(cell_ids):
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    boundary_global = _label_boundary(cell_label)
    nucleus_mask = nucleus_label > 0
    membrane = images[1]
    dapi = images[0]
    # Threshold membrane evidence at the cell-wide median to define "active"
    # membrane pixels for ridge-width and dropout estimation.
    membrane_smooth = _gaussian_blur(membrane, 1.0)
    membrane_high = np.maximum(membrane - membrane_smooth, 0.0)
    for label_value, cell_id in zip(nonzero, cell_ids):
        type_idx = int(cell_id_to_type.get(str(cell_id), 0))
        if type_idx == 0:
            continue
        cell_mask = cell_label == int(label_value)
        cell_pixels = float(np.sum(cell_mask))
        if cell_pixels < 16:
            continue
        cell_boundary = boundary_global & cell_mask
        boundary_count = float(np.sum(cell_boundary))
        if boundary_count < 4:
            continue
        # Ridge width: use inside-cell pixels close to the boundary that are
        # high-frequency-positive. Width ~ (active high-freq pixels near
        # boundary) / boundary length.
        near_boundary = binary_dilation(cell_boundary, iterations=2) & cell_mask
        active_high = (membrane_high > 0.02) & near_boundary
        width_proxy = float(np.sum(active_high)) / max(boundary_count, 1.0)
        per_type_widths.setdefault(type_idx, []).append(min(max(width_proxy, 0.4), 4.5))
        # Dropout-keep proxy: fraction of cell-boundary pixels with positive
        # membrane signal above the cell's local background.
        background_in_cell = cell_mask & ~near_boundary & ~nucleus_mask
        if np.any(background_in_cell):
            local_bg = float(np.median(membrane[background_in_cell]))
        else:
            local_bg = float(np.median(membrane[cell_mask]))
        active_boundary = membrane[cell_boundary] > (local_bg + 0.02)
        dropout_keep = float(np.mean(active_boundary)) if active_boundary.size else 0.0
        per_type_dropout.setdefault(type_idx, []).append(min(max(dropout_keep, 0.05), 0.95))
        # Interior fraction: ratio of mean cytoplasm membrane to mean boundary
        # membrane signal — captures cytoplasmic membrane fill (vs ridges only).
        interior_pixels = cell_mask & ~near_boundary & ~nucleus_mask
        if np.any(interior_pixels) and np.any(cell_boundary):
            interior_mean = float(np.mean(membrane[interior_pixels]))
            boundary_mean = float(np.mean(membrane[cell_boundary]))
            ratio = interior_mean / max(boundary_mean, 1e-3)
            per_type_interior.setdefault(type_idx, []).append(min(max(ratio, 0.0), 0.6))
        # DAPI texture amplitude: relative std of DAPI inside nucleus,
        # normalized by mean (coefficient of variation), capturing chromatin
        # heterogeneity.
        nuc_in_cell = cell_mask & nucleus_mask
        if np.sum(nuc_in_cell) > 8:
            mean_dapi = float(np.mean(dapi[nuc_in_cell]))
            std_dapi = float(np.std(dapi[nuc_in_cell]))
            if mean_dapi > 1e-3:
                per_type_dapi_texture.setdefault(type_idx, []).append(min(std_dapi / mean_dapi, 1.5))


def _accumulate_per_type_polya(
    polya_image: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type: dict[str, int],
    per_type_polya_cytoplasm: dict[int, list[float]],
) -> None:
    """Per-cell mean polyA intensity in cytoplasm (cell minus nucleus), grouped by type."""

    if cell_label.ndim != 2:
        return
    unique = np.unique(cell_label)
    nonzero = unique[unique > 0].tolist()
    if len(nonzero) != len(cell_ids):
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    nucleus_mask = nucleus_label > 0
    for label_value, cell_id in zip(nonzero, cell_ids):
        type_idx = int(cell_id_to_type.get(str(cell_id), 0))
        if type_idx == 0:
            continue
        cell_mask = cell_label == int(label_value)
        cytoplasm = cell_mask & ~nucleus_mask
        if not np.any(cytoplasm):
            continue
        per_type_polya_cytoplasm.setdefault(type_idx, []).append(
            float(np.mean(polya_image[cytoplasm]))
        )


def _augment_with_polya(
    per_type_table: dict[str, dict[str, float]],
    type_names: list[str],
    per_type_polya_cytoplasm: dict[int, list[float]],
    global_polya: float,
) -> None:
    """Add per-type ``polya_efficiency = type_median / global_polya``."""

    for idx, name in enumerate(type_names):
        if name not in per_type_table:
            per_type_table[name] = {}
        record = per_type_table[name]
        if idx == 0:
            record.setdefault("polya_efficiency", 1.0)
            continue
        samples = per_type_polya_cytoplasm.get(idx, [])
        if samples:
            eff = float(np.clip(np.median(samples) / max(global_polya, 1e-6), 0.10, 3.0))
            record["polya_efficiency"] = eff
            record["n_polya"] = int(len(samples))
        else:
            record["polya_efficiency"] = 1.0
            record["n_polya"] = 0


def _accumulate_per_type_shape(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type: dict[str, int],
    pixel_size_um: float,
    per_type_cell_areas: dict[int, list[float]],
    per_type_aspect_ratios: dict[int, list[float]],
    per_type_nucleus_to_cell: dict[int, list[float]],
) -> None:
    """Per-cell area (µm²), aspect ratio (long/short axis from PCA), and
    nucleus-to-cell area ratio, indexed by cell type.
    """

    if cell_label.ndim != 2:
        return
    px_area_um2 = float(pixel_size_um) ** 2 if pixel_size_um else 1.0
    unique = np.unique(cell_label)
    nonzero = unique[unique > 0].tolist()
    if len(nonzero) != len(cell_ids):
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    for label_value, cell_id in zip(nonzero, cell_ids):
        type_idx = int(cell_id_to_type.get(str(cell_id), 0))
        if type_idx == 0:
            continue
        cell_mask = cell_label == int(label_value)
        cell_pixels = int(np.sum(cell_mask))
        if cell_pixels < 16:
            continue
        per_type_cell_areas.setdefault(type_idx, []).append(float(cell_pixels) * px_area_um2)
        ys, xs = np.nonzero(cell_mask)
        if ys.size >= 8:
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
            yy = ys.astype(np.float64) - cy
            xx = xs.astype(np.float64) - cx
            cov = np.array([[np.mean(yy * yy), np.mean(yy * xx)], [np.mean(yy * xx), np.mean(xx * xx)]])
            try:
                evs = np.linalg.eigvalsh(cov)
                lo = max(float(evs[0]), 1e-6)
                hi = max(float(evs[1]), 1e-6)
                aspect = float(np.sqrt(hi / lo))
                per_type_aspect_ratios.setdefault(type_idx, []).append(min(max(aspect, 1.0), 6.0))
            except Exception:  # noqa: BLE001
                pass
        nuc_in_cell = cell_mask & (nucleus_label > 0)
        nuc_pixels = int(np.sum(nuc_in_cell))
        if cell_pixels > 0:
            per_type_nucleus_to_cell.setdefault(type_idx, []).append(
                min(max(float(nuc_pixels) / float(cell_pixels), 0.0), 0.95)
            )


def _augment_with_shape_moments(
    per_type_table: dict[str, dict[str, float]],
    type_names: list[str],
    per_type_cell_areas: dict[int, list[float]],
    per_type_aspect_ratios: dict[int, list[float]],
    per_type_nucleus_to_cell: dict[int, list[float]],
) -> None:
    """Add per-type cell area / aspect ratio / nucleus-fraction moments."""

    for idx, name in enumerate(type_names):
        if name not in per_type_table:
            per_type_table[name] = {}
        record = per_type_table[name]
        if idx == 0:
            record.setdefault("cell_area_um2_mean", 80.0)
            record.setdefault("cell_area_um2_log_std", 0.30)
            record.setdefault("cell_aspect_log_mean", 0.20)
            record.setdefault("cell_aspect_log_std", 0.30)
            record.setdefault("nucleus_to_cell_ratio", 0.30)
            continue
        areas = per_type_cell_areas.get(idx, [])
        aspects = per_type_aspect_ratios.get(idx, [])
        nfracs = per_type_nucleus_to_cell.get(idx, [])
        if areas:
            log_areas = np.log(np.clip(areas, 5.0, None))
            record["cell_area_um2_mean"] = float(np.exp(np.median(log_areas)))
            record["cell_area_um2_log_std"] = float(np.clip(np.std(log_areas), 0.10, 0.80))
        else:
            record["cell_area_um2_mean"] = 80.0
            record["cell_area_um2_log_std"] = 0.30
        if aspects:
            log_aspects = np.log(np.clip(aspects, 1.0, None))
            record["cell_aspect_log_mean"] = float(np.clip(np.median(log_aspects), 0.0, 1.4))
            record["cell_aspect_log_std"] = float(np.clip(np.std(log_aspects), 0.05, 0.6))
        else:
            record["cell_aspect_log_mean"] = 0.2
            record["cell_aspect_log_std"] = 0.3
        if nfracs:
            record["nucleus_to_cell_ratio"] = float(np.clip(np.median(nfracs), 0.05, 0.85))
        else:
            record["nucleus_to_cell_ratio"] = 0.30
        record["n_areas"] = int(len(areas))
        record["n_aspects"] = int(len(aspects))
        record["n_nfracs"] = int(len(nfracs))


def _augment_with_membrane_physics(
    per_type_table: dict[str, dict[str, float]],
    type_names: list[str],
    per_type_widths: dict[int, list[float]],
    per_type_dropout: dict[int, list[float]],
    per_type_interior: dict[int, list[float]],
    per_type_dapi_texture: dict[int, list[float]],
    global_params: MechanisticParams,
) -> None:
    """Add per-type membrane physical parameters to the existing efficiency table."""

    for idx, name in enumerate(type_names):
        if name not in per_type_table:
            per_type_table[name] = {}
        record = per_type_table[name]
        if idx == 0:
            record.setdefault("membrane_width_px", float(global_params.membrane_width_px))
            record.setdefault("membrane_dropout_keep_prob", float(global_params.membrane_dropout_keep_prob))
            record.setdefault("membrane_interior_fraction", float(global_params.membrane_interior_fraction))
            record.setdefault("dapi_texture_scale", float(global_params.dapi_texture_scale))
            continue
        widths = per_type_widths.get(idx, [])
        dropouts = per_type_dropout.get(idx, [])
        interiors = per_type_interior.get(idx, [])
        textures = per_type_dapi_texture.get(idx, [])
        # Width proxy is a coverage ratio, not a true pixel width. Map it
        # multiplicatively around the global width so the per-type spread
        # respects the global ridge-aware fit's physical scale.
        if widths:
            type_ratio = float(np.median(widths))
            global_ratio = float(global_params.membrane_width_px) / max(float(global_params.membrane_width_px), 1.0)
            relative = type_ratio / max(np.median([np.median(v) for v in per_type_widths.values() if v]), 1e-3)
            record["membrane_width_px"] = float(np.clip(
                relative * float(global_params.membrane_width_px), 0.6, 2.5,
            ))
        else:
            record["membrane_width_px"] = float(global_params.membrane_width_px)
        record["membrane_dropout_keep_prob"] = (
            float(np.clip(np.median(dropouts), 0.20, 0.95)) if dropouts else float(global_params.membrane_dropout_keep_prob)
        )
        # Interior fraction: per-cell measurement is noisy and tends to over-
        # estimate because cytoplasm signal includes out-of-plane membrane
        # contributions. Stay near the global default but allow modest
        # variation around it (max 1.5x global).
        if interiors:
            type_interior = float(np.median(interiors))
            scale = type_interior / max(np.median([np.median(v) for v in per_type_interior.values() if v]), 1e-3)
            record["membrane_interior_fraction"] = float(np.clip(
                scale * float(global_params.membrane_interior_fraction),
                0.005, 1.5 * float(global_params.membrane_interior_fraction),
            ))
        else:
            record["membrane_interior_fraction"] = float(global_params.membrane_interior_fraction)
        # Map CoV-of-DAPI to a render texture scale that doesn't blow up; clamp
        # to a sane physical range. Higher type CoV -> stronger granularity.
        record["dapi_texture_scale"] = (
            float(np.clip(np.median(textures), 0.06, 0.30)) if textures else float(global_params.dapi_texture_scale)
        )
        record["n_widths"] = int(len(widths))
        record["n_dropouts"] = int(len(dropouts))
        record["n_interiors"] = int(len(interiors))
        record["n_dapi_textures"] = int(len(textures))


def _per_type_efficiencies(
    type_names: list[str],
    per_type_dapi: dict[int, list[float]],
    per_type_membrane: dict[int, list[float]],
    global_dapi: float,
    global_membrane: float,
    floor: float = 0.10,
    ceiling: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Convert per-type intensity samples into normalized efficiency multipliers.

    Efficiency = type_median / global_median, clipped to ``[floor, ceiling]``.
    The wider [0.10, 2.0] range (vs the original [0.4, 1.8]) preserves
    biology-driven differences between types whose membrane signal is
    genuinely much weaker than epithelial cells (immune / fibroblast /
    endothelial / mural).
    """

    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(type_names):
        if idx == 0:
            out[name] = {"dapi_efficiency": 1.0, "membrane_efficiency": 1.0, "n_dapi": 0, "n_membrane": 0}
            continue
        d_samples = per_type_dapi.get(idx, [])
        m_samples = per_type_membrane.get(idx, [])
        d_eff = float(np.clip(np.median(d_samples) / max(global_dapi, 1e-6), floor, ceiling)) if d_samples else 1.0
        m_eff = float(np.clip(np.median(m_samples) / max(global_membrane, 1e-6), floor, ceiling)) if m_samples else 1.0
        out[name] = {
            "dapi_efficiency": d_eff,
            "membrane_efficiency": m_eff,
            "n_dapi": len(d_samples),
            "n_membrane": len(m_samples),
        }
    return out


def _accumulate_per_type_halo_samples(
    membrane: np.ndarray,
    cell_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type: dict[str, int],
    distance_bins: tuple[float, ...],
    samples_by_type: dict[int, list[list[float]]],
) -> None:
    """Sample real membrane intensity in extracellular shells around each
    cell, grouped by cell type.

    For each cell of a non-background type, compute the distance from its
    footprint to all extracellular pixels (pixels outside *every* cell's
    footprint), bin those distances using ``distance_bins`` (right-open),
    and append the per-bin mean intensity for the cell to
    ``samples_by_type[type_idx]`` as one row of length ``len(distance_bins)``.

    Pixels closer to a different cell than to this one are excluded from
    that cell's samples — the shell only includes pixels for which this
    cell is the *nearest* cell. This prevents double-counting when cells of
    different types neighbour each other.
    """

    if cell_label.ndim != 2 or membrane.ndim != 2:
        return
    try:
        from scipy.ndimage import distance_transform_edt as _edt
    except Exception:  # noqa: BLE001
        return
    cell_mask = cell_label > 0
    extracellular = ~cell_mask
    if not np.any(extracellular):
        return
    unique_labels = np.unique(cell_label)
    nonzero = unique_labels[unique_labels > 0].tolist()
    if len(nonzero) != len(cell_ids):
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    # Distance + nearest-cell-label maps over the whole image, computed once.
    # ``distance_transform_edt`` with ``return_indices`` gives, for each pixel
    # outside ``cell_mask``, the (y, x) of the nearest pixel inside it. We
    # then read ``cell_label`` at those indices to get the nearest cell label.
    inv_cell = ~cell_mask
    dist_from_cells, indices = _edt(inv_cell, return_indices=True)
    nearest_label = cell_label[indices[0], indices[1]].astype(np.int32, copy=False)
    bins = np.asarray(distance_bins, dtype=np.float32)
    for label_value, cell_id in zip(nonzero, cell_ids):
        type_idx = int(cell_id_to_type.get(str(cell_id), 0))
        if type_idx == 0:
            continue
        # Pixels for which THIS cell is the nearest, AND outside all cells.
        nearest_mask = extracellular & (nearest_label == int(label_value))
        if not np.any(nearest_mask):
            continue
        per_bin: list[float] = []
        for k in range(len(bins) - 1):
            lo, hi = float(bins[k]), float(bins[k + 1])
            shell = nearest_mask & (dist_from_cells >= lo) & (dist_from_cells < hi)
            if np.sum(shell) < 4:
                per_bin.append(float("nan"))
            else:
                per_bin.append(float(np.mean(membrane[shell])))
        samples_by_type.setdefault(type_idx, []).append(per_bin)


def _fit_per_type_extracellular_ridge(
    type_names: list[str],
    samples_by_type: dict[int, list[list[float]]],
    distance_bins: tuple[float, ...],
    background_membrane: float,
    min_cells: int = 8,
    amp_floor: float = 0.0,
    amp_ceiling: float = 0.40,
    width_floor: float = 1.0,
    width_ceiling: float = 12.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit per-type extracellular membrane ridge amplitude and width from
    extracellular shell samples.

    Models the extracellular ridge as a Gaussian with respect to distance
    from the cell boundary line:

        intensity(d) = amp * exp(-d^2 / (2 * sigma^2)) + background

    For each type, take the median intensity per distance bin, subtract
    background, and fit by ordinary least squares on
    ``log(I_excess) = log(amp) - d^2 / (2 * sigma^2)`` over bins with
    positive excess. The intercept gives ``log(amp)``; the slope (with
    respect to ``d^2``) gives ``-1/(2 * sigma^2)``, hence
    ``sigma = sqrt(-1 / (2 * slope))``.

    Returns dicts keyed by type *name* matching MechanisticParams fields.
    Types with fewer than ``min_cells`` cells, fewer than two valid bins,
    or a flat/rising profile are omitted (no extracellular ridge).
    """

    bins = np.asarray(distance_bins, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    centers_sq = centers**2
    amp_dict: dict[str, float] = {}
    width_dict: dict[str, float] = {}
    for idx, name in enumerate(type_names):
        if idx == 0:
            continue
        rows = samples_by_type.get(idx, [])
        if len(rows) < min_cells:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(arr, axis=0)
        excess = medians - float(background_membrane)
        valid = np.isfinite(excess) & (excess > 1e-4)
        if int(valid.sum()) < 2:
            continue
        x = centers_sq[valid]
        y = np.log(excess[valid])
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= -1e-4:
            # Flat or rising — no Gaussian-from-boundary signal.
            continue
        sigma = float(np.sqrt(-1.0 / (2.0 * slope)))
        sigma = float(np.clip(sigma, width_floor, width_ceiling))
        amp = float(np.clip(np.exp(intercept), amp_floor, amp_ceiling))
        if amp <= 0.0:
            continue
        amp_dict[name] = amp
        width_dict[name] = sigma
    return amp_dict, width_dict
