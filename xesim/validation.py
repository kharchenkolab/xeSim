from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    ANCHOR_PROFILE_TYPE,
    BACKGROUND_PROFILE_TYPE,
    CELL_APPEARANCE_BENCHMARK_TYPE,
    CELL_FLOW_CORPUS_TYPE,
    CELL_FLOW_SWEEP_TYPE,
    CELL_FLOW_TRAINING_TYPE,
    CROP_MANIFEST_TYPE,
    FIT_REPORT_TYPE,
    MANIFEST_SCHEMA_VERSION,
    MECHANISTIC_REFINER_CORPUS_TYPE,
    MECHANISTIC_REFINER_EVAL_TYPE,
    MECHANISTIC_REFINER_SAMPLES_TYPE,
    MECHANISTIC_REFINER_TRAINING_TYPE,
    QC_REPORT_TYPE,
    SCENE_PROBE_SUMMARY_TYPE,
    SCENE_FIT_TYPE,
    SCENE_REVIEW_TYPE,
    STAIN_MODEL_TYPE,
    SYNTHETIC_MANIFEST_TYPE,
    TRAINING_REPORT_TYPE,
)


class ManifestValidationError(ValueError):
    """Raised when an xeSim artifact does not match the expected v0 contract."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ManifestValidationError(f"{path} is not a JSON object")
    return obj


def validate_crop_manifest(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", CROP_MANIFEST_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "bundle",
            "image_channels",
            "num_crops",
            "crop_selection",
            "split_path",
            "split_method",
            "crops",
        ],
    )
    if not isinstance(obj["crops"], list):
        raise ManifestValidationError(f"{path}: crops must be a list")
    root = path.parent
    for i, crop in enumerate(obj["crops"]):
        _require_keys(crop, path, ["crop_id", "crop_box", "npz_path", "split", "qc"])
        if crop["split"] not in {"train", "val", "test"}:
            raise ManifestValidationError(f"{path}: crop {i} has unknown split {crop['split']!r}")
        if check_files and not (root / crop["npz_path"]).exists():
            raise ManifestValidationError(f"{path}: missing crop npz {crop['npz_path']}")
    split_path = root / obj["split_path"]
    if check_files and not split_path.exists():
        raise ManifestValidationError(f"{path}: missing split manifest {obj['split_path']}")
    return obj


def validate_artifact(
    path: Path,
    kind: str = "auto",
    check_files: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Validate one xeSim JSON artifact and return its detected kind and object."""

    if kind == "crop":
        return "crop", validate_crop_manifest(path, check_files=check_files)
    if kind == "model":
        return "model", validate_stain_model(path, check_source=check_files)
    if kind == "synthetic":
        return "synthetic", validate_synthetic_manifest(path, check_files=check_files)
    if kind == "fit-report":
        return "fit-report", validate_fit_report(path)
    if kind == "qc-report":
        return "qc-report", validate_qc_report(path)
    if kind == "training-report":
        return "training-report", validate_training_report(path)
    if kind == "background-profile":
        return "background-profile", validate_background_profile(path)
    if kind == "anchor-profile":
        return "anchor-profile", validate_anchor_profile(path)
    if kind == "scene-fit":
        return "scene-fit", validate_scene_fit(path)
    if kind == "scene-summary":
        return "scene-summary", validate_scene_probe_summary(path)
    if kind == "scene-review":
        return "scene-review", validate_scene_review(path)
    if kind == "cell-appearance-benchmark":
        return "cell-appearance-benchmark", validate_cell_appearance_benchmark(path)
    if kind == "cell-flow-corpus":
        return "cell-flow-corpus", validate_cell_flow_corpus(path, check_files=check_files)
    if kind == "cell-flow-training":
        return "cell-flow-training", validate_cell_flow_training(path)
    if kind == "cell-flow-sweep":
        return "cell-flow-sweep", validate_cell_flow_sweep(path)
    if kind == "mechanistic-refiner-corpus":
        return "mechanistic-refiner-corpus", validate_mechanistic_refiner_corpus(path, check_files=check_files)
    if kind == "mechanistic-refiner-training":
        return "mechanistic-refiner-training", validate_mechanistic_refiner_training(path)
    if kind == "mechanistic-refiner-samples":
        return "mechanistic-refiner-samples", validate_mechanistic_refiner_samples(path, check_files=check_files)
    if kind == "mechanistic-refiner-eval":
        return "mechanistic-refiner-eval", validate_mechanistic_refiner_eval(path, check_files=check_files)
    if kind != "auto":
        raise ManifestValidationError(f"{path}: unknown validation kind {kind!r}")

    obj = load_json(path)
    artifact_type = obj.get("type")
    model_type = obj.get("model_type")
    if artifact_type == CROP_MANIFEST_TYPE:
        return "crop", validate_crop_manifest(path, check_files=check_files)
    if model_type == STAIN_MODEL_TYPE:
        return "model", validate_stain_model(path, check_source=check_files)
    if artifact_type == SYNTHETIC_MANIFEST_TYPE:
        return "synthetic", validate_synthetic_manifest(path, check_files=check_files)
    if artifact_type == FIT_REPORT_TYPE:
        return "fit-report", validate_fit_report(path)
    if artifact_type == QC_REPORT_TYPE:
        return "qc-report", validate_qc_report(path)
    if artifact_type == TRAINING_REPORT_TYPE:
        return "training-report", validate_training_report(path)
    if artifact_type == BACKGROUND_PROFILE_TYPE:
        return "background-profile", validate_background_profile(path)
    if artifact_type == ANCHOR_PROFILE_TYPE:
        return "anchor-profile", validate_anchor_profile(path)
    if artifact_type == SCENE_FIT_TYPE:
        return "scene-fit", validate_scene_fit(path)
    if artifact_type == SCENE_PROBE_SUMMARY_TYPE:
        return "scene-summary", validate_scene_probe_summary(path)
    if artifact_type == SCENE_REVIEW_TYPE:
        return "scene-review", validate_scene_review(path)
    if artifact_type == CELL_APPEARANCE_BENCHMARK_TYPE:
        return "cell-appearance-benchmark", validate_cell_appearance_benchmark(path)
    if artifact_type == CELL_FLOW_CORPUS_TYPE:
        return "cell-flow-corpus", validate_cell_flow_corpus(path, check_files=check_files)
    if artifact_type == CELL_FLOW_TRAINING_TYPE:
        return "cell-flow-training", validate_cell_flow_training(path)
    if artifact_type == CELL_FLOW_SWEEP_TYPE:
        return "cell-flow-sweep", validate_cell_flow_sweep(path)
    if artifact_type == MECHANISTIC_REFINER_CORPUS_TYPE:
        return "mechanistic-refiner-corpus", validate_mechanistic_refiner_corpus(path, check_files=check_files)
    if artifact_type == MECHANISTIC_REFINER_TRAINING_TYPE:
        return "mechanistic-refiner-training", validate_mechanistic_refiner_training(path)
    if artifact_type == MECHANISTIC_REFINER_SAMPLES_TYPE:
        return "mechanistic-refiner-samples", validate_mechanistic_refiner_samples(path, check_files=check_files)
    if artifact_type == MECHANISTIC_REFINER_EVAL_TYPE:
        return "mechanistic-refiner-eval", validate_mechanistic_refiner_eval(path, check_files=check_files)
    raise ManifestValidationError(f"{path}: could not detect xeSim artifact type")


def validation_summary(kind: str, obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": kind,
        "schema_version": obj.get("schema_version"),
        "valid": True,
    }
    if kind == "crop":
        out.update(
            {
                "type": obj.get("type"),
                "num_crops": len(obj.get("crops", [])),
                "image_channels": len(obj.get("image_channels", [])),
                "split_method": obj.get("split_method"),
            }
        )
    elif kind == "synthetic":
        out.update(
            {
                "type": obj.get("type"),
                "num_samples": len(obj.get("samples", [])),
                "image_channels": len(obj.get("image_channels", [])),
            }
        )
    elif kind == "model":
        out.update(
            {
                "model_type": obj.get("model_type"),
                "image_channels": len(obj.get("image_channels", [])),
                "source_manifest": obj.get("source_manifest"),
            }
        )
    elif kind == "fit-report":
        out.update({"type": obj.get("type"), "num_crops": obj.get("num_crops")})
    elif kind == "qc-report":
        out.update({"type": obj.get("type"), "num_records": obj.get("num_records")})
    elif kind == "training-report":
        out.update({"type": obj.get("type"), "steps": obj.get("steps"), "checkpoint_path": obj.get("checkpoint_path")})
    elif kind == "background-profile":
        out.update(
            {
                "type": obj.get("type"),
                "num_crops_profiled": obj.get("num_crops_profiled"),
                "sheets": obj.get("sheets", {}),
            }
        )
    elif kind == "anchor-profile":
        out.update(
            {
                "type": obj.get("type"),
                "num_crops_profiled": obj.get("num_crops_profiled"),
                "num_anchors": obj.get("num_anchors"),
                "num_high_quality_anchors": obj.get("num_high_quality_anchors"),
            }
        )
    elif kind == "scene-fit":
        out.update(
            {
                "type": obj.get("type"),
                "crop_id": obj.get("crop_id"),
                "mode": obj.get("mode"),
                "image_channels": len(obj.get("image_channels", [])),
                "channel_roles": obj.get("channel_roles", []),
                "latent_cell_budget": obj.get("latent_cell_budget"),
                "num_latent_cells": obj.get("num_latent_cells"),
                "arrays_path": obj.get("arrays_path"),
                "compare_sheet": obj.get("compare_sheet"),
            }
        )
    elif kind == "scene-summary":
        aggregate = obj.get("aggregate", {})
        out.update(
            {
                "type": obj.get("type"),
                "num_scene_fits": obj.get("num_scene_fits"),
                "total_candidates": aggregate.get("total_candidates"),
                "total_accepted_candidates": aggregate.get("total_accepted_candidates"),
                "improved_crops": aggregate.get("improved_crops"),
            }
        )
    elif kind == "scene-review":
        out.update(
            {
                "type": obj.get("type"),
                "num_rows": obj.get("num_rows"),
                "sheet_path": obj.get("sheet_path"),
                "scene_summary": obj.get("scene_summary"),
                "scene_fit": obj.get("scene_fit"),
                "review_mode": obj.get("review_mode"),
            }
        )
    elif kind == "cell-appearance-benchmark":
        holdout = obj.get("holdout", {})
        out.update(
            {
                "type": obj.get("type"),
                "num_reference_anchors": holdout.get("num_reference_anchors"),
                "num_heldout_anchors": holdout.get("num_heldout_anchors"),
                "quality_filter": obj.get("quality_filter"),
            }
        )
    elif kind == "cell-flow-corpus":
        out.update(
            {
                "type": obj.get("type"),
                "num_patches": obj.get("num_patches"),
                "patch_size": obj.get("patch_size"),
                "image_channels": len(obj.get("image_channels", [])),
                "target_mode": obj.get("target_mode"),
            }
        )
    elif kind == "cell-flow-training":
        out.update(
            {
                "type": obj.get("type"),
                "steps": obj.get("steps"),
                "checkpoint_path": obj.get("checkpoint_path"),
                "final_loss": obj.get("final_loss"),
                "target_mode": obj.get("target_mode") or obj.get("model", {}).get("target_mode"),
            }
        )
    elif kind == "cell-flow-sweep":
        best = obj.get("best_candidate", {})
        out.update(
            {
                "type": obj.get("type"),
                "num_candidates": obj.get("num_candidates"),
                "best_candidate": best.get("name"),
                "best_aggregate_distance": (best.get("metrics") or {}).get("aggregate_distance"),
            }
        )
    elif kind == "mechanistic-refiner-corpus":
        out.update(
            {
                "type": obj.get("type"),
                "num_examples": obj.get("num_examples"),
                "condition_channels": len(obj.get("condition_channels", [])),
                "target_channels": len(obj.get("target_channels", [])),
                "npz_path": obj.get("npz_path"),
            }
        )
    elif kind == "mechanistic-refiner-training":
        out.update(
            {
                "type": obj.get("type"),
                "source_corpus": obj.get("source_corpus"),
                "checkpoint_path": obj.get("checkpoint_path"),
                "steps": obj.get("steps"),
            }
        )
    elif kind == "mechanistic-refiner-samples":
        out.update(
            {
                "type": obj.get("type"),
                "source_corpus": obj.get("source_corpus"),
                "num_samples": obj.get("num_samples"),
                "npz_path": obj.get("npz_path"),
            }
        )
    return out


def validate_stain_model(path: Path, check_source: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "model_type", STAIN_MODEL_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["image_channels", "channel_stats", "source_manifest", "device"])
    if not isinstance(obj["channel_stats"], list):
        raise ManifestValidationError(f"{path}: channel_stats must be a list")
    source = Path(str(obj["source_manifest"]))
    if check_source and not source.exists():
        raise ManifestValidationError(f"{path}: source_manifest does not exist: {source}")
    if check_source and source.exists():
        validate_crop_manifest(source, check_files=True)
    return obj


def validate_synthetic_manifest(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", SYNTHETIC_MANIFEST_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["model_path", "image_channels", "samples"])
    if not isinstance(obj["samples"], list):
        raise ManifestValidationError(f"{path}: samples must be a list")
    root = path.parent
    for i, sample in enumerate(obj["samples"]):
        _require_keys(sample, path, ["sample_id", "npz_path", "latent", "image_channels"])
        if check_files and not (root / sample["npz_path"]).exists():
            raise ManifestValidationError(f"{path}: missing sample npz {sample['npz_path']}")
    return obj


def validate_fit_report(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", FIT_REPORT_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["source_manifest", "image_channels", "channel_stats", "real_summary"])
    return obj


def validate_qc_report(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", QC_REPORT_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["manifest_path", "records"])
    return obj


def validate_training_report(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", TRAINING_REPORT_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["source_manifest", "checkpoint_path", "steps", "losses", "final_loss"])
    return obj


def validate_background_profile(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", BACKGROUND_PROFILE_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["source_manifest", "empirical_model", "records", "sheets"])
    return obj


def validate_anchor_profile(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", ANCHOR_PROFILE_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_manifest",
            "image_channels",
            "num_crops_profiled",
            "num_anchors",
            "all_anchor_summary",
            "high_quality_anchor_summary",
            "shape_priors",
            "emission_priors",
            "records",
        ],
    )
    return obj


def validate_scene_fit(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", SCENE_FIT_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_manifest",
            "anchor_profile",
            "crop_id",
            "mode",
            "arrays_path",
            "compare_sheet",
            "metrics",
            "component_summary",
        ],
    )
    return obj


def validate_scene_probe_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", SCENE_PROBE_SUMMARY_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["scene_fit_paths", "num_scene_fits", "aggregate", "records"])
    if not isinstance(obj["scene_fit_paths"], list):
        raise ManifestValidationError(f"{path}: scene_fit_paths must be a list")
    if not isinstance(obj["records"], list):
        raise ManifestValidationError(f"{path}: records must be a list")
    if int(obj["num_scene_fits"]) != len(obj["records"]):
        raise ManifestValidationError(f"{path}: num_scene_fits does not match records length")
    if len(obj["scene_fit_paths"]) != len(obj["records"]):
        raise ManifestValidationError(f"{path}: scene_fit_paths does not match records length")
    if not isinstance(obj["aggregate"], dict):
        raise ManifestValidationError(f"{path}: aggregate must be an object")
    _require_keys(obj["aggregate"], path, ["total_candidates", "total_accepted_candidates", "mae_reduction"])
    return obj


def validate_scene_review(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", SCENE_REVIEW_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["sheet_path", "num_rows", "selection_groups", "rows"])
    if "scene_summary" not in obj and "scene_fit" not in obj:
        raise ManifestValidationError(f"{path}: missing required key: scene_summary or scene_fit")
    if not isinstance(obj["rows"], list):
        raise ManifestValidationError(f"{path}: rows must be a list")
    if int(obj["num_rows"]) != len(obj["rows"]):
        raise ManifestValidationError(f"{path}: num_rows does not match rows length")
    if not isinstance(obj["selection_groups"], dict):
        raise ManifestValidationError(f"{path}: selection_groups must be an object")
    return obj


def validate_cell_appearance_benchmark(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", CELL_APPEARANCE_BENCHMARK_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_anchor_profile",
            "evaluation_frame",
            "quality_filter",
            "feature_names",
            "holdout",
            "reference_summary",
            "heldout_summary",
            "heldout_vs_reference",
        ],
    )
    if obj["evaluation_frame"] != "population_anchor_feature_distribution_v0":
        raise ManifestValidationError(f"{path}: unsupported evaluation_frame {obj['evaluation_frame']!r}")
    if not isinstance(obj["feature_names"], list):
        raise ManifestValidationError(f"{path}: feature_names must be a list")
    if not isinstance(obj["holdout"], dict):
        raise ManifestValidationError(f"{path}: holdout must be an object")
    _require_keys(obj["holdout"], path, ["num_reference_anchors", "num_heldout_anchors"])
    return obj


def validate_cell_flow_corpus(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", CELL_FLOW_CORPUS_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_anchor_profile",
            "source_manifest",
            "npz_path",
            "image_channels",
            "condition_channels",
            "patch_size",
            "num_patches",
            "records",
        ],
    )
    if not isinstance(obj["records"], list):
        raise ManifestValidationError(f"{path}: records must be a list")
    if int(obj["num_patches"]) != len(obj["records"]):
        raise ManifestValidationError(f"{path}: num_patches does not match records length")
    if check_files and not (path.parent / obj["npz_path"]).exists():
        raise ManifestValidationError(f"{path}: missing corpus npz {obj['npz_path']}")
    return obj


def validate_cell_flow_training(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", CELL_FLOW_TRAINING_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_corpus",
            "checkpoint_path",
            "steps",
            "batch_size",
            "device",
            "losses",
            "final_loss",
            "model",
        ],
    )
    if not isinstance(obj["losses"], list):
        raise ManifestValidationError(f"{path}: losses must be a list")
    return obj


def validate_cell_flow_sweep(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", CELL_FLOW_SWEEP_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "checkpoint_path",
            "anchor_profile_path",
            "selection_objective",
            "base_sample_options",
            "benchmark_options",
            "num_candidates",
            "best_candidate",
            "ranking",
            "candidates",
        ],
    )
    if not isinstance(obj["candidates"], list):
        raise ManifestValidationError(f"{path}: candidates must be a list")
    if not isinstance(obj["ranking"], list):
        raise ManifestValidationError(f"{path}: ranking must be a list")
    if int(obj["num_candidates"]) != len(obj["candidates"]):
        raise ManifestValidationError(f"{path}: num_candidates does not match candidates length")
    if len(obj["ranking"]) != len(obj["candidates"]):
        raise ManifestValidationError(f"{path}: ranking length does not match candidates length")
    if not isinstance(obj["best_candidate"], dict):
        raise ManifestValidationError(f"{path}: best_candidate must be an object")
    _require_keys(obj["best_candidate"], path, ["name", "slug", "metrics"])
    return obj


def validate_mechanistic_refiner_corpus(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", MECHANISTIC_REFINER_CORPUS_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_manifest",
            "mechanistic_render_manifest",
            "npz_path",
            "image_channels",
            "condition_channels",
            "target_channels",
            "num_examples",
            "records",
            "split_counts",
            "support_masks",
        ],
    )
    if not isinstance(obj["records"], list):
        raise ManifestValidationError(f"{path}: records must be a list")
    if int(obj["num_examples"]) != len(obj["records"]):
        raise ManifestValidationError(f"{path}: num_examples does not match records length")
    if not isinstance(obj["condition_channels"], list) or not obj["condition_channels"]:
        raise ManifestValidationError(f"{path}: condition_channels must be a nonempty list")
    if obj["target_channels"] not in (["dapi", "membrane"], ["dapi", "membrane", "polya"]):
        raise ManifestValidationError(
            f"{path}: target_channels must be ['dapi', 'membrane'] or ['dapi', 'membrane', 'polya']"
        )
    if check_files and not (path.parent / obj["npz_path"]).exists():
        raise ManifestValidationError(f"{path}: missing corpus npz {obj['npz_path']}")
    if check_files:
        source = Path(str(obj["source_manifest"]))
        if not source.exists():
            raise ManifestValidationError(f"{path}: source_manifest does not exist: {source}")
        render_manifest = Path(str(obj["mechanistic_render_manifest"]))
        if not render_manifest.exists():
            raise ManifestValidationError(f"{path}: mechanistic_render_manifest does not exist: {render_manifest}")
    return obj


def validate_mechanistic_refiner_training(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", MECHANISTIC_REFINER_TRAINING_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["source_corpus", "checkpoint_path", "steps", "losses", "final_loss", "model"])
    if not isinstance(obj["losses"], list):
        raise ManifestValidationError(f"{path}: losses must be a list")
    return obj


def validate_mechanistic_refiner_samples(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", MECHANISTIC_REFINER_SAMPLES_TYPE)
    _require_schema(obj, path)
    _require_keys(obj, path, ["source_corpus", "checkpoint_path", "npz_path", "num_samples", "records"])
    if not isinstance(obj["records"], list):
        raise ManifestValidationError(f"{path}: records must be a list")
    if int(obj["num_samples"]) != len(obj["records"]):
        raise ManifestValidationError(f"{path}: num_samples does not match records length")
    if check_files and not (path.parent / obj["npz_path"]).exists():
        raise ManifestValidationError(f"{path}: missing sample npz {obj['npz_path']}")
    return obj


def validate_mechanistic_refiner_eval(path: Path, check_files: bool = True) -> dict[str, Any]:
    obj = load_json(path)
    _require_type(obj, path, "type", MECHANISTIC_REFINER_EVAL_TYPE)
    _require_schema(obj, path)
    _require_keys(
        obj,
        path,
        [
            "source_corpus",
            "source_samples",
            "patch_size",
            "patch_stride",
            "num_real_patches",
            "num_refined_patches",
            "paired_metrics",
            "spectrum",
            "anisotropy",
            "dino",
        ],
    )
    if check_files:
        sheet = obj.get("sheet_path")
        if sheet and not (path.parent / sheet).exists():
            raise ManifestValidationError(f"{path}: missing eval sheet {sheet}")
        rgb = obj.get("rgb_sheet_path")
        if rgb and not (path.parent / rgb).exists():
            raise ManifestValidationError(f"{path}: missing eval RGB sheet {rgb}")
    return obj


def _require_type(obj: dict[str, Any], path: Path, key: str, expected: str) -> None:
    actual = obj.get(key)
    if actual != expected:
        raise ManifestValidationError(f"{path}: expected {key}={expected!r}, found {actual!r}")


def _require_schema(obj: dict[str, Any], path: Path) -> None:
    actual = obj.get("schema_version")
    if actual != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"{path}: expected schema_version={MANIFEST_SCHEMA_VERSION!r}, found {actual!r}"
        )


def _require_keys(obj: dict[str, Any], path: Path, keys: list[str]) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ManifestValidationError(f"{path}: missing required keys: {', '.join(missing)}")
