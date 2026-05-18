from __future__ import annotations

import contextlib
import gzip
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore

from xesim.cli import main as cli_main
from xesim.canonicalize import canonicalize_bundle, inspect_bundle
from xesim.canonicalize import assign_spatial_splits
from xesim.features import distance_transform_backend
from xesim.images import crop_pixel_bounds
from xesim.metrics import compare_manifests
from xesim.mechanistic_render import render_mechanistic
from xesim.mechanistic_scene import (
    MechanisticCell,
    MechanisticParams,
    MechanisticScene,
    make_tiny_mechanistic_scene,
    validate_mechanistic_scene,
)
from xesim.models import CropBox
from xesim.render import (
    fit_stain_statistics,
    load_tiny_renderer_checkpoint,
    render_parametric,
    sample_from_model,
    save_fit_report,
    save_stain_model,
    train_renderer,
    train_tiny_renderer,
)
from xesim.schema import (
    ANCHOR_PROFILE_TYPE,
    BACKGROUND_PROFILE_TYPE,
    CELL_APPEARANCE_BENCHMARK_TYPE,
    CELL_FLOW_CORPUS_TYPE,
    CELL_FLOW_SWEEP_TYPE,
    CELL_FLOW_TRAINING_TYPE,
    CROP_MANIFEST_TYPE,
    FIT_REPORT_TYPE,
    MECHANISTIC_REFINER_CORPUS_TYPE,
    MECHANISTIC_REFINER_SAMPLES_TYPE,
    MECHANISTIC_REFINER_TRAINING_TYPE,
    QC_REPORT_TYPE,
    SCENE_FIT_TYPE,
    SCENE_PROBE_SUMMARY_TYPE,
    SCENE_REVIEW_TYPE,
    SPLIT_MANIFEST_TYPE,
    STAIN_MODEL_TYPE,
    SYNTHETIC_MANIFEST_TYPE,
    TRAINING_REPORT_TYPE,
)
from xesim.torch_utils import get_device
from xesim.validation import (
    validate_artifact,
    validate_anchor_profile,
    validate_background_profile,
    validate_cell_appearance_benchmark,
    validate_cell_flow_corpus,
    validate_cell_flow_sweep,
    validate_cell_flow_training,
    validate_crop_manifest,
    validate_fit_report,
    validate_mechanistic_refiner_corpus,
    validate_mechanistic_refiner_samples,
    validate_mechanistic_refiner_training,
    validate_qc_report,
    validate_scene_fit,
    validate_scene_probe_summary,
    validate_scene_review,
    validate_training_report,
)
from xesim.xenium import (
    CellSummary,
    choose_crop_boxes,
    read_cells,
    resolve_bundle,
    transcript_counts_by_crop,
    transcript_scan_strategy,
)
from xesim.zarr_masks import decode_cell_id_from_zarr, decode_xenium_base16_letters


class XeSimCoreTests(unittest.TestCase):
    def test_mechanistic_scene_roundtrip_and_render(self) -> None:
        scene = make_tiny_mechanistic_scene()
        obj = scene.to_dict()
        restored = MechanisticScene.from_dict(json.loads(json.dumps(obj)))
        self.assertEqual(restored.scene_id, "tiny_plan3")
        self.assertEqual(restored.image_shape, (64, 64))
        self.assertEqual(len(restored.cells), 2)

        params = MechanisticParams(noise_sigma=0.0, membrane_dropout_keep_prob=1.0, seed=11)
        first = render_mechanistic(restored, params=params, seed=11)
        second = render_mechanistic(restored, params=params, seed=11)
        self.assertEqual(first.images.shape, (3, 64, 64))
        self.assertEqual(first.images.dtype, np.float32)
        self.assertTrue(np.allclose(first.images, second.images))
        self.assertIn("membrane_boundary", first.components)
        self.assertIn("membrane_cell_interior", first.components)
        self.assertIn("dapi_nucleus", first.components)
        boundary = first.components["cell_boundary"] > 0
        interior = (restored.cell_label > 0) & ~boundary
        self.assertGreater(float(np.mean(first.components["membrane_boundary"][boundary])), 0.1)
        self.assertLess(
            float(np.mean(first.components["membrane_cell_interior"][interior])),
            float(np.mean(first.components["membrane_boundary"][boundary])),
        )
        render_manifest = first.to_manifest()
        self.assertEqual(render_manifest["type"], "xesim.mechanistic_render.v0")
        self.assertIn("membrane_degraded", render_manifest["components"])

    def test_mechanistic_scene_validation_rejects_bad_cells(self) -> None:
        scene = make_tiny_mechanistic_scene()
        bad = MechanisticScene(
            image_shape=scene.image_shape,
            pixel_size=scene.pixel_size,
            cell_label=scene.cell_label,
            nucleus_label=scene.nucleus_label,
            cells=(
                MechanisticCell(cell_id="bad", label=1, source="not_a_source"),
            ),
        )
        with self.assertRaises(ValueError):
            validate_mechanistic_scene(bad)

    def test_manifest_csv_loading_and_crop_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp))
            bundle = resolve_bundle(root)
            self.assertEqual(bundle.pixel_size, 0.5)
            cells = read_cells(bundle)
            self.assertEqual(len(cells), 2)
            crops = [CropBox("crop_1", 0, 25, 0, 25)]
            counts = transcript_counts_by_crop(bundle, crops, min_qv=20)
            self.assertEqual(counts["crop_1"], (2, 1))
            summary = inspect_bundle(root)
            self.assertEqual(summary["num_cells_readable"], 2)
            self.assertIn("pyarrow", summary["dependencies"])
            self.assertIn("tifffile", summary["dependencies"])

    def test_crop_pixel_conversion(self) -> None:
        crop = CropBox("c", 10.0, 12.0, 20.0, 23.0)
        self.assertEqual(crop_pixel_bounds(crop, pixel_size=0.5), (20, 24, 40, 46))

    def test_canonicalize_fit_and_sample_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp) / "bundle")
            out = Path(tmp) / "canonical"
            write_cells_zarr(root / "cells.zarr.zip")
            manifest = canonicalize_bundle(
                root,
                out,
                num_crops=1,
                crop_size_um=20.0,
                include_images=False,
                geometry_source="zarr",
            )
            with manifest.open() as handle:
                obj = json.load(handle)
            self.assertEqual(obj["type"], CROP_MANIFEST_TYPE)
            self.assertEqual(obj["schema_version"], "0.1.0")
            self.assertEqual(obj["split_path"], "splits.json")
            self.assertEqual(obj["crop_selection"]["mode"], "spread")
            self.assertEqual(obj["transcript_counting"]["scan_strategy"], "crop_boxes")
            self.assertEqual(obj["image_channels"], [])
            splits = json.loads((out / "splits.json").read_text())
            self.assertEqual(splits["type"], SPLIT_MANIFEST_TYPE)
            self.assertIn("train", splits["splits"])
            self.assertEqual(obj["num_crops"], 1)
            crop_path = out / obj["crops"][0]["npz_path"]
            data = np.load(crop_path, allow_pickle=True)
            self.assertEqual(obj["crops"][0]["qc"]["geometry_source"], "zarr:cells.zarr.zip:masks/1")
            self.assertGreater(float(np.mean(data["cell_label"] > 0)), 0.0)
            self.assertGreater(float(np.mean(data["nucleus_label"] > 0)), 0.0)
            self.assertIn("aaaaaaab-1", data["cell_ids"].tolist())
            self.assertIn("cell_boundary", data.files)
            self.assertIn("cell_interior_distance_um", data.files)
            self.assertGreaterEqual(obj["crops"][0]["qc"]["cell_boundary_fraction"], 0.0)
            self.assertIn(obj["crops"][0]["qc"]["distance_transform_backend"], {"morphology", "scipy"})

            # Inject a small target image so the stain fit path is exercised.
            images = np.stack(
                [
                    (data["nucleus_label"] > 0).astype(np.float32),
                    (data["cell_label"] > 0).astype(np.float32),
                ],
                axis=0,
            )
            np.savez_compressed(
                crop_path,
                images=images,
                cell_label=data["cell_label"],
                nucleus_label=data["nucleus_label"],
                cell_ids=data["cell_ids"],
                nucleus_ids=data["nucleus_ids"],
                crop_bounds_um=data["crop_bounds_um"],
                pixel_size=data["pixel_size"],
            )
            obj["image_channels"] = ["dapi", "membrane"]
            obj["crops"][0]["image_channels"] = ["dapi", "membrane"]
            with manifest.open("w") as handle:
                json.dump(obj, handle)

            mech_dir = Path(tmp) / "mechanistic"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "render-mechanistic",
                        str(manifest),
                        "-o",
                        str(mech_dir),
                        "--max-crops",
                        "1",
                        "--seed",
                        "3",
                        "--membrane-interior-fraction",
                        "0.05",
                    ]
                )
            mech_manifest = json.loads((mech_dir / "mechanistic_render_manifest.json").read_text())
            self.assertEqual(mech_manifest["type"], "xesim.mechanistic_render_manifest.v0")
            self.assertEqual(mech_manifest["num_records"], 1)
            self.assertEqual(mech_manifest["params"]["membrane_interior_fraction"], 0.05)
            self.assertIn("metric_summary", mech_manifest)
            self.assertIn("mechanistic_metrics", mech_manifest["renders"][0])
            self.assertIn(
                "membrane_boundary_to_interior_ratio",
                mech_manifest["renders"][0]["mechanistic_metrics"],
            )
            self.assertIn(
                "rendered_boundary_real_ridge_corr",
                mech_manifest["renders"][0]["mechanistic_metrics"],
            )
            self.assertIn("latent_ridge_num_objects", mech_manifest["renders"][0]["mechanistic_metrics"])
            self.assertIn("latent_ridges", mech_manifest["renders"][0])
            self.assertIn("objects", mech_manifest["renders"][0]["latent_ridges"])
            self.assertTrue((mech_dir / "mechanistic_debug_sheet.png").read_bytes().startswith(b"\x89PNG"))
            mech_data = np.load(mech_dir / mech_manifest["renders"][0]["npz_path"])
            self.assertEqual(mech_data["images"].shape[0], 3)
            self.assertIn("mechanistic_membrane_boundary", mech_data.files)
            self.assertIn("mechanistic_green_tissue_haze", mech_data.files)
            self.assertIn("real_membrane_ridge_evidence", mech_data.files)
            self.assertIn("latent_membrane_ridge", mech_data.files)
            self.assertIn("unexplained_ridge_score", mech_data.files)

            mech_fit_dir = Path(tmp) / "mechanistic_priors"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "fit-mechanistic-priors",
                        str(manifest),
                        "-o",
                        str(mech_fit_dir),
                        "--max-crops",
                        "1",
                    ]
                )
            mech_prior_report = json.loads((mech_fit_dir / "mechanistic_prior_report.json").read_text())
            self.assertEqual(mech_prior_report["type"], "xesim.mechanistic_report.v0")
            self.assertEqual(mech_prior_report["report_kind"], "mechanistic_prior_fit.v0")
            self.assertIn("membrane_ridge_boundary_enrichment", mech_prior_report["summary"])
            self.assertIn("membrane_ridge_width_px", mech_prior_report["summary"])
            self.assertIn("membrane_ridge_near_boundary_coverage", mech_prior_report["records"][0])
            self.assertTrue((mech_fit_dir / "mechanistic_params.json").exists())
            mech_from_params_dir = Path(tmp) / "mechanistic_from_params"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "render-mechanistic",
                        str(manifest),
                        "-o",
                        str(mech_from_params_dir),
                        "--max-crops",
                        "1",
                        "--params",
                        str(mech_fit_dir / "mechanistic_params.json"),
                    ]
                )
            mech_from_params = json.loads((mech_from_params_dir / "mechanistic_render_manifest.json").read_text())
            self.assertEqual(mech_from_params["params"]["type"], "xesim.mechanistic_params.v0")

            refiner_corpus_dir = Path(tmp) / "mechanistic_refiner_corpus"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "build-mechanistic-refiner-corpus",
                        str(manifest),
                        str(mech_dir / "mechanistic_render_manifest.json"),
                        "-o",
                        str(refiner_corpus_dir),
                        "--support-margin-px",
                        "2",
                    ]
                )
            refiner_corpus = validate_mechanistic_refiner_corpus(
                refiner_corpus_dir / "mechanistic_refiner_corpus.json"
            )
            self.assertEqual(refiner_corpus["type"], MECHANISTIC_REFINER_CORPUS_TYPE)
            self.assertEqual(refiner_corpus["num_examples"], 1)
            self.assertEqual(refiner_corpus["split_counts"]["train"], 1)
            self.assertEqual(refiner_corpus["target_channels"], ["dapi", "membrane"])
            self.assertIn("rough_membrane", refiner_corpus["condition_channels"])
            self.assertIn("latent_ridge_map", refiner_corpus["condition_channels"])
            self.assertEqual(refiner_corpus["support_masks"]["foreground_support_mask"]["support_margin_px"], 2)
            self.assertTrue((refiner_corpus_dir / "mechanistic_refiner_corpus_sheet.png").read_bytes().startswith(b"\x89PNG"))
            with np.load(refiner_corpus_dir / refiner_corpus["npz_path"], allow_pickle=True) as refiner_npz:
                self.assertIn("conditioning", refiner_npz.files)
                self.assertIn("target", refiner_npz.files)
                self.assertIn("rough", refiner_npz.files)
                self.assertIn("foreground_support_mask", refiner_npz.files)
                self.assertIn("leakage_mask", refiner_npz.files)
                self.assertIn("valid_mask", refiner_npz.files)
                self.assertEqual(refiner_npz["conditioning"].shape[0], 1)
                self.assertEqual(refiner_npz["conditioning"].shape[1], len(refiner_corpus["condition_channels"]))
                self.assertEqual(refiner_npz["target"].shape[1], 2)
                self.assertGreater(float(np.mean(refiner_npz["foreground_support_mask"])), 0.0)
            kind, detected_refiner_corpus = validate_artifact(refiner_corpus_dir / "mechanistic_refiner_corpus.json")
            self.assertEqual(kind, "mechanistic-refiner-corpus")
            self.assertEqual(detected_refiner_corpus["type"], MECHANISTIC_REFINER_CORPUS_TYPE)

            refiner_checkpoint = Path(tmp) / "mechanistic_refiner.pt"
            refiner_training_path = Path(tmp) / "mechanistic_refiner_training.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "train-mechanistic-refiner",
                        str(refiner_corpus_dir / "mechanistic_refiner_corpus.json"),
                        "-o",
                        str(refiner_checkpoint),
                        "--report-output",
                        str(refiner_training_path),
                        "--steps",
                        "1",
                        "--batch-size",
                        "1",
                        "--device",
                        "cpu",
                        "--hidden-channels",
                        "8",
                        "--val-every",
                        "1",
                    ]
                )
            refiner_training = validate_mechanistic_refiner_training(refiner_training_path)
            self.assertEqual(refiner_training["type"], MECHANISTIC_REFINER_TRAINING_TYPE)
            self.assertEqual(refiner_training["loss_name"], "bounded_residual_refiner_v0")
            self.assertEqual(refiner_training["selected_step"], 1)
            self.assertIn("leakage_residual", refiner_training["loss_weights"])
            self.assertIn("membrane_gradient_l1", refiner_training["losses"][0])
            self.assertTrue(refiner_checkpoint.exists())
            kind, detected_refiner_training = validate_artifact(refiner_training_path)
            self.assertEqual(kind, "mechanistic-refiner-training")
            self.assertEqual(detected_refiner_training["type"], MECHANISTIC_REFINER_TRAINING_TYPE)

            refiner_sample_dir = Path(tmp) / "mechanistic_refiner_samples"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "sample-mechanistic-refiner",
                        str(refiner_corpus_dir / "mechanistic_refiner_corpus.json"),
                        str(refiner_checkpoint),
                        "-o",
                        str(refiner_sample_dir),
                        "--split",
                        "all",
                        "--max-records",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
            refiner_samples = validate_mechanistic_refiner_samples(
                refiner_sample_dir / "mechanistic_refiner_samples.json"
            )
            self.assertEqual(refiner_samples["type"], MECHANISTIC_REFINER_SAMPLES_TYPE)
            self.assertEqual(refiner_samples["num_samples"], 1)
            self.assertIn("mae_delta_mean", refiner_samples["summary"])
            self.assertTrue((refiner_sample_dir / "mechanistic_refiner_samples_sheet.png").read_bytes().startswith(b"\x89PNG"))
            with np.load(refiner_sample_dir / refiner_samples["npz_path"], allow_pickle=True) as refiner_sample_npz:
                self.assertIn("refined", refiner_sample_npz.files)
                self.assertIn("residual", refiner_sample_npz.files)
                self.assertEqual(refiner_sample_npz["refined"].shape[1], 2)
            kind, detected_refiner_samples = validate_artifact(refiner_sample_dir / "mechanistic_refiner_samples.json")
            self.assertEqual(kind, "mechanistic-refiner-samples")
            self.assertEqual(detected_refiner_samples["type"], MECHANISTIC_REFINER_SAMPLES_TYPE)

            model = fit_stain_statistics(manifest, device_name="cpu")
            self.assertEqual(model.to_dict()["model_type"], STAIN_MODEL_TYPE)
            self.assertIn("background_style", model.to_dict())
            self.assertEqual(model.to_dict()["background_style"]["type"], "xesim.background_style.v0")
            self.assertIn("role_calibration", model.to_dict())
            self.assertEqual(model.to_dict()["role_calibration"]["type"], "xesim.role_calibration.v0")
            self.assertIn("residual_style", model.to_dict())
            self.assertEqual(model.to_dict()["residual_style"]["type"], "xesim.residual_style.v0")
            self.assertIn("render_calibration", model.to_dict())
            self.assertEqual(model.to_dict()["render_calibration"]["type"], "xesim.render_calibration.v0")
            self.assertIn("channel_gain", model.to_dict()["render_calibration"])
            self.assertIn("background_gain", model.to_dict()["render_calibration"])
            self.assertIn("background_correction_scale", model.to_dict()["render_calibration"])
            self.assertIn("background_covariance_mae", model.to_dict()["render_calibration"]["score_components"])
            self.assertIn("background_chromaticity_mae", model.to_dict()["render_calibration"]["score_components"])
            self.assertIn("visual_background_whiteness_delta", model.to_dict()["render_calibration"]["score_components"])
            self.assertIn("background_texture_mae", model.to_dict()["render_calibration"]["score_components"])
            checkpoint = train_tiny_renderer(
                manifest,
                Path(tmp) / "tiny.pt",
                steps=1,
                device_name="cpu",
                metadata_path=Path(tmp) / "tiny_training.json",
            )
            self.assertTrue(checkpoint.exists())
            training = validate_training_report(Path(tmp) / "tiny_training.json")
            self.assertEqual(training["type"], TRAINING_REPORT_TYPE)
            self.assertEqual(training["steps"], 1)
            self.assertEqual(training["selected_step"], 1)
            self.assertEqual(training["selection_metric"], "validation_loss")
            self.assertIsNotNone(training["selected_validation_loss"])
            self.assertEqual(training["selected_validation_score"], training["selected_validation_loss"])
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "train-renderer",
                        str(manifest),
                        "-o",
                        str(Path(tmp) / "renderer.pt"),
                        "--report-output",
                        str(Path(tmp) / "renderer_training.json"),
                        "--steps",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
            renderer_training = validate_training_report(Path(tmp) / "renderer_training.json")
            self.assertEqual(renderer_training["completed_steps"], 1)
            self.assertEqual(renderer_training["selected_step"], 1)
            model_path = Path(tmp) / "model.json"
            save_stain_model(model, model_path)
            residual_checkpoint = train_renderer(
                manifest,
                Path(tmp) / "residual.pt",
                report_path=Path(tmp) / "residual_training.json",
                steps=1,
                device_name="cpu",
                renderer_mode="residual",
            )
            residual_training = validate_training_report(Path(tmp) / "residual_training.json")
            self.assertEqual(residual_training["renderer_mode"], "residual")
            self.assertEqual(residual_training["loss_name"], "reconstruction_l1")
            self.assertEqual(residual_training["selected_step"], 1)
            self.assertEqual(load_tiny_renderer_checkpoint(residual_checkpoint, device_name="cpu").step, 1)
            style_checkpoint = train_renderer(
                manifest,
                Path(tmp) / "style_residual.pt",
                report_path=Path(tmp) / "style_residual_training.json",
                steps=1,
                device_name="cpu",
                renderer_mode="style_residual",
            )
            style_training = validate_training_report(Path(tmp) / "style_residual_training.json")
            self.assertEqual(style_training["renderer_mode"], "style_residual")
            self.assertTrue(style_training["baseline_stochastic"])
            self.assertIn("cell_exterior_distance", style_training["input_channels"])
            self.assertEqual(style_training["loss_name"], "component_aware_style_residual_v0")
            self.assertEqual(style_training["visual_background_loss_space"], "raw_plus_display_lut")
            self.assertEqual(style_training["display_lut"]["type"], "xesim.display_lut.v0")
            self.assertIn("background_texture_l1", style_training["loss_weights"])
            self.assertIn("mask_explained_l1", style_training["loss_components"][0])
            self.assertIn("background_chromaticity_l1", style_training["loss_components"][0])
            self.assertIn("background_mean_l1", style_training["validation_losses"][0])
            self.assertIn("background_chromaticity_l1", style_training["validation_losses"][0])
            self.assertIn("stain_ratio_l1", style_training["validation_losses"][0])
            self.assertIn("validation_realism_score", style_training["validation_losses"][0])
            self.assertEqual(style_training["selected_step"], 1)
            self.assertEqual(style_training["selection_metric"], "validation_realism_score")
            self.assertIn("stain_ratio_l1", style_training["selection_weights"])
            self.assertEqual(load_tiny_renderer_checkpoint(style_checkpoint, device_name="cpu").step, 1)
            decomposed_checkpoint = train_renderer(
                manifest,
                Path(tmp) / "decomposed_style.pt",
                report_path=Path(tmp) / "decomposed_style_training.json",
                steps=1,
                device_name="cpu",
                renderer_mode="decomposed_style",
            )
            decomposed_training = validate_training_report(Path(tmp) / "decomposed_style_training.json")
            self.assertEqual(decomposed_training["renderer_mode"], "decomposed_style")
            self.assertTrue(decomposed_training["baseline_stochastic"])
            self.assertEqual(decomposed_training["loss_name"], "decomposed_component_style_v0")
            self.assertEqual(decomposed_training["visual_background_loss_space"], "raw_plus_display_lut")
            self.assertEqual(decomposed_training["display_lut"]["type"], "xesim.display_lut.v0")
            self.assertIn("visual_background_l1", decomposed_training["loss_weights"])
            self.assertIn("texture_component_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_component_target_l1", decomposed_training["loss_components"][0])
            self.assertIn("nucleus_component_target_l1", decomposed_training["loss_components"][0])
            self.assertIn("boundary_component_target_l1", decomposed_training["loss_components"][0])
            self.assertIn("cell_interior_component_target_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_component_mean_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_component_chromaticity_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_component_smoothness_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_global_mean_l1", decomposed_training["loss_components"][0])
            self.assertIn("background_global_chromaticity_l1", decomposed_training["loss_components"][0])
            self.assertIn("texture_mask_leak_l1", decomposed_training["loss_components"][0])
            self.assertEqual(decomposed_training["selected_step"], 1)
            self.assertEqual(decomposed_training["selection_metric"], "validation_realism_score")
            self.assertIsNotNone(decomposed_training["selected_validation_score"])
            self.assertEqual(load_tiny_renderer_checkpoint(decomposed_checkpoint, device_name="cpu").step, 1)
            fit_report_path = Path(tmp) / "fit_report.json"
            save_fit_report(manifest, model, fit_report_path, model_path=model_path)
            fit_report = validate_fit_report(fit_report_path)
            self.assertEqual(fit_report["type"], FIT_REPORT_TYPE)
            self.assertIn("real_summary", fit_report)
            self.assertIn("background_style", fit_report)
            self.assertIn("role_calibration", fit_report)
            self.assertIn("residual_style", fit_report)
            self.assertIn("render_calibration", fit_report)
            self.assertIn("channel_histograms", fit_report)
            self.assertIn("regression_thresholds", fit_report)
            synth_manifest = sample_from_model(model_path, Path(tmp) / "synthetic", num_samples=2, seed=4)
            with synth_manifest.open() as handle:
                synth = json.load(handle)
            self.assertEqual(synth["type"], SYNTHETIC_MANIFEST_TYPE)
            self.assertEqual(synth["schema_version"], "0.1.0")
            self.assertEqual(len(synth["samples"]), 2)
            latent = synth["samples"][0]["latent"]
            self.assertIn("attenuation", latent["stain_state"])
            self.assertIn("background_field", latent["stain_state"])
            self.assertIn("background_style", latent["stain_state"])
            self.assertIn("residual_style", latent["stain_state"])
            self.assertIn("render_calibration", latent["stain_state"])
            self.assertIn("channel_gain", latent["stain_state"]["render_calibration"])
            self.assertIn("background_gain", latent["stain_state"]["render_calibration"])
            self.assertIn("background_correction_scale", latent["stain_state"]["render_calibration"])
            self.assertEqual(latent["stain_state"]["channels"][0]["role_fit"]["method"], "bundle_role_calibration")
            exact_manifest = sample_from_model(
                model_path,
                Path(tmp) / "exact_synthetic",
                num_samples=1,
                seed=4,
                device_name="cpu",
                source_mode="sequential",
                perturb_geometry=False,
            )
            exact = json.loads(exact_manifest.read_text())
            self.assertEqual(exact["source_mode"], "sequential")
            self.assertEqual(exact["source_split"], "all")
            self.assertFalse(exact["perturb_geometry"])
            self.assertEqual(exact["samples"][0]["latent"]["source_crop_id"], obj["crops"][0]["crop_id"])
            self.assertFalse(exact["samples"][0]["latent"]["perturbations"]["applied"])
            train_split_manifest = sample_from_model(
                model_path,
                Path(tmp) / "train_split_synthetic",
                num_samples=1,
                seed=4,
                device_name="cpu",
                source_mode="sequential",
                source_split="train",
                perturb_geometry=False,
            )
            train_split = json.loads(train_split_manifest.read_text())
            self.assertEqual(train_split["source_split"], "train")
            self.assertEqual(train_split["samples"][0]["latent"]["source_crop_id"], obj["crops"][0]["crop_id"])
            synth_data = np.load(Path(tmp) / "synthetic" / synth["samples"][0]["npz_path"])
            self.assertIn("cell_boundary", synth_data.files)
            self.assertIn("pixel_size", synth_data.files)
            learned_manifest = sample_from_model(
                model_path,
                Path(tmp) / "learned_synthetic",
                num_samples=1,
                seed=5,
                device_name="cpu",
                renderer="learned",
                renderer_checkpoint_path=Path(tmp) / "renderer.pt",
            )
            with learned_manifest.open() as handle:
                learned = json.load(handle)
            self.assertEqual(learned["renderer"], "learned")
            self.assertEqual(learned["samples"][0]["latent"]["stain_state"]["renderer"], "learned")
            self.assertEqual(learned["samples"][0]["latent"]["stain_state"]["checkpoint_step"], 1)
            learned_data = np.load(Path(tmp) / "learned_synthetic" / learned["samples"][0]["npz_path"])
            self.assertEqual(learned_data["images"].shape[0], 2)
            residual_manifest = sample_from_model(
                model_path,
                Path(tmp) / "residual_synthetic",
                num_samples=1,
                seed=5,
                device_name="cpu",
                renderer="learned",
                renderer_checkpoint_path=residual_checkpoint,
                perturb_geometry=False,
            )
            residual = json.loads(residual_manifest.read_text())
            residual_state = residual["samples"][0]["latent"]["stain_state"]
            self.assertEqual(residual_state["renderer_mode"], "residual")
            self.assertIn("baseline_state", residual_state)
            style_manifest = sample_from_model(
                model_path,
                Path(tmp) / "style_residual_synthetic",
                num_samples=1,
                seed=5,
                device_name="cpu",
                renderer="learned",
                renderer_checkpoint_path=style_checkpoint,
                perturb_geometry=False,
            )
            style_synthetic = json.loads(style_manifest.read_text())
            style_state = style_synthetic["samples"][0]["latent"]["stain_state"]
            self.assertEqual(style_state["renderer_mode"], "style_residual")
            self.assertTrue(style_state["baseline_stochastic"])
            self.assertEqual(style_state["checkpoint_step"], style_training["selected_step"])
            decomposed_manifest = sample_from_model(
                model_path,
                Path(tmp) / "decomposed_style_synthetic",
                num_samples=1,
                seed=5,
                device_name="cpu",
                renderer="learned",
                renderer_checkpoint_path=decomposed_checkpoint,
                perturb_geometry=False,
            )
            decomposed_synthetic = json.loads(decomposed_manifest.read_text())
            decomposed_state = decomposed_synthetic["samples"][0]["latent"]["stain_state"]
            self.assertEqual(decomposed_state["renderer_mode"], "decomposed_style")
            self.assertIn("mask_linked", decomposed_state["decomposed_components"])
            self.assertIn("nucleus_linked", decomposed_state["decomposed_components"])
            self.assertIn("boundary_linked", decomposed_state["decomposed_components"])
            self.assertIn("cell_interior_linked", decomposed_state["decomposed_components"])
            self.assertIn("background_global_haze", decomposed_state["decomposed_components"])
            self.assertEqual(decomposed_state["checkpoint_step"], decomposed_training["selected_step"])
            decomposed_data = np.load(
                Path(tmp) / "decomposed_style_synthetic" / decomposed_synthetic["samples"][0]["npz_path"]
            )
            self.assertIn("decomposed_mask_linked", decomposed_data.files)
            self.assertIn("decomposed_nucleus_linked", decomposed_data.files)
            self.assertIn("decomposed_boundary_linked", decomposed_data.files)
            self.assertIn("decomposed_cell_interior_linked", decomposed_data.files)
            self.assertIn("decomposed_background_haze", decomposed_data.files)
            self.assertIn("decomposed_background_global_haze", decomposed_data.files)
            self.assertIn("decomposed_residual_texture", decomposed_data.files)
            component_sheet = Path(tmp) / "decomposed_components.png"
            component_report = Path(tmp) / "decomposed_components_qc.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "qc-components-sheet",
                        str(decomposed_manifest),
                        "-o",
                        str(component_sheet),
                        "--report-output",
                        str(component_report),
                    ]
                )
            self.assertTrue(component_sheet.read_bytes().startswith(b"\x89PNG"))
            component_qc = validate_qc_report(component_report)
            self.assertEqual(component_qc["records_key"], "decomposed_components")
            self.assertIn("decomposed_mask_linked", component_qc["records"][0]["components"])
            metrics = compare_manifests(
                manifest,
                synth_manifest,
                Path(tmp) / "metrics.json",
                fit_report_path=fit_report_path,
            )
            self.assertIn("cell_boundary_fraction", metrics["delta"])
            self.assertEqual(metrics["comparison"]["mode"], "paired_by_source_crop_id")
            self.assertEqual(metrics["comparison"]["num_pairs"], 2)
            self.assertIn("threshold_evaluation", metrics)
            self.assertIn("boundary_contrast", metrics["delta"]["channels"][0])
            self.assertIn("membrane_continuity", metrics["delta"]["channels"][0])
            self.assertIn("texture_power", metrics["delta"]["channels"][0])
            self.assertIn("channel_cross_correlation", metrics["delta"])
            self.assertIn("region_intensity", metrics["delta"])
            self.assertIn("background", metrics["delta"]["region_intensity"])
            self.assertIn("stain_ratios", metrics["delta"])
            self.assertIn("mask_explanation", metrics["real"])
            self.assertIn("mask_explanation", metrics["delta"])
            self.assertIn("mean_unexplained_variance_fraction", metrics["real"]["mask_explanation"])
            self.assertIn("visual_background", metrics["real"])
            self.assertIn("visual_background", metrics["delta"])
            self.assertIn("background_whiteness_mae", metrics["delta"]["visual_background"])
            sheet_path = Path(tmp) / "sheet.png"
            qc_report_path = Path(tmp) / "qc_report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "qc-contact-sheet",
                        str(synth_manifest),
                        "-o",
                        str(sheet_path),
                        "--report-output",
                        str(qc_report_path),
                        "--max-records",
                        "2",
                    ]
                )
            self.assertTrue(sheet_path.read_bytes().startswith(b"\x89PNG"))
            qc_report = validate_qc_report(qc_report_path)
            self.assertEqual(qc_report["type"], QC_REPORT_TYPE)
            compare_sheet = Path(tmp) / "compare.png"
            compare_report = Path(tmp) / "compare_qc.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "qc-compare-sheet",
                        str(manifest),
                        str(synth_manifest),
                        "-o",
                        str(compare_sheet),
                        "--report-output",
                        str(compare_report),
                    ]
                )
            self.assertTrue(compare_sheet.read_bytes().startswith(b"\x89PNG"))
            compare_qc = validate_qc_report(compare_report)
            self.assertEqual(compare_qc["records_key"], "pairs")
            self.assertIn("display_lut", compare_qc)
            background_dir = Path(tmp) / "background_profile"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "profile-background",
                        str(manifest),
                        "-o",
                        str(background_dir),
                        "--max-records",
                        "1",
                    ]
                )
            background_profile = validate_background_profile(background_dir / "background_profile.json")
            self.assertEqual(background_profile["type"], BACKGROUND_PROFILE_TYPE)
            self.assertEqual(background_profile["num_crops_profiled"], 1)
            self.assertIn("empirical_model", background_profile)
            self.assertIn("visual_background", background_profile)
            self.assertIn("background_statistics", background_profile)
            self.assertIn("sampler_previews", background_profile)
            self.assertIn("grid_residual", background_profile["sampler_previews"])
            self.assertIn("field_residual", background_profile["sampler_previews"])
            self.assertIn("hybrid_patch_field", background_profile["sampler_previews"])
            self.assertIn("patch_mosaic", background_profile["sampler_previews"])
            self.assertIn("sampler_ranking", background_profile)
            self.assertIn("selected_sampler", background_profile)
            self.assertEqual(background_profile["selected_sampler"], background_profile["sampler_ranking"][0]["sampler"])
            self.assertEqual(background_profile["compare_columns"][0], "real_background")
            self.assertIn("hybrid_patch_field", background_profile["compare_columns"])
            self.assertIn("patch_mosaic", background_profile["compare_columns"])
            self.assertIn("background_statistics_delta", background_profile["sampler_previews"]["grid_residual"])
            self.assertIn("grid_residual_library", background_profile["empirical_model"])
            self.assertIn("field_residual_library", background_profile["empirical_model"])
            self.assertTrue((background_dir / "real_background_sheet.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((background_dir / "synthetic_background_sheet.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(
                (background_dir / "synthetic_background_exemplar_sheet.png").read_bytes().startswith(b"\x89PNG")
            )
            kind, detected_profile = validate_artifact(background_dir / "background_profile.json")
            self.assertEqual(kind, "background-profile")
            self.assertEqual(detected_profile["type"], BACKGROUND_PROFILE_TYPE)
            anchor_dir = Path(tmp) / "anchor_profile"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "profile-anchors",
                        str(manifest),
                        "-o",
                        str(anchor_dir),
                    ]
                )
            anchor_profile = validate_anchor_profile(anchor_dir / "anchor_profile.json")
            self.assertEqual(anchor_profile["type"], ANCHOR_PROFILE_TYPE)
            self.assertGreaterEqual(anchor_profile["num_anchors"], 1)
            self.assertIn("all_anchor_summary", anchor_profile)
            self.assertIn("high_quality_anchor_summary", anchor_profile)
            self.assertIn("shape_priors", anchor_profile)
            self.assertIn("emission_priors", anchor_profile)
            self.assertIn("nucleus", anchor_profile["emission_priors"])
            self.assertIn("boundary_minus_local_background", anchor_profile["emission_priors"])
            first_anchor = anchor_profile["records"][0]["anchors"][0]
            self.assertIn(first_anchor["quality_label"], {"high", "medium", "low"})
            self.assertIn("quality_score", first_anchor)
            kind, detected_anchor = validate_artifact(anchor_dir / "anchor_profile.json")
            self.assertEqual(kind, "anchor-profile")
            self.assertEqual(detected_anchor["type"], ANCHOR_PROFILE_TYPE)
            sampled_cells_dir = Path(tmp) / "sampled_anchor_cells"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "sample-anchor-cells",
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(sampled_cells_dir),
                        "--num-cells",
                        "6",
                        "--canvas-tile-size",
                        "48",
                        "--sheet-tile-size",
                        "64",
                        "--latent-emission-variation-strength",
                        "0.5",
                    ]
                )
            sampled_cell_profile = validate_anchor_profile(sampled_cells_dir / "sampled_anchor_cell_profile.json")
            self.assertEqual(sampled_cell_profile["profile_kind"], "sampled_anchor_cell_profile_v0")
            self.assertEqual(sampled_cell_profile["sampling"]["emission_texture_model"], "anchor_texture_v1")
            self.assertGreaterEqual(sampled_cell_profile["num_anchors"], 1)
            sampled_anchor = sampled_cell_profile["records"][0]["anchors"][0]
            self.assertIn("sampled_cell_source", sampled_anchor)
            self.assertIn("patch_features", sampled_anchor)
            self.assertIn("nucleus_highpass_std", sampled_anchor["patch_features"])
            self.assertIn("boundary_rim_width_proxy", sampled_anchor["patch_features"])
            self.assertIn("cytoplasm_highpass_std", sampled_anchor["patch_features"])
            self.assertTrue((sampled_cells_dir / "sampled_anchor_cells.npz").exists())
            self.assertTrue((sampled_cells_dir / "sampled_anchor_cell_sheet.png").read_bytes().startswith(b"\x89PNG"))
            cell_flow_corpus_dir = Path(tmp) / "cell_flow_corpus"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "build-cell-corpus",
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(cell_flow_corpus_dir),
                        "--patch-size",
                        "32",
                        "--max-anchors",
                        "4",
                        "--target-mode",
                        "foreground_texture_style_v1",
                        "--support-margin-px",
                        "1",
                        "--anchor-sampling-mode",
                        "geometry-stratified",
                        "--anchor-sampling-bins",
                        "3",
                        "--anchor-sampling-seed",
                        "7",
                    ]
                )
            cell_flow_corpus = validate_cell_flow_corpus(cell_flow_corpus_dir / "cell_flow_corpus.json")
            self.assertEqual(cell_flow_corpus["type"], CELL_FLOW_CORPUS_TYPE)
            self.assertEqual(cell_flow_corpus["target_mode"], "foreground_texture_style_v1")
            self.assertEqual(cell_flow_corpus["support_margin_px"], 1)
            self.assertEqual(cell_flow_corpus["anchor_sampling"]["mode"], "geometry-stratified")
            self.assertEqual(cell_flow_corpus["anchor_sampling"]["bins"], 3)
            self.assertEqual(cell_flow_corpus["anchor_sampling"]["seed"], 7)
            self.assertTrue(cell_flow_corpus["texture_style_names"])
            self.assertTrue(any(name.startswith("texture_style_") for name in cell_flow_corpus["condition_channels"]))
            self.assertGreaterEqual(cell_flow_corpus["num_patches"], 1)
            self.assertTrue((cell_flow_corpus_dir / "cell_flow_corpus.npz").exists())
            with np.load(cell_flow_corpus_dir / "cell_flow_corpus.npz", allow_pickle=True) as flow_npz:
                self.assertIn("target_offsets", flow_npz.files)
                self.assertIn("target_scales", flow_npz.files)
                self.assertIn("texture_style_values", flow_npz.files)
                self.assertEqual(flow_npz["texture_style_values"].shape[1], len(cell_flow_corpus["texture_style_names"]))
            kind, detected_cell_flow_corpus = validate_artifact(cell_flow_corpus_dir / "cell_flow_corpus.json")
            self.assertEqual(kind, "cell-flow-corpus")
            self.assertEqual(detected_cell_flow_corpus["type"], CELL_FLOW_CORPUS_TYPE)
            cell_flow_checkpoint = Path(tmp) / "cell_flow.pt"
            cell_flow_training_path = Path(tmp) / "cell_flow_training.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "train-cell-flow",
                        str(cell_flow_corpus_dir / "cell_flow_corpus.json"),
                        "-o",
                        str(cell_flow_checkpoint),
                        "--report-output",
                        str(cell_flow_training_path),
                        "--steps",
                        "2",
                        "--batch-size",
                        "2",
                        "--device",
                        "cpu",
                        "--feature-loss-weight",
                        "0.25",
                        "--feature-nucleus-weight",
                        "2.0",
                        "--feature-boundary-weight",
                        "0.5",
                        "--highfreq-loss-weight",
                        "0.2",
                        "--highfreq-pericellular-weight",
                        "2.5",
                        "--texture-stat-loss-weight",
                        "0.1",
                        "--texture-stat-pericellular-weight",
                        "2.75",
                        "--appearance-loss-weight",
                        "0.05",
                        "--appearance-nucleus-weight",
                        "2.25",
                        "--appearance-boundary-weight",
                        "0.75",
                    ]
                )
            self.assertTrue(cell_flow_checkpoint.exists())
            cell_flow_training = validate_cell_flow_training(cell_flow_training_path)
            self.assertEqual(cell_flow_training["type"], CELL_FLOW_TRAINING_TYPE)
            self.assertEqual(cell_flow_training["target_mode"], "foreground_texture_style_v1")
            self.assertEqual(cell_flow_training["support_margin_px"], 1)
            self.assertAlmostEqual(cell_flow_training["feature_loss_weight"], 0.25)
            self.assertAlmostEqual(cell_flow_training["feature_nucleus_weight"], 2.0)
            self.assertAlmostEqual(cell_flow_training["feature_boundary_weight"], 0.5)
            self.assertAlmostEqual(cell_flow_training["highfreq_loss_weight"], 0.2)
            self.assertAlmostEqual(cell_flow_training["highfreq_pericellular_weight"], 2.5)
            self.assertAlmostEqual(cell_flow_training["texture_stat_loss_weight"], 0.1)
            self.assertAlmostEqual(cell_flow_training["texture_stat_pericellular_weight"], 2.75)
            self.assertAlmostEqual(cell_flow_training["appearance_loss_weight"], 0.05)
            self.assertAlmostEqual(cell_flow_training["appearance_nucleus_weight"], 2.25)
            self.assertAlmostEqual(cell_flow_training["appearance_boundary_weight"], 0.75)
            self.assertIn("foreground_feature_loss", cell_flow_training["losses"][-1])
            self.assertIn("foreground_highfreq_loss", cell_flow_training["losses"][-1])
            self.assertIn("foreground_texture_stat_loss", cell_flow_training["losses"][-1])
            self.assertIn("foreground_appearance_loss", cell_flow_training["losses"][-1])
            kind, detected_cell_flow_training = validate_artifact(cell_flow_training_path)
            self.assertEqual(kind, "cell-flow-training")
            self.assertEqual(detected_cell_flow_training["type"], CELL_FLOW_TRAINING_TYPE)
            flow_sample_dir = Path(tmp) / "flow_sampled_cells"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "sample-cell-flow",
                        str(cell_flow_checkpoint),
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(flow_sample_dir),
                        "--num-cells",
                        "4",
                        "--sample-steps",
                        "2",
                        "--device",
                        "cpu",
                        "--tile-size",
                        "64",
                        "--anchor-sampling-mode",
                        "geometry-stratified",
                        "--anchor-sampling-bins",
                        "3",
                        "--texture-jitter-strength",
                        "0.1",
                    ]
                )
            flow_sample_profile = validate_anchor_profile(flow_sample_dir / "flow_sampled_cell_profile.json")
            self.assertEqual(flow_sample_profile["profile_kind"], "flow_sampled_cell_profile_v0")
            self.assertAlmostEqual(flow_sample_profile["sampling"]["texture_jitter_strength"], 0.1)
            self.assertEqual(flow_sample_profile["sampling"]["support_margin_px"], 1)
            self.assertEqual(flow_sample_profile["sampling"]["anchor_sampling"]["mode"], "geometry-stratified")
            self.assertEqual(flow_sample_profile["sampling"]["anchor_sampling"]["bins"], 3)
            self.assertGreaterEqual(flow_sample_profile["num_anchors"], 1)
            self.assertTrue((flow_sample_dir / "flow_sampled_cells.npz").exists())
            self.assertTrue((flow_sample_dir / "flow_sampled_cell_sheet.png").read_bytes().startswith(b"\x89PNG"))
            component_flow_corpus_dir = Path(tmp) / "component_cell_flow_corpus"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "build-cell-corpus",
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(component_flow_corpus_dir),
                        "--patch-size",
                        "32",
                        "--max-anchors",
                        "4",
                        "--target-mode",
                        "component_texture_style_v1",
                    ]
                )
            component_flow_corpus = validate_cell_flow_corpus(component_flow_corpus_dir / "cell_flow_corpus.json")
            self.assertEqual(component_flow_corpus["target_mode"], "component_texture_style_v1")
            self.assertIn("pericellular_margin", component_flow_corpus["texture_component_names"])
            self.assertTrue(component_flow_corpus["texture_style_names"])
            with np.load(component_flow_corpus_dir / "cell_flow_corpus.npz", allow_pickle=True) as component_npz:
                self.assertEqual(component_npz["target_offsets"].ndim, 3)
                self.assertEqual(component_npz["target_offsets"].shape[2], len(component_flow_corpus["texture_component_names"]))
                self.assertEqual(
                    component_npz["texture_style_values"].shape[1],
                    len(component_flow_corpus["texture_style_names"]),
                )
            rim_flow_corpus_dir = Path(tmp) / "rim_cell_flow_corpus"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "build-cell-corpus",
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(rim_flow_corpus_dir),
                        "--patch-size",
                        "32",
                        "--max-anchors",
                        "4",
                        "--target-mode",
                        "component_membrane_rim_v1",
                    ]
                )
            rim_flow_corpus = validate_cell_flow_corpus(rim_flow_corpus_dir / "cell_flow_corpus.json")
            self.assertEqual(rim_flow_corpus["target_mode"], "component_membrane_rim_v1")
            self.assertEqual(
                rim_flow_corpus["texture_component_names"],
                ["nucleus", "membrane_rim", "cytoplasm_core", "pericellular_margin"],
            )
            self.assertAlmostEqual(rim_flow_corpus["membrane_cytoplasm_offset_scale"], 0.0)
            with np.load(rim_flow_corpus_dir / "cell_flow_corpus.npz", allow_pickle=True) as rim_npz:
                self.assertEqual(rim_npz["target_offsets"].ndim, 3)
                self.assertEqual(rim_npz["target_offsets"].shape[2], len(rim_flow_corpus["texture_component_names"]))
                self.assertTrue(np.allclose(rim_npz["target_offsets"][:, 1, 2], 0.0))
            component_flow_checkpoint = Path(tmp) / "component_cell_flow.pt"
            component_flow_training_path = Path(tmp) / "component_cell_flow_training.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "train-cell-flow",
                        str(component_flow_corpus_dir / "cell_flow_corpus.json"),
                        "-o",
                        str(component_flow_checkpoint),
                        "--report-output",
                        str(component_flow_training_path),
                        "--steps",
                        "1",
                        "--batch-size",
                        "2",
                        "--device",
                        "cpu",
                        "--architecture",
                        "highfreq_two_head_v0",
                        "--hidden-channels",
                        "16",
                        "--highfreq-head-scale",
                        "0.75",
                    ]
                )
            component_flow_training = validate_cell_flow_training(component_flow_training_path)
            self.assertEqual(component_flow_training["target_mode"], "component_texture_style_v1")
            self.assertEqual(component_flow_training["architecture"], "highfreq_two_head_v0")
            self.assertEqual(component_flow_training["hidden_channels"], 16)
            self.assertAlmostEqual(component_flow_training["highfreq_head_scale"], 0.75)
            self.assertIn("pericellular_margin", component_flow_training["model"]["texture_component_names"])
            component_flow_sample_dir = Path(tmp) / "component_flow_sampled_cells"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "sample-cell-flow",
                        str(component_flow_checkpoint),
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(component_flow_sample_dir),
                        "--num-cells",
                        "2",
                        "--sample-steps",
                        "2",
                        "--device",
                        "cpu",
                        "--tile-size",
                        "64",
                        "--region-texture-strength",
                        "0.2",
                        "--owned-texture-strength",
                        "0.3",
                        "--owned-texture-mode",
                        "pca",
                        "--owned-texture-k",
                        "2",
                        "--owned-texture-pca-components",
                        "2",
                        "--owned-texture-nucleus-weight",
                        "1.2",
                        "--owned-texture-membrane-cytoplasm-weight",
                        "0.15",
                        "--owned-intensity-strength",
                        "0.4",
                        "--owned-intensity-cytoplasm-weight",
                        "0.9",
                        "--owned-intensity-membrane-cytoplasm-weight",
                        "0.1",
                        "--membrane-cytoplasm-damping-strength",
                        "0.6",
                        "--membrane-cytoplasm-rim-width-px",
                        "2",
                        "--membrane-cytoplasm-damping-target",
                        "zero",
                        "--membrane-cytoplasm-state-strength",
                        "0.2",
                        "--membrane-cytoplasm-state-texture-strength",
                        "0.3",
                        "--membrane-cytoplasm-state-k",
                        "2",
                        "--membrane-cytoplasm-state-rim-width-px",
                        "2",
                        "--cell-emission-stats-strength",
                        "0.2",
                        "--cell-emission-stats-k",
                        "2",
                        "--cell-emission-stats-cytoplasm-weight",
                        "0.7",
                        "--pericellular-context-strength",
                        "0.25",
                        "--pericellular-context-texture-strength",
                        "0.15",
                        "--pericellular-context-k",
                        "2",
                        "--pericellular-context-margin-px",
                        "1",
                    ]
                )
            component_flow_sample = validate_anchor_profile(component_flow_sample_dir / "flow_sampled_cell_profile.json")
            self.assertEqual(component_flow_sample["sampling"]["target_mode"], "component_texture_style_v1")
            self.assertAlmostEqual(component_flow_sample["sampling"]["region_texture_strength"], 0.2)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_texture_strength"], 0.3)
            self.assertEqual(component_flow_sample["sampling"]["owned_texture_mode"], "pca")
            self.assertEqual(component_flow_sample["sampling"]["owned_texture_k"], 2)
            self.assertEqual(component_flow_sample["sampling"]["owned_texture_pca_components"], 2)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_texture_nucleus_weight"], 1.2)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_texture_membrane_cytoplasm_weight"], 0.15)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_intensity_strength"], 0.4)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_intensity_cytoplasm_weight"], 0.9)
            self.assertAlmostEqual(component_flow_sample["sampling"]["owned_intensity_membrane_cytoplasm_weight"], 0.1)
            self.assertAlmostEqual(component_flow_sample["sampling"]["membrane_cytoplasm_damping_strength"], 0.6)
            self.assertEqual(component_flow_sample["sampling"]["membrane_cytoplasm_rim_width_px"], 2)
            self.assertEqual(component_flow_sample["sampling"]["membrane_cytoplasm_damping_target"], "zero")
            self.assertEqual(component_flow_sample["sampling"]["membrane_cytoplasm_state_model"], "membrane_cytoplasm_state_v1")
            self.assertAlmostEqual(component_flow_sample["sampling"]["membrane_cytoplasm_state_strength"], 0.2)
            self.assertAlmostEqual(component_flow_sample["sampling"]["membrane_cytoplasm_state_texture_strength"], 0.3)
            self.assertEqual(component_flow_sample["sampling"]["membrane_cytoplasm_state_k"], 2)
            self.assertEqual(component_flow_sample["sampling"]["membrane_cytoplasm_state_rim_width_px"], 2)
            self.assertEqual(component_flow_sample["sampling"]["cell_emission_stats_model"], "cell_emission_stats_v1")
            self.assertAlmostEqual(component_flow_sample["sampling"]["cell_emission_stats_strength"], 0.2)
            self.assertEqual(component_flow_sample["sampling"]["cell_emission_stats_k"], 2)
            self.assertAlmostEqual(component_flow_sample["sampling"]["cell_emission_stats_cytoplasm_weight"], 0.7)
            self.assertEqual(component_flow_sample["sampling"]["pericellular_context_model"], "pericellular_context_v1")
            self.assertAlmostEqual(component_flow_sample["sampling"]["pericellular_context_strength"], 0.25)
            self.assertAlmostEqual(component_flow_sample["sampling"]["pericellular_context_texture_strength"], 0.15)
            self.assertEqual(component_flow_sample["sampling"]["pericellular_context_k"], 2)
            self.assertEqual(component_flow_sample["sampling"]["pericellular_context_margin_px"], 1)
            self.assertTrue((component_flow_sample_dir / "flow_sampled_cell_sheet.png").read_bytes().startswith(b"\x89PNG"))
            sweep_profile_path = Path(tmp) / "sweep_anchor_profile.json"
            sweep_profile = json.loads((anchor_dir / "anchor_profile.json").read_text())
            sweep_seed_anchor = sweep_profile["records"][0]["anchors"][0]
            sweep_anchors = []
            for label in range(2):
                anchor_copy = json.loads(json.dumps(sweep_seed_anchor))
                anchor_copy["cell_id"] = f"sweep_cell_{label}"
                anchor_copy["quality_label"] = "high"
                anchor_copy["quality_score"] = 0.9
                anchor_copy["area_um2"] = float(sweep_seed_anchor["area_um2"]) + label
                anchor_copy["intensity"]["nucleus"]["mean"] = [
                    float(value) + 0.01 * label for value in anchor_copy["intensity"]["nucleus"]["mean"]
                ]
                sweep_anchors.append(anchor_copy)
            sweep_profile["records"][0]["anchors"] = sweep_anchors
            sweep_profile["num_anchors"] = len(sweep_anchors)
            sweep_profile["num_high_quality_anchors"] = len(sweep_anchors)
            sweep_profile_path.write_text(json.dumps(sweep_profile))
            sweep_dir = Path(tmp) / "cell_flow_sweep"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "sweep-cell-flow",
                        str(component_flow_checkpoint),
                        str(sweep_profile_path),
                        "-o",
                        str(sweep_dir),
                        "--candidate",
                        "tiny:owned_texture_strength=0.1,owned_texture_mode=pca,owned_texture_pca_components=2,owned_texture_k=2",
                        "--quality-label",
                        "all",
                        "--num-cells",
                        "2",
                        "--sample-steps",
                        "2",
                        "--device",
                        "cpu",
                        "--tile-size",
                        "64",
                        "--max-channels",
                        "2",
                        "--holdout-fraction",
                        "0.5",
                        "--min-stratum-size",
                        "1",
                        "--feature-scope",
                        "foreground-owned",
                        "--num-visual-anchors",
                        "2",
                        "--benchmark-tile-size",
                        "64",
                    ]
                )
            cell_flow_sweep = validate_cell_flow_sweep(sweep_dir / "cell_flow_sweep.json")
            self.assertEqual(cell_flow_sweep["type"], CELL_FLOW_SWEEP_TYPE)
            self.assertEqual(cell_flow_sweep["num_candidates"], 1)
            self.assertEqual(cell_flow_sweep["best_candidate"]["name"], "tiny")
            self.assertEqual(cell_flow_sweep["ranking"][0]["rank"], 1)
            self.assertIn("aggregate_distance", cell_flow_sweep["best_candidate"]["metrics"])
            self.assertTrue(Path(cell_flow_sweep["best_candidate"]["sample_profile"]).exists())
            self.assertTrue(Path(cell_flow_sweep["best_candidate"]["comparison_sheet"]).read_bytes().startswith(b"\x89PNG"))
            kind, detected_cell_flow_sweep = validate_artifact(sweep_dir / "cell_flow_sweep.json")
            self.assertEqual(kind, "cell-flow-sweep")
            self.assertEqual(detected_cell_flow_sweep["type"], CELL_FLOW_SWEEP_TYPE)
            benchmark_profile_path = Path(tmp) / "benchmark_anchor_profile.json"
            benchmark_profile = json.loads((anchor_dir / "anchor_profile.json").read_text())
            seed_anchor = benchmark_profile["records"][0]["anchors"][0]
            benchmark_anchors = []
            for label in range(1, 5):
                anchor_copy = json.loads(json.dumps(seed_anchor))
                anchor_copy["label"] = label
                anchor_copy["cell_id"] = f"benchmark_cell_{label}"
                anchor_copy["area_um2"] = float(seed_anchor["area_um2"]) + label
                anchor_copy["nucleus_fraction"] = min(0.95, float(seed_anchor["nucleus_fraction"]) + 0.02 * label)
                anchor_copy["quality_label"] = "high"
                anchor_copy["quality_score"] = 0.9
                anchor_copy["intensity"]["nucleus"]["mean"] = [
                    float(value) + 0.01 * label for value in anchor_copy["intensity"]["nucleus"]["mean"]
                ]
                anchor_copy["intensity"]["boundary"]["mean"] = [
                    float(value) + 0.005 * label for value in anchor_copy["intensity"]["boundary"]["mean"]
                ]
                anchor_copy["patch_features"] = {
                    "pericellular_margin_mean": [0.10 + 0.01 * label, 0.20 + 0.01 * label],
                    "pericellular_margin_std": [0.02 + 0.001 * label, 0.03 + 0.001 * label],
                    "pericellular_margin_q10": [0.06 + 0.01 * label, 0.16 + 0.01 * label],
                    "pericellular_margin_q50": [0.10 + 0.01 * label, 0.20 + 0.01 * label],
                    "pericellular_margin_q90": [0.14 + 0.01 * label, 0.24 + 0.01 * label],
                    "pericellular_margin_gradient_mean": [0.03 + 0.001 * label, 0.04 + 0.001 * label],
                }
                benchmark_anchors.append(anchor_copy)
            benchmark_profile["records"][0]["anchors"] = benchmark_anchors
            benchmark_profile["num_anchors"] = len(benchmark_anchors)
            benchmark_profile["num_high_quality_anchors"] = len(benchmark_anchors)
            benchmark_profile_path.write_text(json.dumps(benchmark_profile))
            cell_benchmark_path = Path(tmp) / "cell_appearance_benchmark.json"
            cell_benchmark_sheet = Path(tmp) / "cell_appearance_benchmark.png"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "cell-appearance-benchmark",
                        str(benchmark_profile_path),
                        "-o",
                        str(cell_benchmark_path),
                        "--sheet-output",
                        str(cell_benchmark_sheet),
                        "--holdout-fraction",
                        "0.5",
                        "--min-stratum-size",
                        "1",
                        "--feature-scope",
                        "foreground-strict",
                    ]
                )
            cell_benchmark = validate_cell_appearance_benchmark(cell_benchmark_path)
            self.assertEqual(cell_benchmark["type"], CELL_APPEARANCE_BENCHMARK_TYPE)
            self.assertEqual(cell_benchmark["evaluation_frame"], "population_anchor_feature_distribution_v0")
            self.assertEqual(cell_benchmark["feature_scope"], "foreground-strict")
            self.assertGreaterEqual(cell_benchmark["holdout"]["num_reference_anchors"], 1)
            self.assertGreaterEqual(cell_benchmark["holdout"]["num_heldout_anchors"], 1)
            self.assertIn("patch_features", cell_benchmark)
            self.assertIn("visual_sheet", cell_benchmark)
            self.assertTrue(cell_benchmark_sheet.read_bytes().startswith(b"\x89PNG"))
            self.assertIn("heldout_vs_reference", cell_benchmark)
            self.assertIn("distance_summary", cell_benchmark["heldout_vs_reference"])
            self.assertIn("feature_group_summary", cell_benchmark["heldout_vs_reference"])
            self.assertNotIn("background_context", cell_benchmark["heldout_vs_reference"]["feature_group_summary"])
            self.assertNotIn("geometry_quality", cell_benchmark["heldout_vs_reference"]["feature_group_summary"])
            self.assertTrue(any("pericellular_margin" in name for name in cell_benchmark["patch_features"]["feature_names"]))
            self.assertTrue(all("local_background" not in name for name in cell_benchmark["feature_names"]))
            self.assertTrue(all("outer_ring" not in name for name in cell_benchmark["feature_names"]))
            self.assertIn("nearest_neighbor_two_sample_accuracy", cell_benchmark["heldout_vs_reference"])
            kind, detected_cell_benchmark = validate_artifact(cell_benchmark_path)
            self.assertEqual(kind, "cell-appearance-benchmark")
            self.assertEqual(detected_cell_benchmark["type"], CELL_APPEARANCE_BENCHMARK_TYPE)
            cell_owned_benchmark_path = Path(tmp) / "cell_appearance_owned_benchmark.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "cell-appearance-benchmark",
                        str(benchmark_profile_path),
                        "-o",
                        str(cell_owned_benchmark_path),
                        "--holdout-fraction",
                        "0.5",
                        "--min-stratum-size",
                        "1",
                        "--feature-scope",
                        "foreground-owned",
                    ]
                )
            cell_owned_benchmark = validate_cell_appearance_benchmark(cell_owned_benchmark_path)
            self.assertEqual(cell_owned_benchmark["feature_scope"], "foreground-owned")
            self.assertNotIn("pericellular_margin_emission", cell_owned_benchmark["heldout_vs_reference"]["feature_group_summary"])
            self.assertTrue(all("pericellular_margin" not in name for name in cell_owned_benchmark["feature_names"]))
            pericellular_benchmark_path = Path(tmp) / "cell_appearance_pericellular_benchmark.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "cell-appearance-benchmark",
                        str(benchmark_profile_path),
                        "-o",
                        str(pericellular_benchmark_path),
                        "--holdout-fraction",
                        "0.5",
                        "--min-stratum-size",
                        "1",
                        "--feature-scope",
                        "pericellular",
                        "--no-patch-features",
                    ]
                )
            pericellular_benchmark = validate_cell_appearance_benchmark(pericellular_benchmark_path)
            self.assertEqual(pericellular_benchmark["feature_scope"], "pericellular")
            self.assertEqual(list(pericellular_benchmark["heldout_vs_reference"]["feature_group_summary"]), ["pericellular_margin_emission"])
            self.assertTrue(all("pericellular_margin" in name for name in pericellular_benchmark["feature_names"]))
            synthetic_benchmark_path = Path(tmp) / "cell_appearance_synthetic_benchmark.json"
            synthetic_benchmark_sheet = Path(tmp) / "cell_appearance_synthetic_benchmark.png"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "cell-appearance-benchmark",
                        str(benchmark_profile_path),
                        "-o",
                        str(synthetic_benchmark_path),
                        "--synthetic-anchor-profile",
                        str(sampled_cells_dir / "sampled_anchor_cell_profile.json"),
                        "--synthetic-sheet-output",
                        str(synthetic_benchmark_sheet),
                        "--holdout-fraction",
                        "0.5",
                        "--min-stratum-size",
                        "1",
                    ]
                )
            synthetic_benchmark = validate_cell_appearance_benchmark(synthetic_benchmark_path)
            self.assertIn("synthetic_vs_heldout", synthetic_benchmark)
            self.assertIn("feature_group_summary", synthetic_benchmark["synthetic_vs_heldout"])
            self.assertIn("synthetic_acceptance", synthetic_benchmark)
            self.assertIn("feature_groups", synthetic_benchmark["synthetic_acceptance"])
            self.assertIn("passes", synthetic_benchmark["synthetic_acceptance"])
            self.assertTrue(synthetic_benchmark_sheet.read_bytes().startswith(b"\x89PNG"))
            scene_dir = Path(tmp) / "scene_fit"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "fit-scene",
                        str(manifest),
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(scene_dir),
                    ]
                )
            scene_fit = validate_scene_fit(scene_dir / "scene_fit.json")
            self.assertEqual(scene_fit["type"], SCENE_FIT_TYPE)
            self.assertEqual(scene_fit["mode"], "observed_anchor_emission_affine_v0")
            self.assertEqual(len(scene_fit["image_channels"]), 2)
            self.assertEqual(scene_fit["channel_roles"], ["dapi_nuclear", "membrane"])
            self.assertEqual(scene_fit["source_channel_indices"], [0, 1])
            self.assertIn("anchor_deformation", scene_fit)
            self.assertFalse(scene_fit["anchor_deformation"]["enabled"])
            self.assertIn("calibrated_render", scene_fit["metrics"])
            self.assertTrue((scene_dir / "scene_fit_arrays.npz").exists())
            scene_arrays = np.load(scene_dir / "scene_fit_arrays.npz")
            self.assertIn("anchor_deformation_delta_images", scene_arrays.files)
            self.assertIn("deformed_anchor_baseline_images", scene_arrays.files)
            self.assertTrue((scene_dir / "scene_fit_compare.png").read_bytes().startswith(b"\x89PNG"))
            kind, detected_scene = validate_artifact(scene_dir / "scene_fit.json")
            self.assertEqual(kind, "scene-fit")
            self.assertEqual(detected_scene["type"], SCENE_FIT_TYPE)
            latent_scene_dir = Path(tmp) / "scene_fit_latent"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "fit-scene",
                        str(manifest),
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(latent_scene_dir),
                        "--latent-cells",
                        "2",
                        "--latent-candidate-mode",
                        "pericellular",
                    ]
                )
            latent_scene_fit = validate_scene_fit(latent_scene_dir / "scene_fit.json")
            self.assertEqual(latent_scene_fit["mode"], "observed_anchor_latent_cell_affine_v0")
            self.assertEqual(latent_scene_fit["latent_cell_budget"], 2)
            self.assertEqual(latent_scene_fit["latent_initialization"]["candidate_mode"], "pericellular")
            self.assertIn("pericellular_shell_radius_px", latent_scene_fit["latent_initialization"])
            self.assertEqual(
                latent_scene_fit["latent_initialization"]["emission_fit_mode"],
                "visibility_channel_scale_grid_v0",
            )
            self.assertEqual(latent_scene_fit["latent_initialization"]["edge_policy"], "penalize")
            self.assertEqual(latent_scene_fit["latent_initialization"]["min_improved_channels"], 1)
            self.assertLess(latent_scene_fit["latent_initialization"]["min_channel_mae_improvement"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["min_local_support_channels"], 0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["min_visibility"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["min_observed_distance_um"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["observation_modes"], ["whole"])
            self.assertEqual(latent_scene_fit["latent_initialization"]["observed_distance_penalty_um"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["observed_distance_penalty_mae"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["contact_penalty_um"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["contact_penalty_mae"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["interior_min_visibility"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["interior_visibility_penalty_mae"], 0.0)
            self.assertEqual(latent_scene_fit["latent_initialization"]["continuous_refinement_steps"], 4)
            self.assertEqual(latent_scene_fit["latent_initialization"]["emission_variation_strength"], 0.0)
            self.assertEqual(
                latent_scene_fit["latent_initialization"]["emission_texture_model"],
                "compartment_constant_v0",
            )
            self.assertIn("joint_refinement", latent_scene_fit["latent_initialization"])
            self.assertIn("accepted_coordinate_moves", latent_scene_fit["latent_initialization"]["joint_refinement"])
            self.assertIn("latent_improvement", latent_scene_fit["metrics"])
            self.assertIn("latent_candidates", latent_scene_fit)
            if latent_scene_fit["latent_candidates"]:
                self.assertIn("emission_fit", latent_scene_fit["latent_candidates"][0])
                self.assertIn("latent_observation_mode", latent_scene_fit["latent_candidates"][0])
                self.assertIn("local_support", latent_scene_fit["latent_candidates"][0])
                self.assertIn("channel_mae_improvement", latent_scene_fit["latent_candidates"][0])
                self.assertIn("acceptance", latent_scene_fit["latent_candidates"][0])
                self.assertIn("worst_channel_mae_improvement", latent_scene_fit["latent_candidates"][0]["acceptance"])
                self.assertIn("local_support_passed", latent_scene_fit["latent_candidates"][0]["acceptance"])
                self.assertIn("visibility_passed", latent_scene_fit["latent_candidates"][0]["acceptance"])
                self.assertIn("observed_spacing_passed", latent_scene_fit["latent_candidates"][0]["acceptance"])
                self.assertIn(
                    "observed_distance_penalty_mae_applied",
                    latent_scene_fit["latent_candidates"][0]["acceptance"],
                )
                self.assertIn("contact_penalty_mae_applied", latent_scene_fit["latent_candidates"][0]["acceptance"])
                self.assertIn(
                    "interior_visibility_penalty_mae_applied",
                    latent_scene_fit["latent_candidates"][0]["acceptance"],
                )
                self.assertIn("geometry_context", latent_scene_fit["latent_candidates"][0])
                self.assertIn("observed_contact_fraction", latent_scene_fit["latent_candidates"][0]["geometry_context"])
                self.assertIn("plausibility", latent_scene_fit["latent_candidates"][0])
                self.assertIn("cell_area_um2", latent_scene_fit["latent_candidates"][0]["plausibility"])
            self.assertIn("latent_cell_mask_fraction", latent_scene_fit["component_summary"])
            self.assertIn("latent_cell_signal", latent_scene_fit["compare_columns"])
            latent_arrays = np.load(latent_scene_dir / "scene_fit_arrays.npz")
            self.assertIn("latent_cell_delta_images", latent_arrays.files)
            self.assertIn("latent_cell_label", latent_arrays.files)
            scene_summary_path = Path(tmp) / "scene_probe_summary.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "summarize-scenes",
                        str(scene_dir / "scene_fit.json"),
                        str(latent_scene_dir / "scene_fit.json"),
                        "-o",
                        str(scene_summary_path),
                    ]
                )
            scene_summary = validate_scene_probe_summary(scene_summary_path)
            self.assertEqual(scene_summary["type"], SCENE_PROBE_SUMMARY_TYPE)
            self.assertEqual(scene_summary["num_scene_fits"], 2)
            self.assertEqual(scene_summary["aggregate"]["num_scene_fits"], 2)
            self.assertIn("total_candidates", scene_summary["aggregate"])
            self.assertIn("mae_reduction", scene_summary["aggregate"])
            self.assertEqual(len(scene_summary["records"]), 2)
            kind, detected_summary = validate_artifact(scene_summary_path)
            self.assertEqual(kind, "scene-summary")
            self.assertEqual(detected_summary["type"], SCENE_PROBE_SUMMARY_TYPE)
            scene_batch_dir = Path(tmp) / "scene_batch"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "fit-scene-batch",
                        str(manifest),
                        str(anchor_dir / "anchor_profile.json"),
                        "-o",
                        str(scene_batch_dir),
                        "--crop-id",
                        obj["crops"][0]["crop_id"],
                        "--anchor-deform-shell-um",
                        "1.0",
                        "--anchor-deform-mode",
                        "per-anchor-shell",
                        "--anchor-deform-support-z",
                        "0.0",
                        "--anchor-deform-min-support-fraction",
                        "0.0",
                        "--anchor-deform-max-anchors",
                        "4",
                        "--latent-cells",
                        "1",
                        "--latent-candidate-mode",
                        "residual+pericellular",
                        "--latent-observation-modes",
                        "auto",
                        "--latent-partial-mode-penalty-mae",
                        "0.00005",
                        "--latent-scene-prior-penalty-mae",
                        "0.00004",
                        "--latent-scene-prior-min-center-distance-um",
                        "3.0",
                        "--latent-observed-distance-penalty-um",
                        "2.5",
                        "--latent-observed-distance-penalty-mae",
                        "0.00001",
                        "--latent-contact-penalty-um",
                        "1.5",
                        "--latent-contact-penalty-mae",
                        "0.00002",
                        "--latent-interior-min-visibility",
                        "0.5",
                        "--latent-interior-visibility-penalty-mae",
                        "0.00003",
                        "--latent-joint-refine-steps",
                        "1",
                        "--latent-joint-shift-px",
                        "0.5",
                        "--latent-emission-variation-strength",
                        "0.25",
                    ]
                )
            scene_batch_summary = validate_scene_probe_summary(scene_batch_dir / "scene_probe_summary.json")
            self.assertEqual(scene_batch_summary["num_scene_fits"], 1)
            self.assertEqual(scene_batch_summary["records"][0]["crop_id"], obj["crops"][0]["crop_id"])
            self.assertIn("total_accepted_candidates", scene_batch_summary["aggregate"])
            self.assertIn("anchor_deformation_enabled", scene_batch_summary["aggregate"])
            self.assertTrue(scene_batch_summary["records"][0]["anchor_deformation"]["enabled"])
            self.assertEqual(scene_batch_summary["records"][0]["anchor_deformation"]["mode"], "per-anchor-shell")
            self.assertTrue(scene_batch_summary["records"][0]["anchor_deformation"]["support_enabled"])
            self.assertEqual(scene_batch_summary["records"][0]["anchor_deformation"]["support_z"], 0.0)
            self.assertIn("support_fraction", scene_batch_summary["records"][0]["anchor_deformation"])
            self.assertIn("num_anchor_candidates", scene_batch_summary["records"][0]["anchor_deformation"])
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["candidate_mode"],
                "residual+pericellular",
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["observed_distance_penalty_um"],
                2.5,
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["observation_modes"],
                ["whole", "membrane_only", "nucleus_only"],
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["partial_mode_penalty_mae"],
                0.00005,
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["scene_prior_penalty_mae"],
                0.00004,
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["scene_prior_min_center_distance_um"],
                3.0,
            )
            self.assertEqual(scene_batch_summary["records"][0]["latent_initialization"]["contact_penalty_um"], 1.5)
            self.assertEqual(scene_batch_summary["records"][0]["latent_initialization"]["interior_min_visibility"], 0.5)
            self.assertIn("joint_refinement", scene_batch_summary["records"][0]["latent_initialization"])
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["joint_refinement"]["shift_step_px"],
                0.5,
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["emission_variation_strength"],
                0.25,
            )
            self.assertEqual(
                scene_batch_summary["records"][0]["latent_initialization"]["emission_texture_model"],
                "anchor_texture_v1",
            )
            scene_review_path = Path(tmp) / "scene_review.png"
            scene_review_report_path = Path(tmp) / "scene_review.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "scene-review-sheet",
                        str(scene_batch_dir / "scene_probe_summary.json"),
                        "-o",
                        str(scene_review_path),
                        "--report-output",
                        str(scene_review_report_path),
                    ]
                )
            self.assertTrue(scene_review_path.read_bytes().startswith(b"\x89PNG"))
            scene_review = validate_scene_review(scene_review_report_path)
            self.assertEqual(scene_review["type"], SCENE_REVIEW_TYPE)
            self.assertGreaterEqual(scene_review["num_rows"], 1)
            self.assertIn("candidate_overlay", scene_review["review_columns"])
            self.assertIn("label", scene_review["rows"][0]["review_columns"])
            for row in scene_review["rows"]:
                for column in row["review_columns"]:
                    self.assertIn(column, scene_review["review_columns"])
            candidate_review_path = Path(tmp) / "scene_candidate_review.png"
            candidate_review_report_path = Path(tmp) / "scene_candidate_review.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "scene-candidate-sheet",
                        str(scene_batch_dir / obj["crops"][0]["crop_id"] / "scene_fit.json"),
                        "-o",
                        str(candidate_review_path),
                        "--report-output",
                        str(candidate_review_report_path),
                        "--max-candidates",
                        "4",
                    ]
                )
            self.assertTrue(candidate_review_path.read_bytes().startswith(b"\x89PNG"))
            candidate_review = validate_scene_review(candidate_review_report_path)
            self.assertEqual(candidate_review["type"], SCENE_REVIEW_TYPE)
            self.assertEqual(candidate_review["review_mode"], "candidate")
            self.assertIn("scene_fit", candidate_review)
            self.assertIn("candidate_overlay", candidate_review["review_columns"])
            scene_cell_profile_dir = Path(tmp) / "scene_cell_profile"
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(
                    [
                        "profile-scene-cells",
                        str(scene_batch_dir / "scene_probe_summary.json"),
                        "-o",
                        str(scene_cell_profile_dir),
                    ]
                )
            scene_cell_profile = validate_anchor_profile(scene_cell_profile_dir / "scene_cell_profile.json")
            self.assertEqual(scene_cell_profile["type"], ANCHOR_PROFILE_TYPE)
            self.assertEqual(scene_cell_profile["profile_kind"], "scene_latent_cell_profile_v0")
            self.assertIn("source_scene_fits", scene_cell_profile)
            self.assertTrue((scene_cell_profile_dir / "scene_cell_sheet.png").read_bytes().startswith(b"\x89PNG"))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["validate", str(manifest), "--json"])
            validate_summary = json.loads(stdout.getvalue())
            self.assertTrue(validate_summary["valid"])
            self.assertEqual(validate_summary["kind"], "crop")

    def test_visual_check_without_full_canonicalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp) / "bundle")
            out = Path(tmp) / "visual"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "visual-check",
                        str(root),
                        "-o",
                        str(out),
                        "--num-crops",
                        "1",
                        "--crop-size-um",
                        "20",
                        "--no-synthetic",
                    ]
                )
            report = Path(stdout.getvalue().strip())
            obj = json.loads(report.read_text())
            self.assertEqual(obj["type"], "xesim.visual_check.v0")
            self.assertEqual(obj["num_crops"], 1)
            self.assertIn("intensity_lut", obj)
            self.assertIn("display_lut", obj)
            self.assertEqual(obj["comparison_summary"]["num_pairs"], 0)
            self.assertTrue((out / "real_sheet.png").read_bytes().startswith(b"\x89PNG"))

    def test_polygon_fallback_when_zarr_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp) / "bundle")
            out = Path(tmp) / "canonical"
            manifest = canonicalize_bundle(
                root,
                out,
                num_crops=1,
                crop_size_um=20.0,
                include_images=False,
                geometry_source="auto",
            )
            obj = json.loads(manifest.read_text())
            self.assertEqual(obj["crops"][0]["qc"]["geometry_source"], "polygons:boundaries")

    def test_parametric_render_is_seeded(self) -> None:
        cell = np.zeros((16, 16), dtype=np.int32)
        nucleus = np.zeros((16, 16), dtype=np.int32)
        cell[3:13, 3:13] = 1
        nucleus[6:10, 6:10] = 1
        model = type("M", (), {})()
        model.channel_stats = [
            {
                "background_mean": 0.01,
                "cell_mean": 0.2,
                "nucleus_mean": 0.8,
                "boundary_mean": 0.1,
                "noise_std": 0.01,
            }
        ]
        a, state = render_parametric(model, cell, nucleus, seed=1, device_name="cpu")
        b, _ = render_parametric(model, cell, nucleus, seed=1, device_name="cpu")
        self.assertTrue(np.allclose(a, b))
        self.assertIn("attenuation", state)
        self.assertIn("background_field", state)
        self.assertIn("background_style", state)
        self.assertIn("residual_style", state)
        self.assertIn("render_calibration", state)
        self.assertIn("texture", state)
        self.assertIn("speckle_fraction", state["texture"])
        self.assertIn("render_coefficients", state["channels"][0])
        self.assertEqual(state["channels"][0]["role_fit"]["method"], "region_lstsq")

    def test_device_fallback(self) -> None:
        self.assertEqual(get_device("cpu").type, "cpu")

    def test_xenium_zarr_id_decoder(self) -> None:
        self.assertEqual(decode_xenium_base16_letters(15764), "aaaadnje")
        pairs = np.asarray([[15764, 1]], dtype=np.uint32)
        self.assertEqual(decode_cell_id_from_zarr(1, pairs), "aaaadnje-1")

    def test_spatial_splits_are_deterministic(self) -> None:
        crops = [CropBox(f"c{i}", i * 10, i * 10 + 5, 0, 5) for i in range(10)]
        splits = assign_spatial_splits(crops)
        self.assertEqual(splits["c9"], "test")
        self.assertEqual(splits["c8"], "test")
        self.assertEqual(splits["c7"], "val")
        self.assertEqual(splits["c0"], "train")
        y_splits = assign_spatial_splits(crops, method="spatial_y_quantile")
        self.assertEqual(y_splits, splits)
        checker = assign_spatial_splits(
            [
                CropBox("ll", 0, 1, 0, 1),
                CropBox("lr", 10, 11, 0, 1),
                CropBox("ul", 0, 1, 10, 11),
                CropBox("ur", 10, 11, 10, 11),
            ],
            method="spatial_checkerboard",
        )
        self.assertEqual(checker["ur"], "test")
        self.assertEqual(checker["ul"], "val")
        region_splits = assign_spatial_splits(
            [CropBox("a", 0, 2, 0, 2), CropBox("b", 10, 12, 10, 12)],
            val_regions=[CropBox("val", -1, 3, -1, 3)],
            test_regions=[CropBox("test", 9, 13, 9, 13)],
        )
        self.assertEqual(region_splits["a"], "val")
        self.assertEqual(region_splits["b"], "test")
        polygon_splits = assign_spatial_splits(
            [CropBox("a", 0, 2, 0, 2), CropBox("b", 10, 12, 10, 12)],
            val_polygons=[{"points_um": [[-1, -1], [3, -1], [3, 3], [-1, 3]]}],
            test_polygons=[{"points_um": [[9, 9], [13, 9], [13, 13], [9, 13]]}],
        )
        self.assertEqual(polygon_splits["a"], "val")
        self.assertEqual(polygon_splits["b"], "test")

    def test_cli_inspect_bundle_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["inspect-bundle", str(root), "--json"])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["run_name"], "mini")
            self.assertEqual(summary["num_cells_readable"], 2)

    def test_cli_manual_crop_box_overrides_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp) / "bundle")
            out = Path(tmp) / "canonical"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "canonicalize",
                        str(root),
                        "-o",
                        str(out),
                        "--no-images",
                        "--crop-selection",
                        "random",
                        "--crop-box",
                        "0,20,0,20",
                        "--test-box",
                        "0,20,0,20",
                        "--test-polygon",
                        "0,0;20,0;20,20;0,20",
                        "--split-method",
                        "spatial_y_quantile",
                    ]
                )
            manifest = Path(stdout.getvalue().strip())
            obj = json.loads(manifest.read_text())
            self.assertEqual(obj["crop_selection"]["mode"], "manual")
            self.assertEqual(obj["num_crops"], 1)
            self.assertEqual(obj["crops"][0]["crop_box"]["xmax"], 20.0)
            self.assertEqual(obj["crops"][0]["split"], "test")
            self.assertEqual(len(obj["split_regions"]["test"]["boxes"]), 1)
            self.assertEqual(len(obj["split_regions"]["test"]["polygons"]), 1)
            validate_crop_manifest(manifest, check_files=True)

    def test_transcript_scan_strategy_switches_after_filter_limit(self) -> None:
        small = [CropBox(f"c{i}", i * 10, i * 10 + 5, 0, 5) for i in range(4)]
        large = [CropBox(f"c{i}", i * 10, i * 10 + 5, 0, 5) for i in range(65)]
        self.assertEqual(transcript_scan_strategy(small), "crop_boxes")
        self.assertEqual(transcript_scan_strategy(large), "union_bounds")

    def test_crop_selection_modes_are_deterministic(self) -> None:
        cells = [CellSummary(f"c{i}", float(i * 10), float((i % 3) * 10)) for i in range(12)]
        spread = choose_crop_boxes(cells, 4, 4.0, selection="spread")
        random_a = choose_crop_boxes(cells, 4, 4.0, selection="random", seed=3)
        random_b = choose_crop_boxes(cells, 4, 4.0, selection="random", seed=3)
        grid = choose_crop_boxes(cells, 4, 4.0, selection="grid")
        density = choose_crop_boxes(cells, 4, 4.0, selection="density")
        self.assertEqual([c.crop_id for c in spread], ["crop_00001", "crop_00002", "crop_00003", "crop_00004"])
        self.assertEqual([c.center_x for c in random_a], [c.center_x for c in random_b])
        self.assertEqual(len(grid), 4)
        self.assertEqual(len(density), 4)

    def test_distance_transform_backend_reports_available_path(self) -> None:
        self.assertIn(distance_transform_backend(), {"morphology", "scipy"})

    def test_parquet_transcript_counting_when_pyarrow_available(self) -> None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = make_mini_bundle(Path(tmp))
            table = pa.table(
                {
                    "transcript_id": ["p1", "p2", "p3", "p4"],
                    "cell_id": ["cell_1", "cell_2", "UNASSIGNED", "cell_2"],
                    "feature_name": ["G1", "G1", "G2", "G3"],
                    "x_location": [10.0, 50.0, 30.0, 50.0],
                    "y_location": [10.0, 50.0, 30.0, 50.0],
                    "z_location": [0.0, 0.0, 0.0, 0.0],
                    "qv": [30.0, 30.0, 30.0, 5.0],
                }
            )
            pq.write_table(table, root / "transcripts.parquet")
            bundle = resolve_bundle(root)
            crops = [
                CropBox("left", 0.0, 20.0, 0.0, 20.0),
                CropBox("right", 45.0, 55.0, 45.0, 55.0),
            ]
            counts = transcript_counts_by_crop(bundle, crops, min_qv=20)
            self.assertEqual(counts["left"], (1, 1))
            self.assertEqual(counts["right"], (1, 1))


def make_mini_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.xenium").write_text(
        json.dumps(
            {
                "run_name": "mini",
                "pixel_size": 0.5,
                "z_step_size": 3.0,
                "images": {"morphology_focus_filepath": "morphology_focus/morphology_focus_0000.ome.tif"},
            }
        )
    )
    write_gzip(
        root / "cells.csv.gz",
        "cell_id,x_centroid,y_centroid,z_centroid,cell_area,nucleus_area,transcript_counts,cell_type\n"
        "cell_1,10,10,0,100,25,2,type_a\n"
        "cell_2,50,50,0,100,25,1,type_b\n",
    )
    write_gzip(
        root / "transcripts.csv.gz",
        "transcript_id,cell_id,feature_name,x_location,y_location,z_location,qv\n"
        "t1,cell_1,G1,10,10,0,30\n"
        "t2,UNASSIGNED,G2,11,11,0,30\n"
        "t3,cell_2,G1,50,50,0,10\n",
    )
    write_polygon_csv(root / "cell_boundaries.csv.gz", "cell_id", 10, 10, 6, "cell_1")
    append_polygon_csv(root / "cell_boundaries.csv.gz", "cell_id", 50, 50, 6, "cell_2")
    write_polygon_csv(root / "nucleus_boundaries.csv.gz", "cell_id", 10, 10, 3, "cell_1")
    append_polygon_csv(root / "nucleus_boundaries.csv.gz", "cell_id", 50, 50, 3, "cell_2")
    return root


def write_cells_zarr(path: Path) -> None:
    store = ZipStore(str(path), mode="w")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name: .*")
            root = zarr.group(store=store, overwrite=True, zarr_format=2)
            root.create_array("cell_id", data=np.asarray([[1, 1]], dtype=np.uint32), chunks=(1, 1))
            masks = root.create_group("masks")
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0] = 2.0
            transform[1, 1] = 2.0
            masks.create_array("homogeneous_transform", data=transform, chunks=(4, 4))
            nucleus = np.zeros((140, 140), dtype=np.uint32)
            cell = np.zeros((140, 140), dtype=np.uint32)
            # The first deterministic crop is centered at 10um,10um with a 20um width.
            # With 0.5um pixels, that crop spans array rows/cols 0:40.
            nucleus[15:25, 15:25] = 1
            cell[8:32, 8:32] = 1
            masks.create_array("0", data=nucleus, chunks=(40, 40))
            masks.create_array("1", data=cell, chunks=(40, 40))
    finally:
        store.close()


def write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(text)


def polygon_rows(id_col: str, cx: float, cy: float, r: float, oid: str) -> str:
    pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r), (cx - r, cy - r)]
    return "".join(f"{oid},{x},{y}\n" for x, y in pts)


def write_polygon_csv(path: Path, id_col: str, cx: float, cy: float, r: float, oid: str) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(f"{id_col},vertex_x,vertex_y\n")
        handle.write(polygon_rows(id_col, cx, cy, r, oid))


def append_polygon_csv(path: Path, id_col: str, cx: float, cy: float, r: float, oid: str) -> None:
    existing = gzip.open(path, "rt").read()
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(existing)
        handle.write(polygon_rows(id_col, cx, cy, r, oid))


if __name__ == "__main__":
    unittest.main()
