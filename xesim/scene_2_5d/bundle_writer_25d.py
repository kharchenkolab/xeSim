"""Bundle writer for 2.5D scenes.

Layout matches the 2D bundle but with:
  morphology.ome.tif     — (n_z, H, W) DAPI z-stack
  morphology_focus/      — per-channel 2D from the focal-plane render
  transcripts.parquet    — has an extra z_location column
  ground_truth/cells_3d.parquet — per-cell z attrs + tilt vector
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _polygon_area_um2(xs: np.ndarray, ys: np.ndarray) -> float:
    """Shoelace area for a closed polygon, returned in µm² (assuming
    xs/ys already in µm)."""
    if len(xs) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) -
                              np.dot(ys, np.roll(xs, -1))))


def write_bundle_25d(
    *,
    output_dir: str | Path,
    compose_result,                 # Compose25DResult
    model,                          # XesimModel for channel_names + pixel_size
    real_bundle_path: str | Path | None = None,
    config: dict | None = None,
    overwrite: bool = False,
    target_intensity_stats: dict | None = None,
    intensity_mode: str = "scale",
) -> dict[str, Any]:
    """Write a Xenium-compatible bundle with 2.5D additions.

    `compose_result` is the output of `compose_region_scene_25d`.
    `real_bundle_path` is the source bundle for per-cell polygon lookup
    (anchor cells reuse their real cell_boundaries from the bundle).
    """
    import shutil
    from ..scene_2d.bundle_writer import (
        _write_morphology_focus, _write_pyramid_ome_tiff,
        _load_real_polys_lookup, _polygons_to_long_df,
    )
    from ..scene_2d.intensity import calibrate_to_uint16

    out_dir = Path(output_dir)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    written: dict[str, Any] = {"output_dir": str(out_dir)}
    psz = float(compose_result.pixel_size_um)
    channel_names = list(model.manifest.get("channel_names") or [])
    n_ch = len(channel_names)

    # 1. Transcripts — with z_location. feature_name + qv come from the
    # priors-aware emitter when present (real panel genes, qv=40); the
    # legacy stub path (no transcripts_priors on the model) populates
    # neither column so we fall back to "synth" + qv=40 for parity with
    # the 2D writer (synthetic_qv=40 default).
    mol = compose_result.molecules.copy()
    rng_ids = np.random.default_rng(0)
    mol["transcript_id"] = [f"synth_{i:09d}" for i in range(len(mol))]
    feature_name = (mol["gene"].astype(str).to_numpy()
                      if "gene" in mol.columns
                      else np.full(len(mol), "synth", dtype=object))
    qv_vals = (mol["qv"].astype(np.float32).to_numpy()
                 if "qv" in mol.columns
                 else np.full(len(mol), 40.0, dtype=np.float32))
    transcripts_df = pd.DataFrame({
        "transcript_id": mol["transcript_id"],
        "x_location": mol["x_true"].astype(np.float32),
        "y_location": mol["y_true"].astype(np.float32),
        "z_location": mol["z_true"].astype(np.float32),
        "qv": qv_vals,
        "feature_name": feature_name,
        "cell_id": mol["true_cell_id"].astype(str),
        "overlaps_nucleus": [True] * len(mol),
        "nucleus_distance": np.zeros(len(mol), dtype=np.float32),
    })
    transcripts_df.to_parquet(out_dir / "transcripts.parquet", index=False)
    written["transcripts"] = {
        "n_rows": int(len(transcripts_df)),
        "parquet": str(out_dir / "transcripts.parquet"),
    }

    # 2. Cell boundaries (anchors only) — pull from real bundle if available
    cells_3d = compose_result.cells_3d
    observed_ids = set(cells_3d[~cells_3d["is_unobserved"]]["cell_id"].tolist())
    real_lookup = {}
    if real_bundle_path is not None:
        try:
            real_lookup = _load_real_polys_lookup(real_bundle_path, kind="cell",
                                                      cell_ids=observed_ids)
        except Exception as e:
            print(f"[write_bundle_25d] real polygon lookup failed: {e}")
    # Vectorized assembly: per-cell arrays then concat (matches 2D pattern)
    cell_areas_um2: dict[str, float] = {}
    xs_arrs, ys_arrs, cid_arrs, lbl_arrs = [], [], [], []
    cid_to_label: dict[str, int] = {}
    next_label = 1
    for cid in observed_ids:
        if cid not in real_lookup:
            continue
        xs, ys = real_lookup[cid]
        if len(xs) < 3:
            continue
        cell_areas_um2[cid] = _polygon_area_um2(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64))
        cid_to_label[cid] = next_label
        next_label += 1
        lbl = cid_to_label[cid]
        xs_arrs.append(np.asarray(xs, dtype=np.float32))
        ys_arrs.append(np.asarray(ys, dtype=np.float32))
        cid_arrs.append(np.full(len(xs), cid, dtype=object))
        lbl_arrs.append(np.full(len(xs), lbl, dtype=np.int64))
    if xs_arrs:
        cb_df = pd.DataFrame({
            "cell_id": np.concatenate(cid_arrs),
            "vertex_x": np.concatenate(xs_arrs),
            "vertex_y": np.concatenate(ys_arrs),
            "label_id": np.concatenate(lbl_arrs),
        })
    else:
        cb_df = pd.DataFrame({"cell_id": [], "vertex_x": [], "vertex_y": [],
                                 "label_id": []})
    cb_df.to_parquet(out_dir / "cell_boundaries.parquet", index=False)
    cb_df.to_csv(out_dir / "cell_boundaries.csv.gz",
                  index=False, compression="gzip")
    written["cell_boundaries"] = {"n_anchor_cells": int(len(observed_ids))}

    # 2b. Nucleus boundaries — same shape, from real bundle's nucleus polys
    nuc_areas_um2: dict[str, float] = {}
    nuc_lookup: dict = {}
    if real_bundle_path is not None:
        try:
            nuc_lookup = _load_real_polys_lookup(
                real_bundle_path, kind="nucleus", cell_ids=observed_ids)
        except Exception as e:
            print(f"[write_bundle_25d] real nucleus polygon lookup failed: {e}")
    nxs, nys, ncids, nlbls = [], [], [], []
    next_label = 1
    for cid in observed_ids:
        if cid not in nuc_lookup:
            continue
        xs, ys = nuc_lookup[cid]
        if len(xs) < 3:
            continue
        nuc_areas_um2[cid] = _polygon_area_um2(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64))
        lbl = cid_to_label.get(cid, next_label)
        if cid not in cid_to_label:
            next_label += 1
        nxs.append(np.asarray(xs, dtype=np.float32))
        nys.append(np.asarray(ys, dtype=np.float32))
        ncids.append(np.full(len(xs), cid, dtype=object))
        nlbls.append(np.full(len(xs), lbl, dtype=np.int64))
    if nxs:
        nb_df = pd.DataFrame({
            "cell_id": np.concatenate(ncids),
            "vertex_x": np.concatenate(nxs),
            "vertex_y": np.concatenate(nys),
            "label_id": np.concatenate(nlbls),
        })
    else:
        nb_df = pd.DataFrame({"cell_id": [], "vertex_x": [], "vertex_y": [],
                                 "label_id": []})
    nb_df.to_parquet(out_dir / "nucleus_boundaries.parquet", index=False)
    nb_df.to_csv(out_dir / "nucleus_boundaries.csv.gz",
                  index=False, compression="gzip")

    # 2c. cells.parquet — Xenium-spec per-cell metadata (anchors only)
    tx_counts = {}
    if len(transcripts_df) > 0:
        vc = transcripts_df.groupby("cell_id", sort=False).size()
        tx_counts = {str(k): int(v) for k, v in vc.items()}
    cells_3d_obs = cells_3d[~cells_3d["is_unobserved"]].copy()
    n_obs = len(cells_3d_obs)
    cells_pub = pd.DataFrame({
        "cell_id": cells_3d_obs["cell_id"].astype(str).to_numpy(),
        "x_centroid": cells_3d_obs["centroid_x"].astype(np.float64).to_numpy(),
        "y_centroid": cells_3d_obs["centroid_y"].astype(np.float64).to_numpy(),
        "transcript_counts": np.array(
            [tx_counts.get(str(c), 0) for c in cells_3d_obs["cell_id"]],
            dtype=np.int64),
        "control_probe_counts": np.zeros(n_obs, dtype=np.int64),
        "control_codeword_counts": np.zeros(n_obs, dtype=np.int64),
        "unassigned_codeword_counts": np.zeros(n_obs, dtype=np.int64),
        "deprecated_codeword_counts": np.zeros(n_obs, dtype=np.int64),
        "cell_area": np.array(
            [cell_areas_um2.get(str(c), 0.0) for c in cells_3d_obs["cell_id"]],
            dtype=np.float64),
        "nucleus_area": np.array(
            [nuc_areas_um2.get(str(c), 0.0) for c in cells_3d_obs["cell_id"]],
            dtype=np.float64),
    })
    cells_pub["total_counts"] = cells_pub["transcript_counts"].copy()
    # Real-bundle column order
    cells_pub = cells_pub[[
        "cell_id", "x_centroid", "y_centroid", "transcript_counts",
        "control_probe_counts", "control_codeword_counts",
        "unassigned_codeword_counts", "deprecated_codeword_counts",
        "total_counts", "cell_area", "nucleus_area"]]
    cells_pub.to_parquet(out_dir / "cells.parquet", index=False)
    cells_pub.to_csv(out_dir / "cells.csv.gz", index=False, compression="gzip")
    written["cells"] = {
        "n_cells": int(len(cells_pub)),
        "parquet": str(out_dir / "cells.parquet"),
    }

    # 2d. transcripts.csv.gz mirror
    transcripts_df.to_csv(out_dir / "transcripts.csv.gz",
                            index=False, compression="gzip")

    # 3. Morphology: multi-z DAPI in morphology.ome.tif
    dapi_zstack = compose_result.dapi_zstack    # (n_z, H, W) float32 in [0, 1]
    # Apply display-LUT calibration to DAPI channel (matches 2D bundle).
    # Without LUT, plain *4095 saturates DAPI peaks. With LUT lo=43 hi=3841
    # (pancreas), p99=1.0 maps to ≈3841 — preserves channel ratios vs the
    # focal-plane DAPI written below.
    from ..scene_2d.render_tile import load_model_display_lut as _llut
    _lut = _llut(str(model.paths.root))
    _dapi_lo, _dapi_hi = 0.0, 4095.0
    if _lut is not None:
        for c in _lut.get("channels", []):
            if int(c.get("channel_index", -1)) == 0:
                _dapi_lo = float(c.get("lo", 0.0))
                _dapi_hi = float(c.get("hi", 4095.0))
                break
    if target_intensity_stats is not None and "DAPI" in target_intensity_stats:
        # Caller-provided target stats override (rare, opt-in)
        p50, p99 = target_intensity_stats["DAPI"]
        scale = float(p99) / 4095.0
        zstack_u16 = np.clip(dapi_zstack / max(scale, 1e-6) * 4095.0,
                              0, 65535).astype(np.uint16)
    else:
        # Default: LUT-native mapping [0,1] → [lo, hi]
        zstack_u16 = np.clip(_dapi_lo + dapi_zstack * (_dapi_hi - _dapi_lo),
                              0, 65535).astype(np.uint16)
    morph_path = out_dir / "morphology.ome.tif"
    _write_pyramid_ome_tiff(
        morph_path, zstack_u16, axes="ZYX",
        pixel_size_um=psz, channel_names=["DAPI"],
        physical_size_z=float(compose_result.z_slices_um[1] -
                                compose_result.z_slices_um[0]) if len(compose_result.z_slices_um) > 1 else 1.0,
        n_levels=8,
    )
    written["morphology"] = {
        "z_stack": str(morph_path),
        "shape": list(zstack_u16.shape),
        "n_z": int(zstack_u16.shape[0]),
        "z_step_um": float(compose_result.z_slices_um[1] - compose_result.z_slices_um[0]
                            if len(compose_result.z_slices_um) > 1 else 0.0),
    }

    # 4. Morphology focus: per-channel 2D files (focal-plane projection).
    # Pass display_lut so calibrate_to_uint16 uses "lut_native" mode
    # (synth [0,1] → real uint16 [lo, hi] per-channel) — matches the 2D
    # bundle writer. Without it, the writer falls back to pure "off"
    # mode (per-channel p99→4095) which destroys inter-channel ratios
    # and produces DAPI-saturated, aSMA-inflated bundle output.
    from ..scene_2d.render_tile import load_model_display_lut
    display_lut = load_model_display_lut(str(model.paths.root))
    _write_morphology_focus(
        compose_result.focal_2d_render,        # (C, H, W) float32
        channel_names, psz, out_dir,
        target_stats=target_intensity_stats,
        intensity_mode=intensity_mode,
        n_replica_files=max(4, n_ch),
        n_pyramid_levels=8,
        display_lut=display_lut,
    )

    # 5. ground_truth: cells_3d.parquet + molecule_provenance.parquet.
    # molecule_provenance carries Xenium-spec parity with the 2D writer
    # (tile_idx, true_cell_id, is_ghost, gene, true_factor, x_um, y_um)
    # plus 2.5D extensions (x_true / y_true / z_true and true_cell_type).
    cells_3d.to_parquet(gt_dir / "cells_3d.parquet", index=False)
    n_mol = len(mol)
    mol_prov = pd.DataFrame({
        "transcript_id": transcripts_df["transcript_id"].values,
        "tile_idx": np.zeros(n_mol, dtype=np.int64),
        "true_cell_id": mol["true_cell_id"].astype(str).to_numpy(),
        "is_ghost": (mol["is_ghost"].astype(bool).to_numpy()
                       if "is_ghost" in mol.columns
                       else np.zeros(n_mol, dtype=bool)),
        "gene": (mol["gene"].astype(str).to_numpy()
                   if "gene" in mol.columns
                   else np.full(n_mol, "synth", dtype=object)),
        "true_factor": (mol["factor_label"].astype(np.int64).to_numpy()
                          if "factor_label" in mol.columns
                          else np.full(n_mol, -1, dtype=np.int64)),
        "x_um": mol["x_true"].astype(np.float64).to_numpy(),
        "y_um": mol["y_true"].astype(np.float64).to_numpy(),
        "x_true": mol["x_true"].astype(np.float32).to_numpy(),
        "y_true": mol["y_true"].astype(np.float32).to_numpy(),
        "z_true": mol["z_true"].astype(np.float32).to_numpy(),
        "true_cell_type": (mol["true_cell_type"].astype(str).to_numpy()
                              if "true_cell_type" in mol.columns
                              else np.full(n_mol, "unknown", dtype=object)),
    })
    mol_prov.to_parquet(gt_dir / "molecule_provenance.parquet", index=False)

    # 5a. cells_synth.parquet — anchor-cell metadata in the 2D writer's
    # schema (cell_id, cell_type, is_ghost, centroid_x/y, area_um2, source)
    # PLUS the cell-type-resolver provenance columns (cell_type_source,
    # cell_type_confidence, cell_type_evidence) plumbed through cells_3d_df
    # so 2.5D parity with the 2D writer is complete.
    # Excludes unobserved seeds (the 2.5D equivalent of 2D's ghost cells).
    obs = cells_3d[~cells_3d["is_unobserved"]]
    # Optional resolver columns — present when the upstream
    # precompute_scene_25d ran the unified resolver (post-refactor).
    has_resolver = ("cell_type_source" in obs.columns
                      and "cell_type_confidence" in obs.columns
                      and "cell_type_evidence" in obs.columns)
    cells_synth_dict = {
        "cell_id": obs["cell_id"].astype(str).to_numpy(),
        "cell_type": obs["cell_type"].astype(str).to_numpy(),
        "is_ghost": np.zeros(len(obs), dtype=bool),
        "centroid_x": obs["centroid_x"].astype(np.float64).to_numpy(),
        "centroid_y": obs["centroid_y"].astype(np.float64).to_numpy(),
        "area_um2": np.array(
            [cell_areas_um2.get(str(c), 0.0) for c in obs["cell_id"]],
            dtype=np.float64),
        "source": np.full(len(obs), "observed_anchor", dtype=object),
    }
    if has_resolver:
        cells_synth_dict["cell_type_source"] = (
            obs["cell_type_source"].astype(object).to_numpy())
        cells_synth_dict["cell_type_confidence"] = (
            obs["cell_type_confidence"].astype(np.float32).to_numpy())
        cells_synth_dict["cell_type_evidence"] = (
            obs["cell_type_evidence"].astype(str).to_numpy())
    cells_synth = pd.DataFrame(cells_synth_dict)
    # Invariant: no "unknown" / NaN rows. Mirrors the 2D writer.
    if len(cells_synth) > 0:
        bad_mask = (cells_synth["cell_type"]
                      .isin(["unknown", "Unknown", "UNKNOWN", ""])
                      | cells_synth["cell_type"].isna())
        if bad_mask.any():
            sample_ids = cells_synth.loc[bad_mask, "cell_id"].head(5).tolist()
            raise RuntimeError(
                f"cells_synth invariant violated (2.5D): "
                f"{int(bad_mask.sum())} of {len(cells_synth)} cells have "
                f"cell_type ∈ {{'', None, 'unknown'}}. The cell-type "
                f"resolver should produce a real type for every anchor; "
                f"this typically means the model's stain-classifier bank "
                f"is missing. Offending cell_ids: {sample_ids}.")
    cells_synth.to_parquet(gt_dir / "cells_synth.parquet", index=False)
    written["ground_truth_dir"] = str(gt_dir)

    # 5b. gene_panel.json — copy verbatim from real bundle if available
    if real_bundle_path is not None:
        gp_src = Path(real_bundle_path) / "gene_panel.json"
        if gp_src.exists():
            shutil.copy2(gp_src, out_dir / "gene_panel.json")
            written["gene_panel"] = str(out_dir / "gene_panel.json")

    # 5c. experiment.xenium metadata (10x-spec JSON header)
    xmin, ymin, xmax, ymax = compose_result.region_bounds_um
    experiment = {
        "analysis_sw_name": "xeSim",
        "analysis_sw_version": "plan3.2.5D",
        "region_name": "synth_bundle_25d",
        "panel_design_id": "synthetic",
        "panel_name": "synth_panel",
        "num_cells": int(len(cells_pub)),
        "num_unassigned_transcripts": int(0),
        "num_transcripts": int(len(transcripts_df)),
        "pixel_size": float(psz),
        "z_step_size": float(compose_result.z_slices_um[1] -
                                compose_result.z_slices_um[0])
                            if len(compose_result.z_slices_um) > 1 else 0.0,
        "tile_bounds_um": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "num_channels": int(n_ch),
        "channel_names": list(channel_names),
        "synth_metadata": {
            "scene_mode": "2.5d",
            "n_z": int(len(compose_result.z_slices_um)),
            "ground_truth_in_cell_id_column": True,
            "intensity_mode": intensity_mode,
        },
    }
    (out_dir / "experiment.xenium").write_text(json.dumps(experiment, indent=2))
    written["experiment_xenium"] = str(out_dir / "experiment.xenium")

    # 6. config
    cfg = dict(config or {})
    cfg.update({
        "scene_mode": "2.5d",
        "pixel_size_um": psz,
        "z_slices_um": [float(z) for z in compose_result.z_slices_um],
        "region_bounds_um": list(compose_result.region_bounds_um),
        "n_observed_cells": int(len(observed_ids)),
        "n_unobserved_cells": int(cells_3d["is_unobserved"].sum()),
    })
    (gt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    return written


__all__ = ["write_bundle_25d"]
