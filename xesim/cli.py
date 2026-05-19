"""xeSim CLI — clean three-verb surface.

  xesim fit-model BUNDLE --annotations CT --out MODEL_DIR
                          [--no-transcripts | --celladmix-run PATH]
  xesim fit-priors BUNDLE --annotations CT --out PATH
  xesim explain BUNDLE --model MODEL_DIR --out OUT [scope+aug flags]
  xesim generate --model MODEL_DIR --out OUT [--num-scenes N | --guide-crop ID]
  xesim inspect-bundle BUNDLE
  xesim inspect-model MODEL_DIR

`explain` has three scope modes (mutually exclusive):
  --tile X,Y                          single tile centered at (X, Y) µm
  --region XMIN,YMIN,XMAX,YMAX        arbitrary rectangular region in µm
  (neither)                           whole bundle FOV

Augmentation flags work identically in every scope:
  --add-ghosts/--no-ghosts            density-calibrated ghost cells [on]
  --add-transcript-proposed           recover missed cells from orphan transcripts
  --stamp-transcripts                 cellAdmix classification on cell provenance

Output writer (set automatically; override with --format):
  --tile  → scene-dir (PNG, transcripts.parquet, cells.parquet, summary.json)
  --region or whole bundle → Xenium-compatible bundle directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonicalize import inspect_bundle
from .model import XesimModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tile(s: str) -> tuple[float, float]:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 2:
        raise SystemExit("--tile must be 'x,y' (µm)")
    return (parts[0], parts[1])


def _parse_region(s: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 4:
        raise SystemExit("--region must be 'xmin,ymin,xmax,ymax' (µm)")
    return (parts[0], parts[1], parts[2], parts[3])


def _bundle_fov_bounds(bundle_path: str) -> tuple[float, float, float, float]:
    """Infer (xmin, ymin, xmax, ymax) for the whole bundle FOV from its
    morphology image."""
    from .xenium import resolve_bundle
    from .images import ome_image_shape
    bundle = resolve_bundle(bundle_path)
    if not bundle.morphology_focus_paths:
        raise SystemExit(
            f"Cannot infer whole-bundle FOV: {bundle_path} has no morphology image. "
            f"Pass --tile or --region explicitly."
        )
    h, w = ome_image_shape(bundle.morphology_focus_paths[0])
    psz = float(bundle.pixel_size)
    return (0.0, 0.0, w * psz, h * psz)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _fit_model(args: argparse.Namespace) -> None:
    XesimModel.fit(
        bundle_path=args.bundle,
        annotations_path=args.annotations,
        out_dir=args.out,
        num_crops=args.num_crops,
        crop_size_um=args.crop_size_um,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        with_transcripts=args.with_transcripts,
        celladmix_run=getattr(args, "celladmix_run", None),
        crop_selection=args.crop_selection,
        stratified_alpha=args.stratified_alpha,
        stratified_within_pick=args.stratified_within_pick,
    )
    # Diagnostics (opt-in via --diagnostic flag).
    if getattr(args, "diagnostic", None) is not None:
        from .diagnostics import resolve_diagnostic_dir, fit_model_diagnostics
        diag_dir = resolve_diagnostic_dir(args.diagnostic, args.out)
        print(f"[fit-model] writing diagnostics → {diag_dir}")
        written = fit_model_diagnostics(model_dir=args.out,
                                              bundle_path=args.bundle,
                                              out_dir=diag_dir)
        for p in written:
            print(f"  {p}")


def _fit_priors(args: argparse.Namespace) -> None:
    """Fit only the per-cell-type 3D nucleus shape priors. Standalone
    counterpart to `fit-model`: skips the renderer / mechanistic / NMF
    fits, runs only `fit_nucleus_priors_from_bundle` (~5s on a 100k-cell
    bundle). Writes a single nucleus_priors.json — useful for population-
    level shape analyses or for hand-iterating the 2.5D z-prior without
    a 30-min model fit."""
    from pathlib import Path
    from .scene_2_5d.fit_priors import fit_nucleus_priors_from_bundle

    out_path = Path(args.out)
    if out_path.is_dir() or args.out.endswith("/"):
        out_path = out_path / "nucleus_priors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = fit_nucleus_priors_from_bundle(
        bundle_path=args.bundle,
        annotation_path=args.annotations,
        out_path=out_path,
        type_col=args.type_col,
        min_per_type=args.min_per_type,
        n_iter=args.n_iter,
        n_samples=args.n_samples,
        seed=args.seed,
        verbose=True,
    )
    if getattr(args, "diagnostic", None) is not None:
        from .diagnostics import resolve_diagnostic_dir, fit_priors_diagnostics
        diag_dir = resolve_diagnostic_dir(args.diagnostic, out_path.parent)
        written = fit_priors_diagnostics(out_path, diag_dir)
        if written:
            print(f"[fit-priors] diagnostics → {diag_dir}")
            for w in written:
                print(f"  {w}")
    print()
    print(f"[fit-priors] fit {len(res.nucleus_priors)} cell types from "
            f"{res.n_records_used:,} nucleus polygons:")
    print(f"  {'type':30}  {'n':>8}  {'log r̄':>7}  {'log r std':>9}  "
            f"{'κ₁':>6} {'κ₂':>6} {'κ₃':>6}")
    for name, p in res.nucleus_priors.items():
        ks = p.axis_ratio_log_kappa
        print(f"  {name[:30]:30}  {p.n_train:>8}  "
              f"{p.log_radius_mean:>7.3f}  {p.log_radius_std:>9.3f}  "
              f"{ks[0]:>6.3f} {ks[1]:>6.3f} {ks[2]:>6.3f}")
    print()
    print(f"  wrote → {out_path}")
    print(f"  timings: {res.timings_s}")


def _explain(args: argparse.Namespace) -> None:
    """Unified Task-1: bundle → render at tile / region / whole-bundle scope."""
    import numpy as np

    # Validate scope flags (mutually exclusive)
    if args.tile is not None and args.region is not None:
        raise SystemExit("--tile and --region are mutually exclusive")

    # Validate emission backend
    if getattr(args, "emission_backend", "legacy") == "stpuppeteer":
        if not getattr(args, "stpuppeteer_config", None):
            raise SystemExit(
                "--emission-backend stpuppeteer requires --stpuppeteer-config PATH"
            )

    # 2.5D dispatch — independent path
    if getattr(args, "scene_mode", "2d") == "2.5d":
        return _explain_25d(args)

    # Decide scope + default format
    if args.tile is not None:
        scope = "tile"
        default_format = "scene"
    elif args.region is not None:
        scope = "region"
        default_format = "bundle"
    else:
        scope = "whole"
        default_format = "bundle"
    fmt = args.format or default_format
    if fmt == "scene" and scope != "tile":
        raise SystemExit(
            "--format scene is only valid with --tile. "
            "For multi-tile output use --format bundle (the default)."
        )

    model = XesimModel.load(args.model, device=args.device)
    # Resolve --tx-rate-scale → env var so multiprocess workers inherit.
    # tile_pipeline._emit_molecules reads XESIM_TX_RATE_SCALE first, then
    # priors['tx_rate_scale_default']. CLI flag wins when given.
    import os as _os
    if getattr(args, "tx_rate_scale", None) is not None:
        _os.environ["XESIM_TX_RATE_SCALE"] = str(float(args.tx_rate_scale))
        print(f"[explain] --tx-rate-scale {float(args.tx_rate_scale):.3f} "
              f"(overrides priors default)")
    else:
        tx_priors = getattr(model, "transcripts_priors", None) or {}
        priors_default = float(tx_priors.get("tx_rate_scale_default", 1.0))
        if priors_default != 1.0:
            print(f"[explain] tx_rate_scale={priors_default:.3f} "
                  f"(from priors default)")
    rng = np.random.default_rng(args.seed)

    # Compute region bounds in µm
    tile_um = float(model.tile_px) * float(model.pixel_size)
    if scope == "tile":
        x, y = args.tile
        half = tile_um / 2.0
        bounds = (x - half, y - half, x + half, y + half)
        print(f"[explain] tile scope: center ({x:.1f}, {y:.1f}) µm, "
              f"tile size {tile_um:.1f} µm")
    elif scope == "region":
        bounds = args.region
        print(f"[explain] region scope: "
              f"({bounds[0]:.1f}, {bounds[1]:.1f}) — "
              f"({bounds[2]:.1f}, {bounds[3]:.1f}) µm "
              f"({bounds[2]-bounds[0]:.0f} × {bounds[3]-bounds[1]:.0f} µm)")
    else:
        bounds = _bundle_fov_bounds(args.bundle)
        area_um2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if area_um2 > 5_000_000:
            print(f"[explain] WARNING: whole-bundle scope covers "
                  f"{area_um2/1e6:.1f} mm² — this will take a while.")
        print(f"[explain] whole-bundle scope: "
              f"({bounds[0]:.1f}, {bounds[1]:.1f}) — "
              f"({bounds[2]:.1f}, {bounds[3]:.1f}) µm")

    if scope == "tile":
        _explain_tile_path(model, args, bounds, fmt, rng)
    else:
        _explain_multi_path(model, args, bounds, rng)


def _explain_tile_path(model, args, bounds, fmt, rng) -> None:
    """Single-tile path: explain_region → scene-dir or single-tile bundle."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from .scene_2d.explain_region import explain_region

    nf = _resolve_noise_fraction(args.noise_fraction, args.bundle)
    res = explain_region(
        model, args.bundle, region_bounds_um=bounds,
        annotation_path=args.annotation,
        add_ghosts=args.add_ghosts,
        add_transcript_proposed=args.add_transcript_proposed,
        stamp_transcripts=args.stamp_transcripts,
        noise_fraction=nf,
        ghost_count_scale=args.ghost_count_scale,
        sample_molecules=True,
        rng=rng,
        emission_backend=getattr(args, "emission_backend", "legacy"),
        stpuppeteer_config=getattr(args, "stpuppeteer_config", None),
    )
    scene = res.scene
    image = res.image

    print(f"[explain]   {res.n_anchors} anchors"
          f"{f' + {res.n_transcript_proposed} tx-proposed' if res.n_transcript_proposed else ''}"
          f" + {res.n_ghosts} ghosts, {len(scene.molecules)} molecules")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stash the resolved nf so a downstream bundle-writer call doesn't re-tune
    args._resolved_noise_fraction = nf

    if fmt == "scene":
        # Diagnostic scene-dir: PNG + parquet + JSON
        rgb_idx = list(range(min(3, image.shape[0])))
        rgb = np.stack([image[c] for c in rgb_idx], axis=-1)
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(out_dir / "morphology.png")
        np.save(out_dir / "morphology.npy", image)

        scene.molecules.to_parquet(out_dir / "transcripts.parquet")
        cells_rows = [{
            "cell_id": c.cell_id, "cell_type": c.cell_type,
            "label": c.label, "source": c.source,
            "is_ghost": c.provenance.get("is_ghost", False),
        } for c in scene.mech_scene.cells]
        pd.DataFrame(cells_rows).to_parquet(out_dir / "cells.parquet")

        (out_dir / "summary.json").write_text(json.dumps({
            "scope": "tile",
            "region_bounds_um": list(bounds),
            "n_anchors": int(res.n_anchors),
            "n_transcript_proposed": int(res.n_transcript_proposed),
            "n_ghosts": int(res.n_ghosts),
            "n_molecules": int(len(scene.molecules)),
            "geometry_source": scene.provenance.get("geometry_source"),
        }, indent=2))
        print(f"[explain] Wrote scene-dir → {out_dir}")
    else:
        _write_bundle_single(model, args, scene, image, bounds, rng)


def _explain_25d(args: argparse.Namespace) -> None:
    """2.5D scene-mode dispatch.

    Modes:
    - `--tile`: single tile centered at (x, y), one compose call (~40s)
    - `--region`: explicit bounds, one compose call (large regions slow)
    - `--whole-bundle`: tile-and-stitch over the whole bundle's
      cell-vertex extent. Memory-mapped DAPI z-stack so memory is bounded.
    - `--stitch-tile-um`: tile size (default 300 µm) used by whole-bundle
      or large `--region`
    """
    import numpy as np
    from .scene_2_5d.compose import compose_region_scene_25d
    from .scene_2_5d.stitch import build_scene_25d
    from .scene_2_5d.bundle_writer_25d import write_bundle_25d

    model = XesimModel.load(args.model, device=args.device)
    rng = np.random.default_rng(args.seed)

    use_stitch = False
    if args.tile is not None:
        x, y = args.tile
        # crop_size_um from canonical manifest; fall back to 64µm typical
        try:
            import json
            from pathlib import Path
            cm = json.loads((Path(args.model) / "canonical" / "manifest.json").read_text())
            tile_um = float(cm.get("crop_size_um", 64.0))
        except Exception:
            tile_um = 64.0
        half = tile_um / 2.0
        bounds = (x - half, y - half, x + half, y + half)
        print(f"[explain-2.5d] tile scope: center ({x:.1f}, {y:.1f}) µm "
              f"(tile {tile_um}µm)")
    elif args.whole_bundle:
        # Match 2D convention: bundle pixel 0 = µm (0, 0). The morphology
        # image's FOV is the canonical xmin/ymin; cell-extent bounds (which
        # this used to compute via cell_boundaries.vertex_x.min) would put
        # the saved bundle's pixel 0 at the cell-vertex-min absolute µm,
        # breaking downstream readers (e.g. real_tile_image) that assume
        # pixel 0 = µm 0.
        bounds = _bundle_fov_bounds(args.bundle)
        print(f"[explain-2.5d] whole-bundle scope (FOV): "
              f"({bounds[0]:.0f}, {bounds[1]:.0f}) — "
              f"({bounds[2]:.0f}, {bounds[3]:.0f}) µm")
        use_stitch = True
    elif args.region is not None:
        bounds = args.region
        # If the region is large, prefer stitched path (bounded memory).
        area_mm2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1e6
        use_stitch = area_mm2 > 0.3
        print(f"[explain-2.5d] region scope: "
              f"({bounds[0]:.1f}, {bounds[1]:.1f}) — "
              f"({bounds[2]:.1f}, {bounds[3]:.1f}) µm "
              f"({area_mm2:.2f} mm² — {'stitched' if use_stitch else 'monolithic'})")
    else:
        raise SystemExit("--scene-mode 2.5d requires --tile, --region, or --whole-bundle")

    # Resolve intensity calibration for non-DAPI channels
    target_intensity_stats, _, calib_mode = _resolve_intensity_calibration(
        args.intensity_calibration, args.bundle, model)

    if use_stitch:
        stitch_tile_um = float(args.stitch_tile_um)
        stitch_overlap_um = float(args.stitch_overlap_um)
        # 2.5D peaks at ~8 GB GPU per worker (12 z-batched DAPI renders
        # + focal + encoder). On a 38 GB GPU, 4 workers uses ~32 GB
        # peak with ~6 GB headroom — works WITHOUT concurrent GPU load
        # (a contending profile process pushed us to OOM on 2026-05-17,
        # which is why this cap exists). 5+ workers OOMs.
        workers_for_25d = min(int(args.num_workers), 4)
        if workers_for_25d != int(args.num_workers):
            print(f"[explain-2.5d] capping num_workers {args.num_workers} → "
                  f"{workers_for_25d} (12 z-batch peaks ~8 GB/worker; "
                  f"4 is the safe max on a 38 GB GPU)")
        print(f"[explain-2.5d] tile-and-stitch: {stitch_tile_um}µm tiles, "
              f"{stitch_overlap_um}µm overlap")
        sresult = build_scene_25d(
            model, args.bundle, scene_bounds_um=bounds,
            tile_um=stitch_tile_um, overlap_um=stitch_overlap_um,
            rng=rng,
            num_workers=workers_for_25d,
            model_path=str(args.model),
            device=args.device,
            compile_renderer=bool(getattr(args, "compile_renderer", False)),
            model_dir=str(args.model),
            emission_backend=getattr(args, "emission_backend", "legacy"),
            stpuppeteer_config=getattr(args, "stpuppeteer_config", None),
        )
        result = sresult.to_compose_result()
    else:
        result = compose_region_scene_25d(
            model, args.bundle, region_bounds_um=bounds,
            annotation_path=args.annotation,
            rng=rng,
            model_dir=str(args.model),
            emission_backend=getattr(args, "emission_backend", "legacy"),
            stpuppeteer_config=getattr(args, "stpuppeteer_config", None),
        )

    cfg = {
        "scene_mode": "2.5d",
        "scope": ("tile" if args.tile is not None else
                  "whole-bundle" if args.whole_bundle else "region"),
        "stitched": bool(use_stitch),
        "real_bundle": str(args.bundle),
        "model_dir": str(args.model),
        "seed": int(args.seed),
    }
    print(f"[explain-2.5d] writing bundle to {args.out}")
    written = write_bundle_25d(
        output_dir=args.out, compose_result=result, model=model,
        real_bundle_path=args.bundle,
        config=cfg, overwrite=args.overwrite,
        target_intensity_stats=target_intensity_stats,
        intensity_mode=calib_mode,
    )
    import json
    print(json.dumps({k: v for k, v in written.items() if k != "config"},
                       indent=2, default=str))


def _explain_multi_scene_first(model, args, bounds, rng, nf,
                                   target_intensity, target_quantiles, calib_mode):
    """Scene-first path: one explain_region call over the whole region,
    rendered with bridge-tiles. Yields a single-scene 'result' dict shaped
    like build_scene's so the downstream writer is reused."""
    import numpy as np
    from .scene_2d.explain_region import explain_region
    from .scene_2d import Scene2D

    pixel_size = float(model.pixel_size)
    tile_px = int(args.inference_tile_px or model.tile_px)
    print(f"[explain] scene-first: composing whole region "
          f"({bounds[2]-bounds[0]:.0f}×{bounds[3]-bounds[1]:.0f} µm) "
          f"in one pass; render in {tile_px}-px tiles "
          f"({'bridge' if args.bridge_tile else 'single'} grid)")
    res = explain_region(
        model, args.bundle, region_bounds_um=bounds,
        annotation_path=args.annotation,
        add_ghosts=args.add_ghosts,
        add_transcript_proposed=args.add_transcript_proposed,
        stamp_transcripts=args.stamp_transcripts,
        noise_fraction=nf,
        ghost_count_scale=args.ghost_count_scale,
        sample_molecules=True,
        rng=rng,
        tile_render_px=tile_px,
        use_bridge_grid=bool(args.bridge_tile),
        emission_backend=getattr(args, "emission_backend", "legacy"),
        stpuppeteer_config=getattr(args, "stpuppeteer_config", None),
    )

    scene = res.scene
    H, W = res.image.shape[1:]
    scene_bounds_um = (bounds[0], bounds[1],
                         bounds[0] + W * pixel_size,
                         bounds[1] + H * pixel_size)

    # Shape the return like build_scene's so _explain_multi_path doesn't care.
    n_anchors = sum(1 for c in scene.mech_scene.cells
                       if c.source == "observed_anchor")
    n_ghosts = sum(1 for c in scene.mech_scene.cells
                      if c.provenance.get("is_ghost", False))
    n_tx = sum(1 for c in scene.mech_scene.cells
                  if c.source == "tx_inferred")
    return {
        "scenes": [scene],
        "stitched_image": res.image,
        "scene_bounds_um": scene_bounds_um,
        "pixel_size_um": pixel_size,
        "tile_size_um": float(tile_px) * pixel_size,
        "overlap_um": 0.0,
        "grid_nx": 1, "grid_ny": 1, "n_tiles": 1,
        "n_anchor_cells": n_anchors,
        "n_ghost_cells": n_ghosts,
        "n_transcript_proposed": n_tx,
        "n_transcripts": int(len(scene.molecules)),
    }


def _explain_multi_path(model, args, bounds, rng) -> None:
    """Region / whole-bundle path: build_scene → multi-tile stitched bundle."""
    from .scene_2d.bundle_writer import write_bundle
    from .scene_2d.scene_pipeline import build_scene

    nf = _resolve_noise_fraction(args.noise_fraction, args.bundle)
    target_intensity, target_quantiles, calib_mode = _resolve_intensity_calibration(
        args.intensity_calibration, args.bundle, model)

    if args.scene_first:
        result = _explain_multi_scene_first(model, args, bounds, rng, nf,
                                                target_intensity, target_quantiles, calib_mode)
    else:
        result = build_scene(
            model, args.bundle,
            scene_bounds_um=bounds,
            add_ghosts=args.add_ghosts,
            add_transcript_proposed=args.add_transcript_proposed,
            noise_fraction=nf,
            ghost_count_scale=args.ghost_count_scale,
            annotation_path=args.annotation,
            rng=rng,
            target_intensity_stats=target_intensity,
            target_intensity_quantiles=target_quantiles,
            intensity_mode=calib_mode,
            inference_tile_px=args.inference_tile_px,
            num_workers=args.num_workers,
            model_path=str(args.model),
            device=args.device,
            emission_backend=getattr(args, "emission_backend", "legacy"),
            stpuppeteer_config=getattr(args, "stpuppeteer_config", None),
        )

    if args.stamp_transcripts and model.transcripts_priors is not None:
        try:
            classifications = model.classify_cells_by_transcripts()
            result["scenes"] = [
                _stamped_scene(s, model, classifications) for s in result["scenes"]]
        except Exception as e:
            print(f"[explain] stamp_transcripts unavailable: {e}")

    summary_extra = (f" + {result.get('n_transcript_proposed', 0)} tx-proposed"
                       if result.get("n_transcript_proposed") else "")
    print(f"[explain]   {result['n_anchor_cells']} anchors{summary_extra} "
          f"+ {result['n_ghost_cells']} ghosts, "
          f"{result['n_transcripts']} molecules across {result['n_tiles']} tiles")

    gp_src = _gene_panel_source(args.bundle)
    channel_names_out = list(model.manifest.get("channel_names") or [])
    config = {
        "scope": "region" if args.region is not None else "whole",
        "scene_bounds_um": list(result["scene_bounds_um"]),
        "n_tiles": result["n_tiles"],
        "grid_nx": result["grid_nx"], "grid_ny": result["grid_ny"],
        "overlap_um": result["overlap_um"],
        "noise_fraction_target": float(nf),
        "real_bundle": str(args.bundle),
        "model_dir": str(args.model),
        "tile_size_um": float(model.tile_px) * float(model.pixel_size),
        "seed": int(args.seed),
        "ghost_count_scale": float(args.ghost_count_scale),
        "add_transcript_proposed": bool(args.add_transcript_proposed),
        "stamp_transcripts": bool(args.stamp_transcripts),
    }

    # Pass the model's display LUT so the bundle writer maps renderer
    # output → uint16 in the same intensity space the renderer was
    # trained in (lut_native mode preserves channel ratios + matches
    # real Xenium scale, instead of the lossy per-channel p99→4095).
    # Also auto-calibrate per-channel noise (Gaussian read + Poisson shot)
    # from the real bundle so synth's dark regions have the same noise
    # texture as real (the deterministic renderer produces zero variance).
    from .scene_2d.render_tile import load_model_display_lut
    from .scene_2d.intensity import calibrate_noise_stats
    display_lut = load_model_display_lut(str(args.model))
    if display_lut is not None and args.bundle:
        try:
            display_lut["noise_stats"] = calibrate_noise_stats(
                args.bundle, display_lut=display_lut)
            print(f"[explain] calibrated noise: "
                    f"{[(n.get('read_std',0), n.get('shot_k',0)) for n in display_lut['noise_stats']]}")
        except Exception as e:
            print(f"[explain] noise calibration unavailable ({e}); skipping noise injection")
    print(f"[explain] Writing bundle → {args.out}")
    written = write_bundle(
        output_dir=args.out,
        scenes=result["scenes"],
        render_images=result["stitched_image"],
        channel_names=channel_names_out,
        pixel_size_um=float(model.pixel_size),
        gene_panel_source=gp_src,
        config=config,
        rng=rng,
        overwrite=args.overwrite,
        target_intensity_stats=target_intensity,
        target_intensity_quantiles=target_quantiles,
        intensity_mode=calib_mode,
        real_bundle_path=args.bundle,
        display_lut=display_lut,
    )
    print(f"[explain] Done.")
    print(json.dumps({k: v for k, v in written.items() if k != "config"},
                       indent=2, default=str))
    if getattr(args, "diagnostic", None) is not None:
        from .diagnostics import resolve_diagnostic_dir, explain_diagnostics
        diag_dir = resolve_diagnostic_dir(args.diagnostic, args.out)
        print(f"[explain] writing diagnostics → {diag_dir}")
        for p in explain_diagnostics(bundle_path=args.bundle,
                                          synth_dir=args.out,
                                          model_dir=args.model,
                                          out_dir=diag_dir):
            print(f"  {p}")
        # STpuppeteer-backend additions: stratified per-cell-type stats
        # + marker-specificity matrix. Only meaningful when the backend
        # was actually used; legacy runs have no STpuppeteer cfg to read.
        if getattr(args, "emission_backend", "legacy") == "stpuppeteer":
            _write_emission_stpuppeteer_diagnostics(
                args.out, args.stpuppeteer_config, diag_dir,
            )


def _write_bundle_single(model, args, scene, image, bounds, rng) -> None:
    """Single-tile bundle path."""
    from .scene_2d.bundle_writer import write_bundle
    nf_resolved = getattr(args, "_resolved_noise_fraction", None)
    if nf_resolved is None:
        nf_resolved = _resolve_noise_fraction(args.noise_fraction, args.bundle)
    target_intensity, target_quantiles, calib_mode = _resolve_intensity_calibration(
        args.intensity_calibration, args.bundle, model)
    gp_src = _gene_panel_source(args.bundle)
    channel_names_out = list(model.manifest.get("channel_names") or [])
    config = {
        "scope": "tile",
        "tile_bounds_um": list(bounds),
        "noise_fraction_target": float(nf_resolved),
        "real_bundle": str(args.bundle),
        "model_dir": str(args.model),
        "tile_size_um": float(model.tile_px) * float(model.pixel_size),
        "seed": int(args.seed),
        "ghost_count_scale": float(args.ghost_count_scale),
        "add_transcript_proposed": bool(args.add_transcript_proposed),
        "stamp_transcripts": bool(args.stamp_transcripts),
    }
    print(f"[explain] Writing bundle → {args.out}")
    written = write_bundle(
        output_dir=args.out,
        scenes=scene, render_images=image,
        channel_names=channel_names_out,
        pixel_size_um=float(model.pixel_size),
        gene_panel_source=gp_src,
        config=config, rng=rng,
        overwrite=args.overwrite,
        target_intensity_stats=target_intensity,
        target_intensity_quantiles=target_quantiles,
        intensity_mode=calib_mode,
        real_bundle_path=args.bundle,
    )
    print(f"[explain] Done.")
    print(json.dumps({k: v for k, v in written.items() if k != "config"},
                       indent=2, default=str))


def _stamped_scene(scene_2d, model, classifications):
    """Return a copy of scene_2d with its mech_scene cells stamped with
    transcript classifications."""
    from dataclasses import replace
    new_mech = model.stamp_scene_with_transcripts(scene_2d.mech_scene, classifications)
    return replace(scene_2d, mech_scene=new_mech)


def _resolve_noise_fraction(arg_value, bundle_path):
    """Resolve --noise-fraction: 'auto' or None → auto-tune; else float."""
    if arg_value is None or (isinstance(arg_value, str) and arg_value.lower() == "auto"):
        from .scene_2d.bundle_writer import auto_tune_noise_fraction
        nf = auto_tune_noise_fraction(bundle_path)
        print(f"[explain] Auto-tuned noise_fraction from real bundle "
              f"UNASSIGNED rate: {nf:.4f}")
        return nf
    return float(arg_value)


def _resolve_intensity_calibration(mode, bundle_path, model):
    """Tune per-channel calibration stats for the chosen mode."""
    from .scene_2d.bundle_writer import auto_tune_intensity_stats
    from .scene_2d.intensity import auto_tune_intensity_quantiles
    channel_names = list(model.manifest.get("channel_names") or [])
    target_stats = None
    target_quantiles = None
    if mode in ("scale", "match2"):
        target_stats = auto_tune_intensity_stats(bundle_path, channel_names)
        stat_str = ", ".join(f"{n}: p50={p50:.0f} p99.5={p99:.0f}"
                              for n, (p50, p99) in target_stats.items())
        print(f"[explain] Intensity calibration '{mode}' tuned from real "
              f"bundle: {stat_str}")
    elif mode == "histmatch":
        target_quantiles = auto_tune_intensity_quantiles(bundle_path, channel_names)
        print(f"[explain] Intensity calibration 'histmatch': "
              f"tuned {len(target_quantiles)} channel quantile tables")
    else:
        print(f"[explain] Intensity calibration OFF")
    return target_stats, target_quantiles, mode


def _gene_panel_source(bundle_path):
    cand = Path(bundle_path) / "gene_panel.json"
    return cand if cand.exists() else None


def _generate(args: argparse.Namespace) -> None:
    model = XesimModel.load(args.model, device=args.device)
    guide = None
    if args.guide_crop:
        if not args.guide_bundle:
            print("--guide-crop requires --guide-bundle (or MODEL_DIR/canonical/)", file=sys.stderr)
            sys.exit(2)
        guide = (args.guide_bundle, args.guide_crop)
    items = model.generate(
        num_scenes=args.num_scenes,
        guide=guide,
        seed=args.seed,
        target_num_cells=args.num_cells,
        admixture_rate_multiplier=args.admixture_rate,
    )
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in items:
        p = model.write(item, out_dir, name=None)
        paths.append(p)
    (out_dir / "manifest.json").write_text(json.dumps({
        "task": "generate",
        "model": str(Path(args.model).resolve()),
        "guide": {"bundle": args.guide_bundle, "crop_id": args.guide_crop} if guide else None,
        "seed": args.seed,
        "scenes": [{"name": p.name, "path": str(p.relative_to(out_dir))} for p in paths],
    }, indent=2))
    print(f"Wrote {len(paths)} scene(s) → {out_dir}")


def _reemit_molecules(args: argparse.Namespace) -> None:
    """CLI handler for ``xesim re-emit-molecules``."""
    from .emission_stpuppeteer.reemit import reemit_molecules
    print(f"[re-emit] source bundle: {args.bundle}")
    print(f"[re-emit] writing to:    {args.out}")
    print(f"[re-emit] config:        {args.stpuppeteer_config}")
    result = reemit_molecules(
        src_bundle=args.bundle,
        out_bundle=args.out,
        stpuppeteer_config=args.stpuppeteer_config,
        seed=args.seed,
    )
    print(f"[re-emit] done. scene_mode={result.scene_mode} "
          f"n_cells={result.n_cells} n_transcripts={result.n_transcripts:,}")
    print(f"[re-emit] output: {result.out_dir}")

    if getattr(args, "diagnostic", None) is not None:
        from .diagnostics import resolve_diagnostic_dir
        diag_dir = resolve_diagnostic_dir(args.diagnostic, args.out)
        _write_emission_stpuppeteer_diagnostics(
            args.out, args.stpuppeteer_config, diag_dir,
        )


def _write_emission_stpuppeteer_diagnostics(
    bundle_dir: str,
    stpuppeteer_config: str,
    diag_dir,
) -> None:
    """Compute STpuppeteer-emission diagnostics from a written bundle."""
    from pathlib import Path
    import pandas as pd
    from .emission_stpuppeteer.diagnostics import (
        compute_emission_stats, print_emission_summary, write_emission_diagnostics,
    )
    from STpuppeteer.simulation import SimulationConfig

    bundle = Path(bundle_dir)
    tx = pd.read_parquet(bundle / "transcripts.parquet")
    prov_path = bundle / "ground_truth" / "molecule_provenance.parquet"
    if prov_path.exists():
        prov = pd.read_parquet(prov_path)
        if "transcript_id" in tx.columns and "transcript_id" in prov.columns:
            keep = [c for c in ("true_cell_type", "true_cell_id", "is_ghost", "gene")
                       if c in prov.columns]
            tx = tx.merge(prov[["transcript_id", *keep]],
                              on="transcript_id", how="left")
    # 2D bundle writer only stores `true_cell_id` in provenance — recover
    # cell type by joining through `ground_truth/cells_synth.parquet`.
    if "true_cell_type" not in tx.columns and "true_cell_id" in tx.columns:
        cells_synth_path = bundle / "ground_truth" / "cells_synth.parquet"
        if cells_synth_path.exists():
            cs = pd.read_parquet(cells_synth_path)
            if "cell_type" in cs.columns and "cell_id" in cs.columns:
                tx = tx.merge(
                    cs[["cell_id", "cell_type"]].rename(
                        columns={"cell_id": "true_cell_id", "cell_type": "true_cell_type"}),
                    on="true_cell_id", how="left")
    if "true_cell_type" in tx.columns and "source_cell_type" not in tx.columns:
        tx = tx.rename(columns={"true_cell_type": "source_cell_type"})
    if "feature_name" in tx.columns and "gene" not in tx.columns:
        tx = tx.rename(columns={"feature_name": "gene"})
    if "is_leaked" not in tx.columns:
        tx["is_leaked"] = pd.array([False] * len(tx), dtype="boolean")

    cfg = SimulationConfig.from_yaml(stpuppeteer_config)
    stats = compute_emission_stats(tx, cfg)
    print_emission_summary(stats)
    out_path = write_emission_diagnostics(tx, cfg, Path(diag_dir))
    print(f"[emission-diag] wrote {out_path}")


def _diagnostics_model(args: argparse.Namespace) -> None:
    """`xesim diagnostics model MODEL_DIR --bundle BUNDLE`."""
    from pathlib import Path
    from .diagnostics import fit_model_diagnostics
    out_dir = Path(args.out) if args.out else Path(args.model) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diagnostics model] writing → {out_dir}")
    written = fit_model_diagnostics(
        model_dir=args.model, bundle_path=args.bundle, out_dir=out_dir)
    for p in written:
        print(f"  {p}")


def _diagnostics_explain(args: argparse.Namespace) -> None:
    """`xesim diagnostics explain SYNTH --bundle BUNDLE --model MODEL`."""
    from pathlib import Path
    from .diagnostics import explain_diagnostics
    out_dir = Path(args.out) if args.out else Path(args.synth) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diagnostics explain] writing → {out_dir}")
    written = explain_diagnostics(
        bundle_path=args.bundle, synth_dir=args.synth,
        model_dir=args.model, out_dir=out_dir)
    for p in written:
        print(f"  {p}")


def _diagnostics_priors(args: argparse.Namespace) -> None:
    """`xesim diagnostics priors PRIORS_FILE`."""
    from pathlib import Path
    from .diagnostics import fit_priors_diagnostics
    priors_path = Path(args.priors)
    out_dir = Path(args.out) if args.out else priors_path.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diagnostics priors] writing → {out_dir}")
    written = fit_priors_diagnostics(priors_path, out_dir)
    for p in written:
        print(f"  {p}")


def _inspect_bundle(args: argparse.Namespace) -> None:
    info = inspect_bundle(args.bundle)
    print(json.dumps(info, indent=2, default=str))


def _inspect_model(args: argparse.Namespace) -> None:
    m = XesimModel.load(args.model)
    out = {
        "manifest": m.manifest,
        "type_names": m.type_names,
        "n_types": m.n_type_one_hot,
        "latent_dim": m.latent_dim,
    }
    print(json.dumps(out, indent=2, default=str))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xesim",
        description=("Plan3 procedural-mechanistic Xenium simulator. "
                       "Verbs: fit-model, fit-priors, explain, generate, "
                       "inspect-bundle, inspect-model."),
    )

    # Global --diagnostic flag. Each subcommand handler that produces an
    # output directory writes diagnostic plots / panels next to the primary
    # output by default. Pass `--no-diagnostic` to disable, or
    # `--diagnostic DIR` to override the location. Uses the parent-parser
    # pattern so the flag appears on every subcommand (argparse subparsers
    # don't inherit top-level options at parse time).
    diag_parent = argparse.ArgumentParser(add_help=False)
    diag_parent.add_argument(
        "--diagnostic", nargs="?", const="auto", default="auto", metavar="DIR",
        help="write diagnostic plots alongside the primary output (default: "
             "on, into <OUT>/diagnostics/). Pass an explicit DIR to override "
             "the location. Supported by fit-model, fit-priors, and explain.")
    diag_parent.add_argument(
        "--no-diagnostic", dest="diagnostic", action="store_const", const=None,
        help="disable the default diagnostic output.")

    sub = p.add_subparsers(dest="command", required=True)

    # fit-model
    fit = sub.add_parser("fit-model", parents=[diag_parent],
                           help="Fit priors + train renderer from a Xenium bundle")
    fit.add_argument("bundle", help="path to a Xenium bundle (.zip or extracted dir)")
    fit.add_argument("--annotations", required=True,
                       help="cell-type annotation CSV (cell_id, merged_annotation)")
    fit.add_argument("--out", required=True, help="output MODEL_DIR")
    fit.add_argument("--num-crops", type=int, default=128)
    fit.add_argument("--crop-size-um", type=float, default=64.0)
    fit.add_argument("--crop-selection", default="density",
                       choices=["density", "spread", "random", "grid",
                                  "stratified"],
                       help="canonical-crop sampling strategy. density "
                       "(default): high-cell-density windows, biased toward "
                       "dense epithelium/islets. spread: uniform across "
                       "bundle (was v22 default). random/grid: alternatives. "
                       "stratified: k-means on per-window cell-type "
                       "composition, per-cluster quota = n_c^alpha (see "
                       "--stratified-alpha). Needs --annotations.")
    fit.add_argument("--stratified-alpha", type=float, default=0.5,
                       help="cluster-size exponent for stratified selection: "
                       "1.0=proportional (≈spread), 0.0=uniform-per-cluster, "
                       "0.5=moderate rare-upsampling (matches WSI foundation-"
                       "model standard practice).")
    fit.add_argument("--stratified-within-pick", default="centroid",
                       choices=["centroid", "density"],
                       help="within-cluster pick rule for stratified selection. "
                       "centroid (default): cluster's compositional medoid (k-medoid-"
                       "ish). density: cell with highest local density in the "
                       "cluster, then density × spatial-FPS — gives the renderer "
                       "info-rich crops while preserving compositional coverage.")
    fit.add_argument("--steps", type=int, default=8000,
                       help="renderer training steps")
    fit.add_argument("--seed", type=int, default=1)
    fit.add_argument("--device", default=None)
    # Transcript / cellAdmix flow (sister package, default on):
    # by default xeSim runs cellAdmix NMF on the bundle so transcript-
    # based features (per-cell factor scores, tx-classification, etc.) are
    # available. Pass --celladmix-run to reuse an already-fit cellAdmix
    # output; pass --no-transcripts to skip entirely (renderer-only model).
    fit.add_argument("--with-transcripts", dest="with_transcripts",
                       action="store_true", default=True,
                       help="run cellAdmix NMF on the bundle if not already "
                            "fit (default: on).")
    fit.add_argument("--no-transcripts", dest="with_transcripts",
                       action="store_false",
                       help="skip the cellAdmix step; transcript-based features "
                            "will be unavailable in the resulting model.")
    fit.add_argument("--celladmix-run", default=None,
                       help="path to an existing cellAdmix run directory "
                            "(usually <BUNDLE_PARENT>/_xesim_celladmix/runs/...). "
                            "If supplied, xeSim reuses this fit instead of "
                            "running cellAdmix.")
    fit.set_defaults(func=_fit_model)

    # fit-priors — standalone 3D nucleus prior fit (from plan3d)
    fp = sub.add_parser("fit-priors", parents=[diag_parent],
                          help=("Fit only per-cell-type 3D nucleus shape priors "
                                "(plan3d ellipsoid model; ~5s on a 100k-cell bundle)."))
    fp.add_argument("bundle",
                      help="path to a Xenium bundle (.zip or extracted dir)")
    fp.add_argument("--annotations", required=True,
                      help="cell-type annotation CSV (cell_id, merged_annotation "
                           "or specify --type-col)")
    fp.add_argument("--out", required=True,
                      help="output path. Either a directory (writes "
                           "nucleus_priors.json inside) or a .json file.")
    fp.add_argument("--type-col", default="merged_annotation",
                      help="column name in annotation CSV holding the cell-type "
                           "label (default: merged_annotation)")
    fp.add_argument("--min-per-type", type=int, default=30,
                      help="skip cell types with fewer than this many cells")
    fp.add_argument("--n-iter", type=int, default=80,
                      help="gradient-descent iterations per cell type")
    fp.add_argument("--n-samples", type=int, default=4000,
                      help="Monte-Carlo samples per fit iteration")
    fp.add_argument("--seed", type=int, default=0)
    fp.set_defaults(func=_fit_priors)

    # explain — unified Task-1 verb
    exp = sub.add_parser("explain", parents=[diag_parent],
                          help="Render real Xenium tiles/regions/bundles through "
                               "the fitted renderer (Task-1).")
    exp.add_argument("bundle",
                       help="path to a real Xenium bundle (.zip or extracted dir)")
    exp.add_argument("--model", required=True, help="MODEL_DIR")
    exp.add_argument("--out", required=True, help="output directory")

    # Scope (mutually exclusive)
    scope = exp.add_mutually_exclusive_group()
    scope.add_argument("--tile", type=_parse_tile, default=None,
                         help="single tile centered at 'x,y' (µm)")
    scope.add_argument("--region", type=_parse_region, default=None,
                         help="rectangular region 'xmin,ymin,xmax,ymax' (µm)")
    scope.add_argument("--whole-bundle", dest="whole_bundle",
                         action="store_true",
                         help="2.5D only: tile-and-stitch over the bundle's "
                              "cell-vertex extent. 2D ignores this (the default "
                              "no-flag case already means whole-bundle).")
    # (neither flag) → whole bundle FOV (2D), required-error (2.5D)

    # 2.5D stitch parameters
    exp.add_argument("--stitch-tile-um", type=float, default=300.0,
                       help="2.5D stitch: tile size in µm (default 300)")
    exp.add_argument("--stitch-overlap-um", type=float, default=50.0,
                       help="2.5D stitch: tile overlap in µm (default 50)")
    exp.add_argument("--compile", dest="compile_renderer", action="store_true",
                       help="torch.compile(renderer) per worker (~1.5-2x on conv-heavy "
                            "graphs after warmup; first few tiles slower)")

    # Augmentation
    exp.add_argument("--add-ghosts", dest="add_ghosts",
                       action="store_true", default=True,
                       help="add density-calibrated ghost cells (default: on)")
    exp.add_argument("--no-ghosts", dest="add_ghosts", action="store_false")
    exp.add_argument("--add-transcript-proposed", action="store_true",
                       help="recover missed cells from orphan transcripts "
                            "(requires fit-model --with-transcripts)")
    exp.add_argument("--stamp-transcripts", action="store_true",
                       help="stamp each cell's provenance with cellAdmix-implied "
                            "type + uncertainty (requires fit-model --with-transcripts)")
    exp.add_argument("--emission-backend", dest="emission_backend",
                       choices=["legacy", "stpuppeteer"], default="legacy",
                       help="molecular-emission backend. 'legacy' (default): "
                            "cellAdmix-fit NMF priors + LogNormal-Poisson counts "
                            "+ compartment-aware placement. 'stpuppeteer': "
                            "STpuppeteer LMC counts + per-cell-independent halo "
                            "leakage, configured via --stpuppeteer-config. "
                            "Phase 0: stpuppeteer raises NotImplementedError.")
    exp.add_argument("--stpuppeteer-config", dest="stpuppeteer_config",
                       default=None,
                       help="path to STpuppeteer YAML config (required when "
                            "--emission-backend=stpuppeteer)")
    exp.add_argument("--annotation", default=None,
                       help="cell-type annotation CSV (overrides model's)")

    # Output
    exp.add_argument("--format", choices=["scene", "bundle"], default=None,
                       help="output writer. Default: 'scene' for --tile, "
                            "'bundle' for --region and whole-bundle scope.")
    exp.add_argument("--overwrite", action="store_true",
                       help="replace OUT if it already exists (bundle format)")

    # Calibration
    exp.add_argument("--noise-fraction", default=None,
                       help="target ghost-molecule fraction. Default: AUTO-TUNED "
                            "from the real bundle's UNASSIGNED rate.")
    exp.add_argument("--ghost-count-scale", type=float, default=1.0,
                       help="multiplier on calibrated ghost count (default 1.0)")
    exp.add_argument("--tx-rate-scale", type=float, default=None,
                       help="multiplier on every cell's per-type negbin mean. "
                            "Default: priors['tx_rate_scale_default'] if "
                            "present, else 1.0. Pancreas v21 ~1.18 closes "
                            "the per-tile boundary-loss gap. Override here.")
    exp.add_argument("--intensity-calibration",
                       choices=["off", "scale", "match2", "histmatch",
                                "lut_native", "lut_zerofloor"],
                       default="off",
                       help="per-channel brightness handling. "
                            "off: per-render p99.5 → 4095 (viewer-friendly). "
                            "scale: synth_p99.5 → real_bundle_p99.5. "
                            "match2: 2-anchor (p50, p99.5). "
                            "histmatch: full quantile remap.")

    # Misc
    exp.add_argument("--inference-tile-px", type=int, default=None,
                       help="Override inference tile size in pixels. The "
                            "model is fully-convolutional with no norm layers, "
                            "so larger tiles produce identical per-pixel output "
                            "but fewer boundary artifacts and lower stitching "
                            "overhead. Defaults to the model's training tile size.")
    import os as _os
    _default_workers = max(1, min(4, (_os.cpu_count() or 1) // 2))
    exp.add_argument("--num-workers", type=int, default=_default_workers,
                       help="Multiprocess workers for per-tile rendering. The "
                            "per-tile pipeline is CPU-bound (struct channels, "
                            "distance transforms); each worker has its own "
                            "model + CUDA context for true parallel GPU use. "
                            f"Default {_default_workers} on this box "
                            "(min(4, ncpu/2)). Pass 1 for serial.")
    exp.add_argument("--scene-mode", choices=["2d", "2.5d"], default="2d",
                       help="Scene composition mode. '2d' (default): the "
                            "current 2D pipeline (anchors + ghosts + "
                            "tx-proposed, single-plane morphology). '2.5d': "
                            "Tilted-Template SDF — per-cell 3D bodies with "
                            "neighbor-coherent tilt, multi-z DAPI, 3D molecule "
                            "ground truth. Requires --tile or --region.")
    exp.add_argument("--scene-first", action="store_true",
                       help="Compose the whole region into one MechanisticScene "
                            "before rendering, instead of per-tile composition. "
                            "Latents sampled once per cell (no per-tile resampling "
                            "disconnects across seams). Implies a single big "
                            "explain_region call; uses --inference-tile-px for "
                            "memory-bounded rendering.")
    exp.add_argument("--bridge-tile", action="store_true",
                       help="With --scene-first, render via TWO overlapping tile "
                            "grids offset by half a tile. Each output pixel is "
                            "sourced primarily from a tile where it's interior. "
                            "Kills near-edge conv-padding seam artifacts at ~2x "
                            "render cost.")
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--device", default=None)
    exp.set_defaults(func=_explain)

    # generate
    gen = sub.add_parser("generate",
                          help="Sample new scenes (forward or slice-guided) and render them")
    gen.add_argument("--model", required=True, help="MODEL_DIR")
    gen.add_argument("--out", required=True, help="output scene dir")
    gen.add_argument("--num-scenes", type=int, default=1)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--num-cells", type=int, default=70,
                       help="target cell count per scene (forward mode)")
    gen.add_argument("--admixture-rate", type=float, default=1.0,
                       help="scale per-type admixture-factor weight. 1.0=observed, "
                            "0.0=pure-biology cells, >1.0 injects extra admixture")
    gen.add_argument("--guide-bundle", default=None,
                       help="canonical-crop dir to draw the slice-guide tile from")
    gen.add_argument("--guide-crop", default=None,
                       help="crop_id within --guide-bundle to use as guide")
    gen.add_argument("--device", default=None)
    gen.set_defaults(func=_generate)

    # re-emit-molecules — re-sample transcripts on an existing synth bundle.
    re = sub.add_parser(
        "re-emit-molecules", parents=[diag_parent],
        help="Re-sample transcripts on an existing synth bundle without "
             "re-rendering. Copies SRC to --out and rewrites only "
             "transcripts.parquet + ground_truth/molecule_provenance.parquet "
             "against the supplied STpuppeteer config. --out must not exist.")
    re.add_argument("bundle", help="path to existing xeSim synth bundle (SRC)")
    re.add_argument("--out", required=True,
                       help="path to write the new bundle. Must not exist.")
    re.add_argument("--stpuppeteer-config", dest="stpuppeteer_config",
                       required=True,
                       help="path to STpuppeteer YAML config")
    re.add_argument("--seed", type=int, default=0,
                       help="RNG seed (same config + seed → identical output)")
    re.set_defaults(func=_reemit_molecules)

    # diagnostics — emit panels against artifacts that already exist on disk.
    diag = sub.add_parser(
        "diagnostics",
        help="Emit diagnostic plots against existing artifacts (no re-render).")
    diag_sub = diag.add_subparsers(dest="diag_mode", required=True)

    dm = diag_sub.add_parser(
        "model",
        help="Diagnostics for a fitted model dir (training loss, nucleus priors).")
    dm.add_argument("model", help="path to a fitted MODEL_DIR")
    dm.add_argument("--bundle", required=True, help="real Xenium bundle the model was fit from")
    dm.add_argument("--out", default=None,
                       help="output dir (default: MODEL_DIR/diagnostics/)")
    dm.set_defaults(func=_diagnostics_model)

    de = diag_sub.add_parser(
        "explain",
        help="Diagnostics for a synth bundle vs the real bundle "
              "(A1-A4 morphology grids, B-D population panels).")
    de.add_argument("synth", help="path to a synth bundle (xesim explain output)")
    de.add_argument("--bundle", required=True, help="real Xenium bundle to compare against")
    de.add_argument("--model", required=True, help="fitted MODEL_DIR used to render the synth bundle")
    de.add_argument("--out", default=None,
                       help="output dir (default: SYNTH/diagnostics/)")
    de.set_defaults(func=_diagnostics_explain)

    dp = diag_sub.add_parser(
        "priors",
        help="Diagnostics for a standalone 3D nucleus priors file.")
    dp.add_argument("priors", help="path to nucleus_priors.json")
    dp.add_argument("--out", default=None,
                       help="output dir (default: alongside the priors file)")
    dp.set_defaults(func=_diagnostics_priors)

    # inspect-bundle
    ib = sub.add_parser("inspect-bundle", help="Print summary info for a Xenium bundle")
    ib.add_argument("bundle")
    ib.set_defaults(func=_inspect_bundle)

    # inspect-model
    im = sub.add_parser("inspect-model", help="Print summary info for a fitted model dir")
    im.add_argument("model")
    im.set_defaults(func=_inspect_model)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
