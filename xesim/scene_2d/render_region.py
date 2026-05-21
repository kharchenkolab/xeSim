"""Single shared region render+calibrate path.

Both the production CLI (``xesim explain``) and the diagnostics (A3 3-/4-panel)
must turn a bundle region into morphology pixels the *same* way, or they
diverge and the diagnostic stops reflecting what production writes. This
module is that single entry point:

  - :func:`resolve_render_calibration` — the one place that assembles the
    calibration config (display LUT + per-channel sensor noise, plus any
    histmatch/scale targets). Single source of truth.
  - :func:`render_region` — build_scene (raw renderer float) + the writer's
    calibration → returns BOTH the raw float and the calibrated uint16, so a
    caller can show "renderer output" and "what the bundle will contain"
    without reimplementing either step.

``build_scene`` already returns the raw renderer float (calibration happens at
the writer boundary), so the float here is identical to production's. The
calibration delegates to :func:`calibrate_to_uint16` — the exact function the
bundle writer uses — so the uint16 here matches the written morphology for the
same calibration config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def resolve_render_calibration(
    model,
    bundle_path: str | Path | None,
    *,
    model_dir: str | Path | None = None,
    mode: str = "off",
    disable_noise: bool | None = None,
) -> dict[str, Any]:
    """Assemble the per-channel calibration config used to map renderer float
    → uint16. The single source of truth shared by the CLI writer and the
    diagnostics.

    Returns a dict with ``channel_names``, ``mode``, ``target_stats``,
    ``target_quantiles`` and ``display_lut`` (the LUT carries ``noise_stats``
    measured from the real bundle unless noise is disabled). When ``model_dir``
    is None the display LUT is skipped (callers that only need the histmatch/
    scale targets, e.g. the single-tile path).

    ``mode`` follows the ``--intensity-calibration`` choices: ``off`` (→
    lut_native when a display LUT is available), ``scale``/``match2`` (tune
    p50/p99.5 targets from the real bundle), ``histmatch`` (tune per-channel
    quantile tables), or ``lut_native``/``lut_zerofloor``.
    """
    from .bundle_writer import auto_tune_intensity_stats
    from .intensity import (auto_tune_intensity_quantiles, calibrate_noise_stats)
    from .render_tile import load_model_display_lut

    channel_names = list(model.manifest.get("channel_names") or [])
    target_stats = None
    target_quantiles = None
    if bundle_path:
        if mode in ("scale", "match2"):
            target_stats = auto_tune_intensity_stats(str(bundle_path), channel_names)
        elif mode == "histmatch":
            target_quantiles = auto_tune_intensity_quantiles(str(bundle_path),
                                                             channel_names)

    display_lut = load_model_display_lut(str(model_dir)) if model_dir else None
    if display_lut is not None and bundle_path:
        if disable_noise is None:
            disable_noise = os.environ.get("XESIM_DISABLE_NOISE", "").strip() == "1"
        try:
            display_lut["noise_stats"] = ([] if disable_noise
                                          else calibrate_noise_stats(
                                              str(bundle_path), display_lut=display_lut))
        except Exception:
            display_lut.setdefault("noise_stats", [])

    return {
        "channel_names": channel_names,
        "mode": mode,
        "target_stats": target_stats,
        "target_quantiles": target_quantiles,
        "display_lut": display_lut,
    }


def calibrate_float_to_uint16(float_image: np.ndarray,
                              calibration: dict[str, Any]) -> np.ndarray:
    """Apply a resolved calibration config to a renderer float image → uint16,
    via the exact function the bundle writer uses."""
    from .intensity import calibrate_to_uint16
    return calibrate_to_uint16(
        float_image,
        channel_names=calibration.get("channel_names"),
        target_stats=calibration.get("target_stats"),
        target_quantiles=calibration.get("target_quantiles"),
        mode=calibration.get("mode", "off"),
        display_lut=calibration.get("display_lut"),
    )


def render_region(
    model,
    bundle_path: str | Path,
    bounds: tuple[float, float, float, float],
    *,
    calibration: dict[str, Any] | None = None,
    add_ghosts: bool = True,
    add_transcript_proposed: bool = False,
    noise_fraction: float = 0.23,
    ghost_count_scale: float = 1.0,
    inference_tile_px: int | None = None,
    annotation_path: str | None = None,
    num_workers: int = 1,
    model_path: str | None = None,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Render a bundle region exactly as production does.

    Runs ``build_scene`` (non-streaming → raw renderer float) and, when a
    ``calibration`` config is given, the writer's calibration → uint16. Returns
    both so callers can show the renderer output and the bundle-faithful
    morphology without reimplementing either.

    Returns a dict with ``float_image`` ((C,H,W) float32, raw renderer output),
    ``uint16_image`` ((C,H,W) uint16 calibrated, or None when no calibration),
    plus ``scenes``, ``geom_stash``, ``n_anchors``, ``n_ghosts``,
    ``n_transcripts``, ``scene_bounds_um``, ``pixel_size_um``.
    """
    from .scene_pipeline import build_scene

    rng = rng or np.random.default_rng(0)
    res = build_scene(
        model, str(bundle_path),
        scene_bounds_um=tuple(float(v) for v in bounds),
        add_ghosts=add_ghosts,
        add_transcript_proposed=add_transcript_proposed,
        noise_fraction=noise_fraction,
        ghost_count_scale=ghost_count_scale,
        annotation_path=annotation_path,
        inference_tile_px=inference_tile_px,
        num_workers=num_workers,
        model_path=model_path,
        device=device,
        rng=rng,
        progress=progress,
    )
    float_image = np.asarray(res["stitched_image"], dtype=np.float32)
    uint16_image = (calibrate_float_to_uint16(float_image, calibration)
                    if calibration is not None else None)
    return {
        "float_image": float_image,
        "uint16_image": uint16_image,
        "scenes": res["scenes"],
        "geom_stash": res.get("geom_stash"),
        "n_anchors": int(res["n_anchor_cells"]),
        "n_ghosts": int(res["n_ghost_cells"]),
        "n_transcripts": int(res["n_transcripts"]),
        "n_transcript_proposed": int(res.get("n_transcript_proposed", 0)),
        "scene_bounds_um": res["scene_bounds_um"],
        "pixel_size_um": res["pixel_size_um"],
    }


__all__ = ["resolve_render_calibration", "calibrate_float_to_uint16", "render_region"]
