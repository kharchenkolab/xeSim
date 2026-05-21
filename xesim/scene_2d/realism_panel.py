"""Shared utility for real-vs-synth visualization panels.

ALL realism diagnostic scripts should use these helpers — never reinvent
per-script `composite_rgb` or calibration logic. This module wraps the
CLI's exact render path (`explain_region` / `build_scene`) and adds a
single canonical compositing function with full 4-channel hue mixing.

Why: per-script ad-hoc visualization code obscures whether discrepancies
between real and synth come from the renderer or from the diagnostic's
own display logic. See feedback_unified_render_path.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# Canonical channel hue palette (matches xesim_v21_model). DAPI=red,
# membrane=green, 18S=blue, aSMA=magenta. Weight column gates how much
# each channel contributes to the composite (aSMA/18S are dimmed so they
# don't dominate dense fluorescence regions).
CHANNEL_STYLE: dict[str, tuple[tuple[float, float, float], float]] = {
    "DAPI":                       ((1.00, 0.20, 0.20), 1.0),
    "ATP1A1/CD45/E-Cadherin":     ((0.20, 1.00, 0.30), 1.0),
    "18S":                        ((0.30, 0.50, 1.00), 0.55),
    "alphaSMA/Vimentin":          ((0.95, 0.40, 0.85), 0.45),
}


def composite_rgb(arr_nchw: np.ndarray,
                    channel_names: list[str] | None = None,
                    *, p99_norm: bool = True) -> np.ndarray:
    """Composite a (C, H, W) multichannel float array into an (H, W, 3)
    RGB display using the canonical hue palette.

    Always uses ALL channels (don't drop channels — that hides bugs).
    Joint p99 normalization on the composite preserves inter-channel
    intensity ratios. Inputs are clipped to [0, 1] before mixing.

    Parameters
    ----------
    arr_nchw : (C, H, W) float
    channel_names : list of channel names; if None, uses default v21 order
    p99_norm : if True, scale composite by 1/p99 so display fills [0, 1]
        without clipping
    """
    if channel_names is None:
        channel_names = list(CHANNEL_STYLE.keys())
    rgb = np.zeros((arr_nchw.shape[1], arr_nchw.shape[2], 3), dtype=np.float32)
    for ci in range(min(len(channel_names), arr_nchw.shape[0])):
        hue, w = CHANNEL_STYLE.get(channel_names[ci], ((0.8, 0.8, 0.8), 0.5))
        intensity = np.clip(arr_nchw[ci], 0, 1) * w
        for k in range(3):
            rgb[..., k] += intensity * hue[k]
    if p99_norm:
        p = float(np.percentile(rgb, 99))
        if p > 0:
            rgb = rgb / max(p, 1.0)
    return np.clip(rgb, 0, 1)


def real_image_for_bounds(model, bundle_path: str | Path,
                            bounds: tuple[float, float, float, float],
                            ) -> np.ndarray:
    """Read the real morphology image for a region, applying the model's
    display LUT (so intensities match what the model was trained against)."""
    from .render_tile import real_tile_image, load_model_display_lut
    pix = float(model.pixel_size)
    # Resolve the model's checkpoint dir to find its display LUT.
    # XesimModel stores it on .paths.root.
    model_root = None
    if hasattr(model, "paths") and hasattr(model.paths, "root"):
        model_root = str(model.paths.root)
    lut = load_model_display_lut(model_root) if model_root else None
    img = real_tile_image(str(bundle_path),
                            tile_bounds_um=(bounds[0], bounds[2], bounds[1], bounds[3]),
                            pixel_size_um=pix, display_lut=lut)
    return img


def render_region_via_cli_path(
    model,
    bundle_path: str | Path,
    bounds: tuple[float, float, float, float],
    *,
    add_ghosts: bool = False,
    add_transcript_proposed: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Render a region the same way the CLI's `explain --region` does.

    Returns:
        {"image": (C, H, W) float32, "n_anchors": int,
         "n_proposed": int, "scenes": ..., "bounds_um": ...}

    Thin wrapper over the shared :func:`render_region` (calibration=None →
    raw renderer float) so the diagnostic renders through the exact same path
    production uses, instead of a parallel build_scene call that can drift.
    """
    from .render_region import render_region
    res = render_region(
        model, bundle_path, bounds, calibration=None,
        add_ghosts=add_ghosts, add_transcript_proposed=add_transcript_proposed,
        rng=rng)
    return {
        "image": res["float_image"],
        "n_anchors": res["n_anchors"],
        "n_proposed": res["n_transcript_proposed"],
        "scenes": res["scenes"],
        "bounds_um": res["scene_bounds_um"],
    }


def panel_real_vs_synth(
    model,
    bundle_path: str | Path,
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
    *,
    title: str | None = None,
    add_transcript_proposed: bool = False,
    add_ghosts: bool = False,
    rng: np.random.Generator | None = None,
    figsize_per_panel: tuple[float, float] = (7.0, 7.2),
) -> Path:
    """Render a clean 2-panel (real | synth) figure at the given bounds
    using the CLI render path + canonical composite. Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = rng or np.random.default_rng(0)
    synth_res = render_region_via_cli_path(
        model, bundle_path, bounds,
        add_ghosts=add_ghosts,
        add_transcript_proposed=add_transcript_proposed, rng=rng)
    synth = synth_res["image"]
    sH, sW = synth.shape[1:]
    n_anchors = synth_res["n_anchors"]; n_prop = synth_res["n_proposed"]

    real = real_image_for_bounds(model, bundle_path, bounds)
    if real is None:
        real = np.zeros_like(synth)
    real = real[:synth.shape[0], :sH, :sW]

    ch_names = model._channel_names if hasattr(model, "_channel_names") else None
    rgb_real = composite_rgb(real, ch_names)
    rgb_synth = composite_rgb(synth, ch_names)

    fig, axes = plt.subplots(1, 2,
                              figsize=(2 * figsize_per_panel[0], figsize_per_panel[1]),
                              facecolor='white',
                              gridspec_kw={'wspace': 0.03})
    axes[0].imshow(rgb_real)
    axes[0].set_title(f'real ({bounds[2]-bounds[0]:.0f}×{bounds[3]-bounds[1]:.0f} µm)',
                        fontsize=10, fontweight='bold')
    axes[0].axis('off')
    syn_title = f'synth ({n_anchors} anchors'
    if n_prop > 0: syn_title += f' + {n_prop} tx-proposed'
    syn_title += ")"
    axes[1].imshow(rgb_synth)
    axes[1].set_title(syn_title, fontsize=10, fontweight='bold')
    axes[1].axis('off')
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.0)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005)
    out = Path(out_path)
    plt.savefig(out, facecolor='white', dpi=130,
                 bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return out


__all__ = ["CHANNEL_STYLE", "composite_rgb", "real_image_for_bounds",
            "render_region_via_cli_path", "panel_real_vs_synth"]
