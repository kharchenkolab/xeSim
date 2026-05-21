"""3-panel real-vs-render-vs-bundle diagnostic for a region.

Three paths side by side at the same global-µm bounds, in the same
LUT-normalized composite space, to localize where realism breaks:

  panel 1: real     — real_tile_image (real bundle morphology_focus)
  panel 2: m.render — model.render direct output (build_scene single tile)
  panel 3: bundle   — saved synth bundle morphology_focus (lut_native + noise)

  panel 2≈3 ⇒ bundle writer is round-trip faithful
  panel 1≈2 ⇒ renderer output is realistic
  panel 1≈3 ⇒ end-to-end synth is realistic

This is the A3 diagnostic. It is invoked as a SUBPROCESS by
``xesim.diagnostics._plot_tile_3panels`` (``python -m
xesim.scene_2d.region_3panel``) so the model load + CUDA context live in
their own process and are torn down after the panel — keeping the GPU
dependency out of the parent diagnostics process. It can also be called
in-process via :func:`render_region_3panel`, or run directly as a CLI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_bundle_morphology(synth_dir: Path, channel_names, bounds, psz,
                            display_lut) -> np.ndarray:
    """Read the saved synth bundle's morphology_focus and LUT-normalize into
    the same [0,1] space as real_image_for_bounds.

    ``bounds`` are GLOBAL µm; a region synth bundle's image starts at the
    region origin, so map global → local pixels by subtracting the bundle
    origin (0,0 for a full bundle) — the single origin source of truth.
    """
    import tifffile
    from .render_tile import bundle_origin_um

    paths = sorted((synth_dir / "morphology_focus").glob("*.ome.tif"))
    H = int(round((bounds[3] - bounds[1]) / psz))
    W = int(round((bounds[2] - bounds[0]) / psz))
    ox, oy = bundle_origin_um(synth_dir)
    y0 = int(round((bounds[1] - oy) / psz))
    x0 = int(round((bounds[0] - ox) / psz))
    lut_channels = (display_lut or {}).get("channels", []) if display_lut else []
    # Xenium-style OME: file 0 reads as (C, H, W) via cross-refs; older
    # layout: one single-channel file per channel.
    a0 = tifffile.imread(paths[0]).astype(np.float32)
    if a0.ndim == 3:
        planes = [a0[ci] for ci in range(min(len(channel_names), a0.shape[0]))]
    else:
        planes = [tifffile.imread(p).astype(np.float32)
                  for p in paths[:len(channel_names)]]
        planes = [a if a.ndim == 2 else a[a.shape[0] // 2] for a in planes]
    imgs = []
    for ci, img in enumerate(planes):
        ly0 = max(0, min(y0, img.shape[0]))
        lx0 = max(0, min(x0, img.shape[1]))
        crop = img[ly0:ly0 + H, lx0:lx0 + W]
        if ci < len(lut_channels):
            lo = float(lut_channels[ci].get("lo", 0.0))
            hi = float(lut_channels[ci].get("hi", 1.0))
            crop = (crop - lo) / max(hi - lo, 1e-6)
        else:
            p99 = float(np.percentile(crop[crop > 0], 99)) if (crop > 0).any() else 1.0
            if p99 > 0:
                crop = crop / p99
        imgs.append(np.clip(crop, 0, 1))
    return np.stack(imgs, axis=0)


def render_region_3panel(bundle_path: str | Path, synth_dir: str | Path,
                         model_dir: str | Path, out_path: str | Path,
                         region: tuple[float, float, float, float] | None = None,
                         ) -> Path:
    """Build the 3-panel real | m.render | saved-bundle figure for a region.

    ``region`` is global-µm (xmin, ymin, xmax, ymax); when None, defaults to
    the synth bundle's own extent (from its cells)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .realism_panel import (composite_rgb, real_image_for_bounds,
                                render_region_via_cli_path)
    from .render_tile import load_model_display_lut
    from ..model import XesimModel

    bundle_path = str(bundle_path); synth_dir = Path(synth_dir)
    model = XesimModel.load(str(model_dir))
    psz = float(model.pixel_size)
    ch_names = getattr(model, "_channel_names", None)
    lut = load_model_display_lut(str(model_dir))

    if region is not None:
        bounds = tuple(float(v) for v in region)
    else:
        import pandas as pd
        cells = pd.read_parquet(synth_dir / "cells.parquet")
        bounds = (float(cells.x_centroid.min() - 30),
                  float(cells.y_centroid.min() - 30),
                  float(cells.x_centroid.max() + 30),
                  float(cells.y_centroid.max() + 30))

    real = real_image_for_bounds(model, bundle_path, bounds)
    res = render_region_via_cli_path(model, bundle_path, bounds,
                                     add_ghosts=True, add_transcript_proposed=False)
    mrender = res["image"]; n_anchors = res["n_anchors"]
    bundle = _load_bundle_morphology(
        synth_dir, ch_names or ["DAPI", "ATP", "18S", "aSMA"], bounds, psz, lut)

    h = min(real.shape[1], mrender.shape[1], bundle.shape[1])
    w = min(real.shape[2], mrender.shape[2], bundle.shape[2])
    real, mrender, bundle = real[:, :h, :w], mrender[:, :h, :w], bundle[:, :h, :w]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), facecolor="white",
                             gridspec_kw={"wspace": 0.02})
    axes[0].imshow(composite_rgb(real, ch_names))
    axes[0].set_title(f"real\n{bounds[2]-bounds[0]:.0f}×{bounds[3]-bounds[1]:.0f} µm "
                      f"@ ({bounds[0]:.0f},{bounds[1]:.0f})",
                      fontsize=10, fontweight="bold")
    axes[1].imshow(composite_rgb(mrender, ch_names))
    axes[1].set_title(f"m.render (renderer float)\n{n_anchors} anchors",
                      fontsize=10, fontweight="bold")
    axes[2].imshow(composite_rgb(bundle, ch_names))
    axes[2].set_title("saved bundle (uint16 → LUT-norm)\n[lut_native + noise]",
                      fontsize=10, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    plt.subplots_adjust(left=0.003, right=0.997, top=0.94, bottom=0.005)
    plt.savefig(str(out_path), facecolor="white", dpi=120, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return Path(out_path)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="A3 region 3-panel diagnostic")
    ap.add_argument("--bundle", required=True, help="real Xenium bundle dir")
    ap.add_argument("--synth", required=True, help="saved synth bundle dir")
    ap.add_argument("--model", required=True, help="model dir (manifest.json)")
    ap.add_argument("--region", default=None, help="xmin,ymin,xmax,ymax µm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    region = (tuple(float(v) for v in args.region.split(","))
              if args.region else None)
    out = render_region_3panel(args.bundle, args.synth, args.model, args.out, region)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
