"""Render a Scene2D through the renderer (single tile, multi-channel-aware)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import Scene2D


def render_tile(
    scene_2d: Scene2D,
    model,                       # XesimModel
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Run the renderer on the scene's mechanistic scene.

    Returns a (N, H, W) float32 array where N = model's n_channels.
    """
    render = model.render(scene_2d.mech_scene, seed=seed)
    return render


def real_tile_image(
    bundle_path: str,
    tile_bounds_um: tuple[float, float, float, float],
    pixel_size_um: float,
    *,
    display_lut: dict | None = None,
    target_shape: tuple[int, int] | None = None,
    n_channels: int | None = None,
) -> np.ndarray | None:
    """Read the real morphology image for given tile bounds.

    Returns (n_channels, H, W) float32 (in [0, 1] if display_lut is given).
    n_channels = all available if None.
    """
    from ..images import ImageStackReader
    from ..models import CropBox
    from ..xenium import resolve_bundle
    from ..qc import normalize_images_with_lut

    try:
        bundle = resolve_bundle(bundle_path)
        if not bundle.morphology_focus_paths:
            return None
        xmin, xmax, ymin, ymax = tile_bounds_um
        # tile_bounds_um are GLOBAL coords. A region/whole bundle's morphology
        # image has its pixel (0,0) at the bundle's own origin — (0,0) for a
        # full bundle, but the region's (xmin,ymin) for a region explain
        # bundle. Subtract that origin so global bounds map to the right
        # pixels (without this, region bundles crop shifted / out-of-bounds).
        ox, oy = bundle_origin_um(bundle_path)
        crop = CropBox(xmin=xmin - ox, xmax=xmax - ox,
                        ymin=ymin - oy, ymax=ymax - oy,
                        crop_id="real_tile")
        reader = ImageStackReader(bundle.morphology_focus_paths, bundle.pixel_size)
        img = np.asarray(reader.read(crop), dtype=np.float32)
        if img.size == 0:
            return None
        if display_lut is not None:
            img = normalize_images_with_lut(img, display_lut)
        if n_channels is not None:
            img = img[:n_channels]
        if target_shape is not None:
            h_t, w_t = int(target_shape[0]), int(target_shape[1])
        else:
            h_t = int(round((ymax - ymin) / pixel_size_um))
            w_t = int(round((xmax - xmin) / pixel_size_um))
        h, w = img.shape[1], img.shape[2]
        if h != h_t or w != w_t:
            padded = np.zeros((img.shape[0], h_t, w_t), dtype=np.float32)
            padded[:, :min(h, h_t), :min(w, w_t)] = img[:, :min(h, h_t), :min(w, w_t)]
            img = padded
        return img
    except Exception:
        return None


def bundle_origin_um(bundle_path: str | Path) -> tuple[float, float]:
    """Global-µm origin (xmin, ymin) of a bundle's morphology image.

    A whole-bundle render has origin (0, 0); a region explain bundle's
    morphology covers only the region, so its pixel (0,0) sits at the
    region's (xmin, ymin). Read from experiment.xenium tile_bounds_um.
    Returns (0.0, 0.0) when absent (treat as full-bundle / global).
    """
    try:
        m = json.loads((Path(bundle_path) / "experiment.xenium").read_text())
        tb = m.get("tile_bounds_um")
        if tb and len(tb) >= 2:
            return float(tb[0]), float(tb[1])
    except Exception:
        pass
    return 0.0, 0.0


def bundle_bounds_um(bundle_path: str | Path) -> tuple[float, float, float, float] | None:
    """Global-µm bounds (xmin, ymin, xmax, ymax) of a bundle's rendered
    extent, from experiment.xenium tile_bounds_um. None if unavailable.
    Used by diagnostics to restrict example selection to what was actually
    rendered (so region bundles don't pick out-of-region examples)."""
    try:
        m = json.loads((Path(bundle_path) / "experiment.xenium").read_text())
        tb = m.get("tile_bounds_um")
        if tb and len(tb) == 4:
            return tuple(float(v) for v in tb)  # type: ignore[return-value]
    except Exception:
        pass
    return None


def load_model_display_lut(model_dir: str | Path) -> dict | None:
    """Load the display LUT from a model's canonical/manifest.json (or
    via symlink). Resolves symlinks so model dirs that share a canonical
    dir still work.
    """
    p = Path(model_dir) / "canonical" / "manifest.json"
    try:
        p = p.resolve()
    except Exception:
        pass
    if not p.exists():
        return None
    try:
        mf = json.loads(p.read_text())
        return mf.get("image_normalization", {}).get("lut")
    except Exception:
        return None


__all__ = ["render_tile", "real_tile_image", "load_model_display_lut",
           "bundle_origin_um", "bundle_bounds_um"]
