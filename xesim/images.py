from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import CropBox

try:  # pragma: no cover - optional dependency
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tifffile = None


def channel_names(paths: Sequence[Path]) -> tuple[str, ...]:
    """Return per-channel names (one per channel, in canonical order).

    Two layouts seen in the wild:
      1. Each ``morphology_focus_NNNN.ome.tif`` is single-channel, and the
         OME-XML in each file names that single channel. (Xenium synth
         bundles emitted by xesim use this layout.)
      2. The bundle-wide OME-XML lives in file 0 and lists all channels
         (one per file); each file is single-plane. (Standard Xenium real
         bundles.)

    Strategy: try to read a per-file Channel/Name; if every file
    contributes one and they are distinct, return that. Otherwise fall
    back to file 0's full OME-XML channel list. Final fallback: file
    stems.
    """
    if not paths:
        return tuple()
    import re
    import tifffile
    per_file: list[str | None] = []
    for path in paths:
        try:
            with tifffile.TiffFile(path) as tif:
                ome = tif.ome_metadata or ""
            ch = re.findall(r'<Channel[^>]*Name="([^"]*)"', ome)
            per_file.append(ch[0] if ch else None)
        except Exception:
            per_file.append(None)
    # Layout 1: every file has a name and they are distinct.
    if (all(n is not None for n in per_file)
            and len(set(per_file)) == len(per_file)):
        return tuple(n for n in per_file)
    # Layout 2: file 0 carries the bundle-wide channel list.
    try:
        with tifffile.TiffFile(paths[0]) as tif:
            ome = tif.ome_metadata or ""
        full = re.findall(r'<Channel[^>]*Name="([^"]*)"', ome)
        if len(full) >= len(paths):
            return tuple(full[:len(paths)])
        if full:
            return tuple(full)
    except Exception:
        pass
    # Fallback: file stems
    names: list[str] = []
    for path in paths:
        stem = path.stem
        if stem.endswith(".ome"):
            stem = stem[:-4]
        names.append(stem)
    return tuple(names)


# Process-wide cache of decompressed morphology planes. Keyed by
# (absolute path, mtime) so distinct files / re-saves don't alias.
# Bundle morphology TIFFs are JPEG-2000 compressed; decompressing a 64um
# tile costs ~0.5s wall-clock per channel. Caching the whole plane once
# per worker amortizes that across all tiles the worker handles. Memory
# cost: ~3.7GB for a typical 4-channel pancreas bundle.
_PLANE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def clear_plane_cache() -> None:
    """Drop all cached full planes. Useful in tests / long-running services."""
    _PLANE_CACHE.clear()


class OmeCropReader:
    """Read repeated crops from one OME-TIFF efficiently.

    Process-wide cache (``_PLANE_CACHE``) avoids re-decompressing the same
    plane on every crop. The first call decompresses and stores the full
    plane; subsequent calls (in the same process, against the same file)
    slice from memory. ~10x speedup at whole-bundle scale.
    """

    def __init__(self, path: Path, pixel_size: float):
        self.path = path
        self.pixel_size = pixel_size

    def read(self, crop: CropBox) -> np.ndarray:
        plane = self._ensure_full_plane()
        x0, x1, y0, y1 = crop_pixel_bounds(crop, self.pixel_size, shape=plane.shape)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Crop {crop.crop_id} is empty after pixel conversion")
        return plane[y0:y1, x0:x1].astype(np.float32, copy=False)

    def _ensure_full_plane(self) -> np.ndarray:
        import os
        path_str = str(self.path)
        try:
            mtime = int(os.path.getmtime(path_str))
        except OSError:
            mtime = 0
        key = (path_str, mtime)
        cached = _PLANE_CACHE.get(key)
        if cached is None:
            cached = _read_full_plane(self.path)
            _PLANE_CACHE[key] = cached
        return cached


class ImageStackReader:
    """Reusable crop reader for a stack of morphology-focus OME-TIFFs."""

    def __init__(self, paths: Sequence[Path], pixel_size: float):
        self.readers = [OmeCropReader(path, pixel_size) for path in paths]

    def read(self, crop: CropBox) -> np.ndarray:
        if not self.readers:
            return np.zeros((0, 0, 0), dtype=np.float32)
        channels = [reader.read(crop) for reader in self.readers]
        h = min(ch.shape[0] for ch in channels)
        w = min(ch.shape[1] for ch in channels)
        cropped = [ch[:h, :w] for ch in channels]
        return np.stack(cropped, axis=0).astype(np.float32, copy=False)


def crop_pixel_bounds(
    crop: CropBox,
    pixel_size: float,
    shape: tuple[int, int] | None = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[int, int, int, int]:
    x0 = int(math.floor((crop.xmin - x_offset) / pixel_size))
    x1 = int(math.ceil((crop.xmax - x_offset) / pixel_size))
    y0 = int(math.floor((crop.ymin - y_offset) / pixel_size))
    y1 = int(math.ceil((crop.ymax - y_offset) / pixel_size))
    if shape is not None:
        h, w = shape
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
    else:
        x0 = max(0, x0)
        y0 = max(0, y0)
    return x0, x1, y0, y1


def crop_extent_um(
    pixel_bounds: tuple[int, int, int, int],
    pixel_size: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = pixel_bounds
    return (
        x_offset + x0 * pixel_size,
        x_offset + x1 * pixel_size,
        y_offset + y0 * pixel_size,
        y_offset + y1 * pixel_size,
    )


def read_ome_crop(path: Path, crop: CropBox, pixel_size: float) -> np.ndarray:
    """Read one OME-TIFF crop as float32 using tiled selection when available."""

    if tifffile is None:
        raise RuntimeError("tifffile is required for OME-TIFF image crops. Install xesim[io].")
    return OmeCropReader(path, pixel_size).read(crop)


def ome_image_shape(path: Path) -> tuple[int, int]:
    """Return the first 2D plane shape of an OME-TIFF without reading pixels."""

    if tifffile is None:
        raise RuntimeError("tifffile is required for OME-TIFF shape inspection. Install xesim[io].")
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    while len(shape) > 2:
        shape = shape[-2:]
    return int(shape[0]), int(shape[1])


def _read_full_plane(path: Path) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("tifffile is required for OME-TIFF image crops")
    arr = np.asarray(tifffile.imread(path, key=0))
    return _first_plane(arr)


def _first_plane(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    # Keep the cached plane in its NATIVE dtype (typically uint16). The
    # consumer (OmeCropReader.read) casts the per-tile slice to float32.
    # Float32-casting the whole plane here doubled the cache footprint —
    # on a breast 5K bundle that meant 15.4 GB / channel × 4 channels × 3
    # workers = >180 GB and OOM. uint16 native keeps it at ~30 GB total.
    return arr


def read_channel_stack(paths: Sequence[Path], crop: CropBox, pixel_size: float) -> np.ndarray:
    if not paths:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return ImageStackReader(paths, pixel_size).read(crop)


def robust_normalize(images: np.ndarray) -> np.ndarray:
    """Normalize CxHxW image tensor to roughly [0, 1] per channel."""

    if images.size == 0:
        return images.astype(np.float32, copy=False)
    out = np.empty_like(images, dtype=np.float32)
    for c in range(images.shape[0]):
        arr = images[c].astype(np.float32, copy=False)
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.8))
        if hi <= lo:
            hi = lo + 1.0
        out[c] = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out
