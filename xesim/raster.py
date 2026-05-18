from __future__ import annotations

from typing import Sequence

import numpy as np

from .models import CropBox
from .xenium import PolygonRecord


def rasterize_polygons(
    polygons: Sequence[PolygonRecord],
    crop: CropBox,
    shape: tuple[int, int],
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize micron-space polygons into an integer label mask.

    Returns a label image and an array of object IDs where index 1 maps to the
    first ID. Index 0 is background.
    """

    h, w = shape
    label = np.zeros((h, w), dtype=np.int32)
    ids: list[str] = []
    for idx, poly in enumerate(polygons, start=1):
        xmin, xmax, ymin, ymax = poly.bbox
        px0 = max(0, int(np.floor((xmin - crop.xmin) / pixel_size)))
        px1 = min(w, int(np.ceil((xmax - crop.xmin) / pixel_size)) + 1)
        py0 = max(0, int(np.floor((ymin - crop.ymin) / pixel_size)))
        py1 = min(h, int(np.ceil((ymax - crop.ymin) / pixel_size)) + 1)
        if px1 <= px0 or py1 <= py0:
            continue
        xs = crop.xmin + (np.arange(px0, px1, dtype=np.float32) + 0.5) * pixel_size
        ys = crop.ymin + (np.arange(py0, py1, dtype=np.float32) + 0.5) * pixel_size
        xx, yy = np.meshgrid(xs, ys)
        inside = points_in_polygon(xx, yy, poly.x, poly.y)
        if np.any(inside):
            label[py0:py1, px0:px1][inside] = idx
            ids.append(poly.object_id)
    return label, np.asarray(ids, dtype=object)


def points_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test."""

    inside = np.zeros(x.shape, dtype=bool)
    xj = float(poly_x[-1])
    yj = float(poly_y[-1])
    for xi, yi in zip(poly_x, poly_y):
        xi_f = float(xi)
        yi_f = float(yi)
        crosses = ((yi_f > y) != (yj > y)) & (
            x < (xj - xi_f) * (y - yi_f) / ((yj - yi_f) + 1e-12) + xi_f
        )
        inside ^= crosses
        xj, yj = xi_f, yi_f
    return inside


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(max(iterations, 0)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        center = padded[1:-1, 1:-1]
        out = (
            center
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
    return out


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool, copy=True)
    try:
        from scipy.ndimage import binary_dilation as _scipy_binary_dilation
        return _scipy_binary_dilation(mask.astype(bool, copy=False), iterations=int(iterations))
    except Exception:  # noqa: BLE001 — fallback to manual loop
        out = mask.astype(bool, copy=True)
        for _ in range(max(iterations, 0)):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            out = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
                | padded[:-2, :-2]
                | padded[:-2, 2:]
                | padded[2:, :-2]
                | padded[2:, 2:]
            )
        return out


def boundary_mask(mask: np.ndarray, width: int = 1) -> np.ndarray:
    binary = mask.astype(bool)
    eroded = binary_erosion(binary, width)
    return binary & ~eroded
