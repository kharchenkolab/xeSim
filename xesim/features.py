from __future__ import annotations

from typing import Any

import numpy as np

from .raster import binary_dilation, binary_erosion, boundary_mask

try:  # pragma: no cover - exercised only when scipy is installed
    from scipy import ndimage as ndi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ndi = None


def geometry_feature_arrays(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    pixel_size: float,
    max_distance_px: int = 64,
) -> dict[str, np.ndarray]:
    """Build simple mask-derived geometry feature arrays for one crop."""

    cell = cell_label > 0
    nucleus = nucleus_label > 0
    cell_boundary = boundary_mask(cell, width=1)
    nucleus_boundary = boundary_mask(nucleus, width=1)
    return {
        "cell_boundary": cell_boundary.astype(np.uint8),
        "nucleus_boundary": nucleus_boundary.astype(np.uint8),
        "cell_interior_distance_um": _distance_um(cell, "interior", pixel_size, max_distance_px),
        "cell_exterior_distance_um": _distance_um(cell, "exterior", pixel_size, max_distance_px),
    }


def geometry_summary(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    pixel_size: float,
    max_distance_px: int = 64,
    features: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Summarize mask geometry in stable scalar fields."""

    cell = cell_label > 0
    nucleus = nucleus_label > 0
    if features is None:
        features = geometry_feature_arrays(cell_label, nucleus_label, pixel_size, max_distance_px)
    cell_interior = features["cell_interior_distance_um"]
    cell_exterior = features["cell_exterior_distance_um"]
    background = ~cell
    return {
        "cell_mask_fraction": float(np.mean(cell)) if cell.size else 0.0,
        "nucleus_mask_fraction": float(np.mean(nucleus)) if nucleus.size else 0.0,
        "cell_boundary_fraction": float(np.mean(features["cell_boundary"] > 0))
        if cell.size
        else 0.0,
        "nucleus_boundary_fraction": float(np.mean(features["nucleus_boundary"] > 0))
        if nucleus.size
        else 0.0,
        "mean_cell_interior_distance_um": _masked_mean(cell_interior, cell),
        "mean_cell_exterior_distance_um": _masked_mean(cell_exterior, background),
        "distance_transform_backend": distance_transform_backend(),
    }


def distance_transform_backend() -> str:
    return "scipy" if ndi is not None else "morphology"


def _distance_um(
    mask: np.ndarray,
    mode: str,
    pixel_size: float,
    max_distance_px: int,
) -> np.ndarray:
    if ndi is not None:
        if mode == "interior":
            distance = ndi.distance_transform_edt(mask)
        elif mode == "exterior":
            distance = ndi.distance_transform_edt(~mask.astype(bool))
        else:
            raise ValueError(f"unknown distance mode: {mode}")
        distance = np.minimum(distance, float(max_distance_px))
        return (distance.astype(np.float32) * np.float32(pixel_size)).astype(np.float32)
    if mode == "interior":
        distance = _interior_distance_steps(mask, max_distance_px=max_distance_px)
    elif mode == "exterior":
        distance = _exterior_distance_steps(mask, max_distance_px=max_distance_px)
    else:
        raise ValueError(f"unknown distance mode: {mode}")
    return (distance * np.float32(pixel_size)).astype(np.float32)


def _interior_distance_steps(mask: np.ndarray, max_distance_px: int) -> np.ndarray:
    distance = np.zeros(mask.shape, dtype=np.float32)
    current = mask.astype(bool, copy=True)
    for step in range(1, max(1, max_distance_px) + 1):
        if not np.any(current):
            break
        eroded = binary_erosion(current, iterations=1)
        ring = current & ~eroded
        distance[ring] = float(step)
        current = eroded
    if np.any(current):
        distance[current] = float(max_distance_px)
    return distance


def _exterior_distance_steps(mask: np.ndarray, max_distance_px: int) -> np.ndarray:
    distance = np.zeros(mask.shape, dtype=np.float32)
    current = mask.astype(bool, copy=True)
    for step in range(1, max(1, max_distance_px) + 1):
        dilated = binary_dilation(current, iterations=1)
        ring = dilated & ~current
        if not np.any(ring):
            break
        distance[ring] = float(step)
        current = dilated
        if np.all(current):
            break
    return distance


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if arr.shape != mask.shape or not np.any(mask):
        return 0.0
    return float(np.mean(arr[mask]))
