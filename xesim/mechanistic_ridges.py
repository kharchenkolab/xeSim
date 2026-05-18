from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mechanistic_render import _distance_from_mask, _gaussian_blur, _label_boundary
from .raster import binary_dilation


@dataclass(frozen=True)
class RidgeObject:
    object_id: str
    label: int
    classification: str
    area_px: int
    centroid_yx: tuple[float, float]
    bbox_yx: tuple[int, int, int, int]
    mean_score: float
    max_score: float
    mean_distance_to_observed_boundary_px: float
    nucleus_overlap_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": int(self.label),
            "classification": self.classification,
            "area_px": int(self.area_px),
            "centroid_yx": [float(self.centroid_yx[0]), float(self.centroid_yx[1])],
            "bbox_yx": [int(x) for x in self.bbox_yx],
            "mean_score": float(self.mean_score),
            "max_score": float(self.max_score),
            "mean_distance_to_observed_boundary_px": float(self.mean_distance_to_observed_boundary_px),
            "nucleus_overlap_fraction": float(self.nucleus_overlap_fraction),
        }


def membrane_ridge_evidence(membrane: np.ndarray) -> np.ndarray:
    """Return a normalized image-derived membrane ridge evidence map."""

    arr = membrane.astype(np.float32, copy=False)
    smooth = _gaussian_blur(arr, 3.0)
    high = np.maximum(arr - smooth, 0.0)
    gy, gx = np.gradient(smooth.astype(np.float32, copy=False))
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return (0.65 * robust_unit(high) + 0.35 * robust_unit(grad)).astype(np.float32)


def detect_unexplained_membrane_ridges(
    real_membrane: np.ndarray,
    rendered_boundary: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    min_area_px: int = 6,
    max_objects: int = 64,
) -> dict[str, Any]:
    """Detect membrane-like real signal not explained by observed boundary render."""

    real_ridge = membrane_ridge_evidence(real_membrane)
    rendered_support = robust_unit(_gaussian_blur(rendered_boundary.astype(np.float32, copy=False), 1.0))
    observed_boundary = _label_boundary(cell_label.astype(np.int32, copy=False))
    observed_near = binary_dilation(observed_boundary, iterations=2)
    explained = rendered_support > 0.25
    score = np.maximum(real_ridge - 0.55 * rendered_support, 0.0).astype(np.float32)
    score[explained] = 0.0

    candidate_values = score[score > 0]
    threshold = float(np.percentile(candidate_values, 88.0)) if candidate_values.size else 1.0
    threshold = max(threshold, 0.08)
    candidate = score >= threshold
    labels, num_labels = _connected_components(candidate)
    distance_to_boundary = _distance_from_mask(observed_boundary, max_distance=48)

    # Vectorize per-label statistics via scipy.ndimage. Replaces a Python
    # loop over ``num_labels`` components, each making full-image mask copies.
    objects: list[RidgeObject] = []
    latent_map = np.zeros(score.shape, dtype=np.float32)
    if num_labels > 0:
        try:
            from scipy.ndimage import (
                find_objects as _scipy_find_objects,
                mean as _scipy_mean,
                maximum as _scipy_maximum,
                sum_labels as _scipy_sum_labels,
                center_of_mass as _scipy_center_of_mass,
            )
            label_index = np.arange(1, num_labels + 1)
            areas = _scipy_sum_labels(np.ones_like(labels, dtype=np.float32), labels, label_index).astype(np.int64)
            mean_dists = np.asarray(_scipy_mean(distance_to_boundary, labels, label_index), dtype=np.float64)
            nucleus_bool = (nucleus_label > 0).astype(np.float32)
            nucleus_overlap_fracs = np.asarray(_scipy_mean(nucleus_bool, labels, label_index), dtype=np.float64)
            mean_scores = np.asarray(_scipy_mean(score, labels, label_index), dtype=np.float64)
            max_scores = np.asarray(_scipy_maximum(score, labels, label_index), dtype=np.float64)
            observed_near_overlap = np.asarray(
                _scipy_sum_labels(observed_near.astype(np.float32), labels, label_index), dtype=np.float64
            )
            centroids = _scipy_center_of_mass(np.ones_like(labels, dtype=np.float32), labels, label_index)
            bboxes = _scipy_find_objects(labels, max_label=num_labels)
            for li, label_value in enumerate(label_index):
                area = int(areas[li])
                if area < min_area_px:
                    continue
                cy, cx = centroids[li]
                slc = bboxes[li]
                if slc is None:
                    continue
                bbox_yx = (slc[0].start, slc[1].start, slc[0].stop, slc[1].stop)
                obj = RidgeObject(
                    object_id=f"ridge_{len(objects) + 1:05d}",
                    label=int(label_value),
                    classification=_classify_ridge_object(
                        area_px=area,
                        mean_distance_to_boundary_px=float(mean_dists[li]),
                        nucleus_overlap_fraction=float(nucleus_overlap_fracs[li]),
                        touches_observed_near=bool(observed_near_overlap[li] > 0),
                    ),
                    area_px=area,
                    centroid_yx=(float(cy), float(cx)),
                    bbox_yx=bbox_yx,
                    mean_score=float(mean_scores[li]),
                    max_score=float(max_scores[li]),
                    mean_distance_to_observed_boundary_px=float(mean_dists[li]),
                    nucleus_overlap_fraction=float(nucleus_overlap_fracs[li]),
                )
                objects.append(obj)
            # Fill latent_map via vectorized lookup: latent_map = score where labels in kept set.
        except Exception:  # noqa: BLE001 - fallback to per-label Python loop
            for label in range(1, num_labels + 1):
                mask = labels == label
                area = int(np.sum(mask))
                if area < min_area_px:
                    continue
                ys, xs = np.nonzero(mask)
                mean_dist = float(np.mean(distance_to_boundary[mask])) if area else 0.0
                nucleus_overlap = float(np.mean(nucleus_label[mask] > 0)) if area else 0.0
                obj = RidgeObject(
                    object_id=f"ridge_{len(objects) + 1:05d}",
                    label=label,
                    classification=_classify_ridge_object(
                        area_px=area,
                        mean_distance_to_boundary_px=mean_dist,
                        nucleus_overlap_fraction=nucleus_overlap,
                        touches_observed_near=bool(np.any(mask & observed_near)),
                    ),
                    area_px=area,
                    centroid_yx=(float(np.mean(ys)), float(np.mean(xs))),
                    bbox_yx=(int(np.min(ys)), int(np.min(xs)), int(np.max(ys)) + 1, int(np.max(xs)) + 1),
                    mean_score=float(np.mean(score[mask])),
                    max_score=float(np.max(score[mask])),
                    mean_distance_to_observed_boundary_px=mean_dist,
                    nucleus_overlap_fraction=nucleus_overlap,
                )
                objects.append(obj)
    # Build latent_map via single vectorized pass (used by both code paths).
    if objects:
        kept_labels_set = {item.label for item in objects}
        kept_labels_arr = np.asarray(sorted(kept_labels_set), dtype=np.int32)
        keep_mask = np.isin(labels, kept_labels_arr)
        latent_map = np.where(keep_mask, score, 0.0).astype(np.float32)

    objects = sorted(objects, key=lambda item: item.mean_score * item.area_px, reverse=True)[: max(0, max_objects)]
    kept_labels = {item.label for item in objects}
    if kept_labels:
        keep_mask = np.isin(labels, list(kept_labels))
        latent_map = np.where(keep_mask, latent_map, 0.0).astype(np.float32)
    else:
        latent_map.fill(0.0)

    return {
        "real_membrane_ridge_evidence": real_ridge.astype(np.float32),
        "rendered_boundary_support": rendered_support.astype(np.float32),
        "unexplained_ridge_score": score.astype(np.float32),
        "latent_membrane_ridge": latent_map.astype(np.float32),
        "unexplained_ridge_mask": (latent_map > 0).astype(np.uint8),
        "objects": [item.to_dict() for item in objects],
        "summary": _ridge_summary(objects, candidate_fraction=float(np.mean(candidate)) if candidate.size else 0.0),
        "threshold": threshold,
    }


def robust_unit(values: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(values, 5.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _classify_ridge_object(
    area_px: int,
    mean_distance_to_boundary_px: float,
    nucleus_overlap_fraction: float,
    touches_observed_near: bool,
) -> str:
    if area_px <= 12:
        return "debris_or_puncta"
    if touches_observed_near or mean_distance_to_boundary_px <= 3.0:
        return "boundary_offset"
    if nucleus_overlap_fraction > 0.10:
        return "membrane_only_or_sliver_near_nucleus"
    if mean_distance_to_boundary_px <= 12.0:
        return "out_of_plane_membrane"
    return "unassigned_membrane_ridge"


def _ridge_summary(objects: list[RidgeObject], candidate_fraction: float) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    total_area = 0
    scores = []
    for obj in objects:
        by_class[obj.classification] = by_class.get(obj.classification, 0) + 1
        total_area += obj.area_px
        scores.append(obj.mean_score)
    return {
        "num_objects": len(objects),
        "total_area_px": int(total_area),
        "candidate_fraction": float(candidate_fraction),
        "mean_object_score": float(np.mean(scores)) if scores else 0.0,
        "objects_by_class": by_class,
    }


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components. Uses scipy.ndimage.label when available."""

    binary = mask.astype(bool, copy=False)
    try:
        from scipy.ndimage import label as _scipy_label
        # 8-connectivity: 3x3 structure of all True.
        structure = np.ones((3, 3), dtype=np.int8)
        labels, n = _scipy_label(binary, structure=structure)
        return labels.astype(np.int32, copy=False), int(n)
    except Exception:  # noqa: BLE001 — fallback Python flood-fill
        labels = np.zeros(binary.shape, dtype=np.int32)
        current_label = 0
        h, w = binary.shape
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or labels[y, x] != 0:
                    continue
                current_label += 1
                stack = [(y, x)]
                labels[y, x] = current_label
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if labels[ny, nx] == 0 and binary[ny, nx]:
                                labels[ny, nx] = current_label
                                stack.append((ny, nx))
        return labels, current_label

