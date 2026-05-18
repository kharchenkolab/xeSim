"""Stage B / Stage 7 — latent cell proposal.

Implements A1 (DAPI peaks), A2 (unexplained-ridge connected components),
and A3 (matched DAPI+ridge candidates) per misc/stage7_proposal.md and
misc/scene_fit_roadmap.md. Each proposer takes a real tile + the
existing 10x cell_label and emits a list of candidate cells with
provenance scores. The candidates are added to the scene as
``MechanisticCell(source="inferred_latent")`` records before rendering.

The cheapest interpretable proposers, no training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    label as ndi_label,
    maximum_filter,
)


@dataclass
class CandidateCell:
    """Stage B latent-cell proposal."""

    cy: float
    cx: float
    radius_px: float
    cell_pixels: np.ndarray  # (n, 2) (y, x) coords
    nucleus_pixels: np.ndarray | None
    dapi_peak_score: float = 0.0
    ridge_enclosure_score: float = 0.0
    provenance_methods: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cy": float(self.cy),
            "cx": float(self.cx),
            "radius_px": float(self.radius_px),
            "n_cell_pixels": int(self.cell_pixels.shape[0]),
            "n_nucleus_pixels": 0 if self.nucleus_pixels is None else int(self.nucleus_pixels.shape[0]),
            "dapi_peak_score": float(self.dapi_peak_score),
            "ridge_enclosure_score": float(self.ridge_enclosure_score),
            "provenance_methods": list(self.provenance_methods),
        }


def _dapi_peaks_outside_nucleus(
    real_dapi: np.ndarray,
    nucleus_label: np.ndarray,
    *,
    min_distance_px: int = 6,
    threshold: float = 0.30,
    excl_radius_px: int = 4,
) -> list[tuple[float, float, float]]:
    """A1 — DAPI peaks not inside any existing nucleus.

    Returns (cy, cx, peak_value) for each candidate.
    """

    # Local maxima via max-filter equivalence: pixel == max in (2k+1)² nbhd
    fp = int(min_distance_px)
    local_max = maximum_filter(real_dapi, size=(fp * 2 + 1, fp * 2 + 1))
    is_peak = (real_dapi == local_max) & (real_dapi > float(threshold))
    # Exclude pixels inside or near any existing nucleus
    near_nucleus = binary_dilation(nucleus_label > 0, iterations=int(excl_radius_px))
    is_peak &= ~near_nucleus
    ys, xs = np.nonzero(is_peak)
    return [(float(y), float(x), float(real_dapi[y, x])) for y, x in zip(ys, xs)]


def _enclosed_ridge_components(
    unexplained_ridge_score: np.ndarray,
    *,
    threshold: float = 0.10,
    min_area_px: int = 30,
    max_area_px: int = 4000,
    min_enclosure: float = 0.40,
) -> list[dict[str, Any]]:
    """A2 — connected components in unexplained ridge that *enclose* an
    interior region.

    For each component we fill the interior; keep if interior_area >= min
    AND the ratio (component_area / fill_area) is above ``min_enclosure``
    (i.e., the component looks like a closed boundary, not an arc).

    Returns dicts with: centroid (cy, cx), interior_mask (full image
    bool), component_mean_score, n_pixels.
    """

    bin_ridge = unexplained_ridge_score >= float(threshold)
    if not bin_ridge.any():
        return []
    # Dilate ridge a bit to close small gaps before fill
    ridge_closed = binary_dilation(bin_ridge, iterations=1)
    filled = binary_fill_holes(ridge_closed)
    interiors = filled & ~ridge_closed
    label_arr, n_components = ndi_label(interiors)
    out: list[dict[str, Any]] = []
    if n_components == 0:
        return out
    for label_value in range(1, int(n_components) + 1):
        component_mask = label_arr == label_value
        area = int(component_mask.sum())
        if area < int(min_area_px) or area > int(max_area_px):
            continue
        # Enclosure ratio: ratio of ridge perimeter to expected from area
        # (a circle's perimeter scales as sqrt(area)). Use a simpler proxy:
        # ridge-pixels-touching-boundary / boundary-length.
        boundary = binary_dilation(component_mask, iterations=1) & ~component_mask
        ridge_at_boundary = boundary & bin_ridge
        if int(boundary.sum()) == 0:
            continue
        enclosure = float(ridge_at_boundary.sum()) / float(boundary.sum())
        if enclosure < float(min_enclosure):
            continue
        ys, xs = np.nonzero(component_mask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        radius_px = float(np.sqrt(area / np.pi))
        component_mean_score = float(unexplained_ridge_score[component_mask].mean())
        out.append({
            "cy": cy, "cx": cx,
            "radius_px": radius_px,
            "interior_mask": component_mask,
            "score": component_mean_score,
            "enclosure": enclosure,
            "area": area,
        })
    return out


def _circle_mask(cy: float, cx: float, radius_px: float, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius_px ** 2)


def propose_latent_cells(
    real_dapi: np.ndarray,
    real_membrane: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    *,
    dapi_peak_threshold: float = 0.30,
    dapi_peak_min_distance_px: int = 6,
    ridge_threshold: float = 0.10,
    ridge_min_area_px: int = 30,
    ridge_max_area_px: int = 4000,
    ridge_min_enclosure: float = 0.40,
    match_radius_px: int = 8,
    default_radius_px: float = 8.0,
    default_nucleus_radius_px: float = 4.0,
) -> list[CandidateCell]:
    """A3 — combined DAPI-peak + ridge-component proposer.

    Algorithm:
    1. Compute unexplained-ridge map from real membrane vs the existing
       cell_label boundaries (re-uses detect_unexplained_membrane_ridges).
    2. Find DAPI peaks outside existing nuclei.
    3. Find enclosed ridge components.
    4. Match: prefer DAPI peaks; for each peak, snap to nearest ridge
       component within match_radius_px; if found, use the ridge interior
       as the cell mask. Otherwise use a default circle.
    5. Ridge components without a matching peak become nucleus_poor cells.
    """

    from .mechanistic_render import _label_boundary
    from .mechanistic_ridges import detect_unexplained_membrane_ridges

    h, w = cell_label.shape
    # Compute unexplained ridge map.
    cell_boundary = _label_boundary(cell_label).astype(np.float32)
    ridge = detect_unexplained_membrane_ridges(
        real_membrane=real_membrane.astype(np.float32),
        rendered_boundary=cell_boundary,
        cell_label=cell_label,
        nucleus_label=nucleus_label,
    )
    score = ridge["unexplained_ridge_score"].astype(np.float32)

    peaks = _dapi_peaks_outside_nucleus(
        real_dapi.astype(np.float32),
        nucleus_label,
        min_distance_px=dapi_peak_min_distance_px,
        threshold=dapi_peak_threshold,
    )
    components = _enclosed_ridge_components(
        score,
        threshold=ridge_threshold,
        min_area_px=ridge_min_area_px,
        max_area_px=ridge_max_area_px,
        min_enclosure=ridge_min_enclosure,
    )

    # Track which ridge components are claimed by a DAPI-peak match.
    claimed = [False] * len(components)
    candidates: list[CandidateCell] = []

    for cy, cx, peak_value in peaks:
        # Match to nearest ridge component within radius.
        best_idx = -1
        best_dist = float("inf")
        for j, comp in enumerate(components):
            if claimed[j]:
                continue
            d = float(np.hypot(cy - comp["cy"], cx - comp["cx"]))
            if d < best_dist and d <= float(match_radius_px):
                best_dist = d
                best_idx = j
        if best_idx >= 0:
            comp = components[best_idx]
            claimed[best_idx] = True
            interior_mask = comp["interior_mask"]
            cell_pixels = np.argwhere(interior_mask)
            # Nucleus = small disk around the DAPI peak
            nuc_mask = _circle_mask(cy, cx, default_nucleus_radius_px, h, w) & interior_mask
            nuc_pixels = np.argwhere(nuc_mask) if nuc_mask.any() else None
            radius = comp["radius_px"]
            candidates.append(CandidateCell(
                cy=cy, cx=cx, radius_px=float(radius),
                cell_pixels=cell_pixels,
                nucleus_pixels=nuc_pixels,
                dapi_peak_score=peak_value,
                ridge_enclosure_score=comp["enclosure"],
                provenance_methods=("A1_dapi_peak", "A2_ridge_match"),
            ))
        else:
            # DAPI peak without a matching ridge: default circle around peak.
            cell_circle = _circle_mask(cy, cx, default_radius_px, h, w)
            cell_pixels = np.argwhere(cell_circle)
            nuc_mask = _circle_mask(cy, cx, default_nucleus_radius_px, h, w)
            nuc_pixels = np.argwhere(nuc_mask) if nuc_mask.any() else None
            candidates.append(CandidateCell(
                cy=cy, cx=cx, radius_px=float(default_radius_px),
                cell_pixels=cell_pixels,
                nucleus_pixels=nuc_pixels,
                dapi_peak_score=peak_value,
                ridge_enclosure_score=0.0,
                provenance_methods=("A1_dapi_peak",),
            ))

    # Unclaimed ridge components → cells without nucleus (sectioning state).
    for j, comp in enumerate(components):
        if claimed[j]:
            continue
        interior_mask = comp["interior_mask"]
        cell_pixels = np.argwhere(interior_mask)
        candidates.append(CandidateCell(
            cy=comp["cy"], cx=comp["cx"], radius_px=float(comp["radius_px"]),
            cell_pixels=cell_pixels,
            nucleus_pixels=None,
            dapi_peak_score=0.0,
            ridge_enclosure_score=comp["enclosure"],
            provenance_methods=("A2_ridge_only",),
        ))

    return candidates


def stamp_candidates_into_label(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    candidates: list[CandidateCell],
    *,
    label_offset: int = 100000,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stamp Stage B candidates as new (cell_label, nucleus_label) entries.

    First-write-wins on free pixels; existing 10x cells are not
    overwritten. Returns updated arrays and the list of new cell label
    integers assigned (one per candidate, even if some were entirely
    clipped — empty cells are dropped from the return list).
    """

    cell_label = cell_label.astype(np.int32, copy=True)
    nucleus_label = nucleus_label.astype(np.int32, copy=True)
    h, w = cell_label.shape
    next_label = int(label_offset)
    assigned: list[int] = []
    for cand in candidates:
        new_label = next_label
        next_label += 1
        # Cell pixels: only place where currently free.
        cell_pix = cand.cell_pixels
        if cell_pix.size:
            ys = cell_pix[:, 0]
            xs = cell_pix[:, 1]
            in_bounds = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            ys = ys[in_bounds]; xs = xs[in_bounds]
            free = cell_label[ys, xs] == 0
            if free.any():
                cell_label[ys[free], xs[free]] = new_label
        # Did the candidate get any pixels?
        if not (cell_label == new_label).any():
            continue
        # Nucleus: only where this cell ended up + nucleus_label currently
        # free. Use the SAME label int so render_mechanistic's spatial-
        # match nucleus picker finds it.
        if cand.nucleus_pixels is not None and cand.nucleus_pixels.size:
            ys = cand.nucleus_pixels[:, 0]
            xs = cand.nucleus_pixels[:, 1]
            in_bounds = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            ys = ys[in_bounds]; xs = xs[in_bounds]
            in_my_cell = cell_label[ys, xs] == new_label
            free_nuc = nucleus_label[ys, xs] == 0
            place = in_my_cell & free_nuc
            if place.any():
                nucleus_label[ys[place], xs[place]] = new_label
        assigned.append(new_label)
    return cell_label, nucleus_label, assigned


def write_proposals_report(
    output_path: Path,
    candidates: list[CandidateCell],
    *,
    crop_id: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "xesim.scene_proposals.v0",
        "crop_id": crop_id,
        "n_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
        **(extra or {}),
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
