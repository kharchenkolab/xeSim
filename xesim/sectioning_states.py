"""Plan3 D1 — per-type sectioning state distribution.

Section state schema (`mechanistic_schema.MECHANISTIC_SECTIONING_STATES`):
``full``, ``nucleus_rich``, ``nucleus_poor``, ``membrane_only``, ``sliver``,
``unknown``.

The synthetic sampler historically only used ``full`` (nucleus present) and
``nucleus_poor`` (nucleus missing). Real Xenium tissue has a richer mix of
sectioning artifacts: cells clipped at the section boundary, sliver
fragments, membrane-only ghosts where the nucleus is out of plane. This
module fits a per-type empirical distribution over those states from
canonical crops and exposes a sampler the synthetic pipeline draws from.

Per-state parameters:
- ``nucleus_present`` (bool): whether to render a nucleus
- ``visibility_fraction`` (float in (0, 1]): scales rendered DAPI/membrane/
  polyA amplitudes (already used by the renderer LUTs)
- ``nucleus_to_cell_ratio_mult``: multiplier on the type's typical
  nucleus/cell area ratio (e.g., 1.5 for ``nucleus_rich``, 0.5 for
  ``nucleus_poor``)

State rules (rule-based classifier on per-cell measurements):
- ``full`` — nucleus present, ratio in [0.25, 0.6], no edge contact, mid area
- ``nucleus_rich`` — nucleus area / cell area > 0.6
- ``nucleus_poor`` — ratio < 0.10 or nucleus missing-but-cytoplasm-present
- ``membrane_only`` — nucleus missing AND area below 30th percentile
- ``sliver`` — aspect ratio > 3.0 AND area below 30th percentile
- ``unknown`` — fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .mechanistic_schema import MECHANISTIC_SECTIONING_STATES
from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


SECTIONING_STATES_TYPE = "xesim.sectioning_states.v0"

STATE_DEFAULT_PARAMS: dict[str, dict[str, float | bool]] = {
    "full": {"nucleus_present": True, "visibility_fraction": 1.0, "nucleus_ratio_mult": 1.0},
    "nucleus_rich": {"nucleus_present": True, "visibility_fraction": 1.0, "nucleus_ratio_mult": 1.1},
    "nucleus_poor": {"nucleus_present": True, "visibility_fraction": 0.85, "nucleus_ratio_mult": 0.5},
    "membrane_only": {"nucleus_present": False, "visibility_fraction": 0.55, "nucleus_ratio_mult": 0.0},
    "sliver": {"nucleus_present": False, "visibility_fraction": 0.70, "nucleus_ratio_mult": 0.0},
    "unknown": {"nucleus_present": True, "visibility_fraction": 1.0, "nucleus_ratio_mult": 1.0},
}


def _classify_cell_state(
    cell_area_px: int,
    nucleus_area_px: int,
    aspect_ratio: float,
    edge_contact: float,
    area_30th_percentile_px: float,
) -> str:
    """Classify a single cell into a sectioning state."""

    if cell_area_px <= 0:
        return "unknown"
    nucleus_ratio = float(nucleus_area_px) / float(cell_area_px)
    has_nucleus = nucleus_area_px > 0
    is_small = cell_area_px < area_30th_percentile_px
    if has_nucleus and nucleus_ratio > 0.60:
        return "nucleus_rich"
    if has_nucleus and 0.20 <= nucleus_ratio <= 0.60 and edge_contact < 0.05:
        return "full"
    if is_small and aspect_ratio > 3.0:
        return "sliver"
    if not has_nucleus and is_small:
        return "membrane_only"
    if not has_nucleus or nucleus_ratio < 0.10:
        return "nucleus_poor"
    return "full"


def fit_sectioning_states(
    crop_manifest_path: Path,
    cell_types_path: Path,
    output_path: Path,
    edge_margin: int = 2,
) -> Path:
    """Fit per-type sectioning state distribution from canonical crops."""

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    with cell_types_path.open() as handle:
        ct_artifact = json.load(handle)
    type_names = list(ct_artifact["type_names"])
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    root = crop_manifest_path.parent
    per_type_records: dict[int, list[dict[str, float]]] = {}
    all_areas: list[int] = []

    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            nucleus_label = np.asarray(data["nucleus_label"], dtype=np.int32) if "nucleus_label" in data.files else None
            cell_ids = [str(v) for v in data["cell_ids"].tolist()] if "cell_ids" in data.files else []
        h, w = cell_label.shape
        unique = np.unique(cell_label)
        nonzero = unique[unique > 0].tolist()
        if len(cell_ids) != len(nonzero):
            n = min(len(nonzero), len(cell_ids))
            nonzero = nonzero[:n]
            cell_ids = cell_ids[:n]
        for label_val, cid in zip(nonzero, cell_ids):
            type_idx = cell_id_to_type.get(cid, 0)
            cell_mask = cell_label == int(label_val)
            cell_area = int(np.sum(cell_mask))
            if cell_area < 4:
                continue
            ys, xs = np.nonzero(cell_mask)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            edge_contact = float(
                ((y_min < edge_margin) + (y_max >= h - edge_margin)
                 + (x_min < edge_margin) + (x_max >= w - edge_margin))
            ) / 4.0
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
            # Cell labels and nucleus labels use distinct IDs in the
            # canonical NPZ; match by spatial overlap (the largest nucleus
            # blob inside this cell footprint).
            nucleus_area = 0
            if nucleus_label is not None:
                nuc_inside = nucleus_label[cell_mask]
                nuc_inside = nuc_inside[nuc_inside > 0]
                if nuc_inside.size:
                    counts = np.bincount(nuc_inside)
                    if counts.size > 0 and int(counts.max()) > 0:
                        nucleus_area = int(counts.max())
            per_type_records.setdefault(type_idx, []).append({
                "cell_area": cell_area,
                "nucleus_area": nucleus_area,
                "aspect": float(aspect),
                "edge_contact": edge_contact,
            })
            all_areas.append(cell_area)

    if not all_areas:
        raise ValueError(f"{crop_manifest_path}: no cells extracted")
    area_30th = float(np.percentile(np.asarray(all_areas, dtype=np.float32), 30.0))

    distributions: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for type_idx, records in per_type_records.items():
        if type_idx == 0 or type_idx >= len(type_names):
            continue
        name = type_names[type_idx]
        type_counts = {state: 0 for state in MECHANISTIC_SECTIONING_STATES}
        for r in records:
            state = _classify_cell_state(
                int(r["cell_area"]),
                int(r["nucleus_area"]),
                float(r["aspect"]),
                float(r["edge_contact"]),
                area_30th,
            )
            type_counts[state] = type_counts.get(state, 0) + 1
        total = sum(type_counts.values())
        if total <= 0:
            continue
        distributions[name] = {state: type_counts.get(state, 0) / total for state in MECHANISTIC_SECTIONING_STATES}
        counts[name] = type_counts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": SECTIONING_STATES_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(crop_manifest_path),
        "cell_types_path": str(cell_types_path),
        "type_names": type_names,
        "states": list(MECHANISTIC_SECTIONING_STATES),
        "state_params": STATE_DEFAULT_PARAMS,
        "distributions": distributions,
        "counts": counts,
        "area_30th_percentile_px": area_30th,
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def load_sectioning_states(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("type") != SECTIONING_STATES_TYPE:
        raise ValueError(f"{path}: not a sectioning_states artifact")
    return payload


def sample_state_for_type(
    rng: np.random.Generator,
    type_name: str,
    artifact: dict[str, Any],
) -> tuple[str, dict[str, float | bool]]:
    """Sample a sectioning state for a cell of the given type.

    Returns ``(state_name, params)`` where ``params`` includes
    ``nucleus_present``, ``visibility_fraction``, ``nucleus_ratio_mult``.
    """

    distributions = artifact.get("distributions", {})
    state_params = artifact.get("state_params", STATE_DEFAULT_PARAMS)
    states_list = artifact.get("states") or list(MECHANISTIC_SECTIONING_STATES)
    type_dist = distributions.get(type_name)
    if not type_dist:
        return "full", dict(state_params.get("full", STATE_DEFAULT_PARAMS["full"]))
    probs = np.asarray([float(type_dist.get(s, 0.0)) for s in states_list], dtype=np.float64)
    if probs.sum() <= 0:
        return "full", dict(state_params.get("full", STATE_DEFAULT_PARAMS["full"]))
    probs = probs / probs.sum()
    state = str(rng.choice(states_list, p=probs))
    params = dict(state_params.get(state, STATE_DEFAULT_PARAMS.get(state, {})))
    return state, params
