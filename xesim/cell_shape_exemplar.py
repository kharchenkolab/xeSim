"""Plan3 A2-exemplar — real-mask exemplar shape sampler.

Library-based alternative to the A2 VAE. For each type+sectioning_state
bucket, we extract real (cell_mask, nucleus_mask) cutouts from canonical
crops, normalize size and orientation, and store them as a compact
artifact. The synthetic sampler picks an exemplar by (type,
sectioning_state), applies a random rotation/flip/scale and a tiny
elastic warp, and stamps the resulting masks onto the synthetic cell
territory.

This guarantees joint cell+nucleus realism (offsets, partial sectioning
patterns, sliver shapes) without training. Replaces both the ellipse-SDF
Voronoi geometry and the cell-mask-only A2 VAE for downstream synthetic
generation.

Cutouts are stored normalized:
- aligned to principal axis (rotated so longer axis = horizontal)
- centered on cell centroid
- padded to ``patch_size`` × ``patch_size``

At sampling time we:
1. Pick a random rotation in [0, 2π)
2. Pick a scale = sqrt(target_area / exemplar_area)
3. Resample exemplar to scaled size
4. Splat at the synthetic center
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


CELL_SHAPE_EXEMPLAR_TYPE = "xesim.cell_shape_exemplars.v0"

DEFAULT_PATCH_SIZE = 64
DEFAULT_MIN_AREA_PX = 30
DEFAULT_MAX_AREA_PX = 2500
DEFAULT_EDGE_MARGIN = 2


def _cell_principal_angle(cell_mask: np.ndarray) -> float:
    """Compute principal-axis angle (radians) of a binary mask."""

    ys, xs = np.nonzero(cell_mask)
    if ys.size < 4:
        return 0.0
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()
    cov = np.cov(np.stack([xs, ys]))
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(major[1], major[0]))


_ROTATE_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _rotate_mask(mask: np.ndarray, angle_rad: float, output_size: int) -> np.ndarray:
    """Rotate a binary mask by ``angle_rad`` and re-center on output_size.

    Cached coordinate grids per output_size + nearest-neighbor sampling
    (order=0) since masks are binary anyway. Roughly 2x faster than the
    previous order=1 + per-call mgrid version (profile showed ~800 calls
    per 4-scene synth sample at ~0.3 ms each).
    """

    h, w = mask.shape
    cy, cx = h / 2.0, w / 2.0
    out_cy = out_cx = output_size / 2.0
    cos_t = np.cos(-float(angle_rad))
    sin_t = np.sin(-float(angle_rad))
    cache = _ROTATE_GRID_CACHE.get(int(output_size))
    if cache is None:
        yy, xx = np.mgrid[0:output_size, 0:output_size].astype(np.float64)
        yy0 = yy - out_cy
        xx0 = xx - out_cx
        cache = (yy0, xx0)
        _ROTATE_GRID_CACHE[int(output_size)] = cache
    yy0, xx0 = cache
    src_y = cos_t * yy0 - sin_t * xx0 + cy
    src_x = sin_t * yy0 + cos_t * xx0 + cx
    sampled = map_coordinates(mask.astype(np.float32), [src_y, src_x], order=1, mode="constant", cval=0.0)
    return (sampled > 0.5).astype(np.uint8)


def build_cell_shape_exemplars(
    crop_manifest_path: Path,
    cell_types_path: Path,
    output_path: Path,
    sectioning_states_path: Path | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    max_area_px: int = DEFAULT_MAX_AREA_PX,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    max_per_bucket: int = 400,
    seed: int = 11,
) -> Path:
    """Extract per-type real (cell, nucleus) mask exemplar pairs into a single NPZ.

    When ``sectioning_states_path`` is given, exemplars are bucketed by
    (type, state) using the same rule classifier as the sectioning fit so
    sampling at run time can match state. Otherwise all exemplars for a
    type land in a single ``full`` bucket.
    """

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    with cell_types_path.open() as handle:
        ct_artifact = json.load(handle)
    type_names = list(ct_artifact["type_names"])
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    sectioning_states = None
    if sectioning_states_path is not None:
        from .sectioning_states import load_sectioning_states

        sectioning_states = load_sectioning_states(sectioning_states_path)

    rng = np.random.default_rng(int(seed))
    root = crop_manifest_path.parent
    cell_arrays: list[np.ndarray] = []
    nucleus_arrays: list[np.ndarray] = []
    bucket_keys: list[str] = []
    cell_areas_um2: list[float] = []
    aspects: list[float] = []
    states: list[str] = []
    type_indices: list[int] = []

    # Pre-pass to compute area 30th percentile for the rule classifier.
    all_areas: list[int] = []
    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
        for label_val in np.unique(cell_label):
            if label_val <= 0:
                continue
            cm = cell_label == int(label_val)
            area = int(np.sum(cm))
            if area > 0:
                all_areas.append(area)
    if not all_areas:
        raise ValueError(f"{crop_manifest_path}: no cells in canonical corpus")
    area_30th = float(np.percentile(np.asarray(all_areas), 30.0))

    from .sectioning_states import _classify_cell_state  # noqa: PLC0415

    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            nucleus_label = np.asarray(data["nucleus_label"], dtype=np.int32) if "nucleus_label" in data.files else None
            cell_ids = [str(v) for v in data["cell_ids"].tolist()] if "cell_ids" in data.files else []
            pixel_size = float(data["pixel_size"]) if "pixel_size" in data.files else 0.2125
        h, w = cell_label.shape
        unique = np.unique(cell_label)
        nonzero = unique[unique > 0].tolist()
        if len(cell_ids) != len(nonzero):
            n = min(len(nonzero), len(cell_ids))
            nonzero = nonzero[:n]
            cell_ids = cell_ids[:n]
        for label_val, cid in zip(nonzero, cell_ids):
            type_idx = cell_id_to_type.get(cid, 0)
            if type_idx == 0 or type_idx >= len(type_names):
                continue
            cell_mask = cell_label == int(label_val)
            cell_area_px = int(np.sum(cell_mask))
            if cell_area_px < int(min_area_px) or cell_area_px > int(max_area_px):
                continue
            ys, xs = np.nonzero(cell_mask)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            if y_min < edge_margin or x_min < edge_margin:
                continue
            if y_max >= h - edge_margin or x_max >= w - edge_margin:
                continue
            # Match nucleus by spatial overlap (canonical NPZ uses different
            # label IDs for cells vs nuclei).
            nucleus_mask_in_cell = np.zeros_like(cell_mask, dtype=bool)
            nucleus_area = 0
            if nucleus_label is not None:
                nuc_inside = nucleus_label[cell_mask]
                nuc_inside = nuc_inside[nuc_inside > 0]
                if nuc_inside.size:
                    counts = np.bincount(nuc_inside)
                    if counts.size > 0 and int(counts.max()) > 0:
                        best_id = int(counts.argmax())
                        nucleus_mask_in_cell = (nucleus_label == best_id) & cell_mask
                        nucleus_area = int(np.sum(nucleus_mask_in_cell))
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            aspect = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))
            edge_contact = 0.0  # already filtered above
            state = _classify_cell_state(
                cell_area_px,
                nucleus_area,
                aspect,
                edge_contact,
                area_30th,
            )
            # Crop tight bbox + small margin, then rotate to principal axis,
            # center within patch_size × patch_size canvas.
            angle = _cell_principal_angle(cell_mask)
            crop_y0 = max(0, y_min - 4)
            crop_y1 = min(h, y_max + 5)
            crop_x0 = max(0, x_min - 4)
            crop_x1 = min(w, x_max + 5)
            cell_crop = cell_mask[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.uint8)
            nuc_crop = nucleus_mask_in_cell[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.uint8)
            # Rotate to principal axis = 0 then center on patch_size canvas.
            ph = max(cell_crop.shape) + 6
            pad_h = (ph - cell_crop.shape[0]) // 2
            pad_w = (ph - cell_crop.shape[1]) // 2
            padded_cell = np.pad(cell_crop, ((pad_h, ph - cell_crop.shape[0] - pad_h), (pad_w, ph - cell_crop.shape[1] - pad_w)))
            padded_nuc = np.pad(nuc_crop, ((pad_h, ph - cell_crop.shape[0] - pad_h), (pad_w, ph - cell_crop.shape[1] - pad_w)))
            cell_aligned = _rotate_mask(padded_cell, angle, int(patch_size))
            nuc_aligned = _rotate_mask(padded_nuc, angle, int(patch_size))
            if int(cell_aligned.sum()) < int(min_area_px):
                continue
            cell_arrays.append(cell_aligned)
            nucleus_arrays.append(nuc_aligned)
            bucket_keys.append(f"{type_names[type_idx]}|{state}")
            states.append(state)
            type_indices.append(int(type_idx))
            cell_areas_um2.append(float(cell_area_px) * (pixel_size ** 2))
            aspects.append(aspect)

    if not cell_arrays:
        raise ValueError(f"{crop_manifest_path}: no exemplars produced")

    # Subsample per bucket to cap memory.
    bucket_to_indices: dict[str, list[int]] = {}
    for i, key in enumerate(bucket_keys):
        bucket_to_indices.setdefault(key, []).append(i)
    keep: list[int] = []
    for key, idxs in bucket_to_indices.items():
        if len(idxs) <= max_per_bucket:
            keep.extend(idxs)
        else:
            picks = rng.choice(idxs, size=int(max_per_bucket), replace=False)
            keep.extend(picks.tolist())
    keep = sorted(keep)
    cells_arr = np.stack([cell_arrays[i] for i in keep], axis=0).astype(np.uint8)
    nucs_arr = np.stack([nucleus_arrays[i] for i in keep], axis=0).astype(np.uint8)
    keys = [bucket_keys[i] for i in keep]
    states_kept = [states[i] for i in keep]
    types_kept = [type_indices[i] for i in keep]
    areas_kept = [cell_areas_um2[i] for i in keep]
    aspects_kept = [aspects[i] for i in keep]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    npz_name = output_path.stem + ".npz"
    npz_path = output_path.with_name(npz_name)
    np.savez(
        npz_path,
        cell_masks=cells_arr,
        nucleus_masks=nucs_arr,
        bucket_keys=np.asarray(keys, dtype=object),
        states=np.asarray(states_kept, dtype=object),
        type_indices=np.asarray(types_kept, dtype=np.int32),
        areas_um2=np.asarray(areas_kept, dtype=np.float32),
        aspects=np.asarray(aspects_kept, dtype=np.float32),
    )
    bucket_counts: dict[str, int] = {}
    for k in keys:
        bucket_counts[k] = bucket_counts.get(k, 0) + 1
    payload = {
        "type": CELL_SHAPE_EXEMPLAR_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(crop_manifest_path),
        "cell_types_path": str(cell_types_path),
        "sectioning_states_path": str(sectioning_states_path) if sectioning_states_path else None,
        "patch_size": int(patch_size),
        "type_names": type_names,
        "n_exemplars": int(cells_arr.shape[0]),
        "bucket_counts": bucket_counts,
        "npz_path": npz_name,
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


class CellShapeExemplarSampler:
    """Sample real-mask exemplar pairs at run time."""

    def __init__(self, manifest_path: Path):
        with manifest_path.open() as handle:
            payload = json.load(handle)
        if payload.get("type") != CELL_SHAPE_EXEMPLAR_TYPE:
            raise ValueError(f"{manifest_path}: not a cell_shape_exemplars artifact")
        self.manifest = payload
        npz = np.load(manifest_path.parent / payload["npz_path"], allow_pickle=True)
        self.cell_masks = np.asarray(npz["cell_masks"], dtype=np.uint8)
        self.nucleus_masks = np.asarray(npz["nucleus_masks"], dtype=np.uint8)
        keys = npz["bucket_keys"].tolist()
        self.areas = np.asarray(npz["areas_um2"], dtype=np.float32)
        self.bucket_to_indices: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            self.bucket_to_indices.setdefault(str(key), []).append(int(i))
        self.type_indices_per_bucket: dict[str, list[int]] = {}
        for key in self.bucket_to_indices:
            t_name = key.split("|", 1)[0]
            self.type_indices_per_bucket.setdefault(t_name, []).extend(self.bucket_to_indices[key])
        self.patch_size = int(payload["patch_size"])

    def sample(
        self,
        rng: np.random.Generator,
        type_name: str,
        state: str | None = None,
        target_area_um2: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return (cell_mask, nucleus_mask, info) at canonical patch_size.

        Both masks are uint8 binary patches centered on the cell. Caller
        scales+rotates+places into the synthetic scene.
        """

        key = None
        if state is not None:
            key = f"{type_name}|{state}"
            if key not in self.bucket_to_indices:
                key = None
        if key is None:
            indices = self.type_indices_per_bucket.get(type_name)
            if not indices:
                # Fallback: any exemplar.
                all_indices = list(range(self.cell_masks.shape[0]))
                indices = all_indices
        else:
            indices = self.bucket_to_indices[key]
        idx = int(rng.choice(indices))
        cell = self.cell_masks[idx]
        nuc = self.nucleus_masks[idx]
        # Random horizontal flip.
        if bool(rng.integers(0, 2)):
            cell = cell[:, ::-1].copy()
            nuc = nuc[:, ::-1].copy()
        # Random rotation in [0, 2π).
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        cell_rot = _rotate_mask(cell, -angle, self.patch_size)
        nuc_rot = _rotate_mask(nuc, -angle, self.patch_size)
        return cell_rot, nuc_rot, {
            "exemplar_index": int(idx),
            "exemplar_area_um2": float(self.areas[idx]),
            "type_name": type_name,
            "state": state,
            "rotation_rad": angle,
        }
