"""Plan3 — fit per-type intensity scatter (B8) and render-mean calibration (B9).

B8: per-type log-normal scatter of per-cell mean intensity. Adds
``{channel}_efficiency_log_std`` fields so the synthesis-time per-cell
multiplier becomes ``type_efficiency * exp(N(0, log_std))``.

B9: render-mean calibration. The downstream renderer applies several
per-cell reductions (nucleus dilution, sectioning visibility, blur) that
attenuate the rendered per-cell mean below the fitted ``type_efficiency``
target by ~25-40% on membrane and polyA channels. ``calibrate_render_means``
samples a small batch of synthetic scenes, compares per-type per-channel
means to real, and rescales each type's efficiency so the render
matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .validation import validate_crop_manifest


def compute_per_type_channel_means(
    crop_manifest_path: Path,
    cell_types_path: Path,
    out_path: Path,
    *,
    n_channels: int | None = None,
    min_area_px: int = 30,
    max_area_px: int = 3000,
    edge_margin: int = 2,
) -> Path:
    """Compute per-cell-type per-channel mean intensity from canonical
    training crops and save it as a `(T+1, C)` numpy array.

    Row 0 = unknown (type index 0); rows 1..T correspond to
    `cell_types.json`'s ``type_names[1:]`` in order. The array is consumed
    by `build_structural_channels(..., per_type_channel_means=...)` as the
    Phase 2.D ``expected_intensity_{c}`` conditioning — the strongest
    anchor for per-type-specific stain levels.

    Iterates the same canonical crops the renderer trains on; per-cell
    means are computed inside each cell's mask, then averaged within type
    after dropping edge-touching and out-of-range-area cells (same filters
    as `fit_intensity_scatter`).
    """
    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    with cell_types_path.open() as handle:
        ct_artifact = json.load(handle)
    type_names = list(ct_artifact["type_names"])
    n_types = len(type_names)
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    # Discover n_channels from the first crop if not given.
    root = crop_manifest_path.parent
    crops_meta = manifest.get("crops", [])
    if not crops_meta:
        raise RuntimeError(
            f"empty crop manifest at {crop_manifest_path}; cannot compute "
            "per-type channel means")
    if n_channels is None:
        with np.load(root / crops_meta[0]["npz_path"], allow_pickle=True) as d0:
            n_channels = int(np.asarray(d0["images"]).shape[0])

    per_type_means: list[list[list[float]]] = [
        [[] for _ in range(n_channels)] for _ in range(n_types)]

    for crop in crops_meta:
        with np.load(root / crop["npz_path"], allow_pickle=True) as data:
            if "images" not in data.files:
                continue
            images = np.asarray(data["images"], dtype=np.float32)[:n_channels]
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            cell_ids = [str(v) for v in data["cell_ids"].tolist()] \
                       if "cell_ids" in data.files else []
        h, w = cell_label.shape
        nonzero = [int(v) for v in np.unique(cell_label) if int(v) > 0]
        if len(cell_ids) != len(nonzero):
            n = min(len(nonzero), len(cell_ids))
            nonzero, cell_ids = nonzero[:n], cell_ids[:n]
        for label_val, cid in zip(nonzero, cell_ids):
            type_idx = cell_id_to_type.get(cid, 0)
            mask = (cell_label == label_val)
            area = int(mask.sum())
            if area < min_area_px or area > max_area_px:
                continue
            ys, xs = np.nonzero(mask)
            if (int(ys.min()) < edge_margin or int(ys.max()) >= h - edge_margin
                    or int(xs.min()) < edge_margin or int(xs.max()) >= w - edge_margin):
                continue
            for ci in range(n_channels):
                per_type_means[type_idx][ci].append(
                    float(images[ci][mask].mean()))

    out = np.zeros((n_types, n_channels), dtype=np.float32)
    pop_means = np.zeros(n_channels, dtype=np.float32)
    pop_counts = np.zeros(n_channels, dtype=np.int64)
    for ti in range(n_types):
        for ci in range(n_channels):
            if per_type_means[ti][ci]:
                out[ti, ci] = float(np.mean(per_type_means[ti][ci]))
                pop_means[ci] += float(np.sum(per_type_means[ti][ci]))
                pop_counts[ci] += int(len(per_type_means[ti][ci]))
    # Backstop: any type/channel with zero observed cells falls back to the
    # overall population mean for that channel. Beats leaving zeros, which
    # the renderer would interpret as "this type emits no signal at all".
    for ci in range(n_channels):
        backstop = float(pop_means[ci] / pop_counts[ci]) \
                    if pop_counts[ci] > 0 else 0.0
        for ti in range(n_types):
            if out[ti, ci] == 0.0 and not per_type_means[ti][ci]:
                out[ti, ci] = backstop

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return out_path


def fit_intensity_scatter(
    crop_manifest_path: Path,
    cell_types_path: Path,
    params_path: Path,
    min_area_px: int = 30,
    max_area_px: int = 3000,
    edge_margin: int = 2,
) -> Path:
    """Augment an existing mechanistic_params artifact with per-type
    ``{dapi,membrane,polya}_efficiency_log_std`` fields fit from real
    per-cell intensity scatter.
    """

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    with cell_types_path.open() as handle:
        ct_artifact = json.load(handle)
    type_names = list(ct_artifact["type_names"])
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    root = crop_manifest_path.parent
    per_type_per_channel: dict[int, dict[str, list[float]]] = {}

    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            if "images" not in data.files:
                continue
            images = np.asarray(data["images"], dtype=np.float32)[:3]
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
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
            if type_idx == 0:
                continue
            cell_mask = cell_label == int(label_val)
            cell_area = int(cell_mask.sum())
            if cell_area < int(min_area_px) or cell_area > int(max_area_px):
                continue
            ys, xs = np.nonzero(cell_mask)
            if (int(ys.min()) < edge_margin or int(ys.max()) >= h - edge_margin
                or int(xs.min()) < edge_margin or int(xs.max()) >= w - edge_margin):
                continue
            channel_means = {
                "dapi": float(images[0][cell_mask].mean()),
                "membrane": float(images[1][cell_mask].mean()) if images.shape[0] > 1 else 0.0,
                "polya": float(images[2][cell_mask].mean()) if images.shape[0] > 2 else 0.0,
            }
            bucket = per_type_per_channel.setdefault(type_idx, {"dapi": [], "membrane": [], "polya": []})
            for k, v in channel_means.items():
                bucket[k].append(float(v))

    with params_path.open() as handle:
        params = json.load(handle)
    pte = params.setdefault("per_type_efficiencies", {})
    for type_idx, channels in per_type_per_channel.items():
        if type_idx >= len(type_names):
            continue
        name = type_names[type_idx]
        record = pte.setdefault(name, {})
        for ch, values in channels.items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[arr > 1e-4]
            if arr.size < 4:
                record[f"{ch}_efficiency_log_std"] = 0.30  # safe default
                record[f"{ch}_efficiency_log_mean"] = float(np.log(np.mean(arr) + 1e-6)) if arr.size else 0.0
                continue
            log_arr = np.log(arr)
            record[f"{ch}_efficiency_log_std"] = float(np.clip(np.std(log_arr), 0.10, 1.20))
            record[f"{ch}_efficiency_log_mean"] = float(np.mean(log_arr))
            record[f"{ch}_n_cells_for_scatter"] = int(arr.size)

    out_path = params_path
    with out_path.open("w") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
    return out_path


def calibrate_render_means(
    crop_manifest_path: Path,
    cell_types_path: Path,
    synthetic_samples_path: Path,
    params_path: Path,
    min_area_px: int = 30,
    max_area_px: int = 3000,
    cal_min: float = 0.5,
    cal_max: float = 2.5,
) -> Path:
    """B9: rescale per-type {dapi,membrane,polya}_efficiency in ``params``
    so the synthetic per-cell mean (computed from ``synthetic_samples``)
    matches real (computed from ``crop_manifest``) per type.

    Calibration ratio per (type, channel) = real_mean / synth_mean,
    clipped to ``[cal_min, cal_max]`` to avoid runaway scaling on rare
    types. Updates ``params_path`` in place.
    """

    import collections

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    with cell_types_path.open() as handle:
        ct_artifact = json.load(handle)
    type_names = list(ct_artifact["type_names"])
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    real: dict[int, dict[int, list[float]]] = {0: {}, 1: {}, 2: {}}
    root = crop_manifest_path.parent
    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            if "images" not in data.files:
                continue
            images = np.asarray(data["images"], dtype=np.float32)[:3]
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            cell_ids = [str(v) for v in data["cell_ids"].tolist()] if "cell_ids" in data.files else []
        unique = np.unique(cell_label)
        nonzero = unique[unique > 0].tolist()
        if len(cell_ids) != len(nonzero):
            n = min(len(nonzero), len(cell_ids))
            nonzero = nonzero[:n]
            cell_ids = cell_ids[:n]
        for label_val, cid in zip(nonzero, cell_ids):
            type_idx = cell_id_to_type.get(cid, 0)
            if type_idx == 0:
                continue
            cell_mask = cell_label == int(label_val)
            if int(cell_mask.sum()) < int(min_area_px) or int(cell_mask.sum()) > int(max_area_px):
                continue
            for ch in (0, 1, 2):
                real[ch].setdefault(type_idx, []).append(float(images[ch][cell_mask].mean()))

    synth: dict[int, dict[int, list[float]]] = {0: {}, 1: {}, 2: {}}
    with synthetic_samples_path.open() as handle:
        sm = json.load(handle)
    for record in sm.get("records", []):
        npz_path = synthetic_samples_path.parent / record["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            refined = np.asarray(data["refined"], dtype=np.float32)
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            type_label = np.asarray(data["type_label"], dtype=np.int16) if "type_label" in data.files else None
        for label_val in np.unique(cell_label):
            if int(label_val) <= 0:
                continue
            cell_mask = cell_label == int(label_val)
            if int(cell_mask.sum()) < int(min_area_px) or int(cell_mask.sum()) > int(max_area_px):
                continue
            type_idx = 0
            if type_label is not None:
                tin = type_label[cell_mask]
                tin = tin[tin > 0]
                if tin.size:
                    type_idx = int(np.bincount(tin).argmax())
            if type_idx == 0:
                continue
            for ch in (0, 1, 2):
                synth[ch].setdefault(type_idx, []).append(float(refined[ch][cell_mask].mean()))

    with params_path.open() as handle:
        params = json.load(handle)
    pte = params.setdefault("per_type_efficiencies", {})
    ch_keys = ["dapi_efficiency", "membrane_efficiency", "polya_efficiency"]
    calibration_log: dict[str, dict[str, float]] = {}
    for type_idx, name in enumerate(type_names):
        if type_idx == 0 or name not in pte or not isinstance(pte[name], dict):
            continue
        cals: dict[str, float] = {}
        for ch in (0, 1, 2):
            r = np.asarray(real[ch].get(type_idx, []), dtype=np.float64)
            s = np.asarray(synth[ch].get(type_idx, []), dtype=np.float64)
            if r.size < 3 or s.size < 3:
                cals[ch_keys[ch]] = 1.0
                continue
            ratio = float(r.mean() / max(s.mean(), 1e-4))
            cals[ch_keys[ch]] = float(np.clip(ratio, float(cal_min), float(cal_max)))
        for k, c in cals.items():
            pte[name][k] = float(pte[name].get(k, 1.0)) * c
        calibration_log[name] = cals
    params["render_calibration_v0"] = calibration_log
    with params_path.open("w") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
    return params_path
