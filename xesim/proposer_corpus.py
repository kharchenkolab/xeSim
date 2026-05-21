"""Stage E-prime: ablation-pair training corpus for the learned latent cell proposer.

Builds (real_image inputs, ablated_segmentation, recovery target) tuples by
randomly removing cells from canonical128 tiles. The proposer's job is to
recover the ablated cells from real-image evidence alone.

Targets per ablated cell at each pixel inside its original footprint:
- ``cell_present`` ∈ {0, 1}
- ``cell_type_onehot`` (length n_types - 1, one-hot, all zero for unknown)
- ``offset_to_centroid`` (Δy, Δx) in pixels

Inputs to the proposer model (4 channels):
- real DAPI
- real membrane
- ablated cell mask (binary, 1 inside any non-ablated cell)
- ablated nucleus mask (binary)

The proposer is trained to predict the targets only inside ablated regions
(i.e. pixels that became extracellular as a result of the ablation). All
other pixels carry zero in cell_present (whether they're real cells or true
background) so the proposer learns to fire selectively on missed-cell
regions, not on every cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_proposer_corpus(
    canonical_manifest: Path,
    cell_types_path: Path,
    output_path: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    ablation_rates: tuple[float, ...] = (0.10, 0.25, 0.40),
    realizations_per_rate: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Build ablation-pair corpus.

    Saves NPZ with arrays (one per ablation realization, stacked):
    - inputs: (N, 4, H, W) — real_dapi, real_membrane, ablated_cell_mask, ablated_nucleus_mask
    - target_present: (N, 1, H, W)
    - target_offsets: (N, 2, H, W) — Δy, Δx in pixels (zeroed outside ablated)
    - target_type_onehot: (N, T, H, W) — T = n_types - 1 (excluding unknown)
    - crop_ids: (N,) — string ids
    - splits: (N,) — train/val/test
    - ablation_rates: (N,) — float
    - n_ablated: (N,) — number of cells ablated in this realization
    """

    from .validation import validate_crop_manifest

    manifest = validate_crop_manifest(canonical_manifest, check_files=True)
    root = canonical_manifest.parent
    type_artifact = json.loads(Path(cell_types_path).read_text())
    type_names = list(type_artifact["type_names"])
    n_types_minus1 = len(type_names) - 1  # exclude 'unknown'
    cid_to_type: dict[str, dict[str, int]] = {
        str(c["crop_id"]): {str(k): int(v) for k, v in c.get("cell_id_to_type_index", {}).items()}
        for c in type_artifact.get("crops", [])
    }

    rng = np.random.default_rng(seed)
    inputs_list, present_list, offsets_list, onehot_list = [], [], [], []
    crop_ids_list, splits_list, rates_list, n_ablated_list = [], [], [], []

    for record in manifest["crops"]:
        if record.get("split") not in splits:
            continue
        with np.load(root / record["npz_path"], allow_pickle=True) as data:
            real = np.asarray(data["images"], dtype=np.float32)[:2]
            cl_full = np.asarray(data["cell_label"], dtype=np.int32)
            nl_full = np.asarray(data["nucleus_label"], dtype=np.int32)
            cell_ids_full = [str(v) for v in data["cell_ids"]]
        # Crop to common 302x302 (canonical128 tiles are 302x302 with rare +1 px variants).
        H_min = min(real.shape[1], cl_full.shape[0])
        W_min = min(real.shape[2], cl_full.shape[1])
        target_H, target_W = 302, 302
        if H_min < target_H or W_min < target_W:
            continue
        real = real[:, :target_H, :target_W]
        cl_full = cl_full[:target_H, :target_W]
        nl_full = nl_full[:target_H, :target_W]
        live = np.unique(cl_full)
        live = live[live > 0]
        if len(live) < 2:
            continue
        crop_id = str(record.get("crop_id"))
        type_lookup = cid_to_type.get(crop_id, {})
        H, W = cl_full.shape

        for ablation_rate in ablation_rates:
          for _real_idx in range(int(realizations_per_rate)):
            n_ab = max(1, int(round(ablation_rate * len(live))))
            ab_lbls = rng.choice(live, size=n_ab, replace=False)
            ab_set = set(int(x) for x in ab_lbls)
            ablated_mask_full = np.isin(cl_full, list(ab_set))

            cl_ab = cl_full.copy()
            nl_ab = nl_full.copy()
            cl_ab[ablated_mask_full] = 0
            nl_ab[ablated_mask_full] = 0

            # Inputs.
            inp = np.stack([
                real[0],
                real[1],
                (cl_ab > 0).astype(np.float32),
                (nl_ab > 0).astype(np.float32),
            ], axis=0)

            # Targets: cell_present (uint8), offsets (float16, Δy/Δx), type_index (uint8).
            # Pack type-onehot into a single int channel (0 = no target, 1..T = type idx).
            present = np.zeros((H, W), dtype=np.uint8)
            offsets = np.zeros((2, H, W), dtype=np.float16)
            type_idx_map = np.zeros((H, W), dtype=np.uint8)
            for label_value in ab_lbls:
                cell_pix = cl_full == int(label_value)
                if not np.any(cell_pix):
                    continue
                ys, xs = np.where(cell_pix)
                cy = float(ys.mean())
                cx = float(xs.mean())
                present[cell_pix] = 1
                offsets[0, cell_pix] = (cy - ys).astype(np.float16)
                offsets[1, cell_pix] = (cx - xs).astype(np.float16)
                pos = int(np.where(live == int(label_value))[0][0])
                if pos < len(cell_ids_full):
                    cid = cell_ids_full[pos]
                    t = type_lookup.get(cid, 0)
                    if 0 < t <= n_types_minus1:
                        type_idx_map[cell_pix] = t

            # Keep inputs float32: channels 0–1 are continuous real-morphology
            # intensities (DAPI / membrane); float16 (~10-bit mantissa) loses
            # precision on them (esp. raw-uint16 values >2048). The trainer
            # casts to float32 on load anyway, so this only affects stored
            # fidelity, not loader cost. (Binary mask channels 2–3 are exact
            # in float32 too; offsets below stay float16 — sub-pixel, harmless.)
            inputs_list.append(inp.astype(np.float32))
            present_list.append(present)
            offsets_list.append(offsets)
            onehot_list.append(type_idx_map)
            crop_ids_list.append(crop_id)
            splits_list.append(str(record.get("split")))
            rates_list.append(float(ablation_rate))
            n_ablated_list.append(int(n_ab))

    inputs_arr = np.stack(inputs_list, axis=0)  # float32 (N, 4, H, W)
    present_arr = np.stack(present_list, axis=0)  # uint8 (N, H, W)
    offsets_arr = np.stack(offsets_list, axis=0)  # float16 (N, 2, H, W)
    onehot_arr = np.stack(onehot_list, axis=0)  # uint8 (N, H, W)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        inputs=inputs_arr,
        target_present=present_arr,
        target_offsets=offsets_arr,
        target_type_index=onehot_arr,
        crop_ids=np.array(crop_ids_list, dtype=object),
        splits=np.array(splits_list, dtype=object),
        ablation_rates=np.array(rates_list, dtype=np.float32),
        n_ablated=np.array(n_ablated_list, dtype=np.int32),
        type_names=np.array(type_names, dtype=object),
    )
    summary = {
        "num_examples": int(inputs_arr.shape[0]),
        "input_channels": ["real_dapi", "real_membrane", "ablated_cell_mask", "ablated_nucleus_mask"],
        "target_channels": {
            "present": 1,
            "offsets": 2,
            "type_onehot": n_types_minus1,
        },
        "splits_count": {s: int(np.sum(np.array(splits_list) == s)) for s in splits},
        "ablation_rates": list(ablation_rates),
        "shape": list(inputs_arr.shape),
    }
    with output_path.with_suffix(".summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
