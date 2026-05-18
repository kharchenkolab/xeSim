"""Plan3 cell-type annotation join.

Reads the Xenium pancreas annotation CSV (cell_id -> merged_annotation),
joins it to per-crop cell IDs from a canonical manifest, and emits a
``xesim.cell_type_assignment.v0`` artifact. Downstream tools use it to:

* condition the bounded-residual refiner on per-pixel cell type (one-hot),
* stratify population evaluation metrics by cell type,
* later, fit type-specific mechanistic priors and latent emissions.

Plan3 principle: cell type is part of the explicit scene state. Pushing it
into the renderer/refiner conditioning makes more biology auditable and
prevents the texture/perceptual losses from rewarding cross-type confusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


CELL_TYPE_ASSIGNMENT_TYPE = "xesim.cell_type_assignment.v0"

UNKNOWN_TYPE_INDEX = 0
UNKNOWN_TYPE_NAME = "unknown"


def attach_cell_types(
    crop_manifest_path: Path,
    annotation_csv_path: Path,
    output_path: Path,
    type_column: str = "merged_annotation",
    cell_id_column: str = "cell_id",
) -> Path:
    """Join cell-type labels to canonical crops and write a v0 artifact.

    The resulting JSON contains:

    * ``type_names``: ordered list with index 0 reserved for ``unknown``.
    * ``crops``: list of ``{crop_id, npz_path, num_cells, num_typed_cells,
      cell_id_to_type_index}`` records.

    The downstream consumer derives a per-pixel type label image by indexing
    the canonical ``cell_label`` mask through this mapping.
    """

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    annotation_df = pd.read_csv(annotation_csv_path)
    if cell_id_column not in annotation_df.columns:
        raise ValueError(f"annotation CSV missing column {cell_id_column!r}")
    if type_column not in annotation_df.columns:
        raise ValueError(f"annotation CSV missing column {type_column!r}")

    type_names = [UNKNOWN_TYPE_NAME] + sorted(
        annotation_df[type_column].dropna().astype(str).unique().tolist()
    )
    type_to_index = {name: idx for idx, name in enumerate(type_names)}
    cell_id_to_index: dict[str, int] = {}
    for cell_id, type_name in zip(
        annotation_df[cell_id_column].astype(str).tolist(),
        annotation_df[type_column].astype(str).tolist(),
    ):
        cell_id_to_index[cell_id] = type_to_index.get(type_name, UNKNOWN_TYPE_INDEX)

    root = crop_manifest_path.parent
    crop_records: list[dict[str, Any]] = []
    total_cells = 0
    total_typed = 0
    type_counts = [0] * len(type_names)
    for crop in manifest.get("crops", []):
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            cell_ids = [str(value) for value in data["cell_ids"].tolist()]
        mapping = {cid: cell_id_to_index.get(cid, UNKNOWN_TYPE_INDEX) for cid in cell_ids}
        typed = sum(1 for v in mapping.values() if v != UNKNOWN_TYPE_INDEX)
        for cid in cell_ids:
            type_counts[mapping[cid]] += 1
        total_cells += len(cell_ids)
        total_typed += typed
        crop_records.append(
            {
                "crop_id": str(crop["crop_id"]),
                "npz_path": str(crop["npz_path"]),
                "num_cells": len(cell_ids),
                "num_typed_cells": typed,
                "cell_id_to_type_index": mapping,
            }
        )

    payload = {
        "type": CELL_TYPE_ASSIGNMENT_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(crop_manifest_path),
        "annotation_csv": str(annotation_csv_path),
        "type_column": str(type_column),
        "cell_id_column": str(cell_id_column),
        "type_names": type_names,
        "type_counts": type_counts,
        "num_crops": len(crop_records),
        "num_cells_total": total_cells,
        "num_cells_typed": total_typed,
        "crops": crop_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def load_cell_type_assignment(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        obj = json.load(handle)
    if obj.get("type") != CELL_TYPE_ASSIGNMENT_TYPE:
        raise ValueError(f"{path}: expected type {CELL_TYPE_ASSIGNMENT_TYPE}, got {obj.get('type')!r}")
    return obj


def build_per_pixel_type_label(
    cell_label: np.ndarray,
    cell_ids: list[str],
    cell_id_to_type_index: dict[str, int],
    nucleus_label: np.ndarray | None = None,
) -> np.ndarray:
    """Produce an int16 per-pixel type label given a canonical cell-label mask.

    The mapping is: 0 = background or untyped; otherwise the type index.
    Nuclei share the type of the parent cell when ``nucleus_label`` is
    provided; otherwise nucleus pixels follow the cell label.
    """

    # Sort the unique nonzero label values to recover the cell_ids alignment
    # that canonicalize uses (cell_ids is sorted by zarr label value).
    unique_labels = np.unique(cell_label)
    nonzero = unique_labels[unique_labels > 0].tolist()
    if len(nonzero) != len(cell_ids):
        # Fall back to the alignment we have. Shorter list is the limiter.
        n = min(len(nonzero), len(cell_ids))
        nonzero = nonzero[:n]
        cell_ids = list(cell_ids)[:n]
    label_to_type = {
        int(label_value): int(cell_id_to_type_index.get(str(cid), UNKNOWN_TYPE_INDEX))
        for label_value, cid in zip(nonzero, cell_ids)
    }
    if not label_to_type:
        return np.zeros(cell_label.shape, dtype=np.int16)
    keys = np.asarray(list(label_to_type.keys()), dtype=np.int64)
    values = np.asarray(list(label_to_type.values()), dtype=np.int16)
    max_key = int(keys.max())
    lut = np.zeros(max_key + 1, dtype=np.int16)
    lut[keys] = values
    flat = cell_label.astype(np.int64, copy=False)
    flat_clipped = np.where(flat <= max_key, flat, 0)
    type_label = lut[flat_clipped]
    if nucleus_label is not None:
        # Treat nucleus pixels as belonging to the cell-type they sit in.
        # If nucleus is outside any cell, leave it 0 (unknown).
        nuc = nucleus_label.astype(np.int64, copy=False)
        type_label = np.where((nuc > 0) & (type_label == 0), 0, type_label)
    return type_label.astype(np.int16, copy=False)


def per_pixel_type_one_hot(type_label: np.ndarray, num_types: int) -> np.ndarray:
    """Return a ``(num_types, H, W)`` one-hot stack from int type indices.

    Index 0 (unknown) is included; the caller can drop it if they prefer
    a num_types-1 layout.
    """

    out = np.zeros((int(num_types), int(type_label.shape[0]), int(type_label.shape[1])), dtype=np.float32)
    for idx in range(int(num_types)):
        out[idx] = (type_label == idx).astype(np.float32)
    return out


def cell_type_assignment_for_crop(
    cell_type_artifact: dict[str, Any],
    crop_id: str,
) -> dict[str, int] | None:
    for crop in cell_type_artifact.get("crops", []):
        if str(crop.get("crop_id")) == str(crop_id):
            return {str(k): int(v) for k, v in crop.get("cell_id_to_type_index", {}).items()}
    return None
