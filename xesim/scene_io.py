"""Plan3 v37: scene I/O utilities — build MechanisticScene from a canonical
crop NPZ + cell_type_assignment.json. Used to feed Task 1 (tile -> scene
-> edit -> re-render) from canonical data without re-implementing the
loading logic everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .mechanistic_scene import MechanisticCell, MechanisticScene


def scene_from_canonical_crop(
    crop_npz_path: Path,
    cell_types_json_path: Path,
    image_size: int = 256,
    *,
    pixel_size: float = 0.2125,
) -> tuple[MechanisticScene, list[str]]:
    """Load a canonical crop NPZ and assemble a MechanisticScene.

    Returns (scene, type_names). Cells get their type from the
    cell_id_to_type_index map in cell_types_json_path. Nucleus IDs are
    remapped so each cell's nucleus uses the same label as its cell.
    Returns the scene with `cell_label` and `nucleus_label` arrays
    aligned to image_size (cropped from top-left).
    """
    with np.load(crop_npz_path, allow_pickle=True) as d:
        cell_label = np.asarray(d["cell_label"])[:image_size, :image_size].astype(np.int32)
        nucleus_label = np.asarray(d["nucleus_label"])[:image_size, :image_size].astype(np.int32)
        cell_ids = [str(v) for v in d["cell_ids"].tolist()] if "cell_ids" in d.files else []

    ct = json.loads(Path(cell_types_json_path).read_text())
    type_names = list(ct["type_names"])
    cid_to_type: dict[str, int] = {}
    for crop_entry in ct.get("crops", []):
        for cid, idx in crop_entry.get("cell_id_to_type_index", {}).items():
            cid_to_type[str(cid)] = int(idx)

    # Re-key nucleus_label so each cell's nucleus uses the cell's label.
    nz = np.unique(cell_label); nz = nz[nz > 0]
    nuc_remap = np.zeros_like(cell_label)
    cells = []
    for idx_pos, label_val in enumerate(nz):
        cell_pix = cell_label == int(label_val)
        nuc_in = nucleus_label[cell_pix]; nuc_in = nuc_in[nuc_in > 0]
        nucleus_label_for_cell: int | None = None
        if nuc_in.size:
            best = int(np.bincount(nuc_in).argmax())
            nuc_pix = (nucleus_label == best) & cell_pix
            if nuc_pix.sum() > 5:
                nuc_remap[nuc_pix] = int(label_val)
                nucleus_label_for_cell = int(label_val)
        cid = cell_ids[idx_pos] if idx_pos < len(cell_ids) else f"cell_{label_val}"
        type_idx = int(cid_to_type.get(str(cid), 0))
        type_name = type_names[type_idx] if 0 <= type_idx < len(type_names) else "unknown"
        cells.append(MechanisticCell(
            cell_id=str(cid),
            label=int(label_val),
            source="observed_anchor",
            cell_type=type_name,
            nucleus_label=nucleus_label_for_cell,
        ))

    scene = MechanisticScene(
        image_shape=(image_size, image_size),
        pixel_size=float(pixel_size),
        cell_label=cell_label,
        nucleus_label=nuc_remap,
        cells=tuple(cells),
        scene_id=str(crop_npz_path.stem),
    )
    return scene, type_names


def scene_to_npz(scene: MechanisticScene, path: Path) -> None:
    """Save scene's arrays to NPZ (cells records as JSON metadata)."""
    cells_json = json.dumps([c.to_dict() for c in scene.cells])
    np.savez(
        path,
        cell_label=scene.cell_label,
        nucleus_label=scene.nucleus_label,
        image_shape=np.asarray(scene.image_shape, dtype=np.int32),
        pixel_size=np.asarray(scene.pixel_size, dtype=np.float32),
        cells_json=cells_json,
        scene_id=scene.scene_id,
    )


def scene_from_npz(path: Path) -> MechanisticScene:
    with np.load(path, allow_pickle=True) as d:
        cell_label = np.asarray(d["cell_label"])
        nucleus_label = np.asarray(d["nucleus_label"])
        image_shape = tuple(int(x) for x in d["image_shape"].tolist())
        pixel_size = float(d["pixel_size"])
        cells_json = str(d["cells_json"])
        scene_id = str(d["scene_id"])
    cells = tuple(MechanisticCell.from_dict(c) for c in json.loads(cells_json))
    return MechanisticScene(
        image_shape=(image_shape[0], image_shape[1]),
        pixel_size=pixel_size,
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        cells=cells,
        scene_id=scene_id,
    )
