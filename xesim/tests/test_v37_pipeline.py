"""Tests for the v37 production pipeline modules."""

import json
import numpy as np
import pytest
from pathlib import Path

from xesim.mechanistic_scene import MechanisticCell, MechanisticScene
from xesim.scene_editor import SceneEditor
from xesim.structural_refiner import build_structural_channels, PER_CELL_LATENT_DIM


def _tiny_scene():
    cell_label = np.zeros((32, 32), dtype=np.int32)
    cell_label[5:15, 5:15] = 1
    cell_label[20:30, 20:30] = 2
    nuc = np.zeros_like(cell_label)
    nuc[8:12, 8:12] = 1
    nuc[23:27, 23:27] = 2
    cells = (
        MechanisticCell(cell_id="A", label=1, source="observed_anchor",
                        cell_type="Ductal/tumor epithelial", nucleus_label=1),
        MechanisticCell(cell_id="B", label=2, source="observed_anchor",
                        cell_type="Immune", nucleus_label=2),
    )
    return MechanisticScene(
        image_shape=(32, 32), pixel_size=0.2125,
        cell_label=cell_label, nucleus_label=nuc, cells=cells, scene_id="test",
    )


def test_scene_serialization_round_trip():
    scene = _tiny_scene()
    d = scene.to_dict(include_arrays=True)
    restored = MechanisticScene.from_dict(d)
    assert restored.image_shape == scene.image_shape
    assert restored.pixel_size == scene.pixel_size
    assert len(restored.cells) == len(scene.cells)
    assert restored.cells[0].cell_type == "Ductal/tumor epithelial"
    np.testing.assert_array_equal(restored.cell_label, scene.cell_label)


def test_scene_cell_latent_vector_round_trip():
    cell = MechanisticCell(cell_id="A", label=1, latent_vector=(0.1, -0.2, 0.3, 0.4))
    d = cell.to_dict()
    restored = MechanisticCell.from_dict(d)
    assert restored.latent_vector == (0.1, -0.2, 0.3, 0.4)


def test_scene_editor_remove():
    scene = _tiny_scene()
    edited = SceneEditor(scene).remove_cell(1).finalize()
    assert edited.cell_count() if hasattr(edited, "cell_count") else len(edited.cells) == 1
    assert (edited.cell_label == 1).sum() == 0
    assert edited.cells[0].label == 2


def test_scene_editor_remove_by_type():
    scene = _tiny_scene()
    edited = SceneEditor(scene).remove_cells_of_type("Immune").finalize()
    assert len(edited.cells) == 1
    assert edited.cells[0].cell_type == "Ductal/tumor epithelial"


def test_scene_editor_retype():
    scene = _tiny_scene()
    edited = SceneEditor(scene).retype_cell(1, "Endothelial").finalize()
    cell_1 = next(c for c in edited.cells if c.label == 1)
    assert cell_1.cell_type == "Endothelial"


def test_scene_editor_set_latent():
    scene = _tiny_scene()
    edited = (SceneEditor(scene)
              .set_latent(1, np.array([0.5, -0.5, 0.0, 1.0]))
              .finalize())
    cell_1 = next(c for c in edited.cells if c.label == 1)
    assert cell_1.latent_vector is not None
    assert cell_1.latent_vector[0] == pytest.approx(0.5)


def test_scene_editor_resample_all_latents():
    scene = _tiny_scene()
    edited = SceneEditor(scene).resample_all_latents(
        rng=np.random.default_rng(0), latent_dim=4,
    ).finalize()
    for c in edited.cells:
        assert c.latent_vector is not None
        assert len(c.latent_vector) == 4


def test_scene_editor_move():
    scene = _tiny_scene()
    # Move cell 1 by (+5, +5)
    edited = SceneEditor(scene).move_cell(1, dy=5, dx=5).finalize()
    # Cell 1 should now occupy [10:20, 10:20]
    assert (edited.cell_label[10:20, 10:20] == 1).any()
    assert (edited.cell_label[5:10, 5:10] == 1).sum() == 0


def test_scene_editor_add_from_template():
    scene = _tiny_scene()
    editor = SceneEditor(scene)
    new_lbl = editor.add_cell_from_template(
        source_label=1, target_yx=(20, 5), new_type="Fibroblast / CAF",
    )
    assert new_lbl > 0
    edited = editor.finalize()
    assert len(edited.cells) == 3
    new_cell = next(c for c in edited.cells if c.label == new_lbl)
    assert new_cell.cell_type == "Fibroblast / CAF"


def test_build_structural_channels_shapes():
    scene = _tiny_scene()
    # Compactify labels: 1, 2 already compact
    chans, names = build_structural_channels(
        scene.cell_label, scene.nucleus_label,
        cell_type_indices={1: 1, 2: 6},  # Ductal, Immune
        n_type_one_hot=7,
    )
    assert chans.shape == (8 + 7, 32, 32)  # 8 geom + 7 type one-hot
    assert "cell_mask" in names and "nucleus_mask" in names
    assert chans[0].max() == 1.0  # cell_mask binary
    assert chans[1].max() == 1.0  # nucleus_mask binary
    # Type one-hot for Ductal at index 8 (after 8 deterministic channels)
    type_oh_start = 8
    # Cell 1 (Ductal/type 1) has pixels where channel 8 (type 1) is 1
    assert (chans[type_oh_start] > 0).any()


def test_build_structural_intercellular_edge():
    # Two adjacent cells -> intercellular edge should fire
    cell_label = np.zeros((16, 16), dtype=np.int32)
    cell_label[2:8, 2:8] = 1
    cell_label[2:8, 8:14] = 2  # adjacent to cell 1
    chans, names = build_structural_channels(
        cell_label, np.zeros_like(cell_label),
        cell_type_indices={1: 1, 2: 1}, n_type_one_hot=2,
    )
    intercell_idx = names.index("intercellular_edge")
    # Edge column at x=7 and x=8 should be marked
    assert chans[intercell_idx, 4, 7] > 0 or chans[intercell_idx, 4, 8] > 0


def test_scene_finalize_immutable():
    scene = _tiny_scene()
    editor = SceneEditor(scene)
    edited = editor.remove_cell(1).finalize()
    # Editing the editor shouldn't affect already-finalized scene
    editor.remove_cell(2)
    edited2 = editor.finalize()
    assert len(edited.cells) == 1
    assert len(edited2.cells) == 0


def test_scene_provenance():
    scene = _tiny_scene()
    edited = SceneEditor(scene).remove_cell(1).finalize()
    assert "edits" in edited.provenance
