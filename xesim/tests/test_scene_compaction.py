"""Parity tests for `_compact_scene_for_storage`.

The compaction shrinks a finished tile scene's RAM (int32 masks -> uint16,
object molecule columns -> category, float64 -> float32) before it
accumulates in the parent's scenes_out list. The contract is that every
piece of metadata the bundle writer extracts from a scene is *identical*
whether or not the scene was compacted — only the in-RAM representation
changes, never the values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xesim.mechanistic_scene import MechanisticScene, MechanisticCell
from xesim.scene_2d import Scene2D
from xesim.scene_2d.scene_pipeline import _compact_scene_for_storage
from xesim.scene_2d.bundle_writer import (
    _collect_polygons, _build_real_cells_df, _build_cells_df,
)


def _make_scene() -> Scene2D:
    """A 64x64 tile with 3 anchor cells + 1 ghost, blob masks, molecules."""
    H = W = 64
    cell_label = np.zeros((H, W), dtype=np.int32)
    nucleus_label = np.zeros((H, W), dtype=np.int32)
    # three anchor blobs (labels 1,2,3) + one ghost blob (label 4)
    cell_label[5:15, 5:15] = 1;  nucleus_label[8:12, 8:12] = 1
    cell_label[20:32, 20:34] = 2; nucleus_label[24:28, 24:30] = 2
    cell_label[40:50, 10:22] = 3; nucleus_label[43:47, 13:18] = 3
    cell_label[45:55, 45:58] = 4; nucleus_label[48:52, 48:53] = 4
    cells = (
        MechanisticCell(cell_id="cellA", label=1, source="observed_anchor",
                        cell_type="Tumor",
                        provenance={"type_resolution": {
                            "source": "annotation", "confidence": 0.9,
                            "evidence": {"foo": 1}}}),
        MechanisticCell(cell_id="cellB", label=2, source="observed_anchor",
                        cell_type="Bcell",
                        provenance={"type_resolution": {
                            "source": "transcripts", "confidence": 0.7,
                            "evidence": {"bar": 2}}}),
        MechanisticCell(cell_id="cellC", label=3, source="tx_inferred",
                        cell_type="Tcell",
                        provenance={"type_resolution": {
                            "source": "stain_knn", "confidence": 0.5,
                            "evidence": {}}}),
        MechanisticCell(cell_id="ghost_0000007", label=4, source="synthetic",
                        cell_type="Tumor",
                        provenance={"is_ghost": True}),
    )
    mech = MechanisticScene(
        image_shape=(H, W), pixel_size=0.2125,
        cell_label=cell_label, nucleus_label=nucleus_label,
        cells=cells, scene_id="tile_test",
    )
    rng = np.random.default_rng(0)
    n = 400
    cids = rng.choice(["cellA", "cellB", "cellC", "ghost_0000007"], n)
    genes = rng.choice(["GENE1", "GENE2", "GENE3"], n)
    types = rng.choice(["Tumor", "Bcell", "Tcell"], n)
    mols = pd.DataFrame({
        "x": rng.uniform(0, W * 0.2125, n).astype(np.float64),
        "y": rng.uniform(0, H * 0.2125, n).astype(np.float64),
        "gene": genes.astype(object),
        "true_cell_id": cids.astype(object),
        "is_ghost": (cids == "ghost_0000007"),
        "true_factor": rng.integers(0, 5, n).astype(np.int64),
        "qv": rng.uniform(20, 40, n).astype(np.float64),
        "source_cell_type": types.astype(object),
    })
    return Scene2D(mech_scene=mech, molecules=mols,
                   tile_bounds_um=(100.0, 100.0 + W * 0.2125,
                                   200.0, 200.0 + H * 0.2125),
                   pixel_size=0.2125,
                   provenance={"owning_tile_bounds_um": [100, 113.6, 200, 213.6]})


def test_masks_downcast_to_uint16_values_preserved():
    sc = _make_scene()
    comp = _compact_scene_for_storage(sc)
    assert comp.mech_scene.cell_label.dtype == np.uint16
    assert comp.mech_scene.nucleus_label.dtype == np.uint16
    np.testing.assert_array_equal(sc.mech_scene.cell_label,
                                  comp.mech_scene.cell_label.astype(np.int32))
    np.testing.assert_array_equal(sc.mech_scene.nucleus_label,
                                  comp.mech_scene.nucleus_label.astype(np.int32))


def test_molecule_values_preserved_after_compaction():
    sc = _make_scene()
    comp = _compact_scene_for_storage(sc)
    a, b = sc.molecules, comp.molecules
    # String columns are left untouched (already compact str dtype).
    for c in ("gene", "true_cell_id", "source_cell_type"):
        pd.testing.assert_series_equal(
            a[c].reset_index(drop=True), b[c].reset_index(drop=True),
            check_names=False)
    # float64 coords -> float32, values equal to float32 round-trip (which
    # is exactly what the writer emits anyway).
    for c in ("x", "y", "qv"):
        assert b[c].dtype == np.float32
        np.testing.assert_array_equal(a[c].to_numpy().astype(np.float32),
                                      b[c].to_numpy())
    np.testing.assert_array_equal(a["is_ghost"].to_numpy(),
                                  b["is_ghost"].to_numpy())


@pytest.mark.parametrize("kind", ["cell", "nucleus"])
@pytest.mark.parametrize("is_ghost", [False, True])
def test_collect_polygons_identical(kind, is_ghost):
    sc = _make_scene()
    comp = _compact_scene_for_storage(sc)
    a = _collect_polygons([sc], kind=kind, is_ghost=is_ghost)
    b = _collect_polygons([comp], kind=kind, is_ghost=is_ghost)
    assert [t[0] for t in a] == [t[0] for t in b]
    for (_, ax, ay), (_, bx, by) in zip(a, b):
        np.testing.assert_array_equal(ax, bx)
        np.testing.assert_array_equal(ay, by)


def test_build_real_cells_df_identical():
    sc = _make_scene()
    comp = _compact_scene_for_storage(sc)
    # The writer passes a transcripts_df keyed by `cell_id` (true_cell_id is
    # renamed before this call); mirror that here.
    tx_a = sc.molecules.rename(columns={"true_cell_id": "cell_id"})
    tx_b = comp.molecules.rename(columns={"true_cell_id": "cell_id"})
    a = _build_real_cells_df([sc], tx_a)
    b = _build_real_cells_df([comp], tx_b)
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                  b.reset_index(drop=True))


def test_build_cells_df_identical():
    sc = _make_scene()
    comp = _compact_scene_for_storage(sc)
    a = _build_cells_df([sc])
    b = _build_cells_df([comp])
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                  b.reset_index(drop=True))
