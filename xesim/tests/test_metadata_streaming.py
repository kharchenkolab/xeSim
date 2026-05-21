"""Parity tests for per-tile metadata extraction (mask-free bundle write).

The pipeline now extracts each tile's per-cell geometry + contour polygons
(`extract_scene_geometry`) as the tile finishes, stashes it keyed by
cell_id, and frees the label masks. The bundle writer rebuilds the
boundary / cells / cells_synth tables from that stash instead of re-reading
masks. This must be byte-identical to the legacy mask-reading path.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xesim.mechanistic_scene import MechanisticScene, MechanisticCell
from xesim.scene_2d import Scene2D
from xesim.scene_2d.bundle_writer import (
    extract_scene_geometry, write_bundle,
    _collect_polygons, _build_real_cells_df, _build_cells_df,
)
from xesim.scene_2d.scene_pipeline import _extract_and_strip


def _make_tile(tile_idx: int, x0: float, y0: float) -> Scene2D:
    """A 64x64 tile with 2 anchors + 1 ghost, blob masks + molecules.
    cell_ids are globally unique across tiles."""
    H = W = 64
    psz = 0.2125
    cell_label = np.zeros((H, W), dtype=np.int32)
    nucleus_label = np.zeros((H, W), dtype=np.int32)
    cell_label[5:18, 6:20] = 1;  nucleus_label[9:13, 10:15] = 1
    cell_label[30:44, 32:48] = 2; nucleus_label[34:38, 36:42] = 2
    cell_label[46:58, 8:22] = 3;  nucleus_label[49:53, 11:16] = 3   # ghost
    cells = (
        MechanisticCell(cell_id=f"anchorA_{tile_idx}", label=1,
                        source="observed_anchor", cell_type="Tumor",
                        provenance={"type_resolution": {
                            "source": "annotation", "confidence": 0.9,
                            "evidence": {"a": 1}}}),
        MechanisticCell(cell_id=f"anchorB_{tile_idx}", label=2,
                        source="observed_anchor", cell_type="Bcell",
                        provenance={"type_resolution": {
                            "source": "transcripts", "confidence": 0.6,
                            "evidence": {}}}),
        MechanisticCell(cell_id=f"ghost_{tile_idx:07d}", label=3,
                        source="synthetic", cell_type="Tumor",
                        provenance={"is_ghost": True}),
    )
    mech = MechanisticScene(image_shape=(H, W), pixel_size=psz,
                            cell_label=cell_label, nucleus_label=nucleus_label,
                            cells=cells, scene_id=f"tile_{tile_idx}")
    rng = np.random.default_rng(tile_idx)
    n = 300
    cids = rng.choice([f"anchorA_{tile_idx}", f"anchorB_{tile_idx}",
                       f"ghost_{tile_idx:07d}"], n)
    mols = pd.DataFrame({
        "x": rng.uniform(0, W * psz, n),
        "y": rng.uniform(0, H * psz, n),
        "gene": rng.choice(["G1", "G2", "G3"], n),
        "true_cell_id": cids,
        "is_ghost": np.array([c.startswith("ghost") for c in cids]),
        "true_factor": rng.integers(0, 4, n),
        "qv": rng.uniform(20, 40, n),
        "source_cell_type": rng.choice(["Tumor", "Bcell"], n),
    })
    return Scene2D(mech_scene=mech, molecules=mols,
                   tile_bounds_um=(x0, x0 + W * psz, y0, y0 + H * psz),
                   pixel_size=psz, provenance={})


def _read_bundle(d: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for rel in ["transcripts.parquet", "cell_boundaries.parquet",
                "nucleus_boundaries.parquet", "cells.parquet",
                "ground_truth/cells_synth.parquet",
                "ground_truth/ghost_cell_boundaries.parquet",
                "ground_truth/ghost_nucleus_boundaries.parquet"]:
        p = d / rel
        out[rel] = pd.read_parquet(p) if p.exists() else None
    return out


def test_extract_scene_geometry_matches_mask_functions():
    """The stash produced by extract_scene_geometry yields the same
    polygons + centroid/area that the legacy mask-reading functions do."""
    sc = _make_tile(0, 100.0, 200.0)
    stash = extract_scene_geometry(sc)
    for kind, ghost in [("cell", False), ("nucleus", False),
                        ("cell", True), ("nucleus", True)]:
        legacy = _collect_polygons([sc], kind=kind, is_ghost=ghost)
        viastash = _collect_polygons([sc], kind=kind, is_ghost=ghost,
                                     geom_stash=stash)
        assert [t[0] for t in legacy] == [t[0] for t in viastash]
        for (_, ax, ay), (_, bx, by) in zip(legacy, viastash):
            np.testing.assert_array_equal(ax, bx)
            np.testing.assert_array_equal(ay, by)
    tx = sc.molecules.rename(columns={"true_cell_id": "cell_id"})
    pd.testing.assert_frame_equal(
        _build_real_cells_df([sc], tx).reset_index(drop=True),
        _build_real_cells_df([sc], tx, geom_stash=stash).reset_index(drop=True))
    pd.testing.assert_frame_equal(
        _build_cells_df([sc]).reset_index(drop=True),
        _build_cells_df([sc], geom_stash=stash).reset_index(drop=True))


def test_full_bundle_write_parity_legacy_vs_stash():
    """End-to-end: writing a 3-tile bundle via the legacy mask path vs the
    extract-and-strip stash path produces identical metadata parquets."""
    tiles = [_make_tile(0, 0.0, 0.0), _make_tile(1, 14.0, 0.0),
             _make_tile(2, 0.0, 14.0)]

    legacy_dir = Path(tempfile.mkdtemp(prefix="ms_legacy_"))
    stash_dir = Path(tempfile.mkdtemp(prefix="ms_stash_"))
    try:
        # Legacy: scenes keep masks, no stash.
        write_bundle(output_dir=legacy_dir, scenes=[s for s in tiles],
                     render_images=None, channel_names=["DAPI"],
                     pixel_size_um=0.2125, overwrite=True)
        # Stash: extract per tile, strip masks, pass stash.
        stash: dict = {}
        stripped = [_extract_and_strip(s, stash) for s in tiles]
        # masks really are gone
        for s in stripped:
            assert s.mech_scene.cell_label.size == 0
        write_bundle(output_dir=stash_dir, scenes=stripped,
                     render_images=None, channel_names=["DAPI"],
                     pixel_size_um=0.2125, overwrite=True, geom_stash=stash)

        a, b = _read_bundle(legacy_dir), _read_bundle(stash_dir)
        for key in a:
            assert (a[key] is None) == (b[key] is None), key
            if a[key] is None:
                continue
            # transcripts coords are float32 in the stripped path; compare
            # with a float32 round-trip tolerance, exact for everything else.
            if key == "transcripts.parquet":
                # transcript_id is random; compare everything else sorted.
                cols = [c for c in a[key].columns if c != "transcript_id"]
                aa = a[key][cols].sort_values(cols).reset_index(drop=True)
                bb = b[key][cols].sort_values(cols).reset_index(drop=True)
                pd.testing.assert_frame_equal(aa, bb, obj="transcripts")
                continue
            pd.testing.assert_frame_equal(
                a[key].reset_index(drop=True), b[key].reset_index(drop=True),
                check_dtype=True, obj=key)
    finally:
        shutil.rmtree(legacy_dir, ignore_errors=True)
        shutil.rmtree(stash_dir, ignore_errors=True)


def test_ghost_drop_after_strip_excludes_dropped_ghosts():
    """After masks are stripped, dropping a ghost cell from the scene must
    drop it from ghost boundaries too (writer iterates surviving cells)."""
    from dataclasses import replace
    sc = _make_tile(0, 0.0, 0.0)
    stash: dict = {}
    stripped = _extract_and_strip(sc, stash)
    # Drop the ghost from the scene's cells (simulating _drop_ghosts_to_target)
    survivors = tuple(c for c in stripped.mech_scene.cells
                      if not c.provenance.get("is_ghost", False))
    dropped = replace(stripped, mech_scene=replace(stripped.mech_scene,
                                                   cells=survivors))
    ghost_polys = _collect_polygons([dropped], kind="cell", is_ghost=True,
                                    geom_stash=stash)
    assert ghost_polys == []
    # anchors still present
    anchor_polys = _collect_polygons([dropped], kind="cell", is_ghost=False,
                                     geom_stash=stash)
    assert len(anchor_polys) == 2
