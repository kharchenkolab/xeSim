"""Tests for ``xesim re-emit-molecules`` (Phase 3).

These tests focus on the orchestration logic — the underlying emission
(emit_2d / emit_3d) is already covered by test_emission_stpuppeteer.py.
Here we verify:

- The CLI refuses to overwrite an existing directory.
- Source bundle artifacts are preserved byte-for-byte.
- transcripts.parquet schema matches what the original writer produced.
- Different seeds → different transcripts, same config → reproducible.
- Bundle-reader correctly detects 2D vs 2.5D.

We build a tiny synthetic 2.5D bundle in a tmpdir to avoid pulling in
the full pancreas bundle.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic 2.5D bundle fixture
# ---------------------------------------------------------------------------


def _build_minimal_25d_bundle(out: Path) -> None:
    """Build a tiny but schema-correct 2.5D synth bundle in `out`.

    Two cells, 6 z-planes, 30×30 px tile. Enough to exercise the reader's
    polygon rasterisation + 2.5D pipeline without depending on a real
    Xenium bundle on disk.
    """
    out.mkdir(parents=True)
    (out / "ground_truth").mkdir()
    (out / "morphology_focus").mkdir()

    pixel_size = 0.5
    z_step = 3.0
    n_z = 6
    H = W = 30
    tile_xmin, tile_ymin = 100.0, 200.0
    tile_xmax = tile_xmin + W * pixel_size
    tile_ymax = tile_ymin + H * pixel_size

    # experiment.xenium with synth_metadata so the reader picks scene_mode=2.5d.
    exp = {
        "pixel_size": pixel_size,
        "z_step_size": z_step,
        "tile_bounds_um": [tile_xmin, tile_ymin, tile_xmax, tile_ymax],
        "num_cells": 2,
        "num_transcripts": 0,  # to be overwritten by re-emit
        "synth_metadata": {"scene_mode": "2.5d", "n_z": n_z},
    }
    (out / "experiment.xenium").write_text(json.dumps(exp))

    # Two square cells, adjacent in xy, centred at slice 2 with extent 3.
    cells = pd.DataFrame({
        "cell_id": ["c1", "c2"],
        "x_centroid": [tile_xmin + 7 * pixel_size, tile_xmin + 22 * pixel_size],
        "y_centroid": [tile_ymin + 15 * pixel_size, tile_ymin + 15 * pixel_size],
        "transcript_counts": [0, 0],
        "control_probe_counts": [0, 0],
        "control_codeword_counts": [0, 0],
        "unassigned_codeword_counts": [0, 0],
        "deprecated_codeword_counts": [0, 0],
        "total_counts": [0, 0],
        "cell_area": [50.0, 50.0],
        "nucleus_area": [10.0, 10.0],
    })
    cells.to_parquet(out / "cells.parquet")

    # cells_synth.parquet — ground truth with cell_type.
    pd.DataFrame({
        "cell_id": ["c1", "c2"],
        "cell_type": ["Epithelial", "Immune"],
        "is_ghost": [False, False],
        "centroid_x": cells["x_centroid"].values,
        "centroid_y": cells["y_centroid"].values,
        "area_um2": cells["cell_area"].values,
        "source": ["observed_anchor", "observed_anchor"],
    }).to_parquet(out / "ground_truth" / "cells_synth.parquet")

    # cells_3d.parquet — needed by 2.5D bundle_reader to build CellRecord.
    pd.DataFrame({
        "cell_id": ["c1", "c2"],
        "cell_idx": [1, 2],
        "cell_type": ["Epithelial", "Immune"],
        "centroid_x": cells["x_centroid"].values,
        "centroid_y": cells["y_centroid"].values,
        "z_center_um": [6.0, 6.0],            # slice 2
        "z_extent_um": [9.0, 9.0],            # spans ~slices 1..3
        "t_x": [0.0, 0.0],
        "t_y": [0.0, 0.0],
        "t_z": [1.0, 1.0],
        "is_unobserved": [False, False],
    }).to_parquet(out / "ground_truth" / "cells_3d.parquet")

    # Cell polygons: small squares around each centroid (4 verts each).
    half = 4 * pixel_size
    def _poly(cx, cy):
        return np.array([
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ], dtype=np.float32)

    rows = []
    for label, cid, cx, cy in [(1, "c1", cells.loc[0, "x_centroid"], cells.loc[0, "y_centroid"]),
                                  (2, "c2", cells.loc[1, "x_centroid"], cells.loc[1, "y_centroid"])]:
        for vx, vy in _poly(cx, cy):
            rows.append({"cell_id": cid, "vertex_x": float(vx),
                            "vertex_y": float(vy), "label_id": label})
    pd.DataFrame(rows).to_parquet(out / "cell_boundaries.parquet")

    # Smaller nuclei (same square, half size)
    half_n = 2 * pixel_size
    def _nuc(cx, cy):
        return np.array([
            (cx - half_n, cy - half_n),
            (cx + half_n, cy - half_n),
            (cx + half_n, cy + half_n),
            (cx - half_n, cy + half_n),
        ], dtype=np.float32)

    rows = []
    for label, cid, cx, cy in [(1, "c1", cells.loc[0, "x_centroid"], cells.loc[0, "y_centroid"]),
                                  (2, "c2", cells.loc[1, "x_centroid"], cells.loc[1, "y_centroid"])]:
        for vx, vy in _nuc(cx, cy):
            rows.append({"cell_id": cid, "vertex_x": float(vx),
                            "vertex_y": float(vy), "label_id": label})
    pd.DataFrame(rows).to_parquet(out / "nucleus_boundaries.parquet")

    # Stub the files re-emit doesn't read but is supposed to preserve.
    (out / "transcripts.parquet").write_bytes(b"placeholder")
    (out / "transcripts.csv.gz").write_bytes(b"placeholder")
    (out / "morphology.ome.tif").write_bytes(b"placeholder-morphology")
    (out / "gene_panel.json").write_text(json.dumps({"payload": {"targets": []}}))
    (out / "ground_truth" / "molecule_provenance.parquet").write_bytes(b"placeholder")


@pytest.fixture
def synthetic_25d_bundle(tmp_path):
    """Path to a fresh tiny 2.5D bundle in a tmpdir."""
    src = tmp_path / "src_bundle"
    _build_minimal_25d_bundle(src)
    return src


@pytest.fixture
def stpuppeteer_yaml(tmp_path):
    """Two-cell-type explicit STpuppeteer config that uses gene names which
    are NOT in the synthetic bundle's gene panel (panel is empty here, so
    no validation conflict; emit_2d/emit_3d skip the panel check anyway)."""
    yml = tmp_path / "cfg.yml"
    yml.write_text("""
seed: 42
expression_baseline: 0.05
overdispersion_cv: 0.5
overdispersion_exponent: 0.3
leak_dist_factor: 1.0
leakage_by_celltype:
  Epithelial: 0.10
  Immune: 0.10
programs:
  - name: ProgEpi
    loading: {EPCAM: 10.0, KRT7: 8.0}
  - name: ProgImm
    loading: {PTPRC: 11.0, CD3D: 9.0}
cell_type_specs:
  Epithelial:
    proportion: 0.5
    program_activations: {ProgEpi: 1.0}
  Immune:
    proportion: 0.5
    program_activations: {ProgImm: 1.0}
""")
    return yml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reemit_out_must_not_exist(synthetic_25d_bundle, stpuppeteer_yaml, tmp_path):
    """Re-emit refuses to write into an existing directory."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    existing = tmp_path / "already_exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        reemit_molecules(
            src_bundle=synthetic_25d_bundle,
            out_bundle=existing,
            stpuppeteer_config=stpuppeteer_yaml,
        )


def test_reemit_src_eq_out_rejected(synthetic_25d_bundle, stpuppeteer_yaml):
    """Re-emit refuses to write to the same path as the source."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    with pytest.raises(ValueError, match="same as src"):
        reemit_molecules(
            src_bundle=synthetic_25d_bundle,
            out_bundle=synthetic_25d_bundle,
            stpuppeteer_config=stpuppeteer_yaml,
        )


def test_reemit_25d_preserves_source(synthetic_25d_bundle, stpuppeteer_yaml, tmp_path):
    """Re-emit must not modify the source bundle; output must preserve all
    artifacts except transcripts + molecule_provenance + experiment.xenium.
    """
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    out = tmp_path / "out_bundle"
    src_files_before = {
        p.relative_to(synthetic_25d_bundle): hashlib.md5(p.read_bytes()).hexdigest()
        for p in synthetic_25d_bundle.rglob("*") if p.is_file()
    }
    reemit_molecules(
        src_bundle=synthetic_25d_bundle, out_bundle=out,
        stpuppeteer_config=stpuppeteer_yaml,
    )
    # Source must be untouched.
    src_files_after = {
        p.relative_to(synthetic_25d_bundle): hashlib.md5(p.read_bytes()).hexdigest()
        for p in synthetic_25d_bundle.rglob("*") if p.is_file()
    }
    assert src_files_before == src_files_after, "source bundle was modified"

    # Output: morphology + cells + boundaries + ground-truth cells_3d/cells_synth
    # must be byte-identical to source.
    preserved = [
        "cells.parquet", "cell_boundaries.parquet", "nucleus_boundaries.parquet",
        "morphology.ome.tif", "gene_panel.json",
        "ground_truth/cells_3d.parquet", "ground_truth/cells_synth.parquet",
    ]
    for fname in preserved:
        s_h = hashlib.md5((synthetic_25d_bundle / fname).read_bytes()).hexdigest()
        d_h = hashlib.md5((out / fname).read_bytes()).hexdigest()
        assert s_h == d_h, f"{fname} was modified by re-emit"


def test_reemit_25d_schema(synthetic_25d_bundle, stpuppeteer_yaml, tmp_path):
    """transcripts.parquet has the 2.5D bundle writer's schema."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    out = tmp_path / "out_bundle"
    reemit_molecules(
        src_bundle=synthetic_25d_bundle, out_bundle=out,
        stpuppeteer_config=stpuppeteer_yaml,
    )
    df = pd.read_parquet(out / "transcripts.parquet")
    expected_cols = {"transcript_id", "x_location", "y_location", "z_location",
                     "qv", "feature_name", "cell_id", "overlaps_nucleus",
                     "nucleus_distance"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0

    # Provenance has the matching extra columns.
    prov = pd.read_parquet(out / "ground_truth" / "molecule_provenance.parquet")
    assert {"true_cell_id", "true_cell_type", "is_ghost", "x_true", "y_true",
            "z_true", "gene"}.issubset(set(prov.columns))
    assert len(prov) == len(df)


def test_reemit_reproducible(synthetic_25d_bundle, stpuppeteer_yaml, tmp_path):
    """Same seed → byte-identical transcripts.parquet."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    out1 = tmp_path / "out_a"
    out2 = tmp_path / "out_b"
    reemit_molecules(src_bundle=synthetic_25d_bundle, out_bundle=out1,
                     stpuppeteer_config=stpuppeteer_yaml, seed=7)
    reemit_molecules(src_bundle=synthetic_25d_bundle, out_bundle=out2,
                     stpuppeteer_config=stpuppeteer_yaml, seed=7)
    a = pd.read_parquet(out1 / "transcripts.parquet")
    b = pd.read_parquet(out2 / "transcripts.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_reemit_different_seed_different_output(synthetic_25d_bundle,
                                                   stpuppeteer_yaml, tmp_path):
    """Different seed → different transcript counts / locations."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    out1 = tmp_path / "seed_0"
    out2 = tmp_path / "seed_99"
    reemit_molecules(src_bundle=synthetic_25d_bundle, out_bundle=out1,
                     stpuppeteer_config=stpuppeteer_yaml, seed=0)
    reemit_molecules(src_bundle=synthetic_25d_bundle, out_bundle=out2,
                     stpuppeteer_config=stpuppeteer_yaml, seed=99)
    a = pd.read_parquet(out1 / "transcripts.parquet")
    b = pd.read_parquet(out2 / "transcripts.parquet")
    # Counts can match by chance with such a tiny scene, but at least
    # the per-transcript x/y won't match across seeds.
    if len(a) == len(b):
        assert not np.allclose(a["x_location"].values, b["x_location"].values), (
            "different seeds produced identical x_location — RNG seed not threaded"
        )


def test_reemit_updates_experiment_metadata(synthetic_25d_bundle,
                                               stpuppeteer_yaml, tmp_path):
    """experiment.xenium num_transcripts is updated; synth_metadata.reemit flagged."""
    from xesim.emission_stpuppeteer.reemit import reemit_molecules
    out = tmp_path / "out_bundle"
    result = reemit_molecules(
        src_bundle=synthetic_25d_bundle, out_bundle=out,
        stpuppeteer_config=stpuppeteer_yaml,
    )
    with open(out / "experiment.xenium") as f:
        exp = json.load(f)
    assert exp["num_transcripts"] == result.n_transcripts
    assert exp["synth_metadata"]["reemit"] is True


def test_bundle_reader_detects_25d(synthetic_25d_bundle):
    """The reader correctly identifies the synthetic bundle as 2.5D."""
    from xesim.emission_stpuppeteer.bundle_reader import read_bundle_meta
    meta = read_bundle_meta(synthetic_25d_bundle)
    assert meta.scene_mode == "2.5d"
    assert meta.n_z == 6
    assert meta.pixel_size_um == pytest.approx(0.5)
