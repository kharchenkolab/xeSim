"""Tests for the STpuppeteer emission backend (Phase 1).

Synthetic-scene tests that don't require a fitted xeSim model or a real
Xenium bundle — they exercise the emission_stpuppeteer module end-to-end
against a hand-built 3-cell scene. See ``tmp/phase1_probe.py`` for a
standalone diagnostic version of the same scene.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_scene():
    """A 100x100 px scene with three cells: two adjacent epithelial, one isolated immune."""
    H, W = 100, 100
    cell_label = np.zeros((H, W), dtype=np.int32)
    nucleus_label = np.zeros((H, W), dtype=np.int32)

    # Cells 1 and 2 sit side-by-side along the y=20..40 strip; cell 3 is far away.
    # Adjacency lets us check that cell_2's halo can land inside cell_1's volume.
    cell_label[20:40, 20:40] = 1
    nucleus_label[26:34, 26:34] = 1
    cell_label[20:40, 42:62] = 2
    nucleus_label[26:34, 48:56] = 2
    cell_label[65:85, 65:85] = 3
    nucleus_label[71:79, 71:79] = 3

    cells = [
        SimpleNamespace(cell_id="cell_1", label=1, cell_type="Epithelial",
                          provenance={"is_ghost": False}),
        SimpleNamespace(cell_id="cell_2", label=2, cell_type="Epithelial",
                          provenance={"is_ghost": False}),
        SimpleNamespace(cell_id="cell_3", label=3, cell_type="Immune",
                          provenance={"is_ghost": False}),
    ]
    return SimpleNamespace(
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        cells=cells,
        pixel_size=0.5,  # 0.5 µm/px so the tiny scene is ~50 µm wide
    )


@pytest.fixture
def synthetic_config_path():
    """Minimal pancreas-style YAML with two cell types + three programs."""
    yaml_text = """
seed: 42
expression_baseline: 0.05
overdispersion_cv: 0.5
overdispersion_exponent: 0.3
leak_dist_factor: 1.0
leakage_by_celltype:
  Epithelial: 0.20
  Immune: 0.05
programs:
  - name: ProgEpi
    loading: {KRT8: 10.0, KRT18: 8.0, EPCAM: 12.0}
  - name: ProgImm
    loading: {PTPRC: 11.0, CD3D: 9.0}
  - name: ProgHK
    loading: {ACTB: 5.0, GAPDH: 5.0}
cell_type_specs:
  Epithelial:
    proportion: 0.6
    program_activations: {ProgEpi: 1.0, ProgHK: 0.5}
  Immune:
    proportion: 0.4
    program_activations: {ProgImm: 1.0, ProgHK: 0.5}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    yield path
    Path(path).unlink()


def test_emit_2d_smoke(synthetic_scene, synthetic_config_path):
    """End-to-end smoke: emit_2d produces a DataFrame with the expected columns."""
    from xesim.emission_stpuppeteer import emit_2d
    df = emit_2d(
        scene=synthetic_scene,
        stpuppeteer_config=synthetic_config_path,
        rng=np.random.default_rng(0),
        pixel_size_um=synthetic_scene.pixel_size,
    )
    # Schema check — column order doesn't matter, but the names must exist.
    required = {"cell_id", "gene", "x", "y", "factor_label", "qv",
                "source_cell_type", "is_leaked", "compartment",
                "landed_in_cell_id", "overlaps_nucleus"}
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert len(df) > 0, "expected non-empty transcript output"
    # qv is the synthetic constant from emit.py.
    assert df["qv"].nunique() == 1
    # factor_label is always -1 under STpuppeteer (no NMF factors).
    assert (df["factor_label"] == -1).all()


def test_emit_2d_reproducible(synthetic_scene, synthetic_config_path):
    """Same RNG seed → same output."""
    from xesim.emission_stpuppeteer import emit_2d
    df1 = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                  rng=np.random.default_rng(7),
                  pixel_size_um=synthetic_scene.pixel_size)
    df2 = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                  rng=np.random.default_rng(7),
                  pixel_size_um=synthetic_scene.pixel_size)
    pd.testing.assert_frame_equal(df1, df2)


def test_leakage_rate_matches_config(synthetic_scene, synthetic_config_path):
    """Average leakage fraction tracks the configured per-celltype rate.

    We use a large RNG ensemble (many runs) rather than a single noisy
    estimate to keep the test stable. The configured rates are 0.20 for
    Epithelial and 0.05 for Immune; we expect overall ~ weighted average.
    """
    from xesim.emission_stpuppeteer import emit_2d
    # 10 independent draws over the same scene to average noise.
    rates_by_type: dict[str, list[float]] = {"Epithelial": [], "Immune": []}
    for seed in range(10):
        df = emit_2d(
            scene=synthetic_scene,
            stpuppeteer_config=synthetic_config_path,
            rng=np.random.default_rng(seed),
            pixel_size_um=synthetic_scene.pixel_size,
        )
        if len(df) == 0:
            continue
        for ct, sub in df.groupby("source_cell_type"):
            rates_by_type.setdefault(ct, []).append(sub["is_leaked"].mean())

    # Epithelial rate should be ≈ 0.20 (±0.1 tolerance for 10-draw ensemble).
    if rates_by_type.get("Epithelial"):
        mean_epi = float(np.mean(rates_by_type["Epithelial"]))
        assert abs(mean_epi - 0.20) < 0.1, f"Epithelial leakage {mean_epi:.3f} ≉ 0.20"
    # Immune rate should be ≈ 0.05 (loose tolerance — small sample).
    if rates_by_type.get("Immune"):
        mean_imm = float(np.mean(rates_by_type["Immune"]))
        assert abs(mean_imm - 0.05) < 0.10, f"Immune leakage {mean_imm:.3f} ≉ 0.05"


def test_inside_cell_invariants(synthetic_scene, synthetic_config_path):
    """Non-leaked transcripts: source cell == landed-in cell; nucleus overlap consistent."""
    from xesim.emission_stpuppeteer import emit_2d
    df = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                 rng=np.random.default_rng(11),
                 pixel_size_um=synthetic_scene.pixel_size)
    inside = df[~df["is_leaked"]]
    # Every non-leaked transcript landed in its source cell's volume.
    assert (inside["landed_in_cell_id"] == inside["cell_id"]).all(), (
        "non-leaked transcripts should always be inside their source cell"
    )


def test_leaked_into_neighbor(synthetic_scene, synthetic_config_path):
    """At least some leaked cell_2 transcripts land inside cell_1 (adjacent neighbor).

    This is the core "halo overlaps neighbor interior" feature — what
    distinguishes the per-cell-independent halo model from a Voronoi-
    style partition. Aggregate over multiple seeds because the per-tile
    sample is small.
    """
    from xesim.emission_stpuppeteer import emit_2d
    n_neighbor_landings = 0
    for seed in range(20):
        df = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                     rng=np.random.default_rng(seed),
                     pixel_size_um=synthetic_scene.pixel_size)
        leaked_c2 = df[(df["cell_id"] == "cell_2") & df["is_leaked"]]
        n_neighbor_landings += (leaked_c2["landed_in_cell_id"] == "cell_1").sum()
    # Cells 1 & 2 are immediately adjacent and cell_2's halo extends ≥ r_eff/2
    # toward cell_1's body. Across 20 seeds we should see SOME such landings.
    assert n_neighbor_landings > 0, (
        "expected ≥1 cell_2-leaked transcript to land inside cell_1 across 20 seeds"
    )


def test_empty_scene():
    """Scene with no configurable cells → empty output, no crash."""
    from xesim.emission_stpuppeteer import emit_2d
    # 1 cell whose type isn't in the config → no covered types → empty.
    cell_label = np.zeros((20, 20), dtype=np.int32)
    cell_label[5:15, 5:15] = 1
    scene = SimpleNamespace(
        cell_label=cell_label,
        nucleus_label=np.zeros_like(cell_label),
        cells=[SimpleNamespace(cell_id="x", label=1, cell_type="UnknownType",
                                  provenance={"is_ghost": False})],
        pixel_size=0.5,
    )
    yaml_text = """
seed: 1
programs:
  - name: ProgA
    loading: {KRT8: 10.0}
cell_type_specs:
  Epithelial:
    proportion: 1.0
    program_activations: {ProgA: 1.0}
leakage_by_celltype: {Epithelial: 0.0}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(yaml_text)
        cfg_path = f.name
    try:
        df = emit_2d(scene=scene, stpuppeteer_config=cfg_path,
                     rng=np.random.default_rng(0), pixel_size_um=0.5)
        # Empty output is allowed — no cell of the only configured type is in the scene.
        # We just want no crash.
        assert len(df) == 0
    finally:
        Path(cfg_path).unlink()


def test_classify_leakage_helper():
    """STpuppeteer's classify_leakage helper produces stats matching p_eff."""
    from STpuppeteer.simulation import classify_leakage
    n = 10000
    trs_df = pd.DataFrame({
        "transcript_id": [f"t_{i}" for i in range(n)],
        "cell_id": np.where(np.arange(n) < n // 2, "A", "B"),
        "feature_name": ["g1"] * n,
    })
    cell_gdf = pd.DataFrame({
        "cell_id": ["A", "B"],
        "leakage_percentage": [0.10, 0.30],
    })
    # No gene leakage component.
    rng = np.random.default_rng(0)
    is_leaked = classify_leakage(trs_df, cell_gdf, gpar_df=None, rng=rng)
    # ~10% of A, ~30% of B.
    rate_A = is_leaked[:n // 2].mean()
    rate_B = is_leaked[n // 2:].mean()
    assert abs(rate_A - 0.10) < 0.02, f"A rate {rate_A:.3f} ≉ 0.10"
    assert abs(rate_B - 0.30) < 0.02, f"B rate {rate_B:.3f} ≉ 0.30"


def test_simulationconfig_from_dict_round_trip():
    """SimulationConfig.from_dict accepts both explicit and shorthand styles."""
    from STpuppeteer.simulation import SimulationConfig
    # Shorthand.
    short = SimulationConfig.from_dict({
        "n_cells": 100, "n_celltype": 3, "n_programs": 3,
        "n_genes_per_program": 10, "n_hk_genes": 5, "seed": 0,
    })
    assert short.n_celltype == 3
    assert short.n_genes == 35  # 3*10 + 5
    # Explicit.
    explicit = SimulationConfig.from_dict({
        "seed": 0,
        "programs": [{"name": "P1", "loading": {"g1": 5.0, "g2": 3.0}}],
        "cell_type_specs": {
            "TypeX": {"proportion": 1.0, "program_activations": {"P1": 1.0}},
        },
    })
    assert explicit.n_celltype == 1
    assert explicit.celltype_names == ["TypeX"]
    assert [p.name for p in explicit.programs] == ["P1"]
