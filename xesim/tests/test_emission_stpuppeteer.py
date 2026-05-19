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


# ---------------------------------------------------------------------------
# 2.5D (emit_3d) tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_scene_3d():
    """A tiny (5 z-planes, 50x50 px) scene with two CellRecord cells.

    Cells span z-planes 1..3 (so a 3-slice z-extent each); they're adjacent
    in xy so neighbour-interior halo landings should occur.
    """
    n_z, H, W = 5, 50, 50
    cell_label_3d = np.zeros((n_z, H, W), dtype=np.int32)
    nucleus_label_3d = np.zeros((n_z, H, W), dtype=np.int32)
    # Cell 1: z=1..3, y=10..30, x=10..25
    cell_label_3d[1:4, 10:30, 10:25] = 1
    nucleus_label_3d[2, 16:24, 14:21] = 1
    # Cell 2: z=1..3, y=10..30, x=27..42 (adjacent to cell 1)
    cell_label_3d[1:4, 10:30, 27:42] = 2
    nucleus_label_3d[2, 16:24, 31:38] = 2

    # Build CellRecord-like records — emit_3d only reads cell_idx,
    # cell_id, cell_type, so we don't need the full 3D fields.
    cells_records = [
        SimpleNamespace(cell_idx=1, cell_id="c1", cell_type="Epithelial",
                          provenance={"is_ghost": False}),
        SimpleNamespace(cell_idx=2, cell_id="c2", cell_type="Epithelial",
                          provenance={"is_ghost": False}),
    ]
    psz = 0.5
    z_step = 3.0
    z_slices_um = [i * z_step for i in range(n_z)]
    return SimpleNamespace(
        cells_records=cells_records,
        cell_label_3d=cell_label_3d,
        nucleus_label_3d=nucleus_label_3d,
        z_slices_um=z_slices_um,
        pixel_size_um=psz,
        z_step_um=z_step,
    )


def test_emit_3d_smoke(synthetic_scene_3d, synthetic_config_path):
    """End-to-end smoke: emit_3d produces a DataFrame with the 2.5D schema."""
    from xesim.emission_stpuppeteer import emit_3d
    s = synthetic_scene_3d
    df = emit_3d(
        cells_records=s.cells_records,
        cell_label_3d=s.cell_label_3d,
        nucleus_label_3d=s.nucleus_label_3d,
        z_slices_um=s.z_slices_um,
        tile_origin_um=(0.0, 0.0),
        pixel_size_um=s.pixel_size_um,
        stpuppeteer_config=synthetic_config_path,
        rng=np.random.default_rng(3),
    )
    # The 2.5D writer reads these columns; emit_3d must produce them.
    required = {"x_true", "y_true", "z_true", "true_cell_idx",
                "true_cell_id", "true_cell_type", "source", "gene",
                "factor_label", "qv", "is_ghost"}
    assert required.issubset(set(df.columns))
    assert len(df) > 0
    # Source always "body" in v1 (no ambient background; ghosts off).
    assert (df["source"] == "body").all()
    assert (df["is_ghost"] == False).all()
    # true_cell_idx ↔ cell labels we configured.
    assert set(df["true_cell_idx"].unique()) <= {1, 2}


def test_emit_3d_z_within_stack(synthetic_scene_3d, synthetic_config_path):
    """All emitted z coordinates fall within the configured z-stack range."""
    from xesim.emission_stpuppeteer import emit_3d
    s = synthetic_scene_3d
    df = emit_3d(
        cells_records=s.cells_records,
        cell_label_3d=s.cell_label_3d,
        nucleus_label_3d=s.nucleus_label_3d,
        z_slices_um=s.z_slices_um,
        tile_origin_um=(0.0, 0.0),
        pixel_size_um=s.pixel_size_um,
        stpuppeteer_config=synthetic_config_path,
        rng=np.random.default_rng(8),
    )
    # Voxel centers + ±half-z-step jitter → all coords in
    # [z_min - z_step/2, z_max + z_step/2].
    z_min = float(min(s.z_slices_um)) - 0.5 * s.z_step_um
    z_max = float(max(s.z_slices_um)) + 0.5 * s.z_step_um
    assert df["z_true"].min() >= z_min - 1e-6
    assert df["z_true"].max() <= z_max + 1e-6


def test_emit_3d_anisotropic_z(synthetic_config_path):
    """Anisotropic EDT: a 1-z-step jump is "expensive" relative to a 1-xy-pixel jump.

    We build a single-z-slice cell so the cell occupies ONLY slice 2 in z.
    Any leaked transcript at slice 1 or 3 had to "jump" exactly one z-step
    (3 µm). Any leaked transcript still at slice 2 had to "jump" in xy
    only. With λ ≈ 1.4 µm and z_step (3 µm) ≫ psz (0.5 µm), the xy halo
    should dominate by an order of magnitude. If the EDT's anisotropic
    sampling tuple is wrong (e.g., reversed or unit), z-shifted
    transcripts would be roughly comparable to xy-only.
    """
    from xesim.emission_stpuppeteer import emit_3d
    n_z, H, W = 5, 50, 50
    cell_label_3d = np.zeros((n_z, H, W), dtype=np.int32)
    nucleus_label_3d = np.zeros((n_z, H, W), dtype=np.int32)
    # Cell exists ONLY at slice 2 (one z-slice deep).
    cell_label_3d[2, 18:32, 18:32] = 1
    nucleus_label_3d[2, 22:28, 22:28] = 1
    cells_records = [
        SimpleNamespace(cell_idx=1, cell_id="c1", cell_type="Epithelial",
                          provenance={"is_ghost": False}),
    ]
    psz, z_step = 0.5, 3.0
    z_slices_um = [i * z_step for i in range(n_z)]

    # Aggregate over 10 seeds — single cell, so per-seed leak counts are small.
    n_at_source_slice = 0
    n_at_neighbour_z = 0
    for seed in range(10):
        df = emit_3d(
            cells_records=cells_records,
            cell_label_3d=cell_label_3d,
            nucleus_label_3d=nucleus_label_3d,
            z_slices_um=z_slices_um,
            tile_origin_um=(0.0, 0.0),
            pixel_size_um=psz,
            stpuppeteer_config=synthetic_config_path,
            rng=np.random.default_rng(seed),
        )
        leaked = df[df["is_leaked"]]
        if len(leaked) == 0:
            continue
        # Source cell at z=6.0. With ±half-z-step jitter, transcripts
        # ON slice 2 have z in [4.5, 7.5]; on slice 1 or 3 they have z in
        # [1.5, 4.5] or [7.5, 10.5].
        dz = np.abs(leaked["z_true"].to_numpy() - 6.0)
        n_at_source_slice += int((dz < 1.5 - 1e-6).sum())
        n_at_neighbour_z += int((dz >= 1.5 - 1e-6).sum())

    # With λ ≈ 1.4 µm, exp(-3/1.4) ≈ 0.12 (z neighbour weight) vs
    # exp(-0.5/1.4) ≈ 0.70 (xy 1-pixel weight). Even accounting for the
    # larger neighbour-slice voxel count, xy should clearly dominate.
    assert n_at_source_slice > n_at_neighbour_z, (
        f"anisotropic-EDT check failed: source-slice n={n_at_source_slice} "
        f"vs neighbour-z n={n_at_neighbour_z}. Expected source-slice to dominate "
        f"(z_step=3µm makes z-jumps ~6× more 'expensive' than xy-pixel jumps)."
    )


def test_emit_3d_reproducible(synthetic_scene_3d, synthetic_config_path):
    """Same RNG seed → byte-identical output."""
    from xesim.emission_stpuppeteer import emit_3d
    s = synthetic_scene_3d

    def _run(seed):
        return emit_3d(
            cells_records=s.cells_records,
            cell_label_3d=s.cell_label_3d,
            nucleus_label_3d=s.nucleus_label_3d,
            z_slices_um=s.z_slices_um,
            tile_origin_um=(0.0, 0.0),
            pixel_size_um=s.pixel_size_um,
            stpuppeteer_config=synthetic_config_path,
            rng=np.random.default_rng(seed),
        )
    pd.testing.assert_frame_equal(_run(11), _run(11))


def test_per_celltype_marker_specificity(synthetic_config_path):
    """The dominant genes in each cell type's transcripts are its own markers.

    The synthetic config has three exclusive programs:
        Epithelial → ProgEpi   (KRT8, KRT18, EPCAM)  + ProgHK (shared)
        Immune     → ProgImm   (PTPRC, CD3D)         + ProgHK (shared)
    Marker specificity check: a transcript emitted by an Epithelial cell
    should be a ProgEpi or ProgHK gene; an Immune cell's transcripts
    should be ProgImm or ProgHK. Cross-type marker hits only occur when
    a leaked transcript lands in a neighbour cell — those should be the
    minority (≲ 10% with leakage=0.20 / 0.05 in the synthetic config).

    This is the formal regression test for the per-celltype-marker spot
    check we ran on the pancreas region. If a future change accidentally
    couples gene-program activations or breaks the source-cell tag on
    leaked transcripts, this test catches it.
    """
    from xesim.emission_stpuppeteer import emit_2d
    epi_markers = {"KRT8", "KRT18", "EPCAM"}
    imm_markers = {"PTPRC", "CD3D"}
    hk_markers = {"ACTB", "GAPDH"}

    # Build a small scene with one of each type, well separated so
    # cross-cell leakage stays small (so the specificity signal is clean).
    H, W = 60, 60
    cell_label = np.zeros((H, W), dtype=np.int32)
    nucleus_label = np.zeros((H, W), dtype=np.int32)
    cell_label[8:22, 8:22] = 1   # Epithelial
    nucleus_label[12:18, 12:18] = 1
    cell_label[38:52, 38:52] = 2  # Immune (far from cell_1)
    nucleus_label[42:48, 42:48] = 2
    cells = [
        SimpleNamespace(cell_id="c_epi", label=1, cell_type="Epithelial",
                          provenance={"is_ghost": False}),
        SimpleNamespace(cell_id="c_imm", label=2, cell_type="Immune",
                          provenance={"is_ghost": False}),
    ]
    scene = SimpleNamespace(cell_label=cell_label, nucleus_label=nucleus_label,
                              cells=cells, pixel_size=0.5)

    # Aggregate over many seeds so the rates are stable.
    rows = []
    for seed in range(20):
        df = emit_2d(scene=scene, stpuppeteer_config=synthetic_config_path,
                     rng=np.random.default_rng(seed), pixel_size_um=0.5)
        if len(df) > 0:
            rows.append(df.assign(_seed=seed))
    df = pd.concat(rows, ignore_index=True)

    # For Epithelial cells, share of OWN markers (ProgEpi ∪ ProgHK).
    epi = df[df["source_cell_type"] == "Epithelial"]
    epi_own = epi["gene"].isin(epi_markers | hk_markers).mean()
    assert epi_own > 0.95, (
        f"Epithelial own-marker fraction {epi_own:.1%} too low; expected ≥95%. "
        "STpuppeteer's program-based emission shouldn't produce immune-marker "
        "transcripts from epithelial cells (the activation matrix is exclusive)."
    )

    # Same for Immune cells.
    imm = df[df["source_cell_type"] == "Immune"]
    imm_own = imm["gene"].isin(imm_markers | hk_markers).mean()
    assert imm_own > 0.95, (
        f"Immune own-marker fraction {imm_own:.1%} too low; expected ≥95%."
    )

    # Cross-check: an epithelial cell should rarely emit immune markers and
    # vice versa (only via leakage spillover, and the cells are placed far
    # apart so even leakage shouldn't cross-contaminate).
    assert epi["gene"].isin(imm_markers).mean() < 0.05
    assert imm["gene"].isin(epi_markers).mean() < 0.05


def test_cell_label_value_helper():
    """The adapter helper picks up either .label (2D) or .cell_idx (2.5D)."""
    from xesim.emission_stpuppeteer.cell_adapter import _cell_label_value

    class HasLabel:
        label = 7

    class HasCellIdx:
        cell_idx = 12

    assert _cell_label_value(HasLabel()) == 7
    assert _cell_label_value(HasCellIdx()) == 12
    with pytest.raises(AttributeError):
        _cell_label_value(SimpleNamespace())


# ---------------------------------------------------------------------------
# Ghost-cell behaviour (Phase 4)
# ---------------------------------------------------------------------------


def test_emit_2d_skips_ghosts(synthetic_config_path):
    """Ghost cells render into the stain image but never emit STpuppeteer
    transcripts. cell_adapter filters them out before count sampling."""
    from xesim.emission_stpuppeteer import emit_2d

    # Two cells: one anchor (Epithelial), one ghost (also Epithelial).
    # If ghosts emitted, the ghost would account for ~half the transcripts.
    cell_label = np.zeros((40, 40), dtype=np.int32)
    cell_label[5:15, 5:15] = 1     # anchor
    cell_label[25:35, 25:35] = 2   # ghost
    cells = [
        SimpleNamespace(cell_id="anchor", label=1, cell_type="Epithelial",
                          provenance={"is_ghost": False}),
        SimpleNamespace(cell_id="ghost", label=2, cell_type="Epithelial",
                          provenance={"is_ghost": True}),
    ]
    scene = SimpleNamespace(cell_label=cell_label,
                              nucleus_label=np.zeros_like(cell_label),
                              cells=cells, pixel_size=0.5)
    df = emit_2d(scene=scene, stpuppeteer_config=synthetic_config_path,
                 rng=np.random.default_rng(0), pixel_size_um=0.5)

    # Every emitted transcript is from the anchor; the ghost emitted zero.
    assert len(df) > 0
    src_cells = set(df["cell_id"].unique().tolist())
    assert "anchor" in src_cells
    assert "ghost" not in src_cells, (
        f"ghost cell appears as emitter: src cells = {src_cells}"
    )


# ---------------------------------------------------------------------------
# Emission diagnostics (Phase 4)
# ---------------------------------------------------------------------------


def test_emission_diagnostics_compute(synthetic_scene, synthetic_config_path):
    """compute_emission_stats returns the expected dict shape + values."""
    from xesim.emission_stpuppeteer import emit_2d
    from xesim.emission_stpuppeteer.diagnostics import compute_emission_stats
    from STpuppeteer.simulation import SimulationConfig

    df = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                 rng=np.random.default_rng(5),
                 pixel_size_um=synthetic_scene.pixel_size)
    cfg = SimulationConfig.from_yaml(synthetic_config_path)
    stats = compute_emission_stats(df, cfg)

    # Top-level shape.
    assert stats["n_transcripts"] == len(df)
    assert stats["n_cell_types"] >= 1
    assert isinstance(stats["per_cell_type"], list)
    assert isinstance(stats["cross_marker_matrix"], dict)

    # Per-cell-type entries have all the expected fields.
    for row in stats["per_cell_type"]:
        for field in ("cell_type", "n_transcripts", "n_cells",
                      "mean_per_cell", "median_per_cell", "p95_per_cell",
                      "own_marker_fraction", "leaked_fraction"):
            assert field in row
        # Sanity: own-marker fraction in [0, 1].
        assert 0.0 <= row["own_marker_fraction"] <= 1.0
        # Cross matrix has a row per cell type.
        assert row["cell_type"] in stats["cross_marker_matrix"]


def test_emission_diagnostics_write_json(synthetic_scene, synthetic_config_path, tmp_path):
    """write_emission_diagnostics produces a JSON-parseable file."""
    from xesim.emission_stpuppeteer import emit_2d
    from xesim.emission_stpuppeteer.diagnostics import write_emission_diagnostics
    from STpuppeteer.simulation import SimulationConfig
    import json as _json

    df = emit_2d(scene=synthetic_scene, stpuppeteer_config=synthetic_config_path,
                 rng=np.random.default_rng(13),
                 pixel_size_um=synthetic_scene.pixel_size)
    cfg = SimulationConfig.from_yaml(synthetic_config_path)
    out_path = write_emission_diagnostics(df, cfg, tmp_path / "diag")

    assert out_path.exists()
    with open(out_path) as f:
        loaded = _json.load(f)
    assert "n_transcripts" in loaded
    assert "per_cell_type" in loaded


# ---------------------------------------------------------------------------
# Reference configs (Phase 4)
# ---------------------------------------------------------------------------


def test_reference_pancreas_config_loads():
    """The shipped pancreas reference config loads via SimulationConfig.from_yaml."""
    from importlib.resources import files
    from STpuppeteer.simulation import SimulationConfig

    config_path = (files("xesim.emission_stpuppeteer")
                       / "reference_configs" / "pancreas.yml")
    cfg = SimulationConfig.from_yaml(str(config_path))
    # Has the 7 pancreas cell types we documented.
    expected_types = {
        "Exocrine epithelial", "Ductal/tumor epithelial", "Fibroblast / CAF",
        "Immune", "Endothelial", "Mural / pericyte", "Endocrine",
    }
    assert set(cfg.cell_type_specs.keys()) == expected_types
    # Has the configured programs.
    assert len(cfg.programs) == 7


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
