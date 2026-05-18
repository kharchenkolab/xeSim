"""Phase 5 tests for synthetic transcripts in `generate`."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path("/workspace/xeSim/tmp/xesim_v16_model")
PRIORS_JSON = MODEL_DIR / "priors" / "transcripts_nmf.json"
CANONICAL_DIR = MODEL_DIR / "canonical"

requires_priors = pytest.mark.skipif(
    not PRIORS_JSON.exists(),
    reason=f"no priors at {PRIORS_JSON}",
)
requires_canonical = pytest.mark.skipif(
    not (CANONICAL_DIR / "manifest.json").exists(),
    reason=f"no canonical crops at {CANONICAL_DIR}",
)


@pytest.fixture(scope="module")
def priors() -> dict:
    return json.loads(PRIORS_JSON.read_text())


@pytest.fixture(scope="module")
def model():
    from xesim import XesimModel
    return XesimModel.load(MODEL_DIR)


@pytest.fixture(scope="module")
def scene(model):
    """Use a real canonical crop's scene as the substrate for sampling."""
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    return scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )[0]


@requires_priors
@requires_canonical
def test_sample_returns_expected_columns(model, scene):
    from xesim.transcripts import sample_scene_transcripts
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(1))
    assert {"cell_id", "gene", "x", "y", "factor_label",
            "qv", "source_cell_type"}.issubset(set(tx.columns))
    assert len(tx) > 0


@requires_priors
@requires_canonical
def test_sample_positions_inside_cells(model, scene):
    """Sampled (x, y) µm coordinates should map back to pixels with the
    sampled cell's label > 0 in the scene cell_label."""
    from xesim.transcripts import sample_scene_transcripts
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(2))
    h, w = scene.cell_label.shape
    px_size = scene.pixel_size
    py = np.clip((tx["y"].to_numpy() / px_size).astype(int), 0, h - 1)
    pxc = np.clip((tx["x"].to_numpy() / px_size).astype(int), 0, w - 1)
    labels_at_mol = scene.cell_label[py, pxc]
    # Jitter can push molecules just outside the cell mask (≤ 0.5 µm).
    inside_or_near = labels_at_mol > 0
    assert inside_or_near.mean() >= 0.80, \
        f"only {inside_or_near.mean():.1%} of molecules land in any cell"


@requires_priors
@requires_canonical
def test_sample_factor_label_in_biology(model, scene, priors):
    """factor_label values should be in the biology_factors set (1-based)."""
    from xesim.transcripts import sample_scene_transcripts
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(3))
    bio = set(int(f) for f in priors["factor_signatures"]["biology_factors"])
    assert set(int(x) for x in tx["factor_label"].unique()).issubset(bio)


@requires_priors
@requires_canonical
def test_sample_endocrine_emits_ins(model, scene, priors):
    """For an Endocrine cell type with INS as the top loader of the Endocrine
    factor, sampled transcripts should contain INS at a meaningfully higher
    rate than chance."""
    from xesim.transcripts import sample_scene_transcripts
    if "Endocrine" not in priors["per_type_alpha"]:
        pytest.skip("no Endocrine type in priors")
    # Filter the scene to only Endocrine cells. Easier: build a synthetic
    # scene where all cells are Endocrine by reassignment.
    new_cells = tuple(replace(c, cell_type="Endocrine") for c in scene.cells)
    s2 = replace(scene, cells=new_cells)
    tx = sample_scene_transcripts(model.transcripts_priors, s2,
                                     rng=np.random.default_rng(4))
    if len(tx) < 200:
        pytest.skip(f"too few sampled molecules: {len(tx)}")
    gene_count = tx["gene"].value_counts(normalize=True)
    n_genes = len(priors["panel"]["gene_names"])
    chance = 1.0 / n_genes
    ins_rate = float(gene_count.get("INS", 0.0))
    assert ins_rate > 5 * chance, \
        f"INS rate {ins_rate:.4f} is not meaningfully above chance {chance:.4f}"


@requires_priors
@requires_canonical
def test_sample_count_scales_with_negbin_mean(model, priors):
    """Total molecules per cell should track per_type_count_negbin's mean."""
    from xesim.transcripts import sample_scene_transcripts
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )
    # Pick a type with at least 5 cells in the scene
    counts_by_type: dict[str, int] = {}
    for c in scene.cells:
        counts_by_type[c.cell_type] = counts_by_type.get(c.cell_type, 0) + 1
    target = max(counts_by_type, key=counts_by_type.get)
    if counts_by_type[target] < 5:
        pytest.skip(f"too few cells of dominant type: {counts_by_type}")
    nb = priors["per_type_count_negbin"].get(target)
    if nb is None:
        pytest.skip(f"no negbin for {target!r}")
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(5))
    sub = tx[tx["source_cell_type"] == target]
    per_cell = sub.groupby("cell_id").size()
    if len(per_cell) < 3:
        pytest.skip(f"too few sampled cells of {target!r}")
    # Within a 4x window: lognormal samples vary widely on small n
    assert per_cell.mean() >= 0.25 * nb["mean"]
    assert per_cell.mean() <= 4.0 * nb["mean"]


# ---------------------------------------------------------------------------
# Phase 6 — slice-guided transcript guidance
# ---------------------------------------------------------------------------


@requires_priors
@requires_canonical
def test_tile_aware_per_type_alpha(model, priors):
    """Per-type alpha computed over a guide subset should differ from the
    global alpha when the tile composition is biased."""
    from xesim.transcripts import tile_aware_per_type_alpha
    cf = model.cell_factor_fractions(include_annotation=True)
    # Take only Endocrine cells — the tile-aware alpha should look exactly
    # like the global Endocrine alpha (within rounding) when subset == all
    # cells of that type, but diverge if we subset to one specific tile.
    endo_ids = cf[cf["_cell_type"] == "Endocrine"]["cell_id"].head(50).tolist()
    if len(endo_ids) < 5:
        pytest.skip("not enough Endocrine cells in priors fit")
    out = tile_aware_per_type_alpha(model.transcripts_priors, cf,
                                       guide_cell_ids=endo_ids)
    assert "Endocrine" in out
    endo_alpha = np.asarray(out["Endocrine"])
    glob = np.asarray(priors["per_type_alpha"]["Endocrine"])
    # Should be close but not identical (subset of 50 cells, smoothing 0.3)
    assert np.allclose(endo_alpha.sum(), 1.0, atol=1e-3)
    assert not np.allclose(endo_alpha, glob, atol=1e-4)


@requires_priors
@requires_canonical
def test_slice_guided_uses_tile_alpha(model, priors):
    """When guide-mode is on and the guide tile has a known bias toward one
    type, sampled transcripts in the synth scenes should reflect that bias
    (vs the non-guided forward mode)."""
    from xesim.transcripts import (
        sample_scene_transcripts, tile_aware_per_type_alpha,
    )
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )
    cf = model.cell_factor_fractions(include_annotation=True)
    guide_ids = [c.cell_id for c in scene.cells]
    tile_alpha = tile_aware_per_type_alpha(model.transcripts_priors, cf,
                                              guide_cell_ids=guide_ids)
    # The tile-aware alpha for any type with cells in the guide should
    # differ from the global alpha.
    types_in_guide = {c.cell_type for c in scene.cells if c.cell_type}
    n_differs = 0
    for t in types_in_guide:
        if t in tile_alpha and t in priors["per_type_alpha"]:
            a1 = np.asarray(tile_alpha[t])
            a0 = np.asarray(priors["per_type_alpha"][t])
            if not np.allclose(a1, a0, atol=1e-3):
                n_differs += 1
    assert n_differs >= 1, f"tile-aware alpha didn't diverge from global for any of {types_in_guide}"
    # Confirm the sampler accepts the override
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(0),
                                     per_type_alpha_override=tile_alpha)
    assert len(tx) > 0


# ---------------------------------------------------------------------------
# Phase 8 — controlled admixture in generate
# ---------------------------------------------------------------------------


@requires_priors
def test_apply_admixture_rate_zero_removes_admixture(priors):
    """rate_multiplier=0.0 zeros out admixture-flagged factor weights for
    every type that has admixture rules pointing into it."""
    from xesim.transcripts import apply_admixture_rate
    pa0 = apply_admixture_rate(priors, 0.0)
    pa1 = apply_admixture_rate(priors, 1.0)
    bio = list(priors["factor_signatures"]["biology_factors"])
    factor_to_pos = {int(f): i for i, f in enumerate(bio)}
    for t, admix_factors in priors.get("per_type_admixture_factor_indices", {}).items():
        if not admix_factors or t not in pa0:
            continue
        a = np.asarray(pa0[t])
        # Sum should still be 1 (renormalized to native after killing admix)
        assert np.isclose(a.sum(), 1.0, atol=1e-6), \
            f"type {t!r}: sum {a.sum()}"
        # Admix-flagged positions should be ~0 (assuming admix factor was in priors)
        admix_pos = [factor_to_pos[int(f)] for f in admix_factors
                     if int(f) in factor_to_pos]
        assert all(a[p] < 1e-6 for p in admix_pos), \
            f"type {t!r}: admix factors not zeroed: {[a[p] for p in admix_pos]}"


@requires_priors
def test_apply_admixture_rate_high_boosts_admixture(priors):
    """rate_multiplier=3.0 increases admixture mass for affected types."""
    from xesim.transcripts import apply_admixture_rate
    pa1 = apply_admixture_rate(priors, 1.0)
    pa3 = apply_admixture_rate(priors, 3.0)
    bio = list(priors["factor_signatures"]["biology_factors"])
    factor_to_pos = {int(f): i for i, f in enumerate(bio)}
    n_increased = 0
    for t, admix_factors in priors.get("per_type_admixture_factor_indices", {}).items():
        if not admix_factors or t not in pa1:
            continue
        admix_pos = [factor_to_pos[int(f)] for f in admix_factors
                     if int(f) in factor_to_pos]
        m1 = sum(pa1[t][p] for p in admix_pos)
        m3 = sum(pa3[t][p] for p in admix_pos)
        if m3 > m1 + 1e-3:
            n_increased += 1
    assert n_increased >= 1, "no type had admixture boosted by 3x rate"


@requires_priors
@requires_canonical
def test_admixture_rate_affects_sampling(model, priors):
    """A high admixture rate should shift the sampled-transcript factor
    distribution toward admixture-flagged factors."""
    from xesim.transcripts import apply_admixture_rate, sample_scene_transcripts
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )
    # Pick a type whose admixture set is non-empty
    admix_by_type = priors.get("per_type_admixture_factor_indices", {})
    target_type = None
    target_admix = None
    for t, fs in admix_by_type.items():
        if fs:
            target_type, target_admix = t, set(int(x) for x in fs)
            break
    if target_type is None:
        pytest.skip("no admixture types in priors")

    # Force-relabel all scene cells to target_type so we can measure cleanly
    from dataclasses import replace
    s2 = replace(scene, cells=tuple(replace(c, cell_type=target_type)
                                          for c in scene.cells))

    pa_pure = apply_admixture_rate(priors, 0.0)
    pa_observed = apply_admixture_rate(priors, 1.0)
    tx_pure = sample_scene_transcripts(priors, s2, rng=np.random.default_rng(1),
                                          per_type_alpha_override=pa_pure)
    tx_obs = sample_scene_transcripts(priors, s2, rng=np.random.default_rng(1),
                                         per_type_alpha_override=pa_observed)
    if len(tx_pure) < 100 or len(tx_obs) < 100:
        pytest.skip(f"too few sampled mol: pure={len(tx_pure)} obs={len(tx_obs)}")

    pure_admix = tx_pure["factor_label"].isin(target_admix).mean()
    obs_admix = tx_obs["factor_label"].isin(target_admix).mean()
    assert obs_admix > pure_admix, \
        f"observed-rate sampling didn't have more admixture than pure: " \
        f"obs={obs_admix:.3f} pure={pure_admix:.3f}"
    # Pure-biology sampling should have near-zero admixture factor mass
    assert pure_admix < 0.05, f"pure-biology admix rate too high: {pure_admix:.3f}"


@requires_priors
@requires_canonical
def test_write_emits_transcripts_parquet(model, scene, tmp_path):
    """`model.write(item)` with a 'transcripts' DataFrame writes
    transcripts.parquet alongside scene.json/render.png."""
    from xesim.transcripts import sample_scene_transcripts
    tx = sample_scene_transcripts(model.transcripts_priors, scene,
                                     rng=np.random.default_rng(6))
    # Build a minimal item; renderer output is dummy 3-channel zeros
    item = {
        "scene": scene,
        "render": np.zeros((3, scene.image_shape[0], scene.image_shape[1]),
                            dtype=np.float32),
        "seed": 0,
        "transcripts": tx,
    }
    d = model.write(item, tmp_path, name="t1")
    assert (d / "transcripts.parquet").exists()
    import pandas as pd
    back = pd.read_parquet(d / "transcripts.parquet")
    assert list(back.columns) == list(tx.columns)
    assert len(back) == len(tx)
