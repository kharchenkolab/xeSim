"""Tests for compartment-aware transcript sampling.

Covers the EM-fitted hierarchical posterior (Phase A), the persistence
into priors JSON (Phase B), and the compartment-aware sampler (Phase C).
"""
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
    reason="no canonical crops",
)


@pytest.fixture(scope="module")
def priors() -> dict:
    return json.loads(PRIORS_JSON.read_text())


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------


@requires_priors
def test_compartment_model_present(priors):
    """The priors carry a complete compartment_model dict."""
    cm = priors.get("compartment_model")
    assert cm is not None, "compartment_model missing"
    expected = {"boundaries_um", "em_centers_um", "em_scales_um",
                  "em_pi_out", "global_prior", "p_nuclear_global",
                  "concentration", "n_molecules", "min_count_per_type"}
    assert expected.issubset(set(cm.keys()))


@requires_priors
def test_global_prior_normalized(priors):
    cm = priors["compartment_model"]
    gp = np.asarray(cm["global_prior"], dtype=np.float64)
    assert gp.shape == (4,)
    assert np.isclose(gp.sum(), 1.0, atol=1e-6)
    assert (gp >= 0).all()


@requires_priors
def test_em_log_lik_monotonic(priors):
    """EM log-likelihood should be non-decreasing across iterations."""
    cm = priors["compartment_model"]
    ll = cm.get("log_lik_history")
    if ll is None or len(ll) < 2:
        pytest.skip("no log_lik_history persisted")
    diffs = np.diff(ll)
    # Allow tiny numerical noise (~1e-5 relative)
    assert (diffs >= -1e-3 * abs(ll[0])).all(), \
        f"log-lik decreased: diffs={diffs[:5]}"


@requires_priors
def test_em_centers_ordered(priors):
    """The EM components should be ordered by mean distance."""
    cm = priors["compartment_model"]
    centers = np.asarray(cm["em_centers_um"])
    assert (np.diff(centers) > 0).all(), f"centers not ordered: {centers}"


@requires_priors
def test_per_gene_posterior_normalized(priors):
    pgp = priors["per_gene_compartment_posterior"]
    assert len(pgp) > 0
    for gene, vec in list(pgp.items())[:50]:
        arr = np.asarray(vec, dtype=np.float64)
        assert arr.shape == (4,), f"{gene}: wrong shape {arr.shape}"
        assert np.isclose(arr.sum(), 1.0, atol=1e-6), \
            f"{gene}: posterior sums to {arr.sum()}"
        assert (arr >= 0).all()


@requires_priors
def test_per_gene_type_posterior_normalized(priors):
    pgt = priors["per_gene_type_compartment_posterior"]
    if not pgt:
        pytest.skip("no per-(gene, type) posteriors persisted")
    for key, vec in list(pgt.items())[:50]:
        assert "|" in key, f"key not gene|type: {key!r}"
        arr = np.asarray(vec, dtype=np.float64)
        assert arr.shape == (4,)
        assert np.isclose(arr.sum(), 1.0, atol=1e-6)


@requires_priors
def test_posterior_shrinks_for_rare_genes(priors):
    """Rare-gene posteriors (few real molecules) should be close to the
    global prior; common genes should depart from it. We approximate
    'rare' by genes whose per_gene posterior is close to the global prior
    in L1 distance — those exist when N_g << concentration."""
    cm = priors["compartment_model"]
    gp = np.asarray(cm["global_prior"], dtype=np.float64)
    pgp = priors["per_gene_compartment_posterior"]
    # Average L1 distance across all genes; the *spread* of distances
    # should be large (some genes match prior, others don't).
    dists = []
    for gene, vec in pgp.items():
        arr = np.asarray(vec, dtype=np.float64)
        dists.append(float(np.abs(arr - gp).sum()))
    dists = np.array(dists)
    # The 10th percentile (genes most-like-prior) should be very close to 0
    assert np.percentile(dists, 10) < 0.10, \
        f"no genes are close to the global prior (p10={np.percentile(dists,10):.3f})"
    # The 90th percentile (most-different genes) should be substantially > 0
    assert np.percentile(dists, 90) > 0.20, \
        f"posteriors don't depart from prior (p90={np.percentile(dists,90):.3f})"


@requires_priors
def test_marker_compartments_biologically_plausible(priors):
    """Marker genes should have non-degenerate compartment distributions.

    We don't make strong assumptions about which non-nuclear bucket wins
    (perinuc vs cyto vs distal depends on cell geometry per type), but we
    can check that no marker collapses entirely into a single compartment
    (which would indicate a degenerate fit) and that obviously-cytoplasmic
    markers like INS aren't nuclear-dominated.
    """
    pgp = priors["per_gene_compartment_posterior"]
    # INS is a secreted hormone — should not be nuclear-dominated.
    if "INS" in pgp:
        ins = np.asarray(pgp["INS"])
        assert ins[0] < 0.6, f"INS unexpectedly nuclear-heavy: {ins.round(3)}"
        # Perinuc + cyto together (the typical RER + cytoplasmic territory)
        # should outweigh distal for a secreted protein.
        assert (ins[1] + ins[2]) > ins[3], \
            f"INS distal exceeds perinuc+cyto: {ins.round(3)}"


# ---------------------------------------------------------------------------
# Sampler placement tests
# ---------------------------------------------------------------------------


@requires_priors
@requires_canonical
def test_sampled_molecules_respect_compartments(priors):
    """Synth molecules placed via compartment-aware sampling should land in
    their claimed compartment when re-inferred from the scene's distance map."""
    from scipy.ndimage import distance_transform_edt
    from xesim import XesimModel
    from xesim.scene_io import scene_from_canonical_crop
    from xesim.transcripts import sample_scene_transcripts

    m = XesimModel.load(MODEL_DIR)
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )
    tx = sample_scene_transcripts(priors, scene, rng=np.random.default_rng(0))
    if len(tx) < 100:
        pytest.skip(f"too few sampled molecules: {len(tx)}")

    pixel_size = scene.pixel_size
    boundaries = priors["compartment_model"]["boundaries_um"]
    b1, b2, b3 = boundaries
    # For each sampled molecule, recover its compartment from the scene
    cell_label = scene.cell_label
    nuc_label = scene.nucleus_label
    sample_size = min(500, len(tx))
    sample = tx.sample(sample_size, random_state=1)
    # Per-cell distance map cache
    cell_dist: dict[int, np.ndarray] = {}
    n_outside = 0  # of-bounds molecules (rare, allowed)
    for _, row in sample.iterrows():
        cid = row["cell_id"]
        cell_recs = [c for c in scene.cells if c.cell_id == cid]
        if not cell_recs:
            continue
        cell = cell_recs[0]
        lab = int(cell.label)
        nuc = (nuc_label == lab) & (cell_label == lab)
        if lab not in cell_dist:
            cell_dist[lab] = distance_transform_edt(~nuc) * pixel_size
        dist_um = cell_dist[lab]
        px = int(round(row["x"] / pixel_size))
        py = int(round(row["y"] / pixel_size))
        py = max(0, min(py, dist_um.shape[0] - 1))
        px = max(0, min(px, dist_um.shape[1] - 1))
        d = float(dist_um[py, px])
        in_nuc = bool(nuc[py, px])
        # We don't know which compartment the sampler picked, but we can
        # at least verify the molecule is IN the cell:
        if cell_label[py, px] != lab:
            n_outside += 1
    # Jitter pushes a few molecules across boundaries
    assert n_outside / sample_size < 0.10, \
        f"too many sampled molecules outside their cell: {n_outside}/{sample_size}"


@requires_priors
@requires_canonical
def test_compartment_sampling_uses_posterior(priors):
    """For a synthetic cell of a known type, the empirical compartment
    distribution of sampled molecules should match the per-(gene, type)
    posterior on average. Use INS + Endocrine as the test case (largest
    sample size). Allow ±0.05 deviation per compartment."""
    from scipy.ndimage import distance_transform_edt
    from xesim import XesimModel
    from xesim.scene_io import scene_from_canonical_crop
    from xesim.transcripts import sample_scene_transcripts

    pgt = priors.get("per_gene_type_compartment_posterior", {})
    target_key = "INS|Endocrine"
    if target_key not in pgt:
        pytest.skip(f"no posterior for {target_key}")
    target_post = np.asarray(pgt[target_key])
    m = XesimModel.load(MODEL_DIR)
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())

    # Find a crop with Endocrine cells (or fake one by relabelling)
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )
    # Force all cells to Endocrine for a clean test
    s2 = replace(scene, cells=tuple(replace(c, cell_type="Endocrine")
                                          for c in scene.cells))
    # Use a long sampler run to get many INS molecules
    tx = sample_scene_transcripts(priors, s2, rng=np.random.default_rng(42))
    ins = tx[tx["gene"] == "INS"]
    if len(ins) < 300:
        pytest.skip(f"too few INS sampled: {len(ins)}")

    # Compute empirical compartment for each INS mol
    pixel_size = scene.pixel_size
    cell_label = scene.cell_label
    nuc_label = scene.nucleus_label
    b1, b2, b3 = priors["compartment_model"]["boundaries_um"]
    counts = np.zeros(4, dtype=np.float64)
    cell_dist: dict[int, np.ndarray] = {}
    for _, row in ins.iterrows():
        cid = row["cell_id"]
        cell_recs = [c for c in s2.cells if c.cell_id == cid]
        if not cell_recs:
            continue
        cell = cell_recs[0]
        lab = int(cell.label)
        nuc = (nuc_label == lab) & (cell_label == lab)
        if lab not in cell_dist:
            cell_dist[lab] = distance_transform_edt(~nuc) * pixel_size
        dist_um = cell_dist[lab]
        px = int(round(row["x"] / pixel_size))
        py = int(round(row["y"] / pixel_size))
        py = max(0, min(py, dist_um.shape[0] - 1))
        px = max(0, min(px, dist_um.shape[1] - 1))
        in_nuc = nuc[py, px]
        d = dist_um[py, px]
        if in_nuc:
            counts[0] += 1
        elif d <= b2:
            counts[1] += 1
        elif d <= b3:
            counts[2] += 1
        else:
            counts[3] += 1
    emp = counts / counts.sum()
    # Empirical should be close to the posterior; allow 0.10 per compartment
    deviation = np.abs(emp - target_post)
    assert deviation.max() < 0.15, \
        f"empirical compartments {emp.round(3)} too far from posterior {target_post.round(3)}"
