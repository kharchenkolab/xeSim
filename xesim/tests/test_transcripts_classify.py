"""Phase 3 tests for transcript-based type cross-check.

All tests run against the cached fit at `tmp/xesim_v16_model/priors/`;
none of them trigger the renderer or encoder (so this file is fast — well
under a minute total). The full-explain integration is exercised separately
in test_transcripts_explain.py (slow, opt-in).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path("/workspace/xeSim/tmp/xesim_v16_model")
PRIORS_JSON = MODEL_DIR / "priors" / "transcripts_nmf.json"
CANONICAL_DIR = MODEL_DIR / "canonical"

requires_priors = pytest.mark.skipif(
    not PRIORS_JSON.exists(),
    reason=f"no priors at {PRIORS_JSON} — run fit_transcripts_nmf first",
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
def cf(model):
    return model.cell_factor_fractions(include_annotation=True)


@pytest.fixture(scope="module")
def classifications(model):
    """One-time classification of all fit cells (~140k rows)."""
    return model.classify_cells_by_transcripts()


# ---------------------------------------------------------------------------
# Centroid-level smoke tests (synthetic h vectors)
# ---------------------------------------------------------------------------


@requires_priors
def test_classify_uncertainty_zero_for_centroid_cells(priors, model):
    """If h_c == per_type_alpha[t] for some type t, classification returns t."""
    cell_h = {f"_synth_{t}": np.asarray(alpha, dtype=np.float64)
              for t, alpha in priors["per_type_alpha"].items()}
    res = model.classify_cells_by_transcripts(cell_h)
    misses = [t for t in priors["per_type_alpha"]
              if res[f"_synth_{t}"]["cell_type_transcripts"] != t]
    assert not misses, f"centroid cells misclassified: {misses}"


@requires_priors
def test_classify_endocrine_centroid(priors, model):
    alpha = priors["per_type_alpha"].get("Endocrine")
    if alpha is None:
        pytest.skip("no Endocrine type in priors")
    res = model.classify_cells_by_transcripts({"_endo": np.asarray(alpha)})
    assert res["_endo"]["cell_type_transcripts"] == "Endocrine"
    assert res["_endo"]["type_uncertainty"] < 0.5


@requires_priors
def test_classify_zero_norm_cell(model):
    res = model.classify_cells_by_transcripts({"_empty": np.zeros(9)})
    assert res["_empty"]["cell_type_transcripts"] is None
    assert res["_empty"]["type_uncertainty"] is None


# ---------------------------------------------------------------------------
# Real-cell concordance tests (use cached `classifications` fixture)
# ---------------------------------------------------------------------------


def _real_cell_subset(cf, cell_type: str, n: int = 200, min_counts: int = 50):
    return cf[(cf["_cell_type"] == cell_type)
              & (cf["transcript_count"] >= min_counts)].head(n)


@requires_priors
def test_classify_real_well_separated_cells(priors, cf, classifications):
    """Real cells of NMF-distinct types should classify correctly above chance.

    Pancreas K=9 fit cleanly separates Endocrine + Exocrine (high
    centroid distance from each other and from stromal types). Lineage-
    collision cases (Fibroblast/Mural) are tested separately.
    """
    targets = [t for t in ("Endocrine", "Exocrine epithelial")
                 if t in priors["per_type_alpha"]]
    if not targets:
        pytest.skip("no well-separated target types in priors")
    correct = 0; total = 0
    for t in targets:
        sub = _real_cell_subset(cf, t)
        if len(sub) < 30:
            continue
        for cid in sub["cell_id"]:
            r = classifications.get(cid)
            if r is None or r["cell_type_transcripts"] is None:
                continue
            total += 1
            if r["cell_type_transcripts"] == t:
                correct += 1
    assert total >= 30, f"not enough well-separated cells: {total}"
    rate = correct / total
    assert rate >= 0.7, f"well-separated concordance only {rate:.2%}"


@requires_priors
def test_classify_fibroblast_lineage_collision(priors, cf, classifications):
    """Document the K=9 limitation: Fibroblast/Mural centroids cosine≈1
    (both load 66% on the SFRP2/VCAN fibroblast factor). Fibroblast cells
    should classify *somewhere in the fibroblast lineage*, not at random.
    """
    pta = priors["per_type_alpha"]
    if "Fibroblast / CAF" not in pta or "Mural / pericyte" not in pta:
        pytest.skip("need both Fibroblast and Mural")
    sub = _real_cell_subset(cf, "Fibroblast / CAF")
    if len(sub) < 30:
        pytest.skip(f"too few Fibroblast cells: {len(sub)}")
    lineage = {"Fibroblast / CAF", "Mural / pericyte"}
    in_lineage = 0; total = 0
    for cid in sub["cell_id"]:
        r = classifications.get(cid)
        if r is None or r["cell_type_transcripts"] is None:
            continue
        total += 1
        if r["cell_type_transcripts"] in lineage:
            in_lineage += 1
    rate = in_lineage / total
    assert rate >= 0.8, f"Fibroblast lineage concordance only {rate:.2%}"


# ---------------------------------------------------------------------------
# Scene-stamping (no rendering)
# ---------------------------------------------------------------------------


@requires_priors
@requires_canonical
def test_stamp_scene_with_transcripts(priors, model, classifications):
    """stamp_scene_with_transcripts adds a `transcripts` block to each
    cell's provenance dict. Skips encoder + renderer — fast."""
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    crop = manifest["crops"][0]
    scene, _ = scene_from_canonical_crop(
        CANONICAL_DIR / crop["npz_path"],
        CANONICAL_DIR / "cell_types.json",
    )
    stamped = model.stamp_scene_with_transcripts(scene, classifications)

    with_txn = [c for c in stamped.cells if "transcripts" in c.provenance]
    assert len(with_txn) > 0, "no scene cell got a transcripts stamp"
    sample = with_txn[0].provenance["transcripts"]
    expected = {"cell_type_transcripts", "type_uncertainty", "cosine_top", "type_scores"}
    assert expected.issubset(sample.keys())
    valid_types = set(priors["per_type_alpha"].keys()) | {None}
    for c in with_txn:
        assert c.provenance["transcripts"]["cell_type_transcripts"] in valid_types


@requires_priors
@requires_canonical
def test_stamp_scene_concordance(priors, model, cf, classifications):
    """For well-populated, annotation-typed scene cells, the transcript-
    implied type should agree with the annotated type on a majority of
    cells when the annotated type is NMF-distinguishable. (Same K=9
    lineage-collision caveat as above; we group fibroblast lineage.)"""
    from xesim.scene_io import scene_from_canonical_crop
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    tc_lookup = dict(zip(cf["cell_id"], cf["transcript_count"]))
    lineage_groups = {
        "Fibroblast / CAF": "stromal",
        "Mural / pericyte": "stromal",
    }
    def collapse(t): return lineage_groups.get(t, t)

    n_total = 0; n_concordant = 0
    for crop in manifest["crops"][:2]:
        scene, _ = scene_from_canonical_crop(
            CANONICAL_DIR / crop["npz_path"],
            CANONICAL_DIR / "cell_types.json",
        )
        stamped = model.stamp_scene_with_transcripts(scene, classifications)
        for c in stamped.cells:
            txn = c.provenance.get("transcripts")
            if txn is None or txn.get("cell_type_transcripts") is None:
                continue
            if c.cell_type is None or c.cell_type not in priors["per_type_alpha"]:
                continue
            if tc_lookup.get(c.cell_id, 0) < 30:
                continue
            n_total += 1
            if collapse(c.cell_type) == collapse(txn["cell_type_transcripts"]):
                n_concordant += 1
    if n_total < 20:
        pytest.skip(f"too few annotated well-populated scene cells: {n_total}")
    rate = n_concordant / n_total
    assert rate >= 0.6, f"scene concordance (lineage-collapsed) only {rate:.2%}"
