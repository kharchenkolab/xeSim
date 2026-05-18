"""Phase 4 tests for transcript-based latent cell completion."""

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
    reason=f"no priors at {PRIORS_JSON} — run fit_transcripts_nmf first",
)
requires_canonical = pytest.mark.skipif(
    not (CANONICAL_DIR / "manifest.json").exists(),
    reason=f"no canonical crops at {CANONICAL_DIR}",
)


@pytest.fixture(scope="module")
def model():
    from xesim import XesimModel
    return XesimModel.load(MODEL_DIR)


@pytest.fixture(scope="module")
def crops():
    manifest = json.loads((CANONICAL_DIR / "manifest.json").read_text())
    return manifest["crops"]


def _load_scene(crop):
    from xesim.scene_io import scene_from_canonical_crop
    return scene_from_canonical_crop(
        Path(CANONICAL_DIR / crop["npz_path"]),
        CANONICAL_DIR / "cell_types.json",
    )[0]


def _drop_cells(scene, fraction: float, rng):
    """Remove `fraction` of cells from a scene; return (scene_dropped,
    dropped_cell_masks)."""
    labels = sorted(int(c.label) for c in scene.cells)
    if len(labels) == 0:
        return scene, {}
    n = max(1, int(fraction * len(labels)))
    to_drop = set(int(x) for x in rng.choice(labels, size=n, replace=False))
    dropped_masks = {lab: (scene.cell_label == lab) for lab in to_drop}
    new_cell_label = scene.cell_label.copy()
    for lab in to_drop:
        new_cell_label[new_cell_label == lab] = 0
    new_cells = tuple(c for c in scene.cells if int(c.label) not in to_drop)
    return replace(scene, cell_label=new_cell_label, cells=new_cells), dropped_masks


# ---------------------------------------------------------------------------


@requires_priors
@requires_canonical
def test_proposer_no_double_coverage(model, crops):
    """Proposed cells must not overlap existing segmented cells or each other."""
    from xesim.transcript_proposer import propose_transcript_cells
    rng = np.random.default_rng(0)
    crop = crops[0]
    scene, dropped_masks = _drop_cells(_load_scene(crop), 0.3, rng)
    cands, _ = propose_transcript_cells(model, scene, crop, rng=rng)
    h, w = scene.image_shape
    existing = scene.cell_label > 0
    union = np.zeros((h, w), dtype=bool)
    for c in cands:
        cm = np.zeros((h, w), dtype=bool)
        cm[c.cell_pixels[:, 0], c.cell_pixels[:, 1]] = True
        assert not (cm & existing).any(), "proposed cell overlaps an existing cell"
        assert not (cm & union).any(), "two proposed cells overlap each other"
        union |= cm


@requires_priors
@requires_canonical
def test_proposer_zero_on_intact_scene(model, crops):
    """If no cells are dropped, the proposer should add ≤ a tiny number
    (orphan transcripts from cellAdmix's confidence-margin filter)."""
    from xesim.transcript_proposer import propose_transcript_cells
    crop = crops[0]
    scene = _load_scene(crop)
    cands, diag = propose_transcript_cells(model, scene, crop,
                                              rng=np.random.default_rng(0))
    # The intact scene shouldn't yield many candidates; a few may slip in
    # from orphan-prone tile margins, but it should be tiny relative to
    # the existing cell count.
    assert len(cands) <= max(2, int(0.1 * len(scene.cells))), \
        f"unexpectedly many candidates on intact scene: {len(cands)}"


@requires_priors
@requires_canonical
def test_drop_and_recover_above_chance(model, crops):
    """Drop 30% of cells across the first few crops and verify the
    transcript proposer recovers a meaningful fraction at IoU ≥ 0.3.

    Real-data caveat: per-cell molecule count on a 64µm tile is small
    (avg ~6 mol/cell on this pancreas panel), so DBSCAN clusters are
    coarse and IoU ≥ 0.5 is rare. We target above-chance recall
    (≥ 15%) and well-above-chance type precision on what is recovered.
    """
    from xesim.transcript_proposer import propose_transcript_cells
    total_dropped = 0
    total_iou03 = 0
    for crop in crops[:5]:
        rng = np.random.default_rng(int(crop["crop_id"].split("_")[-1]))
        scene_full = _load_scene(crop)
        if len(scene_full.cells) < 5:
            continue
        scene_drop, dropped_masks = _drop_cells(scene_full, 0.3, rng)
        cands, _ = propose_transcript_cells(model, scene_drop, crop, rng=rng)
        h, w = scene_full.image_shape
        for lab, dmask in dropped_masks.items():
            best = 0.0
            for c in cands:
                cm = np.zeros((h, w), dtype=bool)
                cm[c.cell_pixels[:, 0], c.cell_pixels[:, 1]] = True
                inter = (dmask & cm).sum(); un = (dmask | cm).sum()
                iou = inter / un if un else 0.0
                if iou > best:
                    best = iou
            total_dropped += 1
            if best >= 0.3:
                total_iou03 += 1
    assert total_dropped >= 50, f"too few dropped cells across crops: {total_dropped}"
    rate = total_iou03 / total_dropped
    assert rate >= 0.15, f"transcript-proposer recovery only {rate:.1%} at IoU≥0.3"


@requires_priors
@requires_canonical
def test_stamp_proposals_into_scene(model, crops):
    """The proposer's CandidateCell outputs flow cleanly through
    `stamp_candidates_into_label` to give a new (cell_label, nucleus_label)
    pair where the proposed cells have unique labels not present in the
    original scene.

    Tests the wiring that `explain(complete=True)` does end-to-end,
    minus the slow encoder + renderer pass."""
    from xesim.transcript_proposer import propose_transcript_cells
    from xesim.scene_completion import stamp_candidates_into_label
    rng = np.random.default_rng(7)
    crop = crops[0]
    scene_full = _load_scene(crop)
    scene_drop, _ = _drop_cells(scene_full, 0.3, rng)
    cands, _ = propose_transcript_cells(model, scene_drop, crop, rng=rng)
    if not cands:
        pytest.skip("no candidates produced on this crop")
    new_cl, new_nl, assigned = stamp_candidates_into_label(
        scene_drop.cell_label, scene_drop.nucleus_label, cands,
        label_offset=int(scene_drop.cell_label.max() + 1000),
    )
    # New labels must not collide with existing scene labels.
    existing = set(int(l) for l in np.unique(scene_drop.cell_label) if l > 0)
    new_labels = set(int(l) for l in assigned)
    assert not (new_labels & existing), f"label collision: {new_labels & existing}"
    # And every assigned new label must actually appear in the new raster.
    for lab in assigned:
        assert int((new_cl == lab).sum()) > 0


@requires_priors
@requires_canonical
def test_explain_complete_raises_without_priors(crops):
    """If transcripts priors aren't loaded, complete=True should still
    raise (preserving the old behavior of the flag for legacy callers)."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    # Force-disable priors for this call to mimic a model fit without
    # --with-transcripts.
    m.transcripts_priors = None
    with pytest.raises(NotImplementedError):
        m.explain(CANONICAL_DIR, num_crops=1, complete=True)
