"""Unit tests for xesim.cell_type_resolver."""
from __future__ import annotations

import json

import pytest

from xesim.cell_type_resolver import (
    ALLOWED_SOURCES, TypeResolution, assert_all_resolved, evidence_to_json,
    resolve_cell_types, tier_counts,
)


def _resolve(cell_ids, **kwargs):
    """Helper: drop the unresolved list for tests that only check resolutions."""
    out, _ = resolve_cell_types(cell_ids, **kwargs)
    return out


# ---------------------------------------------------------------------------
# TypeResolution contract
# ---------------------------------------------------------------------------

class TestTypeResolutionContract:
    def test_valid_resolution_constructs(self):
        r = TypeResolution(cell_type="Immune", source="annotation",
                            confidence=1.0, evidence={})
        assert r.cell_type == "Immune"
        assert r.source == "annotation"

    def test_empty_cell_type_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            TypeResolution(cell_type="", source="annotation",
                            confidence=1.0, evidence={})

    def test_unknown_cell_type_rejected_in_all_cases(self):
        # The whole point of the resolver: zero "unknown" rows.
        for variant in ("unknown", "Unknown", "UNKNOWN"):
            with pytest.raises(ValueError, match="not allowed"):
                TypeResolution(cell_type=variant, source="annotation",
                                confidence=1.0, evidence={})

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="not in"):
            TypeResolution(cell_type="Immune", source="vibes",
                            confidence=1.0, evidence={})

    def test_confidence_must_be_in_unit_interval(self):
        for bad in (-0.1, 1.1, 2.0, -1.0):
            with pytest.raises(ValueError, match="confidence"):
                TypeResolution(cell_type="Immune", source="annotation",
                                confidence=bad, evidence={})

    def test_to_dict_roundtrip(self):
        r = TypeResolution(cell_type="Ductal", source="transcripts",
                            confidence=0.7, evidence={"cosine_top": 0.7})
        d = r.to_dict()
        assert d == {"cell_type": "Ductal", "source": "transcripts",
                      "confidence": 0.7, "evidence": {"cosine_top": 0.7}}


# ---------------------------------------------------------------------------
# Cascade ordering
# ---------------------------------------------------------------------------

TYPE_NAMES = ["unknown", "Ductal", "Endocrine", "Immune", "Endothelial"]


class TestCascadeOrdering:
    def test_annotation_wins_over_transcripts_when_both_present(self):
        out = _resolve(
            ["c1"],
            annotation_map={"c1": "Ductal"},
            transcript_classifications={
                "c1": {"cell_type_transcripts": "Endocrine",
                        "cosine_top": 0.9, "type_uncertainty": 0.1},
            },
        )
        assert out["c1"].cell_type == "Ductal"
        assert out["c1"].source == "annotation"
        assert out["c1"].confidence == 1.0

    def test_transcripts_used_when_annotation_missing(self):
        out = _resolve(
            ["c1"],
            transcript_classifications={
                "c1": {"cell_type_transcripts": "Endocrine",
                        "cosine_top": 0.73, "type_uncertainty": 0.27},
            },
        )
        assert out["c1"].source == "transcripts"
        assert out["c1"].cell_type == "Endocrine"
        assert out["c1"].confidence == pytest.approx(0.73)
        assert out["c1"].evidence["cosine_top"] == pytest.approx(0.73)

    def test_training_used_when_annotation_and_transcripts_missing(self):
        out = _resolve(
            ["c1"],
            training_map={"c1": 3},   # index 3 in TYPE_NAMES → "Immune"
            training_type_names=TYPE_NAMES,
        )
        assert out["c1"].source == "training"
        assert out["c1"].cell_type == "Immune"
        assert out["c1"].confidence == 1.0

    def test_stain_knn_used_as_last_resort(self):
        out = _resolve(
            ["c1"],
            stain_predictions={"c1": "Endothelial"},
        )
        assert out["c1"].source == "stain_knn"
        assert out["c1"].cell_type == "Endothelial"
        # Default confidence for stain_knn is the low-confidence flag.
        assert 0.0 < out["c1"].confidence < 0.5

    def test_stain_knn_uses_supplied_confidence_when_given(self):
        out = _resolve(
            ["c1"],
            stain_predictions={"c1": "Endothelial"},
            stain_confidences={"c1": 0.65},
        )
        assert out["c1"].confidence == pytest.approx(0.65)
        assert out["c1"].evidence["knn_default_confidence"] is False


# ---------------------------------------------------------------------------
# Tier skipping when claims are empty / "unknown"
# ---------------------------------------------------------------------------

class TestEmptyClaimsSkipped:
    def test_annotation_value_unknown_is_skipped(self):
        # An annotation row that explicitly says "unknown" should NOT
        # block the cascade — the resolver moves to the next tier.
        out = _resolve(
            ["c1"],
            annotation_map={"c1": "unknown"},
            transcript_classifications={
                "c1": {"cell_type_transcripts": "Ductal",
                        "cosine_top": 0.6, "type_uncertainty": 0.4},
            },
        )
        assert out["c1"].source == "transcripts"
        assert out["c1"].cell_type == "Ductal"

    def test_transcripts_unknown_value_is_skipped(self):
        out = _resolve(
            ["c1"],
            transcript_classifications={
                "c1": {"cell_type_transcripts": "unknown", "cosine_top": 0.1},
            },
            training_map={"c1": 2},
            training_type_names=TYPE_NAMES,
        )
        assert out["c1"].source == "training"
        assert out["c1"].cell_type == "Endocrine"

    def test_training_unknown_index_is_skipped(self):
        # Index 0 in TYPE_NAMES is "unknown" — should fall through.
        out = _resolve(
            ["c1"],
            training_map={"c1": 0},
            training_type_names=TYPE_NAMES,
            stain_predictions={"c1": "Immune"},
        )
        assert out["c1"].source == "stain_knn"
        assert out["c1"].cell_type == "Immune"

    def test_transcripts_missing_cell_type_field_is_skipped(self):
        out = _resolve(
            ["c1"],
            transcript_classifications={
                "c1": {"cosine_top": 0.9},  # no cell_type_transcripts
            },
            training_map={"c1": 1},
            training_type_names=TYPE_NAMES,
        )
        assert out["c1"].source == "training"


# ---------------------------------------------------------------------------
# Hard-error when no tier resolves
# ---------------------------------------------------------------------------

class TestUnresolvedHardError:
    def test_resolver_returns_unresolved_list_without_raising(self):
        # The resolver itself is lenient — it reports unresolved cells
        # via the second return value rather than raising. Callers that
        # need the contract use assert_all_resolved (below).
        resolutions, unresolved = resolve_cell_types(["c1", "c2", "c3"])
        assert resolutions == {}
        assert sorted(unresolved) == ["c1", "c2", "c3"]

    def test_assert_all_resolved_raises_on_missing(self):
        resolutions, _ = resolve_cell_types(
            ["c1", "c2"], annotation_map={"c1": "Ductal"})
        with pytest.raises(ValueError, match="could not be resolved"):
            assert_all_resolved(resolutions, ["c1", "c2"])

    def test_assert_all_resolved_passes_when_complete(self):
        resolutions, _ = resolve_cell_types(
            ["c1", "c2"],
            annotation_map={"c1": "Ductal", "c2": "Immune"})
        assert_all_resolved(resolutions, ["c1", "c2"])  # no raise


# ---------------------------------------------------------------------------
# Confidence values are tier-comparable
# ---------------------------------------------------------------------------

class TestConfidenceRanking:
    def test_curated_tiers_outrank_classifier_tiers(self):
        ann = _resolve(["c1"], annotation_map={"c1": "Ductal"})
        train = _resolve(
            ["c1"], training_map={"c1": 1}, training_type_names=TYPE_NAMES)
        tx_low = _resolve(
            ["c1"], transcript_classifications={
                "c1": {"cell_type_transcripts": "Ductal",
                        "cosine_top": 0.4, "type_uncertainty": 0.6}})
        stain = _resolve(
            ["c1"], stain_predictions={"c1": "Ductal"})
        # Curated should be 1.0; classifier tiers should be < 1.
        assert ann["c1"].confidence == 1.0
        assert train["c1"].confidence == 1.0
        assert tx_low["c1"].confidence < 1.0
        assert stain["c1"].confidence < 1.0

    def test_transcripts_confidence_clipped_to_unit_interval(self):
        # Pathological cosine_top outside [0,1] gets clipped.
        out = _resolve(
            ["c1"],
            transcript_classifications={
                "c1": {"cell_type_transcripts": "Ductal", "cosine_top": 1.5}})
        assert out["c1"].confidence == 1.0


# ---------------------------------------------------------------------------
# Evidence + diagnostics helpers
# ---------------------------------------------------------------------------

class TestEvidenceHelpers:
    def test_transcripts_evidence_includes_top_5_types(self):
        many = {f"Type{i}": 0.5 - 0.01 * i for i in range(10)}
        out = _resolve(
            ["c1"],
            transcript_classifications={
                "c1": {"cell_type_transcripts": "Type0",
                        "cosine_top": 0.5, "type_uncertainty": 0.5,
                        "type_scores": many},
            })
        evidence = out["c1"].evidence
        assert "top_type_scores" in evidence
        assert len(evidence["top_type_scores"]) == 5
        # The top entry should be the winning type.
        assert "Type0" in evidence["top_type_scores"]

    def test_tier_counts_returns_per_source_histogram(self):
        out = _resolve(
            ["c1", "c2", "c3", "c4"],
            annotation_map={"c1": "Ductal"},
            transcript_classifications={
                "c2": {"cell_type_transcripts": "Endocrine",
                        "cosine_top": 0.9}},
            training_map={"c3": 2},
            training_type_names=TYPE_NAMES,
            stain_predictions={"c4": "Endothelial"},
        )
        counts = tier_counts(out)
        assert counts == {"annotation": 1, "transcripts": 1,
                            "training": 1, "stain_knn": 1}

    def test_evidence_to_json_round_trip(self):
        s = evidence_to_json({"cosine_top": 0.73,
                                "top_type_scores": {"A": 0.5, "B": 0.3}})
        parsed = json.loads(s)
        assert parsed["cosine_top"] == 0.73
        assert parsed["top_type_scores"]["A"] == 0.5

    def test_evidence_to_json_handles_numpy_floats(self):
        # Numpy float types must round-trip via the default=float hook.
        import numpy as np
        s = evidence_to_json({"x": np.float32(0.5), "y": np.float64(0.25)})
        parsed = json.loads(s)
        assert parsed["x"] == 0.5
        assert parsed["y"] == 0.25


# ---------------------------------------------------------------------------
# Source set contract
# ---------------------------------------------------------------------------

class TestAllowedSources:
    def test_synth_sources_are_in_allowed_set(self):
        # ghost_prior and tx_proposer aren't produced by the resolver, but
        # they're allowed on TypeResolution so synthetic-cell construction
        # sites can stamp them.
        for src in ("ghost_prior", "tx_proposer"):
            assert src in ALLOWED_SOURCES
            r = TypeResolution(cell_type="Immune", source=src,
                                confidence=0.5, evidence={})
            assert r.source == src
