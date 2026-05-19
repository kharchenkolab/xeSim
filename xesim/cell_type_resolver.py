"""Unified cell-type resolution for explain-mode rendering.

Every anchor cell that flows into rendering or emission MUST exit with a
non-empty, non-"unknown" cell type AND a record of which classification
tier produced the call + a comparable confidence score. This module is
the single entry point that enforces both.

Tiers, in cascade order (first non-empty wins):

  1. ``annotation``  — user-supplied cell_id → type CSV. Curated; trust 1.0.
  2. ``transcripts`` — cellAdmix cosine classifier
     (``model.classify_cells_by_transcripts``). Confidence = cosine_top.
  3. ``training``    — cells the model saw during fit (model.cid_to_type).
                       Curated subset of (1); trust 1.0.
  4. ``stain_knn``   — V37b encoder-latent kNN against training cells.
                       Confidence from kNN vote-margin (Phase 4); for now
                       a low default (0.30) flags it as best-guess.

The cascade has NO confidence threshold — even a low-confidence
classifier output is preferred over "unknown". For cells with stain
pixels, tier 4 is guaranteed to produce a prediction (assuming the
model's stain bank was built at fit time, which is mandatory in the
post-refactor pipeline).

Synthetic cell sources stamp themselves directly and bypass the resolver:

  - ``ghost_prior``  — density-conditioned sampler. Stamped at ghost
                       construction.
  - ``tx_proposer``  — orphan-transcript cluster cell. Stamped at
                       transcript-proposed cell construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# Sources allowed on TypeResolution.source. Kept as a module constant so
# downstream consumers can validate without copying the list.
ALLOWED_SOURCES = frozenset({
    "annotation", "transcripts", "training", "stain_knn",
    "ghost_prior", "tx_proposer",
})


@dataclass(frozen=True)
class TypeResolution:
    """One cell's resolved type + provenance."""
    cell_type: str          # non-empty; never "unknown" / "" / None
    source: str             # one of ALLOWED_SOURCES
    confidence: float       # in [0, 1]; comparable across tiers (annotation=1, knn≈0.3, …)
    evidence: dict[str, Any]   # tier-specific debug blob (serialisable to JSON)

    def __post_init__(self):
        # Contract checks: every TypeResolution carries a real type.
        if not isinstance(self.cell_type, str) or not self.cell_type:
            raise ValueError(f"cell_type must be non-empty str, got {self.cell_type!r}")
        if self.cell_type.lower() == "unknown":
            raise ValueError(
                "cell_type='unknown' is not allowed by the resolver contract. "
                "Every cell must end with a real type.")
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source {self.source!r} not in {sorted(ALLOWED_SOURCES)}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "source": self.source,
            "confidence": float(self.confidence),
            "evidence": dict(self.evidence),
        }


# Default confidence for tier-4 (stain kNN) until the API exposes a real
# vote-margin score. Picked low so downstream consumers treat these as
# best-guess rather than curated. Replaced with the real per-cell margin
# in Phase 4 of the refactor.
_STAIN_KNN_DEFAULT_CONFIDENCE = 0.30


def resolve_cell_types(
    cell_ids: list[str],
    *,
    annotation_map: dict[str, str] | None = None,
    transcript_classifications: dict[str, dict[str, Any]] | None = None,
    training_map: dict[str, int] | None = None,
    training_type_names: list[str] | None = None,
    stain_predictions: dict[str, str] | None = None,
    stain_confidences: dict[str, float] | None = None,
) -> tuple[dict[str, TypeResolution], list[str]]:
    """Resolve every cell in ``cell_ids`` to a `TypeResolution`.

    Inputs are PRE-COMPUTED prediction maps; this function only runs the
    cascade. The caller is responsible for invoking each classifier and
    building the maps. Keeping the resolver pure makes it cheap to unit
    test and deterministic across call sites.

    Parameters
    ----------
    cell_ids
        Anchor cell_ids in the order they should appear in the output dict.
    annotation_map
        ``cell_id → cell_type``. Anything in this map wins outright.
    transcript_classifications
        cellAdmix output: ``cell_id → {cell_type_transcripts, cosine_top,
        type_uncertainty, type_scores}``. Pass ``model.classify_cells_by_transcripts()``
        directly.
    training_map
        ``cell_id → type_idx`` from the model's canonical training crops.
    training_type_names
        Type-name table used to translate ``training_map`` indices to
        strings. Required if ``training_map`` is given.
    stain_predictions
        ``cell_id → predicted_type`` from the V37b encoder-latent kNN.
        Caller is responsible for skipping the bank-missing case.
    stain_confidences
        Optional ``cell_id → confidence_in_[0,1]`` matching
        ``stain_predictions``. Falls back to
        :data:`_STAIN_KNN_DEFAULT_CONFIDENCE` if missing.

    Returns
    -------
    (resolutions, unresolved)
        ``resolutions`` is a dict[cell_id → TypeResolution] containing
        only cells that were resolved by some tier. ``unresolved`` is the
        list of cell_ids that fell through all four tiers (in input
        order). The resolver does NOT raise on unresolved cells —
        callers use :func:`assert_all_resolved` when the contract demands
        zero unresolved.
    """
    annotation_map = annotation_map or {}
    transcript_classifications = transcript_classifications or {}
    training_map = training_map or {}
    stain_predictions = stain_predictions or {}
    stain_confidences = stain_confidences or {}

    if training_map and not training_type_names:
        raise ValueError(
            "training_map provided without training_type_names; cannot "
            "translate type indices to names.")

    out: dict[str, TypeResolution] = {}
    unresolved: list[str] = []

    for cid in cell_ids:
        cid_str = str(cid)

        # Tier 1: annotation
        ann = annotation_map.get(cid_str)
        if ann and ann.lower() != "unknown":
            out[cid_str] = TypeResolution(
                cell_type=str(ann), source="annotation",
                confidence=1.0,
                evidence={"annotation_table_hit": True},
            )
            continue

        # Tier 2: transcripts
        tx = transcript_classifications.get(cid_str)
        if tx and tx.get("cell_type_transcripts"):
            tx_type = str(tx["cell_type_transcripts"])
            if tx_type.lower() != "unknown":
                # cosine_top in [0,1]; type_uncertainty in [0,1] (1 - margin).
                # Use cosine_top directly as confidence — it's the model's
                # similarity score against the winning prototype.
                cosine_top = float(tx.get("cosine_top", 0.0))
                evidence = {
                    "cosine_top": cosine_top,
                    "type_uncertainty": float(tx.get("type_uncertainty", 0.0)),
                }
                # type_scores can be a dict[type → score]; keep only the
                # top-5 to bound serialised size.
                ts = tx.get("type_scores")
                if isinstance(ts, dict) and ts:
                    top5 = sorted(ts.items(), key=lambda kv: -float(kv[1]))[:5]
                    evidence["top_type_scores"] = {k: float(v) for k, v in top5}
                out[cid_str] = TypeResolution(
                    cell_type=tx_type, source="transcripts",
                    confidence=max(0.0, min(1.0, cosine_top)),
                    evidence=evidence,
                )
                continue

        # Tier 3: training
        tr_idx = training_map.get(cid_str)
        if tr_idx is not None:
            tr_idx = int(tr_idx)
            if (0 < tr_idx < len(training_type_names)
                    and training_type_names[tr_idx].lower() != "unknown"):
                out[cid_str] = TypeResolution(
                    cell_type=str(training_type_names[tr_idx]),
                    source="training", confidence=1.0,
                    evidence={"type_idx": tr_idx},
                )
                continue

        # Tier 4: stain kNN
        sk = stain_predictions.get(cid_str)
        if sk and sk.lower() != "unknown":
            conf = float(stain_confidences.get(
                cid_str, _STAIN_KNN_DEFAULT_CONFIDENCE))
            out[cid_str] = TypeResolution(
                cell_type=str(sk), source="stain_knn",
                confidence=max(0.0, min(1.0, conf)),
                evidence={"knn_default_confidence":
                            cid_str not in stain_confidences},
            )
            continue

        # All four tiers produced nothing — this means the caller didn't
        # arrange for tier-4 to be available. Record the unresolved id
        # and continue so we can report ALL of them in one shot.
        unresolved.append(cid_str)

    return out, unresolved


def assert_all_resolved(
    resolutions: dict[str, TypeResolution], expected_ids: list[str],
) -> None:
    """Raise if any expected cell_id is missing from ``resolutions``.

    Callers that enforce the "zero unknowns" contract (most production
    explain paths) wrap their resolver call with this — the resolver
    itself stays lenient so unit tests and lower-tier code paths don't
    have to mock out every cascade source.
    """
    missing = [c for c in expected_ids if c not in resolutions]
    if missing:
        raise ValueError(
            f"{len(missing)} cell(s) could not be resolved by any tier. "
            "Ensure the stain-classifier bank is built at fit time so "
            f"tier-4 is guaranteed to fire. First unresolved ids: "
            f"{missing[:5]}")


def resolve_from_model(
    cell_ids: list[str],
    centroids_um: "np.ndarray | None",
    model,
    bundle_path: str | "Path",
    *,
    annotation_map: dict[str, str] | None = None,
    annotation_path: str | "Path | None" = None,
    progress: bool = True,
) -> tuple[dict[str, TypeResolution], list[str]]:
    """Single entry point: turn a fitted model + Xenium bundle into a
    resolved cell-type dict for ``cell_ids``.

    This is the shared cascade implementation that the 2D
    (``explain_region``) and 2.5D (``scene_2_5d.scene_first``) pipelines
    both invoke. It owns:

      - tier-1 annotation table loading (from ``annotation_path``,
        the model's bundled annotation copy, or the explicit
        ``annotation_map`` the caller passed in),
      - tier-2 cellAdmix transcript classifier invocation (with a
        process-wide cache keyed by ``id(model)``),
      - tier-3 model.cid_to_type lookup,
      - tier-4 stain-classifier kNN — run ONLY for the cells that
        fell through tiers 1-3, since encoding is the expensive step,
      - the final cascade via :func:`resolve_cell_types`.

    Parameters
    ----------
    cell_ids
        Anchor cell ids in the order they should appear in the output.
    centroids_um
        Optional ``(N, 2)`` array of cell centroids in µm. Required only
        if stain_knn will need to encode some cells (i.e. the bank
        exists AND some cells fall through tiers 1-3). Pass ``None`` to
        skip tier-4 entirely.
    model
        Fitted ``XesimModel``; supplies the transcripts classifier,
        ``cid_to_type``, ``type_names``, and the model dir (used to
        locate ``cell_latent_bank.npz``).
    bundle_path
        Path to the source Xenium bundle (stain_knn reads crops from
        it).
    annotation_map
        Pre-loaded ``cell_id → type_name`` map. If provided, takes
        precedence over the CSV at ``annotation_path``.
    annotation_path
        Path to a ``cell_id, merged_annotation`` CSV. Falls back to
        ``MODEL_DIR/annotations/annotation.csv.gz`` when not given.
    progress
        Whether to print a one-line per-tier summary after resolution.

    Returns
    -------
    (resolutions, unresolved)
        Same shape as :func:`resolve_cell_types`. Use
        :func:`assert_all_resolved` to enforce zero-unknowns at the
        caller boundary.
    """
    from pathlib import Path as _Path
    import numpy as _np

    # ---- Tier 1: annotation map ---------------------------------------
    if annotation_map is None:
        annotation_map = {}
        if annotation_path is None and hasattr(model, "paths"):
            cand = _Path(model.paths.root) / "annotations" / "annotation.csv.gz"
            if cand.exists():
                annotation_path = cand
        if annotation_path is not None and _Path(annotation_path).exists():
            try:
                import pandas as _pd
                ann_df = _pd.read_csv(annotation_path, compression="infer")
                if {"cell_id", "merged_annotation"}.issubset(ann_df.columns):
                    annotation_map = dict(zip(
                        ann_df["cell_id"].astype(str),
                        ann_df["merged_annotation"].astype(str)))
            except Exception:
                pass

    # ---- Tier 2: cellAdmix classifier, with process-wide cache --------
    transcript_classifications: dict[str, dict] = {}
    if getattr(model, "transcripts_priors", None) is not None:
        cache_key = id(model)
        cached = _MODEL_TX_CACHE.get(cache_key)
        if cached is not None:
            transcript_classifications = cached
        else:
            try:
                transcript_classifications = model.classify_cells_by_transcripts()
                _MODEL_TX_CACHE[cache_key] = transcript_classifications
            except Exception as e:
                if progress:
                    print(f"[resolve_from_model] cellAdmix unavailable: {e}")
                _MODEL_TX_CACHE[cache_key] = {}

    # ---- Tier 3: training cid_to_type ---------------------------------
    training_map: dict[str, int] = {}
    training_type_names: list[str] = []
    if hasattr(model, "cid_to_type"):
        training_map = {str(k): int(v) for k, v in model.cid_to_type.items()}
        training_type_names = list(model.type_names)

    # Pre-resolve with tiers 1-3 to find the (small) set that needs
    # the expensive tier-4 encoder pass.
    _pre, unresolved_pre = resolve_cell_types(
        cell_ids,
        annotation_map=annotation_map,
        transcript_classifications=transcript_classifications,
        training_map=training_map,
        training_type_names=training_type_names,
    )

    # ---- Tier 4: stain kNN on the leftovers ---------------------------
    stain_predictions: dict[str, str] = {}
    if unresolved_pre and centroids_um is not None and hasattr(model, "paths"):
        bank_path = _Path(model.paths.root) / "cell_latent_bank.npz"
        if bank_path.exists():
            id_to_xy = {cid: (float(centroids_um[i, 0]), float(centroids_um[i, 1]))
                        for i, cid in enumerate(cell_ids)
                        if i < len(centroids_um)}
            triples = [(cid, *id_to_xy[cid]) for cid in unresolved_pre
                       if cid in id_to_xy]
            if triples:
                try:
                    from .stain_classifier import (
                        classify_cells_by_centroid, load_bank)
                    bk_key = str(bank_path.resolve())
                    bank = _STAIN_BANK_CACHE.get(bk_key)
                    if bank is None:
                        bank = load_bank(bank_path)
                        _STAIN_BANK_CACHE[bk_key] = bank
                    stain_predictions = classify_cells_by_centroid(
                        model, bundle_path, triples, bank=bank)
                except Exception as e:
                    if progress:
                        print(f"[resolve_from_model] stain_knn skipped: {e}")

    resolutions, unresolved = resolve_cell_types(
        cell_ids,
        annotation_map=annotation_map,
        transcript_classifications=transcript_classifications,
        training_map=training_map,
        training_type_names=training_type_names,
        stain_predictions=stain_predictions,
    )

    if progress and resolutions:
        from collections import Counter as _Counter
        ctr = _Counter(r.source for r in resolutions.values())
        print(f"[resolve_from_model] {len(cell_ids)} cells; per tier: "
              + ", ".join(f"{k}={v}" for k, v in sorted(ctr.items()))
              + (f"; UNRESOLVED={len(unresolved)}" if unresolved else ""))
    return resolutions, unresolved


# Process-wide caches used by `resolve_from_model`. cellAdmix
# classifier output is keyed by id(model); stain banks by absolute path.
# Lifting these from `explain_region` so both 2D and 2.5D share them.
_MODEL_TX_CACHE: dict[int, dict[str, dict]] = {}
_STAIN_BANK_CACHE: dict[str, tuple] = {}


def tier_counts(resolutions: dict[str, TypeResolution]) -> dict[str, int]:
    """Aggregate per-source counts — useful for diagnostics + summary stats."""
    from collections import Counter
    return dict(Counter(r.source for r in resolutions.values()))


def evidence_to_json(evidence: dict[str, Any]) -> str:
    """Serialise an evidence dict to a parquet-friendly JSON string.

    Kept stable so downstream consumers can parse it without depending on
    parquet's struct support (universally available; structs are not).
    """
    return json.dumps(evidence, sort_keys=True, default=float, ensure_ascii=False)
