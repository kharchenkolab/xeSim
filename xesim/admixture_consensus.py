"""Consensus + calibration over multiple admixture-detection methods.

Each method (M=membrane, A=neighbor-cooccurrence, D=EM-reassignment)
produces a rule list keyed by (factor, source, target) with a per-rule
strength. None is "ground truth". This module fuses them into:

  - a tier per rule (1=all 3 agree, 2=any 2 agree, 3=single-method)
  - a calibrated strength per method (raw / threshold maximizing F1
    against the ≥2-method pseudo-truth set)
  - a combined score (geometric mean of calibrated strengths over the
    methods that detected the rule)
  - a naive-Bayes posterior P(rule is real | detection pattern)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _scores_from_rules(rules: list[dict[str, Any]],
                          strength_key: str) -> dict[tuple, float]:
    """Pivot rule list into a (factor, source, target) → strength map."""
    out: dict[tuple, float] = {}
    for r in rules:
        out[(int(r["factor"]), r["source"], r["target"])] = float(r[strength_key])
    return out


def _calibrate_f1(scores: dict[tuple, float],
                    truth: set,
                    candidate_universe: set) -> dict[str, float]:
    """Find the score threshold that maximizes F1 against `truth`, with
    detection = (score >= threshold). Returns dict with threshold, F1,
    precision, recall.

    `candidate_universe` includes rules not in `scores` (where the method
    didn't detect — treated as score=0, so any positive threshold excludes
    them). This makes the false-positive count correct."""
    if not scores:
        return {"threshold": float("nan"), "f1": 0.0,
                "precision": 0.0, "recall": 0.0, "n_detected": 0}
    # Candidate thresholds: every unique score value.
    unique_vals = sorted(set(scores.values()))
    best = {"threshold": unique_vals[0], "f1": -1.0,
              "precision": 0.0, "recall": 0.0, "n_detected": 0}
    for thr in unique_vals:
        detected = {k for k, v in scores.items() if v >= thr}
        tp = len(detected & truth)
        if tp == 0:
            continue
        fp = len(detected - truth)
        fn = len(truth - detected)
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1),
                      "precision": float(prec), "recall": float(rec),
                      "n_detected": int(tp + fp)}
    return best


def _naive_bayes_posteriors(
    detect_patterns: list[tuple[bool, bool, bool]],
    truth_flag: list[bool],
    prior_real: float | None = None,
    eps: float = 0.02,
) -> dict[tuple[bool, bool, bool], float]:
    """For each detection pattern (m, a, d), compute P(real | pattern)
    via naive Bayes. We need per-method likelihoods P(detect=T | real)
    and P(detect=T | not real), estimated empirically from the truth labels.

    The 'truth' here is the pseudo-truth (≥2 methods agree). This is
    circular (truth depends on detections) but the resulting posterior
    is still a useful soft consensus score.

    Returns dict mapping (m_in, a_in, d_in) → posterior P(real).
    """
    if prior_real is None:
        prior_real = sum(truth_flag) / max(len(truth_flag), 1)
    # Per-method TPR and FPR
    tpr = [eps] * 3; fpr = [eps] * 3
    for i in range(3):
        det_real = sum(1 for p, t in zip(detect_patterns, truth_flag) if t and p[i])
        n_real = sum(truth_flag)
        det_notreal = sum(1 for p, t in zip(detect_patterns, truth_flag) if not t and p[i])
        n_notreal = len(truth_flag) - n_real
        tpr[i] = max(min(det_real / max(n_real, 1), 1 - eps), eps)
        fpr[i] = max(min(det_notreal / max(n_notreal, 1), 1 - eps), eps)
    # Posterior for each possible pattern
    out: dict[tuple[bool, bool, bool], float] = {}
    for m in (False, True):
        for a in (False, True):
            for d in (False, True):
                pat = (m, a, d)
                p_real = math.log(max(prior_real, eps))
                p_not = math.log(max(1 - prior_real, eps))
                for i, det in enumerate(pat):
                    if det:
                        p_real += math.log(tpr[i])
                        p_not += math.log(fpr[i])
                    else:
                        p_real += math.log(1 - tpr[i])
                        p_not += math.log(1 - fpr[i])
                # Normalize
                lr = p_real - p_not
                posterior = 1.0 / (1.0 + math.exp(-lr))
                out[pat] = posterior
    return out, {"tpr": tpr, "fpr": fpr, "prior_real": prior_real}


def combine_admixture_methods(
    M_rules: list[dict[str, Any]],
    A_rules: list[dict[str, Any]],
    D_rules: list[dict[str, Any]],
    *,
    M_strength_key: str = "membrane_neg_log10_p",
    A_strength_key: str = "neg_log10_p",
    D_strength_key: str = "reassignment_rate",
    min_methods_for_truth: int = 2,
    eps: float = 0.02,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse three admixture rule lists into a single tiered list.

    Each output rule has fields:
      factor, source, target
      tier ∈ {1, 2, 3}                   (1 = all methods agree)
      n_methods                          (1..3)
      detected_by_{M,A,D}                (bool)
      {M,A,D}_strength                   (raw)
      {M,A,D}_calibrated                 (raw / threshold; 0 if not detected)
      combined_score                     (geometric mean over detected,
                                          calibrated values)
      posterior                          (naive-Bayes P(real | pattern))

    Returns (rules_list, diagnostics).
    """
    M_scores = _scores_from_rules(M_rules, M_strength_key)
    A_scores = _scores_from_rules(A_rules, A_strength_key)
    D_scores = _scores_from_rules(D_rules, D_strength_key)

    all_keys = set(M_scores) | set(A_scores) | set(D_scores)
    truth = {k for k in all_keys
              if (k in M_scores) + (k in A_scores) + (k in D_scores)
              >= min_methods_for_truth}

    # Per-method F1 calibration
    M_cal = _calibrate_f1(M_scores, truth, all_keys)
    A_cal = _calibrate_f1(A_scores, truth, all_keys)
    D_cal = _calibrate_f1(D_scores, truth, all_keys)

    # Naive-Bayes posteriors over the 8 possible (m, a, d) patterns
    patterns = [(k in M_scores, k in A_scores, k in D_scores) for k in all_keys]
    truth_flags = [k in truth for k in all_keys]
    posteriors, posterior_diag = _naive_bayes_posteriors(
        patterns, truth_flags, eps=eps,
    )

    out: list[dict[str, Any]] = []
    for k in all_keys:
        f, s, t = k
        m_in = k in M_scores; a_in = k in A_scores; d_in = k in D_scores
        n_methods = m_in + a_in + d_in
        tier = 1 if n_methods >= 3 else (2 if n_methods >= 2 else 3)

        def _cal_strength(scores, cal, in_):
            if not in_ or cal["threshold"] == 0 or math.isnan(cal["threshold"]):
                return 0.0
            return float(scores.get(k, 0.0) / cal["threshold"])

        m_cal = _cal_strength(M_scores, M_cal, m_in)
        a_cal = _cal_strength(A_scores, A_cal, a_in)
        d_cal = _cal_strength(D_scores, D_cal, d_in)

        detected_strengths = [x for x in (m_cal, a_cal, d_cal) if x > 0]
        combined = (float(np.exp(np.log(detected_strengths).mean()))
                    if detected_strengths else 0.0)
        out.append({
            "factor": f, "source": s, "target": t,
            "tier": tier,
            "n_methods": n_methods,
            "detected_by_M": m_in, "detected_by_A": a_in, "detected_by_D": d_in,
            "M_strength": float(M_scores.get(k, 0.0)),
            "A_strength": float(A_scores.get(k, 0.0)),
            "D_strength": float(D_scores.get(k, 0.0)),
            "M_calibrated": m_cal,
            "A_calibrated": a_cal,
            "D_calibrated": d_cal,
            "combined_score": combined,
            "posterior": float(posteriors[(m_in, a_in, d_in)]),
        })
    out.sort(key=lambda r: (r["tier"], -r["combined_score"], -r["posterior"]))
    diagnostics = {
        "n_total_candidates": len(all_keys),
        "n_pseudo_truth": len(truth),
        "M_calibration": M_cal,
        "A_calibration": A_cal,
        "D_calibration": D_cal,
        "posterior_diag": posterior_diag,
        "pattern_posteriors": {f"M={m},A={a},D={d}": p
                                  for (m, a, d), p in posteriors.items()},
    }
    return out, diagnostics
