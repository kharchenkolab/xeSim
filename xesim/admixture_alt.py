"""Alternative admixture-detection methods that don't need membrane staining.

Two approaches, both operating on cellAdmix's NMF outputs + the bundle's
cell segmentation:

  A. `detect_admixture_neighbor_cooccurrence`
     Bundle-scale, cell-level. For each (target type T, foreign factor F
     native to source S): is alpha_T[F] significantly higher in T-cells
     that have an S-typed xy-neighbor compared to T-cells that don't?
     Welch's t-test → rule list matching cellAdmix's schema.

  D. `detect_admixture_em_reassignment`
     Bundle-scale, molecule-level. EM where each molecule's host cell is
     a latent variable competed over by self vs centroid-knn neighbors.
     Posterior weighted by factor compatibility × spatial-kernel distance,
     with a self-bonus λ to avoid runaway reassignment.
     After convergence, per (initial_type, factor) rate of reassignment to
     each (reassigned_type) → rule list.

Both share `build_cell_adjacency_centroid_knn`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind


# ---------------------------------------------------------------------------
# Phase 0 — cell adjacency
# ---------------------------------------------------------------------------


def build_cell_adjacency_centroid_knn(
    cells_df: pd.DataFrame,
    *,
    k: int = 12,
    max_distance_um: float = 15.0,
) -> dict[str, list[str]]:
    """Build cell→neighbors adjacency from cell centroids.

    Uses k-nearest neighbors with a distance cap. Returns {cell_id → [neighbor_id, ...]}.
    Symmetric in expectation (k-NN doesn't enforce symmetry exactly, but the cap does).
    """
    coords = cells_df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    cell_ids = cells_df["cell_id"].to_numpy()
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=min(k + 1, len(coords)))
    if dists.ndim == 1:
        dists = dists[:, None]; idxs = idxs[:, None]
    adjacency: dict[str, list[str]] = {}
    for i, cid in enumerate(cell_ids):
        nbrs: list[str] = []
        for j_pos in range(1, dists.shape[1]):     # skip self at index 0
            j = int(idxs[i, j_pos]); d = float(dists[i, j_pos])
            if d <= max_distance_um:
                nbrs.append(str(cell_ids[j]))
        adjacency[str(cid)] = nbrs
    return adjacency


# ---------------------------------------------------------------------------
# Phase A — neighbor co-occurrence test
# ---------------------------------------------------------------------------


def detect_admixture_neighbor_cooccurrence(
    cf: pd.DataFrame,                       # per-cell factor fractions + _cell_type
    adjacency: dict[str, list[str]],
    factor_source_labels: dict[str, dict[str, Any]],
    biology_factors: list[int],
    *,
    min_stratum_size: int = 30,
    p_thresh: float = 0.01,
    min_effect: float = 0.005,              # minimum delta to consider
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """For each (target T, factor F): stratify T-cells by whether they have
    a neighbor of F's native source type S. Compare alpha_T[F] between
    strata via Welch's t-test. Emit a rule iff:
      - both strata ≥ min_stratum_size
      - mean_with_S_neighbor > mean_without (positive effect)
      - delta ≥ min_effect
      - p-value < p_thresh
    """
    type_lookup = dict(zip(cf["cell_id"].astype(str), cf["_cell_type"]))
    # Pre-compute the *set* of neighbor types per cell — drops duplicates,
    # gives O(1) "has S-neighbor?" lookup.
    neighbor_type_sets: dict[str, set] = {}
    for cid, nbrs in adjacency.items():
        s = set()
        for n in nbrs:
            t = type_lookup.get(str(n))
            if t is not None and not (isinstance(t, float) and np.isnan(t)):
                s.add(t)
        neighbor_type_sets[cid] = s

    types = sorted({t for t in type_lookup.values()
                       if t is not None and not (isinstance(t, float) and np.isnan(t))})
    rules: list[dict[str, Any]] = []

    cf_indexed = cf.set_index("cell_id")
    for f in biology_factors:
        f_id = int(f)
        src = factor_source_labels.get(str(f_id), {}).get("type")
        if src is None:
            continue
        col = f"factor_{f_id}_fraction"
        if col not in cf_indexed.columns:
            continue
        f_vals = cf_indexed[col]

        for t in types:
            if t == src:
                continue
            t_cells = cf_indexed[cf_indexed["_cell_type"] == t].index
            if len(t_cells) < 2 * min_stratum_size:
                continue
            # Stratify
            with_s = []
            without_s = []
            for cid in t_cells:
                cid_s = str(cid)
                if src in neighbor_type_sets.get(cid_s, set()):
                    with_s.append(f_vals.loc[cid])
                else:
                    without_s.append(f_vals.loc[cid])
            with_s = np.asarray(with_s, dtype=np.float64)
            without_s = np.asarray(without_s, dtype=np.float64)
            n_with = len(with_s); n_without = len(without_s)
            if n_with < min_stratum_size or n_without < min_stratum_size:
                continue
            stat, pval = ttest_ind(with_s, without_s, equal_var=False)
            delta = float(with_s.mean() - without_s.mean())
            if stat <= 0 or pval >= p_thresh or delta < min_effect:
                continue
            rules.append({
                "factor": f_id,
                "source": src,
                "target": t,
                "mean_with_neighbor": float(with_s.mean()),
                "mean_without": float(without_s.mean()),
                "delta": delta,
                "neg_log10_p": float(-np.log10(max(float(pval), 1e-300))),
                "n_with": n_with,
                "n_without": n_without,
                "method": "neighbor_cooccurrence",
            })
    rules.sort(key=lambda r: -r["neg_log10_p"])
    if verbose:
        print(f"[admix-A] {len(rules)} rules (top p={rules[0]['neg_log10_p']:.2f})"
              if rules else "[admix-A] no rules above threshold")
    return rules


# ---------------------------------------------------------------------------
# Phase D — EM reassignment
# ---------------------------------------------------------------------------


def detect_admixture_em_reassignment(
    molecules: pd.DataFrame,                # x, y, cell_idx, factor
    cells_df: pd.DataFrame,                 # cell_idx, cell_id, centroid_x, centroid_y, factor_K_fraction
    adjacency: dict[str, list[str]],        # cell_id → neighbor cell_ids
    cell_id_to_type: dict[str, str],
    biology_factors: list[int],
    *,
    n_iter: int = 4,
    self_bonus: float = 2.0,
    candidate_radius_um: float = 8.0,
    k_max: int = 8,
    min_rule_count: int = 30,
    min_rule_rate: float = 0.05,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """EM over molecule → cell assignment.

    For each molecule m at position (x, y) in cell c(m): candidate cells
    are those whose centroid lies within `candidate_radius_um` µm of m
    (always including c(m) itself). Posterior P(z_m = c' | m) ∝ h_{c'}[F_m]
    with a `self_bonus` multiplier on the molecule's current host cell.

    This means molecules deep inside a cell have only their own cell as a
    candidate (can't move), and only near-boundary molecules face
    real competition. After convergence, count molecules whose final cell
    differs from initial; per (initial_type, factor, final_type) groups,
    derive admixture rules in cellAdmix's schema.

    Returns (rules, diagnostics).
    """
    # ---- Build arrays --------------------------------------------------
    cell_idx = cells_df["cell_idx"].to_numpy(dtype=np.int64)
    n_cells = int(cell_idx.max() + 1)
    cell_id_arr = cells_df.sort_values("cell_idx")["cell_id"].to_numpy()
    centroids = (cells_df.sort_values("cell_idx")
                  [["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32))
    # Per-cell type as int (-1 = unknown)
    type_list = sorted({t for t in cell_id_to_type.values()
                          if t is not None and not (isinstance(t, float) and np.isnan(t))})
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    cell_type = np.full(n_cells, -1, dtype=np.int32)
    for i, cid in enumerate(cell_id_arr):
        t = cell_id_to_type.get(str(cid))
        if t is not None and not (isinstance(t, float) and np.isnan(t)):
            cell_type[i] = type_to_idx[t]

    # Per-cell h vector over biology_factors
    K = len(biology_factors)
    h = np.zeros((n_cells, K), dtype=np.float32)
    for i_pos, f in enumerate(biology_factors):
        col = f"factor_{int(f)}_fraction"
        if col in cells_df.columns:
            h[:, i_pos] = cells_df.sort_values("cell_idx")[col].to_numpy(dtype=np.float32)
    # Normalize each cell's h (some cellAdmix-skipped cells may have all zeros)
    row_sums = h.sum(axis=1, keepdims=True)
    h = np.where(row_sums > 0, h / row_sums.clip(min=1e-9), 0.0)

    # ---- Molecule arrays ----------------------------------------------
    factor_to_pos = {int(f): i for i, f in enumerate(biology_factors)}
    mol_factor_raw = molecules["factor"].to_numpy(dtype=np.int64)
    in_panel = np.isin(mol_factor_raw,
                          np.asarray(biology_factors, dtype=np.int64))
    mol = molecules[in_panel].copy().reset_index(drop=True)
    if verbose:
        print(f"[admix-D] using {len(mol):,} of {len(molecules):,} molecules "
              f"(factor in biology_factors)")
    mol_factor_pos = np.asarray(
        [factor_to_pos[int(f)] for f in mol["factor"].to_numpy()],
        dtype=np.int64,
    )
    mol_cell_idx0 = mol["cell_idx"].to_numpy(dtype=np.int64)
    mol_xy = mol[["x", "y"]].to_numpy(dtype=np.float32)

    # Drop molecules whose cell_idx is out of range or whose host cell has
    # no annotated type.
    valid_initial = (mol_cell_idx0 >= 0) & (mol_cell_idx0 < n_cells)
    valid_initial &= cell_type[mol_cell_idx0] >= 0
    if verbose:
        print(f"[admix-D] {valid_initial.sum():,} molecules after dropping "
              f"out-of-range / unannotated host cells")

    mol_cell_idx0 = mol_cell_idx0[valid_initial]
    mol_factor_pos = mol_factor_pos[valid_initial]
    mol_xy = mol_xy[valid_initial]
    M = len(mol_cell_idx0)

    # ---- Build per-molecule candidate cells -----------------------------
    # Candidates = cells whose centroid is within candidate_radius_um of m.
    # Always includes self (m's assigned cell). Padded to k_max.
    if verbose:
        print(f"[admix-D] building per-molecule candidate sets "
              f"(radius={candidate_radius_um}µm)...")
    tree = cKDTree(centroids)
    # Returns list of lists; convert to padded (M, k_max) array.
    raw_candidates = tree.query_ball_point(mol_xy, r=candidate_radius_um)
    candidates = -np.ones((M, k_max), dtype=np.int64)
    for i, cands in enumerate(raw_candidates):
        # Put self first, then up to (k_max - 1) others (closest first not
        # important — likelihood doesn't depend on order).
        self_c = mol_cell_idx0[i]
        others = [c for c in cands if c != self_c]
        sel = [self_c] + others[: k_max - 1]
        candidates[i, : len(sel)] = sel
    n_cand = (candidates >= 0).sum(axis=1)
    if verbose:
        print(f"[admix-D] avg candidates / mol: {n_cand.mean():.2f}  "
              f"(max {n_cand.max()})  "
              f"{(n_cand == 1).sum():,} molecules have only self as candidate "
              f"({(n_cand == 1).mean()*100:.1f}%)")

    # ---- EM iterations -------------------------------------------------
    log_lik_history = []
    cur_cell = mol_cell_idx0.copy()

    for it in range(n_iter):
        cand_valid = candidates >= 0
        # Factor compatibility for each candidate
        cand_h = h[np.where(cand_valid, candidates, 0),
                     mol_factor_pos[:, None]]    # (M, k_max)
        cand_h = np.where(cand_valid, cand_h, 0.0)
        # Self bonus on the molecule's CURRENT cell (re-found inside the
        # candidate matrix on each iteration so the "self" tracks moves)
        is_self = candidates == cur_cell[:, None]
        # Likelihood
        like = cand_h.copy()
        like = np.where(is_self, like * self_bonus, like)
        row_sum = like.sum(axis=1, keepdims=True).clip(min=1e-30)
        post = like / row_sum
        log_lik = float(np.log(row_sum.clip(min=1e-30)).sum())
        log_lik_history.append(log_lik)

        argmax_col = np.argmax(post, axis=1)
        new_cell = candidates[np.arange(M), argmax_col]
        new_cell = np.where(new_cell >= 0, new_cell, cur_cell)
        n_changed = int((new_cell != cur_cell).sum())
        if verbose:
            print(f"[admix-D] EM iter {it+1}: log_lik~{log_lik:,.0f}  "
                  f"{n_changed:,} mol moved this iter "
                  f"({n_changed/M*100:.2f}%)")
        cur_cell = new_cell

    moved = cur_cell != mol_cell_idx0
    n_moved = int(moved.sum())
    diagnostics = {
        "n_molecules_em": M,
        "n_moved": n_moved,
        "move_rate": n_moved / max(M, 1),
        "log_lik_history": log_lik_history,
        "self_bonus": self_bonus,
        "candidate_radius_um": candidate_radius_um,
        "n_iter": n_iter,
        "avg_candidates_per_mol": float(n_cand.mean()),
        "boundary_mol_rate": float((n_cand > 1).mean()),
    }
    if verbose:
        print(f"[admix-D] {n_moved:,} of {M:,} molecules moved "
              f"({n_moved/M*100:.2f}%)")

    # ---- Derive rules ---------------------------------------------------
    # For each (initial_type T, factor F, reassigned_type S):
    #   count of molecules moved
    # If count >= min_rule_count and rate >= min_rule_rate, emit rule.
    initial_type = cell_type[mol_cell_idx0]
    final_type = cell_type[cur_cell]
    # For each (T, F), totals and reassignment breakdown.
    rules: list[dict[str, Any]] = []
    biology_arr = np.asarray(biology_factors, dtype=np.int64)
    for t_idx, t_name in enumerate(type_list):
        t_mask = initial_type == t_idx
        if not t_mask.any():
            continue
        for f_pos, f_id in enumerate(biology_factors):
            f_mask = mol_factor_pos == f_pos
            tf_mask = t_mask & f_mask
            n_tf = int(tf_mask.sum())
            if n_tf < min_rule_count:
                continue
            # Distribution of final_type among moved tf molecules
            moved_tf = tf_mask & moved
            if not moved_tf.any():
                continue
            dest_types = final_type[moved_tf]
            dest_counts = np.bincount(dest_types[dest_types >= 0],
                                         minlength=len(type_list))
            top_idx = int(np.argmax(dest_counts))
            s_name = type_list[top_idx]
            if s_name == t_name:
                continue
            n_moved_to_s = int(dest_counts[top_idx])
            rate = n_moved_to_s / n_tf
            if rate < min_rule_rate:
                continue
            rules.append({
                "factor": int(f_id),
                "source": s_name,
                "target": t_name,
                "n_in_target": n_tf,
                "n_moved_to_source": n_moved_to_s,
                "reassignment_rate": float(rate),
                # An effective "neg_log10_p"-like rank metric for sorting:
                # use a binomial tail-prob approximation (very rough).
                "neg_log10_p": float(-np.log10(max(_binom_tail_pvalue(
                    n_moved_to_s, n_tf, 0.05), 1e-300))),
                "method": "em_reassignment",
            })
    rules.sort(key=lambda r: -r["neg_log10_p"])
    if verbose:
        print(f"[admix-D] {len(rules)} rules derived")
    return rules, diagnostics


def _binom_tail_pvalue(k: int, n: int, p0: float) -> float:
    """Rough binomial-tail p-value for n trials, k successes, null p0.
    Used as a sortable significance proxy."""
    # Normal approximation
    if n <= 0:
        return 1.0
    mu = n * p0
    sigma = (n * p0 * (1 - p0)) ** 0.5
    if sigma == 0:
        return 1.0 if k <= mu else 1e-12
    from math import erfc, sqrt
    z = (k - mu) / sigma
    # Upper tail
    return 0.5 * erfc(z / sqrt(2))
