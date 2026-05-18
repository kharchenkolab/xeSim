"""Round-trip admixture validation: inject known admixture into real molecule
data, run detection methods, measure recovery vs ground truth.

Production note: this is a *validation* tool. Production xeSim
(`apply_admixture_rate`, etc.) continues to use cellAdmix's M-derived
rules from priors as the canonical admixture-rule source. This module's
output tells us how reliable that source is, not what to put in
production.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def inject_boundary_admixture(
    real_molecules: pd.DataFrame,           # x, y, cell_idx, factor (+ optional gene_idx)
    cells_df: pd.DataFrame,                 # cell_idx, cell_id, centroid_x, centroid_y, _cell_type
    rule: dict[str, Any],                   # {'factor': int, 'source': str, 'target': str}
    rate: float,                             # fraction of (S, F)-molecules to move
    *,
    boundary_shift_factor: float = 0.7,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move `rate` fraction of (source-cell, factor F) molecules to nearest
    target-typed neighbor cell. Position shifted toward the S-T midline so
    boundary-proximity methods see them at the cell-cell interface.

    Parameters
    ----------
    boundary_shift_factor : float
        Fraction of the way from S centroid → T centroid to place the
        moved molecule. 0.5 = exactly on the midline; 0.7 = inside T.

    Returns
    -------
    (modified_molecules, ground_truth)
        modified_molecules has the same shape as real_molecules with
        `cell_idx` and `x, y` columns updated for moved molecules.
        ground_truth has columns: molecule_idx, orig_cell_idx,
        new_cell_idx, rule_factor, rule_source, rule_target.
    """
    rng = rng or np.random.default_rng(0)
    factor = int(rule["factor"]); source = rule["source"]; target = rule["target"]

    s_cells = cells_df[cells_df["_cell_type"] == source].copy()
    t_cells = cells_df[cells_df["_cell_type"] == target].copy()
    if len(s_cells) == 0 or len(t_cells) == 0:
        if verbose:
            print(f"[inject] no {source} or {target} cells")
        return real_molecules.copy(), pd.DataFrame(columns=["molecule_idx"])

    # For each S-cell, find its nearest T-typed neighbor by centroid.
    t_coords = t_cells[["centroid_x", "centroid_y"]].to_numpy()
    t_idxs = t_cells["cell_idx"].to_numpy()
    tree = cKDTree(t_coords)
    s_coords = s_cells[["centroid_x", "centroid_y"]].to_numpy()
    s_idxs = s_cells["cell_idx"].to_numpy()
    dists, nearest_pos = tree.query(s_coords, k=1)
    s_to_t = {int(si): int(t_idxs[int(np)]) for si, np in zip(s_idxs, nearest_pos)}
    s_to_dist = {int(si): float(d) for si, d in zip(s_idxs, dists)}

    # Filter to S-cells whose nearest T is plausibly a real neighbor (<25 µm)
    s_with_t = {si for si, d in s_to_dist.items() if d <= 25.0}
    if verbose:
        print(f"[inject] {len(s_with_t)} {source} cells have a {target} "
              f"neighbor within 25 µm (of {len(s_cells)} total)")

    # All (S, F) molecules
    mol = real_molecules.reset_index(drop=False).rename(columns={"index": "molecule_idx"})
    s_idx_set = np.fromiter(s_with_t, dtype=np.int64)
    in_s = np.isin(mol["cell_idx"].to_numpy(), s_idx_set)
    in_f = mol["factor"].to_numpy() == factor
    cand_mask = in_s & in_f
    cand_indices = np.where(cand_mask)[0]
    if verbose:
        print(f"[inject] {len(cand_indices):,} candidate ({source}, F={factor}) molecules")

    n_to_move = int(rate * len(cand_indices))
    if n_to_move == 0:
        return real_molecules.copy(), pd.DataFrame(columns=["molecule_idx"])
    moved_indices = rng.choice(cand_indices, size=n_to_move, replace=False)

    # Build cell_idx → centroid lookup
    centroid_lookup = dict(zip(cells_df["cell_idx"].astype(int),
                                  zip(cells_df["centroid_x"], cells_df["centroid_y"])))

    modified = real_molecules.copy().reset_index(drop=True)
    gt_rows = []
    for mi in moved_indices:
        orig_cell = int(mol.at[mi, "cell_idx"])
        new_cell = s_to_t.get(orig_cell)
        if new_cell is None:
            continue
        # Shift position toward T's centroid
        sx, sy = centroid_lookup[orig_cell]
        tx, ty = centroid_lookup[new_cell]
        u = boundary_shift_factor
        # Place along the line from orig_cell centroid toward T centroid,
        # with small Gaussian jitter to avoid all moved molecules at the
        # exact same point.
        new_x = (1 - u) * sx + u * tx + rng.normal(0, 0.5)
        new_y = (1 - u) * sy + u * ty + rng.normal(0, 0.5)
        modified.at[mi, "cell_idx"] = new_cell
        modified.at[mi, "x"] = new_x
        modified.at[mi, "y"] = new_y
        gt_rows.append({"molecule_idx": int(mi),
                          "orig_cell_idx": orig_cell, "new_cell_idx": new_cell,
                          "rule_factor": factor, "rule_source": source,
                          "rule_target": target})

    ground_truth = pd.DataFrame(gt_rows)
    if verbose:
        print(f"[inject] moved {len(ground_truth)} molecules ({rate*100:.0f}% rate)")
    return modified, ground_truth


def recompute_cf_from_molecules(
    modified_molecules: pd.DataFrame,
    cells_df: pd.DataFrame,                 # has cell_idx, cell_id, _cell_type
    biology_factors: list[int],
) -> pd.DataFrame:
    """Reproduce a `cell_factor_fractions`-like DataFrame from modified
    molecules. Required because both A and D consume cf, and the original
    cf reflects the un-injected molecule distribution.

    Returns columns: cell_id, transcript_count, dominant_factor,
    dominant_fraction, factor_K_fraction (1..max K), and _cell_type.
    """
    counts = (modified_molecules.groupby(["cell_idx", "factor"])
                                 .size()
                                 .reset_index(name="count"))
    pivot = counts.pivot(index="cell_idx", columns="factor",
                            values="count").fillna(0)
    transcript_count = pivot.sum(axis=1)
    pivot_n = pivot.div(transcript_count.replace(0, 1), axis=0)
    pivot_n.columns = [f"factor_{int(c)}_fraction" for c in pivot_n.columns]
    # Make sure all biology factors are present
    for f in biology_factors:
        col = f"factor_{int(f)}_fraction"
        if col not in pivot_n.columns:
            pivot_n[col] = 0.0
    pivot_n = pivot_n.reset_index()
    pivot_n["transcript_count"] = pivot_n["cell_idx"].map(transcript_count)
    # Join with cells_df
    out = pivot_n.merge(cells_df[["cell_idx", "cell_id", "_cell_type"]],
                          on="cell_idx", how="inner")
    return out


def evaluate_admixture_detection(
    modified_molecules: pd.DataFrame,
    cells_df_merged: pd.DataFrame,           # has cell_idx, cell_id, centroid_x,_y, _cell_type
    adjacency: dict[str, list[str]],
    factor_source_labels: dict[str, Any],
    biology_factors: list[int],
    cell_id_to_type: dict[str, str],
    rule_under_test: dict[str, Any],
    ground_truth: pd.DataFrame,
    *,
    n_em_iter: int = 4,
    self_bonus: float = 2.0,
    candidate_radius_um: float = 10.0,
    run_d: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run detection methods on the modified data and compare to ground truth."""
    from xesim.admixture_alt import (
        detect_admixture_neighbor_cooccurrence,
        detect_admixture_em_reassignment,
    )

    f = int(rule_under_test["factor"])
    s = rule_under_test["source"]
    t = rule_under_test["target"]

    # Recompute cf on modified molecules
    cf_modified = recompute_cf_from_molecules(
        modified_molecules, cells_df_merged, biology_factors,
    )
    # Merge centroid info for downstream (D needs it)
    cf_with_centroid = cells_df_merged[["cell_idx", "cell_id", "centroid_x",
                                              "centroid_y"]].merge(
        cf_modified, on=["cell_idx", "cell_id"], how="inner",
    )

    result = {
        "rule": rule_under_test,
        "n_injected": len(ground_truth),
    }

    # --- A ---
    rules_A = detect_admixture_neighbor_cooccurrence(
        cf_modified, adjacency,
        factor_source_labels=factor_source_labels,
        biology_factors=biology_factors,
        verbose=verbose,
    )
    a_match = next(
        (r for r in rules_A if r["factor"] == f and r["source"] == s and r["target"] == t),
        None,
    )
    result["A_detected"] = a_match is not None
    result["A_strength"] = a_match["neg_log10_p"] if a_match else 0.0
    result["A_delta"] = a_match["delta"] if a_match else 0.0
    result["A_n_other_rules"] = len(rules_A)
    result["A_rules_other_rules_with_same_target"] = [
        {"factor": r["factor"], "source": r["source"], "delta": r["delta"],
         "neg_log10_p": r["neg_log10_p"]}
        for r in rules_A if r["target"] == t and r != a_match
    ][:3]

    # --- D ---
    if run_d:
        rules_D, diag_D = detect_admixture_em_reassignment(
            modified_molecules, cf_with_centroid, adjacency,
            cell_id_to_type, biology_factors,
            n_iter=n_em_iter, self_bonus=self_bonus,
            candidate_radius_um=candidate_radius_um,
            min_rule_count=50, min_rule_rate=0.02,
            verbose=verbose,
        )
        d_match = next(
            (r for r in rules_D if r["factor"] == f and r["source"] == s and r["target"] == t),
            None,
        )
        result["D_detected"] = d_match is not None
        result["D_rate"] = d_match["reassignment_rate"] if d_match else 0.0
        # Per-molecule recovery: of the injected molecules, how many did D
        # reassign back to a source-typed cell?
        # We can't easily get D's per-molecule output without modifying the
        # function — use the fact that ground_truth.molecule_idx tells us
        # exactly which were moved. We'd need to refactor D to expose
        # per-mol cur_cell. For now, just report the rule-level metrics.
        result["D_n_other_rules"] = len(rules_D)
        result["D_n_moved_em"] = diag_D["n_moved"]
        result["D_move_rate_total"] = diag_D["move_rate"]
    else:
        result["D_detected"] = None; result["D_rate"] = None

    return result
