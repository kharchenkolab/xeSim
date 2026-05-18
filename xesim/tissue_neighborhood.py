"""Plan3 A1 — per-type empirical neighborhood model.

Fits the per-type-pair conditional P(neighbor_type | center_type, distance)
from real canonical crops. At synthesis, Gibbs sampling over cell types
respects this empirical neighbor distribution, producing tissue-level
clustering (islets, acini, immune infiltrates) that arises from real
biology rather than from heuristic per-type cluster sizes.

This is the empirical-data version of the Plan3 frontier A1 hierarchical
GNN tissue layer. A small MLP/GNN can replace the empirical estimate
later if needed; for now empirical conditionals are simpler, easier to
audit, and capture the same biology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cell_types import load_cell_type_assignment
from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


TISSUE_NEIGHBORHOOD_TYPE = "xesim.tissue_neighborhood.v0"


def fit_tissue_neighborhood(
    crop_manifest_path: Path,
    cell_types_path: Path,
    output_path: Path,
    n_neighbors: int = 8,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    """Fit per-type empirical neighborhood matrix from canonical crops.

    For each cell of type ``i``, count types of its ``n_neighbors`` nearest
    other cells. Aggregate across crops + cells of the same type to form
    ``P[i, j]`` = fraction of a type-i cell's neighbors that are type-j.
    """

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    ct_artifact = load_cell_type_assignment(cell_types_path)
    type_names = list(ct_artifact["type_names"])
    n_types = len(type_names)
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    accept = set(splits)
    root = crop_manifest_path.parent
    counts = np.zeros((n_types, n_types), dtype=np.float64)
    type_totals = np.zeros(n_types, dtype=np.float64)
    distance_bins = np.array([5, 10, 20, 40, 80], dtype=np.float32)
    distance_counts = np.zeros((n_types, n_types, len(distance_bins) + 1), dtype=np.float64)

    for crop in manifest.get("crops", []):
        if str(crop.get("split", "train")) not in accept:
            continue
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            cell_ids = [str(v) for v in data["cell_ids"].tolist()]
        unique = np.unique(cell_label)
        nonzero = unique[unique > 0].tolist()
        if len(nonzero) != len(cell_ids):
            n = min(len(nonzero), len(cell_ids))
            nonzero = nonzero[:n]
            cell_ids = cell_ids[:n]
        if len(nonzero) <= 1:
            continue
        # Per-cell centroid + type.
        centroids = np.zeros((len(nonzero), 2), dtype=np.float32)
        type_idx_arr = np.zeros(len(nonzero), dtype=np.int64)
        for ci, (lab, cid) in enumerate(zip(nonzero, cell_ids)):
            ys, xs = np.nonzero(cell_label == int(lab))
            if ys.size == 0:
                continue
            centroids[ci] = [float(np.mean(ys)), float(np.mean(xs))]
            type_idx_arr[ci] = int(cell_id_to_type.get(cid, 0))
        # k-NN neighbors per cell.
        n_cells = len(nonzero)
        diffs = centroids[:, None, :] - centroids[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        np.fill_diagonal(d2, np.inf)
        k = min(int(n_neighbors), n_cells - 1)
        if k <= 0:
            continue
        nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        for ci in range(n_cells):
            ti = int(type_idx_arr[ci])
            if ti == 0:
                continue
            type_totals[ti] += 1
            for nb in nn_idx[ci]:
                tj = int(type_idx_arr[nb])
                if tj == 0:
                    continue
                counts[ti, tj] += 1
                # Distance bin.
                dist = float(np.sqrt(d2[ci, nb]))
                bi = int(np.searchsorted(distance_bins, dist))
                distance_counts[ti, tj, bi] += 1
    # Normalize to conditional P[neighbor | center].
    row_sums = counts.sum(axis=1, keepdims=True)
    cond_prob = np.where(row_sums > 0, counts / np.maximum(row_sums, 1.0), 0.0)
    # Type prior from totals (fraction of cells of each type, training set).
    type_prior = type_totals / max(type_totals.sum(), 1.0)

    payload = {
        "type": TISSUE_NEIGHBORHOOD_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(crop_manifest_path),
        "cell_types_path": str(cell_types_path),
        "type_names": type_names,
        "n_neighbors": int(n_neighbors),
        "splits": list(splits),
        "type_prior": type_prior.tolist(),
        "neighbor_conditional": cond_prob.tolist(),
        "neighbor_counts": counts.tolist(),
        "type_totals": type_totals.tolist(),
        "distance_bins_px": distance_bins.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def load_tissue_neighborhood(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        obj = json.load(handle)
    if obj.get("type") != TISSUE_NEIGHBORHOOD_TYPE:
        raise ValueError(f"{path}: not a tissue_neighborhood artifact")
    return obj


def gibbs_assign_types(
    centers: np.ndarray,
    type_names: list[str],
    type_prior: np.ndarray,
    neighbor_conditional: np.ndarray,
    n_neighbors: int = 8,
    n_iterations: int = 5,
    seed: int = 11,
    type_frequencies: dict[str, float] | None = None,
    prior_weight: float = 1.0,
    position_log_prior: np.ndarray | None = None,
    position_prior_weight: float = 1.0,
    rebalance_to_prior: bool = True,
) -> list[str]:
    """Iteratively assign types via per-cell Gibbs sampling.

    For each cell, sample its type from a distribution proportional to:
      P(t) ∝ type_prior[t]^prior_weight * Π over n of P(t_n | t)
    where P(t_n | t) is the empirical neighbor-conditional. Iterating a
    few rounds produces local clustering (islets, acini) without any
    explicit per-type cluster-size heuristic.

    With ``prior_weight = 1`` (default), each neighbor contributes
    log P(t_n | t) which adds k×log_cond magnitude to the score, while
    the prior adds only log P(t) — so the conditional dominates and the
    chain drifts from prior. With ``prior_weight > 1``, the prior gets
    a stronger pull (e.g. prior_weight=2 doubles the prior's log-weight
    contribution; useful for forward-mode marginal stability).
    """

    rng = np.random.default_rng(int(seed))
    n_cells = int(centers.shape[0])
    n_types = int(type_prior.shape[0])
    if n_cells == 0:
        return []
    candidate_indices = [i for i, name in enumerate(type_names) if name and name != "unknown"]
    if not candidate_indices:
        return [""] * n_cells
    candidate_set = np.asarray(candidate_indices)

    # Override prior with explicit frequencies if provided.
    prior = np.asarray(type_prior, dtype=np.float64).copy()
    if type_frequencies:
        for name, freq in type_frequencies.items():
            if name in type_names:
                prior[type_names.index(name)] = float(freq)
    prior_candidates = prior[candidate_set]
    prior_candidates = prior_candidates / max(prior_candidates.sum(), 1e-12)

    # Initialize types from prior.
    init_choice = rng.choice(candidate_set, size=n_cells, p=prior_candidates)
    types = init_choice.astype(np.int64)

    # k-NN.
    diffs = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    np.fill_diagonal(d2, np.inf)
    k = min(int(n_neighbors), max(1, n_cells - 1))
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    cond = np.asarray(neighbor_conditional, dtype=np.float64)
    cond = np.clip(cond, 1e-6, None)
    log_cond = np.log(cond)
    log_prior = np.log(np.clip(prior, 1e-6, None))
    pw = float(prior_weight)

    # Optional per-cell spatial prior. Shape (n_cells, n_types) in log space.
    # Contribution to each cell's score is position_prior_weight * row.
    if position_log_prior is not None:
        pos_log = np.asarray(position_log_prior, dtype=np.float64)
        assert pos_log.shape == (n_cells, n_types), \
            f"position_log_prior shape {pos_log.shape} != ({n_cells}, {n_types})"
    else:
        pos_log = None
    ppw = float(position_prior_weight)
    for _ in range(int(n_iterations)):
        order = rng.permutation(n_cells)
        for ci in order:
            neighbor_types = types[nn_idx[ci]]
            # Score for each candidate type t:
            #   log P(t | neighbors, position) ~ pw*log P(t) + ppw*log P_spatial(t|x_ci)
            #                                  + sum_n log P(t_n | t)
            log_scores = pw * log_prior[candidate_set]
            if pos_log is not None:
                log_scores = log_scores + ppw * pos_log[ci, candidate_set]
            for nt in neighbor_types:
                log_scores = log_scores + log_cond[candidate_set, int(nt)]
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            probs = probs / max(probs.sum(), 1e-12)
            types[ci] = int(rng.choice(candidate_set, p=probs))
    # Plan3 v34 forward-mode fix: post-Gibbs marginal rebalancing.
    # Gibbs's stationary distribution drifts from the prior to dense
    # same-type clusters when neighbor_conditional has high diagonals
    # (rare types like Mural over-cluster ~40x their prior frequency).
    # Rebalance by relabelling a fraction of cells from over-represented
    # to under-represented types, weighted by the prior gap. Spatial
    # structure produced by Gibbs is mostly preserved because we only
    # touch the marginal-imbalance portion.
    if rebalance_to_prior:
        types = _rebalance_to_prior(types, prior_candidates, candidate_set, rng)
    return [type_names[int(t)] for t in types]


def _rebalance_to_prior(
    types: np.ndarray,
    prior_candidates: np.ndarray,
    candidate_set: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Post-Gibbs marginal rebalancing toward ``prior_candidates``.

    Counts current type assignments; for over-represented types, randomly
    selects ``excess`` cells and relabels them to under-represented types
    in proportion to the prior gap. After rebalancing, the type marginal
    matches the prior; spatial structure is mostly preserved because only
    the marginal-imbalance subset is touched.
    """
    n = int(types.shape[0])
    if n == 0:
        return types
    target = prior_candidates * float(n)
    counts = np.zeros_like(prior_candidates, dtype=np.float64)
    cs_list = candidate_set.tolist()
    for t in types:
        try:
            local = cs_list.index(int(t))
            counts[local] += 1.0
        except ValueError:
            continue
    excess = counts - target
    over_local = np.where(excess > 0)[0]
    under_local = np.where(excess < 0)[0]
    if len(over_local) == 0 or len(under_local) == 0:
        return types
    under_demand = -excess[under_local]
    under_dist = under_demand / max(under_demand.sum(), 1e-12)
    out = types.copy()
    for ol in over_local:
        type_idx = int(candidate_set[int(ol)])
        cells = np.where(out == type_idx)[0]
        n_take = int(round(excess[int(ol)]))
        if n_take <= 0 or len(cells) == 0:
            continue
        n_take = min(n_take, len(cells))
        picked = rng.choice(cells, size=n_take, replace=False)
        new_under = rng.choice(under_local, size=n_take, p=under_dist)
        for p, nl in zip(picked, new_under):
            out[int(p)] = int(candidate_set[int(nl)])
    return out


def refine_assignment_gibbs(
    centers: np.ndarray,
    initial_types: list[str],
    type_names: list[str],
    type_prior: np.ndarray,
    neighbor_conditional: np.ndarray,
    n_neighbors: int = 8,
    n_iterations: int = 2,
    seed: int = 11,
) -> list[str]:
    """Refine an existing per-cell type assignment via a few Gibbs sweeps
    using the empirical neighbor conditional. Used by A1v2 to add local
    neighborhood realism on top of domain-first assignment.
    """

    rng = np.random.default_rng(int(seed))
    n_cells = int(centers.shape[0])
    if n_cells == 0:
        return []
    candidate_indices = [i for i, name in enumerate(type_names) if name and name != "unknown"]
    if not candidate_indices:
        return list(initial_types)
    candidate_set = np.asarray(candidate_indices)
    name_to_idx = {name: i for i, name in enumerate(type_names)}
    types = np.asarray(
        [int(name_to_idx.get(t, candidate_indices[0])) for t in initial_types],
        dtype=np.int64,
    )

    diffs = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    np.fill_diagonal(d2, np.inf)
    k = min(int(n_neighbors), max(1, n_cells - 1))
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    cond = np.clip(np.asarray(neighbor_conditional, dtype=np.float64), 1e-6, None)
    log_cond = np.log(cond)
    log_prior = np.log(np.clip(np.asarray(type_prior, dtype=np.float64), 1e-6, None))

    for _ in range(int(n_iterations)):
        order = rng.permutation(n_cells)
        for ci in order:
            neighbor_types = types[nn_idx[ci]]
            log_scores = log_prior[candidate_set]
            for nt in neighbor_types:
                log_scores = log_scores + log_cond[candidate_set, int(nt)]
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            probs = probs / max(probs.sum(), 1e-12)
            types[ci] = int(rng.choice(candidate_set, p=probs))
    return [type_names[int(t)] for t in types]
