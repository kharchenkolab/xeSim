"""Geometry-free emission decisions: counts + per-transcript leakage.

This module produces a long-format DataFrame of per-transcript decisions
— ``(cell_id, feature_name, is_leaked, compartment)`` — with NO
spatial coordinates. Placement happens later (see ``placement_2d`` /
``placement_3d``). All work here is dimension-agnostic.

Steps:

1. Sample the count matrix via STpuppeteer's program-based generative
   model (``sample_counts_program_model``).
2. Expand to long format (``counts_to_transcript_df``).
3. Decide per-transcript leakage via the union rule
   (``classify_leakage``).
4. Stamp a compartment column. Phase 1 ships with all transcripts marked
   ``"cyto"`` since STpuppeteer doesn't yet model compartment preference
   (Phase 5 extension; see §6.1 of tmp/cellAdmix-integration.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from STpuppeteer.simulation import SimulationConfig


def emit_decisions(
    cell_gdf: pd.DataFrame,
    cfg: "SimulationConfig",
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run STpuppeteer count + leakage decisions for the cells in ``cell_gdf``.

    Parameters
    ----------
    cell_gdf : pd.DataFrame
        Output of :func:`cell_adapter.build_stpuppeteer_cell_gdf` —
        columns ``cell_id``, ``celltype``, ``scale``, ``leakage_percentage``
        (plus ``visible_voxels`` and ``celltype_original`` which we ignore).
    cfg : SimulationConfig
        Loaded STpuppeteer config. Provides programs, cell_type_specs,
        leakage_by_gene, dispersion params, etc.
    rng : np.random.Generator
        NumPy RNG; threading downstream from xeSim's seeded run.

    Returns
    -------
    pd.DataFrame
        Long transcript table. Columns:

        - ``cell_id`` (str) — source cell
        - ``feature_name`` (str) — gene name
        - ``is_leaked`` (bool)
        - ``compartment`` (str) — placeholder "cyto" in Phase 1

        No ``transcript_id`` column — xeSim's bundle writer / reemit
        assigns its own uint64 IDs at output time.
    """
    from STpuppeteer.simulation import (
        sample_counts_program_model,
        classify_leakage,
    )

    if len(cell_gdf) == 0:
        # No emittable cells (e.g., tile had no cells with configured types).
        # Returning an empty frame keeps the caller's downstream code simple.
        return pd.DataFrame({
            "cell_id": [], "feature_name": [],
            "is_leaked": [], "compartment": [],
        })

    # Step 1: per-(cell, gene) count draw via the LMC model. The sampler
    # reads `celltype` and `scale` from cell_gdf and produces an integer
    # (n_cells × n_genes) array.
    count_array = sample_counts_program_model(cell_gdf, cfg, rng)

    # Step 2: expand the dense count matrix to a long per-transcript frame.
    # Inlined np.nonzero + np.repeat rather than calling
    # counts_to_transcript_df — the helper additionally synthesizes
    # f"tr_{i}" Python-string transcript IDs that no xeSim consumer reads
    # (bundle writer / reemit assign their own uint64 IDs at output time),
    # which adds ~2s + ~200MB at 6M-transcript scale.
    gene_names = cfg.resolve_gene_names()
    counts_arr = np.asarray(count_array)
    ci, gi = np.nonzero(counts_arr)
    cv = counts_arr[ci, gi].astype(int)
    if cv.sum() == 0:
        return pd.DataFrame({
            "cell_id": pd.array([], dtype=object),
            "feature_name": pd.array([], dtype=object),
            "is_leaked": pd.array([], dtype="boolean"),
            "compartment": pd.array([], dtype=object),
        })
    pair = np.repeat(np.arange(len(cv)), cv)
    trs_df = pd.DataFrame({
        "cell_id": np.asarray(cell_gdf["cell_id"].values)[ci[pair]],
        "feature_name": np.asarray(gene_names)[gi[pair]],
    })

    # Step 3: per-transcript Bernoulli leakage. classify_leakage only reads
    # gpar_df["gene_leakage"]; build a minimal 2-column frame from
    # cfg.leakage_by_gene rather than calling build_program_gpar_df, whose
    # μ/Φ/z-score columns we don't use.
    gpar_df = pd.DataFrame({
        "feature_name": gene_names,
        "gene_leakage": _resolve_gene_leakage(cfg, gene_names),
    })
    is_leaked = classify_leakage(trs_df, cell_gdf, gpar_df, rng)
    trs_df["is_leaked"] = is_leaked

    # Step 4: compartment stamp. Until §6.1 lands in STpuppeteer, every
    # non-leaked transcript is "cyto"; xeSim's existing compartment
    # placement falls through to a uniform-within-cell sampler in this
    # case (compartments are a Phase 5 extension).
    trs_df["compartment"] = "cyto"

    return trs_df


def _resolve_gene_leakage(cfg: "SimulationConfig", gene_names: list[str]) -> np.ndarray:
    """Resolve cfg.leakage_by_gene into a (n_genes,) array aligned with gene_names.

    Mirrors SpotlessSimulator._add_gene_leakage_column without requiring a
    full simulator instance. The leakage_by_gene attribute can be:

    - None: zeros for every gene.
    - float / int: same value for every gene.
    - list / ndarray: positional, length must equal n_genes.
    - dict[str -> float]: gene-name-keyed, missing keys default to 0.

    int keys (positional dict) are also supported, matching the simulator's
    behaviour.
    """
    raw = cfg.leakage_by_gene
    n = len(gene_names)
    if raw is None:
        return np.zeros(n, dtype=float)
    if isinstance(raw, (int, float)):
        return np.full(n, float(raw), dtype=float)
    if isinstance(raw, list) or hasattr(raw, "__len__") and not isinstance(raw, dict):
        arr = np.asarray(raw, dtype=float)
        if arr.shape[0] != n:
            raise ValueError(
                f"leakage_by_gene list length {arr.shape[0]} != n_genes {n}"
            )
        return arr
    if isinstance(raw, dict):
        vals = np.zeros(n, dtype=float)
        name_to_pos = {nm: i for i, nm in enumerate(gene_names)}
        for key, v in raw.items():
            if isinstance(key, int):
                vals[key] = float(v)
            elif key in name_to_pos:
                vals[name_to_pos[key]] = float(v)
            # Unknown gene names are silently dropped (xeSim's gene panel
            # validator in config_loader already would have errored if any
            # were truly outside the bundle's panel).
        return vals
    raise TypeError(f"leakage_by_gene must be None, float, list, or dict (got {type(raw).__name__})")
