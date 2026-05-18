"""Geometry-free emission decisions: counts + per-transcript leakage.

This module produces a long-format DataFrame of per-transcript decisions
— ``(transcript_id, cell_id, gene, is_leaked, compartment)`` — with NO
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

        - ``transcript_id`` (str)
        - ``cell_id`` (str) — source cell
        - ``feature_name`` (str) — gene name (STpuppeteer's column name; the
          xeSim bundle writer renames to ``gene`` / ``feature_name`` at output)
        - ``is_leaked`` (bool)
        - ``compartment`` (str) — placeholder "cyto" in Phase 1
    """
    from STpuppeteer.simulation import (
        sample_counts_program_model,
        counts_to_transcript_df,
        classify_leakage,
        build_program_gpar_df,
    )

    if len(cell_gdf) == 0:
        # No emittable cells (e.g., tile had no cells with configured types).
        # Returning an empty frame keeps the caller's downstream code simple.
        return pd.DataFrame({
            "transcript_id": [], "cell_id": [], "feature_name": [],
            "is_leaked": [], "compartment": [],
        })

    # Step 1: per-(cell, gene) count draw via the LMC model. The sampler
    # reads `celltype` and `scale` from cell_gdf and produces an integer
    # (n_cells × n_genes) array.
    count_array = sample_counts_program_model(cell_gdf, cfg, rng)

    # Step 2: expand counts to long format.
    gene_names = cfg.resolve_gene_names()
    trs_df = counts_to_transcript_df(
        count_array, cell_gdf["cell_id"].values, gene_names
    )
    if len(trs_df) == 0:
        trs_df["is_leaked"] = pd.array([], dtype="boolean")
        trs_df["compartment"] = pd.array([], dtype="object")
        return trs_df

    # Step 3: per-transcript Bernoulli leakage via the union rule.
    # classify_leakage needs gpar_df["gene_leakage"]; build_program_gpar_df
    # produces that column from cfg.leakage_by_gene (populating zeros where
    # absent). This also gives us per-gene summaries we could attach for
    # diagnostics, but for counts we only need the leakage column.
    gpar_df = build_program_gpar_df(cfg)
    # gene_leakage column is added by SpotlessSimulator's
    # _add_gene_leakage_column; replicate that minimal step here so we
    # don't depend on simulator state. cfg.leakage_by_gene is None by
    # default (no per-gene leakage) — in that case classify_leakage
    # treats every gene as 0 and the cell-type rate dominates.
    gpar_df["gene_leakage"] = _resolve_gene_leakage(cfg, gene_names)

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
