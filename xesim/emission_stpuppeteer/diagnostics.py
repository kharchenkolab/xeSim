"""Diagnostics for STpuppeteer-emitted transcripts.

Surfaces what STpuppeteer's config actually produced — per-cell-type
counts, leakage rates, and marker specificity vs the configured programs.
The numbers are observation-only; nothing feeds back into the sampler.
This is the "what you configured is what you get" loop we want users to
inspect, replacing the auto-rescale knob we deliberately don't have.

Triggered by the existing ``--diagnostic [DIR]`` CLI flag (see
``xesim/diagnostics.py::resolve_diagnostic_dir``). For ``explain`` and
``re-emit-molecules`` we add ``emission_stpuppeteer.json`` under the
diagnostic dir alongside the existing diagnostic outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from STpuppeteer.simulation import SimulationConfig

logger = logging.getLogger(__name__)


# Column names produced by emit_2d / emit_3d that this module consumes.
# Provided as constants so a future schema change shows up here loudly
# instead of as silent KeyErrors in the stats computation.
_COL_GENE_2D = "gene"
_COL_GENE_3D = "gene"
_COL_CELLTYPE_2D = "source_cell_type"
_COL_CELLTYPE_25D = "true_cell_type"
_COL_LEAKED = "is_leaked"


def compute_emission_stats(
    mol_df: pd.DataFrame,
    cfg: "SimulationConfig",
) -> dict[str, Any]:
    """Per-cell-type emission stats + marker-specificity table.

    Parameters
    ----------
    mol_df : pd.DataFrame
        Output of ``emit_2d`` / ``emit_3d`` — must carry ``gene`` (or
        ``feature_name``), per-transcript cell-type, and ``is_leaked``.
    cfg : SimulationConfig
        The config that drove the emission. We extract the "own marker"
        set for each cell type from its ``program_activations`` to
        compute specificity.

    Returns
    -------
    dict
        JSON-serialisable summary. Keys: ``n_transcripts``, ``n_cell_types``,
        ``per_cell_type`` (list of per-type stats), ``cross_marker_matrix``
        (cell-type × cell-type, fraction of source-type's transcripts that
        are the column-type's markers; diagonal ≈ specificity).
    """
    if len(mol_df) == 0:
        return {
            "n_transcripts": 0,
            "n_cell_types": 0,
            "per_cell_type": [],
            "cross_marker_matrix": {},
        }

    # Locate cell-type column — emit_2d uses 'source_cell_type', emit_3d
    # uses 'true_cell_type'. Either is fine; we pick whichever exists.
    if _COL_CELLTYPE_2D in mol_df.columns:
        ct_col = _COL_CELLTYPE_2D
    elif _COL_CELLTYPE_25D in mol_df.columns:
        ct_col = _COL_CELLTYPE_25D
    else:
        raise KeyError(
            f"mol_df has neither {_COL_CELLTYPE_2D!r} nor "
            f"{_COL_CELLTYPE_25D!r}; cannot stratify by cell type."
        )

    # Build per-program-name → set of marker gene names from the config.
    # Programs with non-dict loading (ndarray) are skipped — they don't
    # carry semantic gene names.
    prog_markers: dict[str, set[str]] = {}
    for p in cfg.programs:
        if isinstance(p.loading, dict):
            prog_markers[p.name] = {g for g in p.loading.keys() if isinstance(g, str)}

    # Build per-cell-type → set of "own markers" = union of marker sets of
    # all programs the cell type activates (with non-zero activation).
    ct_markers: dict[str, set[str]] = {}
    for ct_name, spec in cfg.cell_type_specs.items():
        if isinstance(spec.program_activations, dict):
            active = [pname for pname, act in spec.program_activations.items()
                      if act > 0]
        else:
            # List-form activations were normalised to dict by config validation,
            # but be defensive.
            active = []
        own: set[str] = set()
        for pname in active:
            own |= prog_markers.get(pname, set())
        ct_markers[ct_name] = own

    per_ct: list[dict[str, Any]] = []
    cross: dict[str, dict[str, float]] = {}

    # Restrict the stats to cell types actually present in the output;
    # the matrix shape grows with N_cell_types and we don't want to
    # report zeros for types that never emitted.
    cts_present = sorted(set(mol_df[ct_col].dropna().astype(str).unique().tolist()))

    for src_ct in cts_present:
        sub = mol_df[mol_df[ct_col].astype(str) == src_ct]
        n = int(len(sub))
        n_unique_cells = int(sub["true_cell_id"].nunique()) if "true_cell_id" in sub.columns \
                              else int(sub["cell_id"].nunique()) if "cell_id" in sub.columns \
                              else 0
        leaked_frac = float(sub[_COL_LEAKED].mean()) if _COL_LEAKED in sub.columns else float("nan")

        # Own-marker fraction: how much of this type's emission is "in genre".
        own = ct_markers.get(src_ct, set())
        gene_col = _COL_GENE_3D if _COL_GENE_3D in sub.columns else "feature_name"
        own_frac = float(sub[gene_col].isin(own).mean()) if own else 0.0

        # Per-cell median count, useful for spotting clipped /
        # under-emitting types.
        if "true_cell_id" in sub.columns:
            per_cell = sub.groupby("true_cell_id").size()
        elif "cell_id" in sub.columns:
            per_cell = sub.groupby("cell_id").size()
        else:
            per_cell = pd.Series([], dtype=int)
        per_cell_arr = per_cell.to_numpy() if len(per_cell) > 0 else np.array([0])

        per_ct.append({
            "cell_type": src_ct,
            "n_transcripts": n,
            "n_cells": n_unique_cells,
            "mean_per_cell": float(per_cell_arr.mean()),
            "median_per_cell": float(np.median(per_cell_arr)),
            "p95_per_cell": float(np.percentile(per_cell_arr, 95)),
            "own_marker_fraction": own_frac,
            "leaked_fraction": leaked_frac,
        })

        # Cross-marker matrix: row = source cell type, column = "is gene
        # a marker for column type". Diagonal ≈ own-marker fraction;
        # off-diagonal = foreign-marker contamination from leakage.
        row: dict[str, float] = {}
        for col_ct in cts_present:
            col_markers = ct_markers.get(col_ct, set())
            row[col_ct] = float(sub[gene_col].isin(col_markers).mean()) if col_markers else 0.0
        cross[src_ct] = row

    return {
        "n_transcripts": int(len(mol_df)),
        "n_cell_types": int(len(cts_present)),
        "per_cell_type": per_ct,
        "cross_marker_matrix": cross,
    }


def print_emission_summary(stats: dict[str, Any]) -> None:
    """Compact stdout summary; suitable for the CLI tail-of-run log."""
    n_tot = stats.get("n_transcripts", 0)
    if n_tot == 0:
        print("[emission-diag] no transcripts produced")
        return
    print(f"[emission-diag] {n_tot:,} transcripts across {stats['n_cell_types']} cell types")
    print(f"[emission-diag] {'cell_type':<28s} {'cells':>6s} {'tx':>8s} "
          f"{'mean':>7s} {'median':>7s} {'own%':>7s} {'leak%':>7s}")
    print(f"[emission-diag] {'-' * 75}")
    for row in stats["per_cell_type"]:
        print(f"[emission-diag] {row['cell_type']:<28s} "
              f"{row['n_cells']:>6d} {row['n_transcripts']:>8d} "
              f"{row['mean_per_cell']:>7.1f} {row['median_per_cell']:>7.1f} "
              f"{row['own_marker_fraction'] * 100:>6.1f}% "
              f"{row['leaked_fraction'] * 100:>6.1f}%")


def write_emission_diagnostics(
    mol_df: pd.DataFrame,
    cfg: "SimulationConfig",
    out_dir: str | Path,
) -> Path:
    """Write a JSON diagnostic + return the path.

    Called from the CLI after emission completes when ``--diagnostic [DIR]``
    is set. The JSON contains the same content as ``compute_emission_stats``
    and is meant to be diffed across runs / configs.
    """
    stats = compute_emission_stats(mol_df, cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emission_stpuppeteer.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    return out_path
