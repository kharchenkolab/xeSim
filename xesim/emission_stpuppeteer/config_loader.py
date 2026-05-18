"""YAML → SimulationConfig loader, with xeSim-side validation.

Thin adapter over ``STpuppeteer.simulation.SimulationConfig.from_yaml``.
The YAML parsing and dict-to-dataclass construction live in STpuppeteer
itself (see ``SimulationConfig.from_dict`` for the schema); this module
adds xeSim-specific validation hooks:

- **Cell-type coverage**: the STpuppeteer config's ``cell_type_specs``
  keys should cover the cell types that appear in the xeSim scene. When
  they don't, we substitute the most common configured type with a
  warning (see §8.3 of tmp/cellAdmix-integration.md). Hard error only if
  no bundle type is covered at all.
- **Gene panel match**: every gene referenced by ``programs[*].loading``
  must exist in the bundle's ``gene_panel.json``. Hard error on mismatch
  (would otherwise emit transcripts for genes the bundle doesn't carry).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from STpuppeteer.simulation import SimulationConfig

logger = logging.getLogger(__name__)


class STpuppeteerConfigError(ValueError):
    """Raised when the STpuppeteer config is incompatible with the bundle."""


def load_stpuppeteer_config(
    path: str | Path,
    *,
    bundle_cell_types: "pd.Series | list[str] | None" = None,
    bundle_gene_panel: list[str] | None = None,
) -> "SimulationConfig":
    """Load + validate a STpuppeteer YAML config for use with an xeSim bundle.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file. Both flat and ``simulation:``-wrapped
        layouts are accepted (see ``SimulationConfig.from_yaml``).
    bundle_cell_types : pd.Series, list of str, or None
        Cell types present in the bundle (one entry per cell). When
        provided, we check that the config's ``cell_type_specs`` covers
        them and warn (with auto-fallback) on missing types. When None,
        cell-type validation is skipped.
    bundle_gene_panel : list of str or None
        Gene names from the bundle's ``gene_panel.json``. When provided,
        we hard-error if any program references a gene name not in the
        panel. When None, gene validation is skipped.

    Returns
    -------
    SimulationConfig
        The loaded STpuppeteer config object.

    Raises
    ------
    STpuppeteerConfigError
        If a hard validation error occurs (no covered cell types,
        unknown genes referenced).
    """
    # Defer the import so xesim users who never touch this backend don't pay
    # the STpuppeteer import cost up front.
    from STpuppeteer.simulation import SimulationConfig

    cfg = SimulationConfig.from_yaml(str(path))

    if bundle_gene_panel is not None:
        _validate_gene_panel(cfg, bundle_gene_panel, source=str(path))
    if bundle_cell_types is not None:
        _validate_cell_type_coverage(cfg, bundle_cell_types, source=str(path))

    return cfg


def _validate_gene_panel(
    cfg: "SimulationConfig",
    bundle_gene_panel: list[str],
    source: str,
) -> None:
    """Hard-error if any program loading references a gene not in the panel."""
    panel = set(bundle_gene_panel)
    missing: list[tuple[str, str]] = []  # (program_name, gene_name)
    for prog in cfg.programs:
        loading = prog.loading
        if isinstance(loading, dict):
            for gene_id in loading:
                if isinstance(gene_id, str) and gene_id not in panel:
                    missing.append((prog.name, gene_id))
    if missing:
        # Truncate long lists in the error message for readability.
        preview = ", ".join(f"{p}:{g}" for p, g in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise STpuppeteerConfigError(
            f"STpuppeteer config {source!r} references {len(missing)} gene(s) "
            f"not in the bundle's gene panel: {preview}{more}. "
            f"Either remove these from program loadings or update the bundle/panel."
        )


def _validate_cell_type_coverage(
    cfg: "SimulationConfig",
    bundle_cell_types,
    source: str,
) -> None:
    """Warn about uncovered cell types; hard-error only if none are covered.

    The auto-fallback (substitution with the most common configured type)
    is applied at emission time, not here — see §8.3 of
    tmp/cellAdmix-integration.md.
    """
    # Accept either a pandas Series or a plain list of cell-type labels.
    if hasattr(bundle_cell_types, "value_counts"):
        bundle_set = set(bundle_cell_types.dropna().astype(str).unique().tolist())
    else:
        bundle_set = {str(t) for t in bundle_cell_types if t is not None}

    config_set = set(cfg.cell_type_specs.keys())
    covered = bundle_set & config_set
    if not covered:
        raise STpuppeteerConfigError(
            f"STpuppeteer config {source!r} has no cell types in common "
            f"with the bundle annotation. "
            f"Config types: {sorted(config_set)}. "
            f"Bundle types: {sorted(bundle_set)}. "
            f"Either rename cell_type_specs keys to match the bundle, or "
            f"re-annotate the bundle."
        )

    missing = bundle_set - config_set
    if missing:
        # The fallback substitution itself happens at emission time;
        # surface here so the user knows it will fire and can opt to
        # extend the config.
        logger.warning(
            "STpuppeteer config %r is missing specs for %d bundle cell type(s): %s. "
            "Emission will fall back to the most common configured type for those cells.",
            source, len(missing), sorted(missing)
        )
