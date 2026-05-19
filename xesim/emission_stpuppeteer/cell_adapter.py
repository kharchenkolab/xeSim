"""Adapt xeSim's scene cells into the cell_gdf shape STpuppeteer expects.

STpuppeteer's ``sample_counts_program_model`` consumes a per-cell table
with ``cell_id``, ``celltype``, ``scale``, and ``leakage_percentage``
columns. It does not need any geometry — counts are a per-cell-type
draw. Geometry is xeSim's job at the placement step.

The adapter does three things:

1. **Compute visible mask sizes** from xeSim's rasterised ``cell_label``
   (post-overlap-resolution). Cells that got fully buried in
   rasterisation have zero visible voxels and are dropped.
2. **Resolve cell types** against the STpuppeteer config. Unknown types
   are substituted with the auto-fallback (§8.3 of the integration plan).
3. **Compute size factors** ``scale = visible_voxels / mean_visible_voxels(celltype)``.
   Voxel count is used (not µm²) so the same logic works for 2D and 3D
   arrays — the per-celltype normalisation cancels units anyway.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from STpuppeteer.simulation import SimulationConfig

logger = logging.getLogger(__name__)


def _is_ghost(cell) -> bool:
    """Return True if the cell is a synthetic ghost (don't emit for it).

    Ghost cells are tagged via ``cell.provenance["is_ghost"] = True`` by
    xeSim's 2D scene composition. 2.5D bundles have no ghosts.
    """
    prov = getattr(cell, "provenance", None) or {}
    if isinstance(prov, dict):
        return bool(prov.get("is_ghost", False))
    return False


def _cell_label_value(cell) -> int:
    """Read the integer key used to index cell_label / cell_label_3d.

    xeSim has two cell dataclasses that mean the same thing but spell
    the field differently:

    - ``MechanisticCell.label`` (2D pipeline, ``scene_2d``)
    - ``CellRecord.cell_idx``  (2.5D pipeline, ``scene_2_5d/sdf_tess``)

    Both are the integer that ``cell_label[...]`` / ``cell_label_3d[...]``
    voxels are populated with (1-indexed; 0 = background). Rather than
    rename across the codebase, we centralise the lookup here.
    """
    val = getattr(cell, "label", None)
    if val is None:
        val = getattr(cell, "cell_idx", None)
    if val is None:
        raise AttributeError(
            f"cell {cell!r} has neither .label nor .cell_idx; "
            "cannot determine its cell_label index."
        )
    return int(val)


def build_stpuppeteer_cell_gdf(
    scene_cells: Iterable,
    cell_label: np.ndarray,
    cfg: "SimulationConfig",
) -> pd.DataFrame:
    """Build the per-cell table STpuppeteer's count sampler needs.

    Parameters
    ----------
    scene_cells : iterable of MechanisticCell (or anything with .cell_id,
        .label, .cell_type attributes)
        Cells in the current scene/tile. Their ``.label`` field is the
        integer key used by ``cell_label``.
    cell_label : np.ndarray of int
        Rasterised cell label array (2D or 3D). Each voxel is either 0
        (background) or a cell's label.
    cfg : SimulationConfig
        Loaded STpuppeteer config. Used for:

        - ``cfg.cell_type_specs.keys()`` — the set of configured cell types,
          which determines which scene-cell types are recognised and which
          fall through to the auto-fallback.
        - ``cfg.leakage_by_celltype`` — per-cell-type leak probability,
          mapped onto each cell as ``leakage_percentage`` (matches the
          column STpuppeteer's ``classify_leakage`` reads).

    Returns
    -------
    pd.DataFrame
        One row per cell with positive visible-voxel count. Columns:

        - ``cell_id`` (str)
        - ``celltype`` (str) — after fallback substitution
        - ``celltype_original`` (str) — pre-substitution, for provenance
        - ``visible_voxels`` (int)
        - ``scale`` (float) — within-celltype-normalised voxel count
        - ``leakage_percentage`` (float) — per-cell leak prob from cfg
    """
    configured_types = set(cfg.cell_type_specs.keys())

    # Count visible voxels per cell label in ONE pass over cell_label —
    # cheaper than (cell_label == c).sum() in a loop, which is O(N_cells × tile_size).
    # np.bincount on the flattened label image; index = label, value = pixel count.
    flat = cell_label.ravel()
    max_label = int(flat.max()) if flat.size else 0
    voxel_count_by_label = np.bincount(flat, minlength=max_label + 1) if max_label > 0 else np.zeros(1, dtype=np.int64)

    # Materialise scene_cells once so we can iterate twice (for fallback
    # picking, then again to build rows).
    cells = list(scene_cells)

    # ---- Step 0: filter out ghost cells.
    # Ghost cells exist in the scene to add visual realism to the stain
    # rendering (out-of-plane / partial-visibility cells). They render
    # into the morphology image like any other cell, but they don't
    # represent real-cell biology, so they shouldn't drive STpuppeteer's
    # per-cell-type emission — otherwise we'd be making up transcript
    # counts for synthetic cells with no biological identity, which is
    # exactly what the STpuppeteer backend is meant to avoid.
    #
    # Concretely:
    #   - explain path: ghost cells stay in scene.cells (so the renderer
    #     paints them); cell_adapter drops them here.
    #   - re-emit path: bundle_reader loads ghosts from
    #     ghost_cell_boundaries.parquet so they appear in cell_label
    #     (leak placement still respects ghost territory); cell_adapter
    #     drops them here.
    # In both cases the morphology image is unchanged.
    n_ghosts_before = sum(
        1 for c in cells if _is_ghost(c)
    )
    cells = [c for c in cells if not _is_ghost(c)]
    if n_ghosts_before > 0:
        logger.info(
            "Skipping %d ghost cell(s) for emission "
            "(they remain in the rendered stain; STpuppeteer-driven "
            "emission is anchors-only).",
            n_ghosts_before,
        )

    # ---- Step 1: pick fallback type ----
    # Tile-local fallback: most common bundle type that's *also* in the config.
    # Bundle-wide fallback would be more stable but needs a precomputed stat
    # that's not available at this layer. Tile-local is good enough for v1;
    # surfacing in the warning lets the user adjust the config if it bites.
    raw_types = [c.cell_type for c in cells if c.cell_type]
    type_counts = pd.Series(raw_types).value_counts()
    configured_in_tile = type_counts[type_counts.index.isin(configured_types)]
    fallback = configured_in_tile.idxmax() if not configured_in_tile.empty else None
    if fallback is None:
        # No bundle cell in this tile has a configured type — emission would
        # have nothing meaningful to do. Caller (emit_2d) returns an empty
        # frame in this case rather than failing hard.
        logger.warning(
            "No scene cells have a cell_type in the STpuppeteer config "
            "(configured: %s). Emission will be skipped for this scene.",
            sorted(configured_types),
        )

    # ---- Step 2: build rows ----
    rows: list[dict] = []
    n_substituted = 0
    substituted_types: dict[str, int] = {}
    for c in cells:
        # Works for both MechanisticCell.label (2D) and CellRecord.cell_idx (2.5D).
        label = _cell_label_value(c)
        if label <= 0 or label > max_label:
            continue
        n_vox = int(voxel_count_by_label[label])
        if n_vox == 0:
            continue  # fully buried in overlap-resolution
        original_type = c.cell_type or ""
        if original_type in configured_types:
            resolved_type = original_type
        else:
            if fallback is None:
                continue  # nothing usable, drop the cell
            resolved_type = fallback
            n_substituted += 1
            substituted_types[original_type] = substituted_types.get(original_type, 0) + 1
        rows.append({
            "cell_id": c.cell_id,
            "celltype": resolved_type,
            "celltype_original": original_type,
            "visible_voxels": n_vox,
            # Leakage probability is per-celltype after STpuppeteer normalises
            # at construction time; cfg.leakage_by_celltype is always a dict.
            "leakage_percentage": float(cfg.leakage_by_celltype.get(resolved_type, 0.0)),
        })

    if n_substituted > 0:
        logger.warning(
            "Substituted %d cell(s) of unconfigured type(s) %s with '%s' "
            "(most common configured type in scene).",
            n_substituted, substituted_types, fallback,
        )

    if not rows:
        return pd.DataFrame(columns=[
            "cell_id", "celltype", "celltype_original",
            "visible_voxels", "scale", "leakage_percentage",
        ])

    df = pd.DataFrame(rows)
    # ---- Step 3: within-celltype size factor ----
    # scale = visible_voxels / mean(visible_voxels within this celltype).
    # Buried cells (zero voxels) were already dropped, so the mean is well-defined.
    df["scale"] = (
        df["visible_voxels"]
        / df.groupby("celltype")["visible_voxels"].transform("mean")
    )
    return df
