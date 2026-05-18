"""Public entry points for the STpuppeteer emission backend.

``emit_2d`` and ``emit_3d`` are the call targets used by xeSim's existing
dispatch sites in ``scene_2d/tile_pipeline.py::_emit_molecules`` and
``scene_2_5d/compose.py``. They produce a DataFrame in xeSim's expected
schema (the same columns ``sample_scene_transcripts`` returns) so the
downstream bundle writer and provenance code work unchanged.

End-to-end flow per call:

1. Load the STpuppeteer config (``config_loader.load_stpuppeteer_config``).
2. Build the per-cell table (``cell_adapter.build_stpuppeteer_cell_gdf``).
3. Sample counts + leakage decisions (``counts.emit_decisions``).
4. Derive per-cell λ / max_dist for the halo (STpuppeteer's
   ``leak_dist_factor`` × per-cell radius).
5. Place transcripts in space (``placement_2d.place_2d`` or
   ``placement_3d.place_3d``).
6. Rename + add the columns xeSim expects (``true_cell_id``, ``gene``,
   ``x``, ``y``, ``true_factor``, ``qv``, ``source_cell_type``,
   ``is_ghost``). STpuppeteer-specific columns (``is_leaked``,
   ``compartment``, ``landed_in_cell_id``) are carried along for the
   ground-truth/provenance writer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from STpuppeteer.simulation import SimulationConfig
    from ..mechanistic_scene import MechanisticScene

logger = logging.getLogger(__name__)

# Constant qv for STpuppeteer-emitted transcripts. Matches the
# `synthetic_qv` default in xeSim's legacy sample_scene_transcripts.
_DEFAULT_QV = 40.0

# Default coverage fraction used to convert STpuppeteer's `max_dist`
# (a hard cutoff) into the exponential decay constant λ. 0.97 means
# "97% of the exponential mass lies within max_dist". Matches
# transcripts.sample_points_outside_polygon's default.
_DEFAULT_COVERAGE = 0.97


def emit_2d(
    scene: "MechanisticScene",
    *,
    stpuppeteer_config: str,
    rng: np.random.Generator,
    pixel_size_um: float | None = None,
    **_extra,
) -> pd.DataFrame:
    """2D STpuppeteer emission.

    Returns a DataFrame in xeSim's molecule schema (columns:
    ``true_cell_id``, ``gene``, ``x``, ``y``, ``true_factor``, ``qv``,
    ``source_cell_type``, ``is_ghost``) plus STpuppeteer provenance
    columns (``is_leaked``, ``compartment``, ``landed_in_cell_id``).

    Empty result if the scene has no cells or no configurable cell types.
    """
    if stpuppeteer_config is None:
        raise ValueError(
            "emit_2d requires --stpuppeteer-config PATH; got None. "
            "This should have been caught at CLI validation time."
        )

    from .config_loader import load_stpuppeteer_config
    from .cell_adapter import build_stpuppeteer_cell_gdf
    from .counts import emit_decisions
    from .placement_2d import place_2d

    cfg = load_stpuppeteer_config(stpuppeteer_config)

    cell_label = scene.cell_label
    nucleus_label = scene.nucleus_label
    psz = float(pixel_size_um or getattr(scene, "pixel_size", 0.25))

    # Step 1: build cell table for STpuppeteer.
    cell_gdf = build_stpuppeteer_cell_gdf(scene.cells, cell_label, cfg)
    if len(cell_gdf) == 0:
        return _empty_2d_output()

    # Step 2: count + leakage decisions. Returns long DataFrame.
    trs_df = emit_decisions(cell_gdf, cfg, rng)
    if len(trs_df) == 0:
        return _empty_2d_output()

    # Step 3: thread the cell_id ↔ integer label mapping through, since
    # the placement core needs labels (for cell_label indexing) but
    # trs_df only carries cell_id (strings).
    cell_id_to_label = {c.cell_id: int(c.label) for c in scene.cells}
    trs_df["_label"] = trs_df["cell_id"].map(cell_id_to_label).astype("int64")

    # Step 4: per-cell λ and max_dist for the halo. STpuppeteer's
    # parameterisation: max_dist = leak_dist_factor · r_eff;
    # λ = -max_dist / log(1 - coverage). r_eff is derived from each
    # cell's VISIBLE area (post overlap-resolution) so buried cells
    # automatically get a smaller halo, matching their reduced footprint.
    psz2 = psz * psz
    cell_area_um2 = cell_gdf["visible_voxels"].to_numpy(dtype=np.float64) * psz2
    r_eff = np.sqrt(cell_area_um2 / np.pi)
    max_dist_arr = float(cfg.leak_dist_factor) * r_eff
    coverage = _DEFAULT_COVERAGE
    # Avoid divide-by-zero in log(1 - coverage = 0.03); coverage=0.97 → log≈-3.51.
    lam_arr = np.where(
        max_dist_arr > 0.0,
        max_dist_arr / (-np.log(1.0 - coverage)),
        0.0,
    )
    label_arr = cell_gdf["cell_id"].map(cell_id_to_label).to_numpy(dtype=np.int64)
    leak_lam = dict(zip(label_arr.tolist(), lam_arr.tolist()))
    max_dist = dict(zip(label_arr.tolist(), max_dist_arr.tolist()))

    # Step 5: place transcripts. Tile origin is (y, x) in µm — derive
    # from scene if it knows; default (0, 0).
    tile_origin_um = _scene_tile_origin(scene)
    placed = place_2d(
        trs_df=trs_df,
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        psz_um=psz,
        tile_origin_um=tile_origin_um,
        leak_lam_per_cell=leak_lam,
        max_dist_per_cell=max_dist,
        rng=rng,
    )

    # Step 6: schema mapping for xeSim's downstream consumers.
    return _to_xesim_schema_2d(placed, cell_gdf, scene)


def emit_3d(*args, **kwargs) -> pd.DataFrame:
    """2.5D STpuppeteer emission. Phase 2 — not implemented yet."""
    raise NotImplementedError(
        "2.5D STpuppeteer emission is not yet implemented (Phase 2). "
        "See tmp/cellAdmix-integration.md §9 for the plan."
    )


def _scene_tile_origin(scene) -> tuple[float, float]:
    """Extract tile (y0, x0) µm origin from the scene if it has one.

    MechanisticScene exposes ``tile_bounds_um`` via the Scene2D wrapper —
    but the inner ``MechanisticScene`` itself doesn't carry the origin.
    Most legacy 2D code emits TILE-LOCAL coordinates that the bundle
    writer then offsets to scene coords. We follow the same convention:
    voxel index 0 → µm 0 (tile-local), and let downstream code add the
    tile origin if needed.
    """
    # If a scene gave us bounds, use the (ymin, xmin) corner; else (0,0).
    bounds = getattr(scene, "tile_bounds_um", None)
    if bounds is not None and len(bounds) >= 4:
        # Convention varies; the 2D legacy emits TILE-LOCAL coords,
        # matching origin (0, 0). Stay tile-local for parity.
        return (0.0, 0.0)
    return (0.0, 0.0)


def _to_xesim_schema_2d(
    placed: pd.DataFrame,
    cell_gdf: pd.DataFrame,
    scene,
) -> pd.DataFrame:
    """Map STpuppeteer's per-transcript output to xeSim's molecule schema.

    Matches the columns ``sample_scene_transcripts`` returns so the rest
    of xeSim's 2D pipeline (rename in _emit_molecules, bundle writer)
    works unchanged. Carries STpuppeteer-specific provenance columns
    alongside for the ground-truth writer.
    """
    # Build lookups once: source cell type (post-fallback) and ghost flag.
    celltype_lookup = dict(zip(cell_gdf["cell_id"], cell_gdf["celltype"]))
    is_ghost_lookup = {
        c.cell_id: bool(c.provenance.get("is_ghost", False))
        for c in scene.cells
    }

    n = len(placed)
    out = pd.DataFrame({
        # Legacy column names (sample_scene_transcripts compatible)
        "cell_id": placed["cell_id"].values,
        "gene": placed["feature_name"].values,
        "x": placed["x_location"].values,
        "y": placed["y_location"].values,
        # STpuppeteer has no NMF factors; sentinel value.
        "factor_label": np.full(n, -1, dtype=np.int64),
        "qv": np.full(n, _DEFAULT_QV, dtype=np.float32),
        "source_cell_type": placed["cell_id"].map(celltype_lookup).fillna("").values,
        # STpuppeteer-specific provenance (drop in bundle writer if not used)
        "is_leaked": placed["is_leaked"].astype(bool).values,
        "compartment": placed["compartment"].values,
        "landed_in_cell_id": placed["landed_in_cell_id"].values,
        "overlaps_nucleus": placed["overlaps_nucleus"].astype(np.uint8).values,
    })
    # is_ghost is annotated by xeSim's _emit_molecules wrapper *after* this
    # call (it renames cell_id → true_cell_id and adds is_ghost there);
    # we don't need to add it here.
    return out


def _empty_2d_output() -> pd.DataFrame:
    """Empty DataFrame with the xeSim molecule schema."""
    return pd.DataFrame({
        "cell_id": pd.array([], dtype="object"),
        "gene": pd.array([], dtype="object"),
        "x": pd.array([], dtype="float64"),
        "y": pd.array([], dtype="float64"),
        "factor_label": pd.array([], dtype="int64"),
        "qv": pd.array([], dtype="float32"),
        "source_cell_type": pd.array([], dtype="object"),
        "is_leaked": pd.array([], dtype="bool"),
        "compartment": pd.array([], dtype="object"),
        "landed_in_cell_id": pd.array([], dtype="object"),
        "overlaps_nucleus": pd.array([], dtype="uint8"),
    })
