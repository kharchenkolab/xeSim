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


def emit_3d(
    cells_records,
    cell_label_3d: np.ndarray,
    nucleus_label_3d: np.ndarray,
    *,
    z_slices_um,
    tile_origin_um: tuple[float, float],
    pixel_size_um: float,
    stpuppeteer_config: str,
    rng: np.random.Generator,
    **_extra,
) -> pd.DataFrame:
    """2.5D STpuppeteer emission.

    Parallel to ``emit_2d``: load config → build cell table → sample
    counts + leakage → place via per-cell SDF. Differences from 2D:

    - **Input shape**: ``cell_label_3d`` / ``nucleus_label_3d`` are 3D
      ``(n_z, H, W)`` arrays. ``cells_records`` is a list of
      ``CellRecord`` (2.5D's cell dataclass) — adapter handles the
      ``.cell_idx`` vs ``.label`` field-name difference.
    - **Output schema**: matches the legacy
      ``emit_molecules_3d_from_priors`` columns
      (``x_true, y_true, z_true, true_cell_idx, true_cell_id,
      true_cell_type, source, gene, factor_label, qv, is_ghost``) so the
      2.5D bundle writer reads from it unchanged.

    Empty result if the scene has no cells or no configurable cell types.
    """
    if stpuppeteer_config is None:
        raise ValueError(
            "emit_3d requires --stpuppeteer-config PATH; got None. "
            "This should have been caught at CLI validation time."
        )

    from .config_loader import load_stpuppeteer_config
    from .cell_adapter import build_stpuppeteer_cell_gdf, _cell_label_value
    from .counts import emit_decisions
    from .placement_3d import place_3d

    cfg = load_stpuppeteer_config(stpuppeteer_config)

    # Step 1: build cell table. Same adapter as 2D — it consumes the 3D
    # cell_label via np.bincount on the flattened array (dimension-agnostic).
    cell_gdf = build_stpuppeteer_cell_gdf(cells_records, cell_label_3d, cfg)
    if len(cell_gdf) == 0:
        return _empty_3d_output()

    # Step 2: count + leakage decisions (geometry-free).
    trs_df = emit_decisions(cell_gdf, cfg, rng)
    if len(trs_df) == 0:
        return _empty_3d_output()

    # Step 3: thread cell_id ↔ integer label, plus pre-compute the
    # cell_idx and cell_type lookups we'll need for the output schema.
    cell_id_to_label = {c.cell_id: _cell_label_value(c) for c in cells_records}
    cell_id_to_type = {c.cell_id: (c.cell_type or "") for c in cells_records}
    trs_df["_label"] = trs_df["cell_id"].map(cell_id_to_label).astype("int64")

    # Step 4: per-cell λ / max_dist from cell-VISIBLE volumes. For 2.5D
    # we use the same area-based formula as 2D for simplicity (§6.4 of
    # the design doc): r_eff = sqrt(visible_area_um² / π). The
    # "visible area" used here is voxel_count × psz² (so it's actually
    # volume-divided-by-psz, not 2D footprint area). This makes r_eff
    # proportionally larger for thicker cells — a defensible
    # approximation; revisit if calibration evidence prefers a true
    # 2D-projection footprint or a volume-derived r_eff.
    psz2 = pixel_size_um * pixel_size_um
    cell_area_um2 = cell_gdf["visible_voxels"].to_numpy(dtype=np.float64) * psz2
    r_eff = np.sqrt(cell_area_um2 / np.pi)
    max_dist_arr = float(cfg.leak_dist_factor) * r_eff
    coverage = _DEFAULT_COVERAGE
    lam_arr = np.where(
        max_dist_arr > 0.0,
        max_dist_arr / (-np.log(1.0 - coverage)),
        0.0,
    )
    label_arr = cell_gdf["cell_id"].map(cell_id_to_label).to_numpy(dtype=np.int64)
    leak_lam = dict(zip(label_arr.tolist(), lam_arr.tolist()))
    max_dist = dict(zip(label_arr.tolist(), max_dist_arr.tolist()))

    # Step 5: place transcripts in 3D.
    placed = place_3d(
        trs_df=trs_df,
        cell_label_3d=cell_label_3d,
        nucleus_label_3d=nucleus_label_3d,
        psz_um=pixel_size_um,
        z_slices_um=z_slices_um,
        tile_origin_um=tile_origin_um,
        leak_lam_per_cell=leak_lam,
        max_dist_per_cell=max_dist,
        rng=rng,
    )

    # Step 6: schema mapping for the 2.5D bundle writer.
    return _to_xesim_schema_3d(placed, cell_gdf, cell_id_to_type, cell_id_to_label)


def _to_xesim_schema_3d(
    placed: pd.DataFrame,
    cell_gdf: pd.DataFrame,
    cell_id_to_type: dict,
    cell_id_to_label: dict,
) -> pd.DataFrame:
    """Map placed transcripts to the 2.5D writer's expected columns.

    Output column order matches ``emit_molecules_3d_from_priors._OUT_COLS``
    so ``bundle_writer_25d`` reads from it unchanged.
    """
    n = len(placed)
    # source: STpuppeteer emits "body" transcripts only (no ghost source
    # in v1, no ambient background). is_ghost is False for the same reason.
    # If/when ghost integration lands (Phase 6) this will need updating.
    return pd.DataFrame({
        "x_true": placed["x_location"].astype(np.float32).values,
        "y_true": placed["y_location"].astype(np.float32).values,
        "z_true": placed["z_location"].astype(np.float32).values,
        "true_cell_idx": placed["cell_id"].map(cell_id_to_label)
                                              .astype(np.int64).values,
        "true_cell_id": placed["cell_id"].astype(object).values,
        "true_cell_type": placed["cell_id"].map(cell_id_to_type)
                                              .fillna("").astype(object).values,
        "source": np.full(n, "body", dtype=object),
        "gene": placed["feature_name"].astype(object).values,
        "factor_label": np.full(n, -1, dtype=np.int64),
        "qv": np.full(n, _DEFAULT_QV, dtype=np.float32),
        "is_ghost": np.zeros(n, dtype=bool),
        # STpuppeteer provenance columns alongside the legacy schema.
        # The bundle writer's transcripts.parquet will drop these; the
        # ground-truth provenance file may keep them when wired in.
        "is_leaked": placed["is_leaked"].astype(bool).values,
        "compartment": placed["compartment"].astype(object).values,
        "landed_in_cell_id": placed["landed_in_cell_id"].astype(object).values,
        "overlaps_nucleus": placed["overlaps_nucleus"].astype(np.uint8).values,
    })


def _empty_3d_output() -> pd.DataFrame:
    """Empty DataFrame with the 2.5D output schema."""
    return pd.DataFrame({
        "x_true": pd.array([], dtype="float32"),
        "y_true": pd.array([], dtype="float32"),
        "z_true": pd.array([], dtype="float32"),
        "true_cell_idx": pd.array([], dtype="int64"),
        "true_cell_id": pd.array([], dtype="object"),
        "true_cell_type": pd.array([], dtype="object"),
        "source": pd.array([], dtype="object"),
        "gene": pd.array([], dtype="object"),
        "factor_label": pd.array([], dtype="int64"),
        "qv": pd.array([], dtype="float32"),
        "is_ghost": pd.array([], dtype="bool"),
        "is_leaked": pd.array([], dtype="bool"),
        "compartment": pd.array([], dtype="object"),
        "landed_in_cell_id": pd.array([], dtype="object"),
        "overlaps_nucleus": pd.array([], dtype="uint8"),
    })


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
