"""Tile-level pipeline: compose Scene2D (anchors + ghosts) and emit molecules.

This is the main entry point for Phase 1 tile-level work.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..mechanistic_scene import MechanisticCell, MechanisticScene
from ..raster import rasterize_polygons
from ..xenium import PolygonRecord, read_boundary_polygons, resolve_bundle
from ..models import CropBox
from .ghost_cells import (
    GhostCellRecord,
    calibrate_n_ghosts,
    sample_ghost_cells,
)
from . import Scene2D


# ---------------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------------


def _load_annotation(annotation_path: str | None) -> dict[str, str]:
    """Load full bundle cell-id → type mapping from annotation.csv.gz.

    Returns {} if path is None or file unreadable.
    """
    if not annotation_path:
        return {}
    # Cache by absolute path — build_scene calls this thousands of times
    # with the same annotation_path.
    from pathlib import Path as _P
    key = str(_P(annotation_path).resolve())
    if key in _ANN_CACHE_TP:
        return _ANN_CACHE_TP[key]
    try:
        ann = pd.read_csv(annotation_path, compression="infer")
    except Exception:
        _ANN_CACHE_TP[key] = {}
        return {}
    if "cell_id" not in ann.columns or "merged_annotation" not in ann.columns:
        _ANN_CACHE_TP[key] = {}
        return {}
    out = dict(zip(ann["cell_id"].astype(str), ann["merged_annotation"].astype(str)))
    _ANN_CACHE_TP[key] = out
    return out


_ANN_CACHE_TP: dict[str, dict[str, str]] = {}


def _load_anchor_records(
    bundle_path: str,
    tile_bounds_um: tuple[float, float, float, float],
    *,
    context_buffer_um: float,
    cell_id_to_type: dict[str, str],
) -> pd.DataFrame:
    """Read anchor polygons + types from a Xenium bundle within
    tile + context_buffer.

    Returns a DataFrame indexed by anchor cell id with cell + nucleus
    polygons (vertex arrays in µm) and centroids.
    """
    bundle = resolve_bundle(bundle_path)
    xmin, xmax, ymin, ymax = tile_bounds_um
    crop = CropBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    crop_id="anchor_load")

    cell_polys = read_boundary_polygons(
        bundle.cell_boundaries_path, crop, padding_um=context_buffer_um,
    )
    nuc_polys = read_boundary_polygons(
        bundle.nucleus_boundaries_path, crop, padding_um=context_buffer_um,
    )
    nuc_by_id = {p.object_id: p for p in nuc_polys}

    rows = []
    for poly in cell_polys:
        cid = str(poly.object_id)
        bx = np.asarray(poly.x, dtype=np.float64)
        by = np.asarray(poly.y, dtype=np.float64)
        cx = float(bx.mean()); cy = float(by.mean())
        cell_type = cell_id_to_type.get(cid, "unknown")
        nuc = nuc_by_id.get(cid)
        if nuc is not None and len(nuc.x) >= 3:
            nuc_x = np.asarray(nuc.x, dtype=np.float64)
            nuc_y = np.asarray(nuc.y, dtype=np.float64)
        else:
            nuc_x, nuc_y = bx, by
        rows.append({
            "cell_id": cid,
            "cell_type": cell_type,
            "centroid_x": cx,
            "centroid_y": cy,
            "contour_x": bx,
            "contour_y": by,
            "nucleus_x": nuc_x,
            "nucleus_y": nuc_y,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def _make_polygon_record(
    object_id: str, contour_x: np.ndarray, contour_y: np.ndarray,
) -> PolygonRecord:
    return PolygonRecord(
        object_id=object_id,
        x=np.asarray(contour_x, dtype=np.float32),
        y=np.asarray(contour_y, dtype=np.float32),
    )


def _build_mech_scene(
    anchor_df: pd.DataFrame,
    ghosts: list[GhostCellRecord],
    tile_bounds_um: tuple[float, float, float, float],
    pixel_size_um: float,
    cell_id_to_type: dict[str, str],
    target_shape: tuple[int, int] | None = None,
) -> MechanisticScene:
    """Compose a MechanisticScene from anchor + ghost cells.

    Rasterize all polygons into cell_label / nucleus_label, build
    per-cell MechanisticCell records, return MechanisticScene.
    """
    xmin, xmax, ymin, ymax = tile_bounds_um
    crop = CropBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    crop_id="tile_synth")
    if target_shape is not None:
        h, w = int(target_shape[0]), int(target_shape[1])
    else:
        h = max(1, int(round((ymax - ymin) / pixel_size_um)))
        w = max(1, int(round((xmax - xmin) / pixel_size_um)))

    # Build polygon lists in label order: anchors first, then ghosts.
    cell_polygons: list[PolygonRecord] = []
    nucleus_polygons: list[PolygonRecord] = []
    is_ghost_list: list[bool] = []
    type_list: list[str] = []
    cell_id_list: list[str] = []
    # Anchor cells
    for _, row in anchor_df.iterrows():
        cell_polygons.append(_make_polygon_record(
            row["cell_id"], row["contour_x"], row["contour_y"]))
        nucleus_polygons.append(_make_polygon_record(
            row["cell_id"], row.get("nucleus_x", row["contour_x"]),
            row.get("nucleus_y", row["contour_y"])))
        is_ghost_list.append(False)
        type_list.append(row["cell_type"])
        cell_id_list.append(row["cell_id"])
    # Ghost cells
    for g in ghosts:
        cell_polygons.append(_make_polygon_record(
            g.cell_id, g.contour_x, g.contour_y))
        # Use the cell contour as a stand-in for nucleus (no separate nucleus
        # polygon for ghosts in this MVP). Shrink to ~50% for nucleus mass.
        cx_nuc = g.centroid_x + (g.contour_x - g.centroid_x) * 0.5
        cy_nuc = g.centroid_y + (g.contour_y - g.centroid_y) * 0.5
        nucleus_polygons.append(_make_polygon_record(g.cell_id, cx_nuc, cy_nuc))
        is_ghost_list.append(True)
        type_list.append(g.cell_type)
        cell_id_list.append(g.cell_id)

    cell_label, _ = rasterize_polygons(cell_polygons, crop, (h, w), pixel_size_um)
    nucleus_label, _ = rasterize_polygons(nucleus_polygons, crop, (h, w), pixel_size_um)

    # Build MechanisticCell objects. Note: rasterize_polygons starts labels at 1.
    mech_cells = []
    for idx, (cid, t, is_ghost) in enumerate(zip(cell_id_list, type_list, is_ghost_list), start=1):
        # Skip cells that didn't actually rasterize to any pixel
        if not (cell_label == idx).any():
            continue
        mech_cells.append(MechanisticCell(
            cell_id=cid,
            label=idx,
            source="observed_anchor" if not is_ghost else "synthetic",
            cell_type=t if t else None,
            provenance={"is_ghost": bool(is_ghost)},
        ))

    return MechanisticScene(
        image_shape=(h, w),
        pixel_size=float(pixel_size_um),
        cell_label=cell_label.astype(np.int32),
        nucleus_label=nucleus_label.astype(np.int32),
        cells=tuple(mech_cells),
        scene_id="tile_synth",
        provenance={
            "tile_bounds_um": list(tile_bounds_um),
            "n_anchors": int(sum(1 for g in is_ghost_list if not g)),
            "n_ghosts": int(sum(1 for g in is_ghost_list if g)),
        },
    )


# ---------------------------------------------------------------------------
# Molecule emission
# ---------------------------------------------------------------------------


def _emit_molecules(
    mech_scene: MechanisticScene,
    transcripts_priors: dict[str, Any],
    rng: np.random.Generator,
    *,
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> pd.DataFrame:
    """Emit molecules per cell. Dispatches between the legacy cellAdmix-fit
    backend (``transcripts.sample_scene_transcripts``) and the STpuppeteer
    backend (``emission_stpuppeteer.emit_2d``).

    Both backends produce the same pre-rename schema (``cell_id``, ``gene``,
    ``x``, ``y``, ``factor_label``, ``qv``, ``source_cell_type``). The xeSim
    post-processing (rename ``cell_id`` → ``true_cell_id`` / ``factor_label``
    → ``true_factor``, attach ``is_ghost`` from MechanisticCell provenance)
    is applied here, AFTER the dispatch, so it works for both paths and the
    downstream bundle writer reads a consistent schema.
    """
    if emission_backend == "stpuppeteer":
        from ..emission_stpuppeteer import emit_2d
        df = emit_2d(
            scene=mech_scene,
            stpuppeteer_config=stpuppeteer_config,
            rng=rng,
        )
    else:
        from ..transcripts import sample_scene_transcripts
        import os as _os
        # Per-cell tx-rate calibration. Precedence (highest first):
        #   1. XESIM_TX_RATE_SCALE env var (debug override)
        #   2. transcripts_priors["tx_rate_scale_default"] (baked-in per-
        #      bundle calibration from fit-priors)
        #   3. 1.0 (no scaling)
        # Why: cellAdmix-fit per-type negbin means match real per-cell tx
        # exactly, but per-tile sampling loses ~13-20% to visible_fraction
        # scaling at tile boundaries. The default lets each bundle bake in
        # its measured loss factor; explain stays single-knob from the CLI.
        env_scale = _os.environ.get("XESIM_TX_RATE_SCALE")
        if env_scale is not None:
            tx_scale = float(env_scale)
        else:
            tx_scale = float(transcripts_priors.get("tx_rate_scale_default", 1.0))
        df = sample_scene_transcripts(
            priors=transcripts_priors,
            scene=mech_scene,
            rng=rng,
            pixel_size_um=float(mech_scene.pixel_size),
            tx_rate_scale=tx_scale,
        )

    # ---- xeSim post-processing (applies to both backends) ----
    # Rename to the schema downstream code (bundle writer, provenance file)
    # expects. Then annotate is_ghost from MechanisticCell provenance.
    cell_id_to_is_ghost = {
        c.cell_id: bool(c.provenance.get("is_ghost", False))
        for c in mech_scene.cells
    }
    df = df.rename(columns={"cell_id": "true_cell_id",
                              "factor_label": "true_factor"})
    df["is_ghost"] = df["true_cell_id"].map(cell_id_to_is_ghost).fillna(False).astype(bool)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_tile(
    model,                                # XesimModel
    bundle_path: str,
    *,
    tile_center_um: tuple[float, float],
    tile_size_um: float | None = None,
    add_ghosts: bool = True,
    _suppress_deprecation: bool = False,
    noise_fraction: float = 0.25,
    ghost_count_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    context_buffer_um: float = 25.0,
    ghost_mol_count_mu: float = 10.0,
    ghost_mol_count_dispersion: float = 1.0,
    annotation_path: str | None = None,
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> Scene2D:
    """Compose + emit a tile-level synth scene.

    Parameters
    ----------
    model
        Loaded XesimModel (has priors + cell types). Uses
        ``model.transcripts_priors`` for emission, ``model.cid_to_type`` for
        anchor cell typing.
    bundle_path
        Real Xenium bundle path for anchor cells.
    tile_center_um
        (x, y) center of the tile in scene coordinates.
    tile_size_um
        Tile extent (defaults to ``model.tile_px × model.pixel_size``).
    add_ghosts
        If False, only anchor cells (no ghost emission).
    noise_fraction
        Target fraction of molecules attributable to ghost cells.
    ghost_count_scale
        Multiplier on the calibrated ghost count.
    rng
        Random generator (default seeded from system).
    context_buffer_um
        Buffer beyond tile bounds to consider for local context
        (anchor cell loading + ghost neighbor lookup).
    ghost_mol_count_mu, ghost_mol_count_dispersion
        NB(μ, φ) per-ghost molecule count target distribution.
        Informational; actual count comes from compartment-aware emission.

    Returns
    -------
    Scene2D
    """
    if not _suppress_deprecation:
        import warnings
        warnings.warn(
            "build_tile is deprecated since Phase 2.E. Use "
            "xesim.scene_2d.explain_region.explain_region(model, bundle, "
            "region_bounds_um, ...) instead — it unifies the explain and "
            "build_tile code paths (same cell selection, encoder used for "
            "real-image-backed cells, optional ghosts/molecules). build_tile "
            "is kept only for backward compatibility and will be removed in "
            "a future release.",
            DeprecationWarning, stacklevel=2,
        )
    rng = rng or np.random.default_rng(0)
    pixel_size = float(model.pixel_size)
    if tile_size_um is None:
        tile_size_um = float(model.tile_px) * pixel_size
    half = tile_size_um / 2
    cx, cy = tile_center_um
    tile_bounds_um = (cx - half, cx + half, cy - half, cy + half)

    # 1. Load anchor cells in tile + buffer
    # Prefer full annotation file (bundle-wide); fall back to model's
    # crop-restricted cid_to_type.
    cell_id_to_type = _load_annotation(annotation_path)
    if not cell_id_to_type:
        cell_id_to_type = {cid: model.type_names[idx]
                             for cid, idx in model.cid_to_type.items()}
    anchor_df = _load_anchor_records(
        bundle_path, tile_bounds_um,
        context_buffer_um=context_buffer_um,
        cell_id_to_type=cell_id_to_type,
    )
    if len(anchor_df) == 0:
        n_anchors_in_tile = 0
    else:
        n_anchors_in_tile = int(((anchor_df["centroid_x"] >= tile_bounds_um[0]) &
                                  (anchor_df["centroid_x"] <= tile_bounds_um[1]) &
                                  (anchor_df["centroid_y"] >= tile_bounds_um[2]) &
                                  (anchor_df["centroid_y"] <= tile_bounds_um[3])).sum())
    if n_anchors_in_tile == 0:
        # Empty tile — still produce a valid scene with no cells
        empty_scene = MechanisticScene(
            image_shape=(int(tile_size_um / pixel_size), int(tile_size_um / pixel_size)),
            pixel_size=pixel_size,
            cell_label=np.zeros((int(tile_size_um / pixel_size), int(tile_size_um / pixel_size)), dtype=np.int32),
            nucleus_label=np.zeros((int(tile_size_um / pixel_size), int(tile_size_um / pixel_size)), dtype=np.int32),
            cells=tuple(), scene_id="tile_synth_empty",
        )
        return Scene2D(
            mech_scene=empty_scene,
            molecules=pd.DataFrame(columns=["x", "y", "gene", "true_cell_id", "is_ghost"]),
            tile_bounds_um=tile_bounds_um,
            pixel_size=pixel_size,
            provenance={"empty": True},
        )

    # 2. Sample ghost cells (if enabled)
    ghosts: list[GhostCellRecord] = []
    if add_ghosts:
        # Estimate per-anchor expected molecule count for calibration
        # (rough estimate from priors' per_type_negbin)
        priors = model.transcripts_priors
        per_type_negbin = priors.get("per_type_count_negbin", {}) if priors else {}
        anchor_in_tile = anchor_df[
            (anchor_df["centroid_x"] >= tile_bounds_um[0]) &
            (anchor_df["centroid_x"] <= tile_bounds_um[1]) &
            (anchor_df["centroid_y"] >= tile_bounds_um[2]) &
            (anchor_df["centroid_y"] <= tile_bounds_um[3])
        ]
        est_anchor_mols = 0.0
        for _, row in anchor_in_tile.iterrows():
            nb = per_type_negbin.get(row["cell_type"])
            mu = float(nb["mean"]) if nb else 20.0
            est_anchor_mols += mu

        # Calibrate n_ghosts to hit target noise fraction
        n_ghosts_target = calibrate_n_ghosts(
            noise_fraction, int(est_anchor_mols), mol_count_mu=ghost_mol_count_mu,
        )
        n_ghosts_target = int(round(n_ghosts_target * ghost_count_scale))

        ghosts = sample_ghost_cells(
            anchor_df, tile_bounds_um,
            n_ghosts=n_ghosts_target,
            rng=rng,
            mol_count_mu=ghost_mol_count_mu,
            mol_count_dispersion=ghost_mol_count_dispersion,
        )

    # 3. Compose MechanisticScene with anchors + ghosts
    mech_scene = _build_mech_scene(
        anchor_df=anchor_df,
        ghosts=ghosts,
        tile_bounds_um=tile_bounds_um,
        pixel_size_um=pixel_size,
        cell_id_to_type=cell_id_to_type,
        target_shape=(model.tile_px, model.tile_px),
    )

    # 4. Emit molecules
    molecules = _emit_molecules(
        mech_scene=mech_scene,
        transcripts_priors=model.transcripts_priors,
        rng=rng,
        emission_backend=emission_backend,
        stpuppeteer_config=stpuppeteer_config,
    )

    return Scene2D(
        mech_scene=mech_scene,
        molecules=molecules,
        tile_bounds_um=tile_bounds_um,
        pixel_size=pixel_size,
        provenance={
            "bundle_path": str(bundle_path),
            "tile_center_um": list(tile_center_um),
            "tile_size_um": float(tile_size_um),
            "add_ghosts": bool(add_ghosts),
            "noise_fraction_target": float(noise_fraction),
            "n_anchors_in_tile": int(n_anchors_in_tile),
            "n_ghosts": len(ghosts),
            "n_ghosts_target": int(round(noise_fraction * 1)),  # diagnostic
        },
    )


__all__ = ["build_tile"]
