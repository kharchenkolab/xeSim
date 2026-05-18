"""Unified Task-1 operation: explain a region of a real bundle.

The previous architecture had two separate code paths doing essentially the
same thing:
 * `m.explain(canonical_dir, crop_ids)` — operated on pre-canonicalized
   crops from disk; used encoder; no ghosts; filtered cells.
 * `build_tile(bundle, tile_center)` — operated at arbitrary bundle regions;
   loaded all cells fresh via polygons; used random latents; added ghosts.

These differences were accidents of when each path was built, not deliberate.
This module is the single replacement: given a real bundle and a region, do
the full Task-1 "explain" operation — load all bundle cells, optionally
augment with mechanistic calls (transcript-proposed cells, ghosts), use the
encoder for cells with real-image pixels, render, and return scene + image
+ ground truth.

Task 2 (pure forward synthesis from priors without a real bundle) is out of
scope for this module.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..mechanistic_scene import MechanisticCell, MechanisticScene
from ..models import CropBox
from ..xenium import resolve_bundle
from . import Scene2D
from .load_geometry import load_geometry


@dataclass
class ExplainRegionResult:
    """Output of :func:`explain_region`."""
    scene: Scene2D                       # composed scene (mech + molecules)
    image: np.ndarray                    # (C, H, W) float32, renderer output
    real_image: np.ndarray | None        # (C, H, W) float32 at this region or None
    cell_latents: dict[int, np.ndarray]  # original_label -> latent (encoded cells only)
    n_anchors: int
    n_transcript_proposed: int
    n_ghosts: int
    region_bounds_um: tuple[float, float, float, float]


_ANN_CACHE: dict[str, dict[str, str]] = {}
_TX_CLASSIFY_CACHE: dict[int, dict[str, str]] = {}


def _transcript_classified_types(model) -> dict[str, str]:
    """Get cell_id → cell_type from the model's cellAdmix-based classifier,
    cached per-process per-model so the (somewhat expensive) classifier
    runs once per build_scene invocation, not per tile.

    Returns {} if transcripts_priors is unavailable or classification fails.
    """
    if model is None or model.transcripts_priors is None:
        return {}
    key = id(model)
    if key in _TX_CLASSIFY_CACHE:
        return _TX_CLASSIFY_CACHE[key]
    try:
        classifications = model.classify_cells_by_transcripts()
        out = {str(cid): str(cls["cell_type_transcripts"])
                for cid, cls in classifications.items()
                if cls.get("cell_type_transcripts")}
        _TX_CLASSIFY_CACHE[key] = out
        return out
    except Exception as e:
        print(f"[explain_region] cellAdmix classification unavailable: {e}")
        _TX_CLASSIFY_CACHE[key] = {}
        return {}


def _load_annotation(annotation_path: str | None) -> dict[str, str]:
    """Load `cell_id → merged_annotation` map from a CSV. Cached in-process
    by absolute path — build_scene calls explain_region thousands of times
    with the same annotation, so re-parsing 125k-row CSVs each call would
    dominate runtime.
    """
    if not annotation_path: return {}
    key = str(Path(annotation_path).resolve())
    if key in _ANN_CACHE:
        return _ANN_CACHE[key]
    try:
        ann = pd.read_csv(annotation_path, compression="infer")
    except Exception:
        _ANN_CACHE[key] = {}
        return {}
    if "cell_id" not in ann.columns or "merged_annotation" not in ann.columns:
        _ANN_CACHE[key] = {}
        return {}
    out = dict(zip(ann["cell_id"].astype(str),
                    ann["merged_annotation"].astype(str)))
    _ANN_CACHE[key] = out
    return out


def explain_region(
    model,
    bundle_path: str | Path,
    region_bounds_um: tuple[float, float, float, float],
    *,
    annotation_path: str | None = None,
    add_ghosts: bool = False,
    add_transcript_proposed: bool = False,
    stamp_transcripts: bool = False,
    sample_molecules: bool = False,
    noise_fraction: float = 0.23,
    ghost_count_scale: float = 1.0,
    context_buffer_um: float = 25.0,
    rng: np.random.Generator | None = None,
    geometry_source: str = "auto",
    background_mask_sigma: float | None = 3.0,
    tile_render_px: int | None = None,
    use_bridge_grid: bool = True,
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> ExplainRegionResult:
    """Render a real-bundle region under the mechanistic model.

    Parameters
    ----------
    model
        Loaded `XesimModel`.
    bundle_path
        Real Xenium bundle directory.
    region_bounds_um
        ``(xmin, ymin, xmax, ymax)`` in world µm.
    annotation_path
        Optional `annotation.csv.gz` (bundle-wide cell types).
    add_ghosts
        Sample synthetic ghost cells inside the region to absorb noise
        molecules. Off by default (use for forward-synth bundles).
    add_transcript_proposed
        (Future hook) augment scene with proposed cells from clusters of
        orphan transcripts. Not yet implemented; ignored for now.
    sample_molecules
        Sample molecules from priors for each cell. Off by default.
    noise_fraction
        Target ghost-molecule fraction when add_ghosts=True.
    ghost_count_scale
        Multiplier on the calibrated ghost count.
    context_buffer_um
        Load anchor cells within region + this buffer (to provide rasterization
        context near edges).
    rng
        Random generator.
    geometry_source
        "auto" (default — try zarr, fall back to polygons), "zarr", or
        "polygons".
    background_mask_sigma
        Post-render soft cell-mask sigma in pixels. Suppresses background
        leakage. Pass None to disable.

    Returns
    -------
    ExplainRegionResult
    """
    rng = rng or np.random.default_rng(0)
    bundle = resolve_bundle(str(bundle_path))
    pixel_size = float(bundle.pixel_size)
    xmin, ymin, xmax, ymax = region_bounds_um

    # Auto-discover the annotation file when no explicit path was given.
    # Without per-cell types, model.cid_to_type only covers cells from
    # training crops (~7% of the bundle); the other 93% fall back to
    # "unknown" type and per-type-conditioned channels (especially
    # membrane) render dim.
    # Lookup priority:
    #   1. explicit annotation_path arg
    #   2. <MODEL_DIR>/annotations/annotation.csv.gz   (stashed by fit-model)
    #   3. <BUNDLE>/annotations/annotation.csv.gz       (legacy fallback)
    #   4. <BUNDLE_parent>/annotations/annotation.csv.gz
    if annotation_path is None:
        from pathlib import Path as _P
        candidates = []
        if hasattr(model, "paths") and hasattr(model.paths, "root"):
            candidates.append(_P(model.paths.root) / "annotations" / "annotation.csv.gz")
        candidates.append(_P(bundle_path) / "annotations" / "annotation.csv.gz")
        candidates.append(_P(bundle_path).parent / "annotations" / "annotation.csv.gz")
        for cand in candidates:
            if cand.exists():
                annotation_path = str(cand)
                break
    h = max(1, int(round((ymax - ymin) / pixel_size)))
    w = max(1, int(round((xmax - xmin) / pixel_size)))

    # 1. Load anchor cells via the shared zarr-or-polygon path.
    crop = CropBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    crop_id="explain_region")
    cell_label, nucleus_label, cell_ids, nucleus_ids, geom_src = load_geometry(
        bundle, crop, shape=(h, w), geometry_source=geometry_source)

    # 2. Optionally augment with transcript-proposed cells (Phase 4).
    # Clusters orphan transcripts (those not inside any anchor cell) and
    # proposes new cells where 10x's segmentation likely missed them.
    n_transcript_proposed = 0
    transcript_candidates: list = []
    transcript_proposed_labels: list[int] = []
    if add_transcript_proposed:
        if model.transcripts_priors is None:
            print("[explain_region] add_transcript_proposed=True but model has "
                  "no transcripts_priors; skipping. Re-fit with --with-transcripts.")
        else:
            try:
                from ..transcript_proposer import (
                    propose_transcript_cells, CropGeometry)
                from ..scene_completion import stamp_candidates_into_label
                from ..mechanistic_scene import MechanisticScene as _MS

                # Build a temporary mech_scene for the proposer (needs
                # cell_label to identify orphan transcripts).
                tmp_mech = _MS(
                    image_shape=(h, w), pixel_size=pixel_size,
                    cell_label=cell_label.astype(np.int32),
                    nucleus_label=nucleus_label.astype(np.int32),
                    cells=(), scene_id="tmp_for_proposer", provenance={},
                )
                geom = CropGeometry.from_region(
                    region_bounds_um, pixel_size_um=pixel_size, image_size=h)
                transcript_candidates, _diag = propose_transcript_cells(
                    model, tmp_mech, geom=geom, rng=rng,
                )
                n_transcript_proposed = len(transcript_candidates)
                if transcript_candidates:
                    new_cell_label, new_nuc_label, transcript_proposed_labels = (
                        stamp_candidates_into_label(
                            cell_label, nucleus_label, transcript_candidates,
                            label_offset=int(cell_label.max() + 1000),
                        ))
                    cell_label = new_cell_label
                    nucleus_label = new_nuc_label
            except (ImportError, RuntimeError, FileNotFoundError) as e:
                # cellAdmix-core is required for transcript priors at inference
                # time. If unavailable, skip transcript-proposed cells.
                print(f"[explain_region] transcript proposer unavailable: {e}")
                print("  Install cellAdmix-core or skip add_transcript_proposed.")

    # 3. Optionally add ghost cells. Ghost sampling needs the anchor polygons
    # (to borrow shapes) and a per-anchor type-aware DataFrame — build that
    # separately from cell_boundaries.parquet (independent of mask source).
    n_ghosts_target = 0
    ghost_records: list[Any] = []
    if add_ghosts:
        from .ghost_cells import calibrate_n_ghosts, sample_ghost_cells
        from .tile_pipeline import _load_anchor_records
        cell_id_to_type_map = _load_annotation(annotation_path)
        anchor_df = _load_anchor_records(
            str(bundle_path),
            (xmin, xmax, ymin, ymax),
            context_buffer_um=context_buffer_um,
            cell_id_to_type=cell_id_to_type_map,
        )

        # Estimate per-anchor mol count for noise calibration
        priors = model.transcripts_priors
        per_type_negbin = priors.get("per_type_count_negbin", {}) if priors else {}
        # Empty tiles (no anchor cells in or near them) produce a
        # zero-column DataFrame from _load_anchor_records. Guard against
        # that so the parallel render loop doesn't crash a single tile
        # and bring down the whole 9150-tile run.
        if len(anchor_df) > 0 and "centroid_x" in anchor_df.columns:
            anchor_in_tile = anchor_df[
                (anchor_df["centroid_x"] >= xmin) &
                (anchor_df["centroid_x"] <= xmax) &
                (anchor_df["centroid_y"] >= ymin) &
                (anchor_df["centroid_y"] <= ymax)
            ]
        else:
            anchor_in_tile = anchor_df  # empty
        est_anchor_mols = 0.0
        for _, row in anchor_in_tile.iterrows():
            nb = per_type_negbin.get(row["cell_type"])
            mu = float(nb["mean"]) if nb else 20.0
            est_anchor_mols += mu
        n_ghosts_target = calibrate_n_ghosts(
            noise_fraction, int(est_anchor_mols), mol_count_mu=10.0)
        n_ghosts_target = int(round(n_ghosts_target * ghost_count_scale))
        if n_ghosts_target > 0 and len(anchor_df) > 0:
            ghost_records = sample_ghost_cells(
                anchor_df, (xmin, xmax, ymin, ymax),
                n_ghosts=n_ghosts_target, rng=rng,
            )
            # Rasterize ghosts and merge into mask space.
            from ..raster import rasterize_polygons
            from ..xenium import PolygonRecord
            n_anchor_max = int(cell_label.max())
            cell_polys = [PolygonRecord(
                object_id=g.cell_id,
                x=g.contour_x.astype(np.float32),
                y=g.contour_y.astype(np.float32)) for g in ghost_records]
            nuc_polys = []
            for g in ghost_records:
                cx_n = g.centroid_x + (g.contour_x - g.centroid_x) * 0.5
                cy_n = g.centroid_y + (g.contour_y - g.centroid_y) * 0.5
                nuc_polys.append(PolygonRecord(
                    object_id=g.cell_id,
                    x=cx_n.astype(np.float32), y=cy_n.astype(np.float32)))
            ghost_label, _ = rasterize_polygons(
                cell_polys, crop, (h, w), pixel_size)
            ghost_nuc, _ = rasterize_polygons(
                nuc_polys, crop, (h, w), pixel_size)
            ghost_label = np.where(ghost_label > 0,
                                    ghost_label + n_anchor_max, 0)
            ghost_nuc = np.where(ghost_nuc > 0,
                                  ghost_nuc + n_anchor_max, 0)
            empty = (cell_label == 0)
            cell_label = np.where(empty, ghost_label, cell_label)
            nucleus_label = np.where(empty, ghost_nuc, nucleus_label)

    # 4. Read real image at region.
    from .render_tile import real_tile_image, load_model_display_lut
    display_lut = load_model_display_lut(
        str(Path(model.paths.root)))   # model dir
    real_image = real_tile_image(
        str(bundle_path), (xmin, xmax, ymin, ymax), pixel_size,
        display_lut=display_lut, target_shape=(h, w))

    # 5. Build MechanisticScene. Use np.unique(cell_label) to walk actual
    # labels (zarr labels are large integers, not 1..N), matching the logic
    # in scene_io.scene_from_canonical_crop. Also do nucleus-cell remapping
    # so each cell's nucleus uses the cell's label.
    #
    # Cell-type assignment, in priority order (we want NO untyped cells in
    # explain output — silent skipping causes per-cell molecule undercount
    # vs real, AND the renderer collapses to a dim "unknown-type" mode):
    #   1. Explicit annotation file (cell_id → merged_annotation)
    #   2. cellAdmix transcript-based classifier (per-cell type from
    #      cosine similarity to per-type alpha centroids) — covers cells
    #      that have transcripts but no annotation row
    #   3. Model's training-time cid_to_type (subset of cells from
    #      canonical crops)
    # Future: also a stain-based classifier (encoder-latent kNN on
    # training cells) — see task #395.
    cell_id_to_type = _load_annotation(annotation_path) or {}
    # Augment with transcript-classified types for cells the annotation missed
    tx_types = _transcript_classified_types(model)
    if tx_types:
        # Only fill in missing entries — explicit annotation wins
        for cid, t in tx_types.items():
            cell_id_to_type.setdefault(cid, t)
    # Final fallback: model's cid_to_type from training data
    if hasattr(model, "cid_to_type"):
        for cid, idx in model.cid_to_type.items():
            if cid not in cell_id_to_type and 0 <= idx < len(model.type_names):
                cell_id_to_type[cid] = model.type_names[idx]
    # 4th-tier fallback: stain-based encoder-latent kNN (task 9.AA).
    # Catches anchor cells that fall through annotation, transcript
    # classifier, and training cid_to_type — typically small / tx-poor.
    # Only runs if the model has a precomputed latent bank.
    try:
        bank_path = Path(model.paths.root) / "cell_latent_bank.npz"
        nz_check = np.unique(cell_label); nz_check = nz_check[nz_check > 0]
        # Build list of untyped anchor cells
        untyped: list[tuple[str, float, float]] = []
        for i, lbl in enumerate(nz_check):
            cid = cell_ids[i] if i < len(cell_ids) else None
            if cid is None or cid in cell_id_to_type:
                continue
            # centroid from cell_label mask
            ys, xs = np.where(cell_label == int(lbl))
            if ys.size == 0: continue
            cy_um = float(ys.mean()) * float(model.pixel_size) + float(region_bounds_um[1])
            cx_um = float(xs.mean()) * float(model.pixel_size) + float(region_bounds_um[0])
            untyped.append((str(cid), cx_um, cy_um))
        if bank_path.exists() and untyped:
            from ..stain_classifier import classify_cells_by_centroid, load_bank
            bank = load_bank(bank_path)
            preds = classify_cells_by_centroid(
                model, bundle_path, untyped, bank=bank)
            for cid, t in preds.items():
                if t and t != 'unknown':
                    cell_id_to_type[cid] = t
    except Exception as e:
        print(f"[explain_region] stain-classifier fallback skipped: {e}")
    nz = np.unique(cell_label); nz = nz[nz > 0]
    nuc_remap = np.zeros_like(cell_label)
    cells: list[MechanisticCell] = []
    cell_id_arr = np.asarray(cell_ids)
    n_anchor_labels = int(len(cell_id_arr))

    # Build a fast lookup: zarr-label → cell_id. Anchor cell_ids correspond
    # to anchor labels in sorted order (preserved from the zarr loader).
    # Ghost labels were shifted by + n_anchor_max in the merge step.
    ghost_cid_by_label: dict[int, str] = {}
    ghost_type_by_label: dict[int, str | None] = {}
    if ghost_records:
        n_anchor_max = int(cell_id_arr.size and np.max(nz[:n_anchor_labels]) or 0)
        # But we shifted ghost labels by +n_anchor_max BEFORE the merge into
        # cell_label. So a ghost's label in the merged cell_label is
        # n_anchor_max + (ghost's original label, which was 1..n_ghosts).
        for gi, g in enumerate(ghost_records):
            ghost_cid_by_label[n_anchor_max + gi + 1] = g.cell_id
            ghost_type_by_label[n_anchor_max + gi + 1] = g.cell_type or None

    # Transcript-proposed cells: their labels were assigned by
    # stamp_candidates_into_label starting at label_offset = max(orig)+1000.
    tprop_meta_by_label: dict[int, tuple[str, str | None, dict]] = {}
    for lab, cand in zip(transcript_proposed_labels, transcript_candidates):
        cell_type = (cand.provenance_methods[-1]
                     if len(cand.provenance_methods) >= 3 else None)
        tprop_meta_by_label[int(lab)] = (
            f"transcripts_lab{int(lab)}", cell_type,
            {"transcripts_proposer": {
                "factor": (cand.provenance_methods[1]
                            if len(cand.provenance_methods) >= 2 else None),
                "cluster_n_molecules": float(cand.ridge_enclosure_score),
                "cy": float(cand.cy), "cx": float(cand.cx),
                "radius_px": float(cand.radius_px),
            }},
        )

    # Vectorized "dominant nucleus per cell" pre-computation. Compact
    # both label spaces first (cell labels can be sparse millions when
    # proposer/ghost label offsets are large), then 2D bincount.
    dominant_nuc_per_cell: dict[int, int] = {}
    if nz.size > 0 and nucleus_label.max() > 0:
        flat_mask = (cell_label > 0) & (nucleus_label > 0)
        if flat_mask.any():
            cl_flat = cell_label[flat_mask].astype(np.int64)
            nl_flat = nucleus_label[flat_mask].astype(np.int64)
            # Compact cell labels to 0..C-1 via searchsorted into the sorted unique set.
            uniq_cl = np.unique(cl_flat)
            uniq_nl = np.unique(nl_flat)
            cl_compact = np.searchsorted(uniq_cl, cl_flat)
            nl_compact = np.searchsorted(uniq_nl, nl_flat)
            n_cl, n_nl = len(uniq_cl), len(uniq_nl)
            encoded = cl_compact * n_nl + nl_compact
            counts = np.bincount(encoded, minlength=n_cl * n_nl)
            # Reshape to (n_cl, n_nl) and argmax across nucleus axis.
            counts2d = counts.reshape(n_cl, n_nl)
            best_nl_idx = counts2d.argmax(axis=1)
            best_nl_val = uniq_nl[best_nl_idx]
            best_count = counts2d[np.arange(n_cl), best_nl_idx]
            keep = best_count > 5
            for i in np.where(keep)[0]:
                dominant_nuc_per_cell[int(uniq_cl[i])] = int(best_nl_val[i])

    # Build nuc_remap in one vectorized pass using the dominant lookup.
    if dominant_nuc_per_cell:
        max_cl = int(nz.max())
        cell_to_dom_nuc = np.zeros(max_cl + 1, dtype=np.int64)
        for cl_lbl, dn in dominant_nuc_per_cell.items():
            if 0 <= cl_lbl <= max_cl:
                cell_to_dom_nuc[cl_lbl] = dn
        cl_clipped = np.clip(cell_label, 0, max_cl)
        expected_nuc = cell_to_dom_nuc[cl_clipped]
        nuc_remap = np.where((expected_nuc > 0) & (nucleus_label == expected_nuc),
                                cell_label, 0).astype(np.int32)

    # Walk all labels in sorted order. Anchor labels appear first (zarr
    # assigns lower numbers to anchors); ghost labels (shifted) come later.
    anchor_idx = 0
    for label_val in nz:
        label_int = int(label_val)
        # Per-cell nucleus_label_for_cell: present iff dominant nucleus
        # exists for this cell.
        nucleus_label_for_cell: int | None = (label_int
            if label_int in dominant_nuc_per_cell else None)

        # Determine if this label is an anchor, ghost, or transcript-proposed
        if label_int in ghost_cid_by_label:
            cells.append(MechanisticCell(
                cell_id=ghost_cid_by_label[label_int], label=label_int,
                source="synthetic",
                cell_type=ghost_type_by_label[label_int],
                nucleus_label=nucleus_label_for_cell,
                provenance={"is_ghost": True},
            ))
        elif label_int in tprop_meta_by_label:
            cid, ctype, prov = tprop_meta_by_label[label_int]
            cells.append(MechanisticCell(
                cell_id=cid, label=label_int,
                source="tx_inferred",
                cell_type=ctype,
                nucleus_label=nucleus_label_for_cell,
                provenance={**prov, "is_ghost": False},
            ))
        else:
            cid = (str(cell_id_arr[anchor_idx])
                   if anchor_idx < n_anchor_labels else f"cell_{label_int}")
            cells.append(MechanisticCell(
                cell_id=cid, label=label_int,
                source="observed_anchor",
                cell_type=cell_id_to_type.get(cid),
                nucleus_label=nucleus_label_for_cell,
                provenance={"is_ghost": False, "source_tag": geom_src},
            ))
            anchor_idx += 1

    mech = MechanisticScene(
        image_shape=(h, w), pixel_size=pixel_size,
        cell_label=cell_label.astype(np.int32),
        nucleus_label=nuc_remap.astype(np.int32),
        cells=tuple(cells), scene_id="explain_region",
        provenance={
            "region_bounds_um": list(region_bounds_um),
            "n_anchors": int(len(cell_id_arr)),
            "n_ghosts": int(len(ghost_records)),
            "n_transcript_proposed": int(n_transcript_proposed),
            "geometry_source": geom_src,
        },
    )

    # Optional: stamp transcript classifications into per-cell provenance
    # (cellAdmix-implied cell type + cosine-margin uncertainty). Skips
    # gracefully if cellAdmix is unavailable.
    if stamp_transcripts and model.transcripts_priors is not None:
        try:
            classifications = model.classify_cells_by_transcripts()
            mech = model.stamp_scene_with_transcripts(mech, classifications)
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            print(f"[explain_region] stamp_transcripts unavailable: {e}")

    # 6. Per-cell latents: encoder for cells with real pixels (anchors only),
    #    random for ghosts AND orphan-injected cells. Orphan cells are placed
    #    over extracellular regions by construction — the encoder has no
    #    real-image evidence to give them, so we fall back to N(0, I) like
    #    ghosts. Only `observed_anchor` cells get encoded latents.
    encoded_latents: dict[int, np.ndarray] = {}
    if real_image is not None and len(cells) > 0:
        # Build a sub-scene containing only observed anchors for the encoder.
        anchor_cells = tuple(c for c in cells if c.source == "observed_anchor")
        if anchor_cells:
            # Mask out ghost pixels for the encoder pass (set their cell_label=0)
            anchor_mask = np.zeros_like(cell_label)
            for c in anchor_cells:
                anchor_mask = np.where(cell_label == c.label, c.label, anchor_mask)
            anchor_scene = MechanisticScene(
                image_shape=(h, w), pixel_size=pixel_size,
                cell_label=anchor_mask.astype(np.int32),
                nucleus_label=mech.nucleus_label,
                cells=anchor_cells, scene_id="anchors_only",
                provenance=dict(mech.provenance or {}),
            )
            try:
                encoded_latents = model.encode_real(anchor_scene, real_image)
            except Exception as e:
                print(f"[explain_region] encoder failed, falling back to random: {e}")
                encoded_latents = {}

    # 7. Render. If tile_render_px is set, use the bridge-tile driver
    # which renders TWO overlapping grids offset by half a tile and
    # weights each tile by distance-to-its-center. Pixels are sourced
    # primarily from a tile where they are interior (no edge-of-RF
    # artifacts). Falls back to a single model.render call for the
    # whole region otherwise.
    #
    # Mask out cells that should NOT contribute to the rendered morphology
    # image but DO emit molecules (for ground-truth attribution):
    #   - tx_inferred: orphan-transcript-cluster cells (no image evidence)
    #   - synthetic (ghost): density-calibrated background noise — adding
    #     extra cell-shaped stain to the morphology would over-render the
    #     image vs real (we already have all the real anchors painting
    #     stains; ghosts represent the noise floor, which by definition
    #     shouldn't be discrete cells in the image).
    # The molecular signal is preserved because emit_molecules works off
    # the cells tuple, not the masked cell_label.
    no_render_sources = {"tx_inferred", "synthetic"}
    no_render_labels = {c.label for c in mech.cells if c.source in no_render_sources}
    if no_render_labels:
        no_render_mask = np.isin(mech.cell_label, list(no_render_labels))
        render_cell_label = np.where(no_render_mask, 0, mech.cell_label).astype(np.int32)
        render_nucleus_label = np.where(
            np.isin(mech.nucleus_label, list(no_render_labels)),
            0, mech.nucleus_label).astype(np.int32)
        mech_for_render = MechanisticScene(
            image_shape=mech.image_shape, pixel_size=mech.pixel_size,
            cell_label=render_cell_label, nucleus_label=render_nucleus_label,
            cells=tuple(c for c in mech.cells
                          if c.source not in no_render_sources),
            scene_id=mech.scene_id + "_render", provenance=mech.provenance,
        )
    else:
        mech_for_render = mech

    if tile_render_px is not None and (h > tile_render_px or w > tile_render_px):
        from .bridge_render import render_with_bridge_tiles
        image = render_with_bridge_tiles(
            model, mech_for_render, tile_size_px=int(tile_render_px),
            cell_latents=encoded_latents or None,
            background_mask_sigma=background_mask_sigma,
            use_bridge_grid=use_bridge_grid,
            seed=0,
        )
    else:
        image = model.render(
            mech_for_render, cell_latents=encoded_latents or None,
            background_mask_sigma=background_mask_sigma)

    # 8. Optionally sample molecules from priors.
    molecules = pd.DataFrame({
        "x": [], "y": [], "gene": [], "true_cell_id": [],
        "is_ghost": [], "true_factor": [], "qv": [],
    })
    if sample_molecules and (model.transcripts_priors is not None
                                or emission_backend == "stpuppeteer"):
        from .tile_pipeline import _emit_molecules
        try:
            molecules = _emit_molecules(
                mech, model.transcripts_priors, rng,
                emission_backend=emission_backend,
                stpuppeteer_config=stpuppeteer_config,
            )
        except Exception as e:
            print(f"[explain_region] molecule sampling failed: {e}")

    scene_2d = Scene2D(
        mech_scene=mech, molecules=molecules,
        tile_bounds_um=(xmin, xmax, ymin, ymax),
        pixel_size=pixel_size,
        provenance={
            "operation": "explain_region",
            "n_anchors": int(len(cell_id_arr)),
            "n_ghosts": int(len(ghost_records)),
            "n_transcript_proposed": int(n_transcript_proposed),
            "geometry_source": geom_src,
            "encoder_used": bool(encoded_latents),
            "annotation_path": str(annotation_path) if annotation_path else None,
        },
    )

    return ExplainRegionResult(
        scene=scene_2d, image=image, real_image=real_image,
        cell_latents=encoded_latents,
        n_anchors=int(len(cell_id_arr)),
        n_transcript_proposed=int(n_transcript_proposed),
        n_ghosts=int(len(ghost_records)),
        region_bounds_um=tuple(region_bounds_um),
    )


__all__ = ["explain_region", "ExplainRegionResult"]
