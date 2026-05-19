"""Scene-first 2.5D: one bundle-wide setup pass, many per-tile renders.

Parallels the 2D `--scene-first` pattern. The expensive setup (bank
loads, cell filtering, unobserved-seed sampling, tilt MRF, CellRecord
building) runs ONCE over the full scene bounds, producing a
`SharedScene25D` that the per-tile renderer consumes.

Two wins over per-tile compose:
  1. ~7 min saved on parquet bank reloads × 348 tiles.
  2. The MRF tilt field is solved ONCE on all cells — neighbor coherence
     no longer breaks at 300 µm tile boundaries.

Per-tile render still does its own SDF tessellation, encoder pass on
the local real_img patch, and multi-z DAPI render. Those are
bandwidth-bound to the tile region so there's no win moving them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .sdf_tess import CellRecord


@dataclass
class SharedScene25D:
    """Bundle-wide precomputed state, passed to each per-tile render.

    Pickleable — workers receive this via `Pool` initargs and reuse it
    across all tiles they process.
    """
    all_cells: list[CellRecord]                       # CellRecord baked with tilt + z + template
    nuc_templates_by_id: dict[str, tuple]              # cell_id → (xs_rel, ys_rel)
    owned_cell_ids: set[str]                           # observed cells (not "__unobs_") = molecule emitters
    z_slices_um: np.ndarray
    pixel_size_um: float
    bundle_path: str
    scene_bounds_um: tuple[float, float, float, float]
    annotation_path: str | None
    imaged_depth_um: float
    emission_backend: str = "legacy"
    stpuppeteer_config: str | None = None


def precompute_scene_25d(
    model,
    bundle_path: str | Path,
    *,
    scene_bounds_um: tuple[float, float, float, float],
    annotation_path: str | Path | None = None,
    add_unobserved: bool = True,
    z_step_um: float = 3.0,
    imaged_depth_um: float = 33.0,
    mrf_sweeps: int = 30,
    mrf_kappa: float = 2.0,
    mrf_anchor_weight: float = 1.0,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    model_dir: str | Path | None = None,
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> SharedScene25D:
    """Bundle-wide preparation. Runs once before per-tile rendering."""
    from ..xenium import resolve_bundle
    from .z_attrs import load_or_fit_cells_z
    from .templates import TemplateBank, NucleusBank
    from .seeds import sample_unobserved_seeds
    from .tilt import initialize_tilts, mrf_gibbs_sweep

    rng = rng or np.random.default_rng(0)
    bundle = resolve_bundle(str(bundle_path))
    psz = float(bundle.pixel_size)
    xmin, ymin, xmax, ymax = scene_bounds_um
    if progress:
        print(f"[scene_first] precompute over {xmin:.0f}-{xmax:.0f}, "
              f"{ymin:.0f}-{ymax:.0f} µm")

    z_attrs = load_or_fit_cells_z(bundle_path)
    if annotation_path is None:
        cand = Path(bundle_path).parent / "annotations" / "annotation.csv.gz"
        if cand.exists():
            annotation_path = cand
    if annotation_path is not None:
        ann = pd.read_csv(annotation_path, compression="infer")
    else:
        ann = pd.DataFrame({"cell_id": [], "merged_annotation": []})
    cf = model.cell_factor_fractions(only_biology=True, include_annotation=False)
    bank = TemplateBank.from_bundle(bundle_path, cell_factor_fractions=cf,
                                       cell_annotation_df=ann, rng=rng)
    nuc_bank = NucleusBank.from_bundle(bundle_path, cell_factor_fractions=cf,
                                          cell_annotation_df=ann, rng=rng)

    # 1. Observed cells in scene bounds (no halo — bundle scope IS the scene)
    obs_mask = ((bank.df['centroid_x'] >= xmin) &
                 (bank.df['centroid_x'] < xmax) &
                 (bank.df['centroid_y'] >= ymin) &
                 (bank.df['centroid_y'] < ymax))
    obs = bank.df[obs_mask].copy()
    obs = obs.merge(z_attrs[["cell_id", "z_center_um", "z_extent_um"]],
                      on="cell_id", how="left")
    obs["z_center_um"] = obs["z_center_um"].fillna(imaged_depth_um / 2.0)
    obs["z_extent_um"] = obs["z_extent_um"].fillna(15.0)
    if progress: print(f"[scene_first] {len(obs)} observed cells")

    # Unified cell-type resolution (same cascade as the 2D explain path).
    # Every observed cell ends with a TypeResolution (source/confidence/
    # evidence), stamped onto its CellRecord below so the writer can
    # populate ground_truth/cells_synth.parquet's resolver columns.
    from .typing_fallback import resolve_observed_cell_types
    obs, _obs_type_resolutions = resolve_observed_cell_types(
        obs, model, bundle_path,
        annotation_path=annotation_path, progress=progress)

    # 2. Unobserved seeds over the full scene
    unobs_seeds = []
    if add_unobserved and len(obs) > 0:
        unobs_seeds = sample_unobserved_seeds(
            scene_bounds_um, obs, z_attrs=z_attrs, cell_annotation_df=ann,
            imaged_depth_um=imaged_depth_um, rng=rng,
        )
        if progress: print(f"[scene_first] {len(unobs_seeds)} unobserved seeds sampled")

    # 3. Build cell rows for tilt + unobs template
    obs_rows = obs[["cell_id", "cell_type", "z_extent_um",
                       "vertex_x_rel", "vertex_y_rel"]].copy()
    unobs_rows = []
    for i, seed in enumerate(unobs_seeds):
        xs_rel, ys_rel = bank.sample_template(seed.cell_type, seed.zone, rng=rng)
        unobs_rows.append({
            "cell_id": f"__unobs_{i:06d}",
            "cell_type": seed.cell_type,
            "z_extent_um": seed.z_extent,
            "vertex_x_rel": xs_rel, "vertex_y_rel": ys_rel,
        })
    if unobs_rows:
        all_rows = pd.concat([obs_rows, pd.DataFrame(unobs_rows)], ignore_index=True)
    else:
        all_rows = obs_rows

    # 4. ONE MRF over all cells in the scene (consistent tilt field across tiles)
    import time as _t
    if progress: print(f"[scene_first] initialize tilts for {len(all_rows)} cells")
    _t0 = _t.perf_counter()
    tilts = initialize_tilts(all_rows, rng=rng, model_dir=model_dir)
    if progress: print(f"  tilt init: {_t.perf_counter() - _t0:.1f}s")
    obs_pos = obs[["centroid_x", "centroid_y"]].to_numpy()
    unobs_pos = np.array([s.xy_seed for s in unobs_seeds]) if unobs_seeds else np.empty((0, 2))
    positions = np.concatenate([obs_pos, unobs_pos], axis=0) if len(unobs_pos) else obs_pos
    if progress: print(f"[scene_first] MRF {mrf_sweeps} sweeps over {len(tilts)} cells "
                          f"(vectorized)")
    _t0 = _t.perf_counter()
    tilts = mrf_gibbs_sweep(tilts, positions=positions, n_sweeps=mrf_sweeps,
                              kappa=mrf_kappa, anchor_weight=mrf_anchor_weight, rng=rng,
                              verbose=progress)
    if progress: print(f"  MRF total: {_t.perf_counter() - _t0:.1f}s")

    # 5. Build CellRecord list (one per cell, indexed 1..N globally).
    # cell_idx will be re-mapped per-tile to a local index for the renderer.
    z_centers = obs["z_center_um"].to_numpy(dtype=np.float32).tolist() + \
                [s.z_center for s in unobs_seeds]
    seeds_xy = obs[["centroid_x", "centroid_y"]].to_numpy().tolist() + \
                [list(s.xy_seed) for s in unobs_seeds]
    all_cells: list[CellRecord] = []
    for i, (tlt, row) in enumerate(zip(tilts, all_rows.itertuples(index=False))):
        # Type-resolver provenance: observed anchors get the cascade
        # result; unobserved seeds are synthetic ("__unobs_…") and stamp
        # source='ghost_prior' with confidence 0.5 (parity with 2D).
        cid = str(tlt.cell_id)
        if cid.startswith("__unobs_"):
            tr = {"cell_type": tlt.cell_type, "source": "ghost_prior",
                  "confidence": 0.5,
                  "evidence": {"unobserved_seed": True}}
        else:
            res = _obs_type_resolutions.get(cid)
            tr = res.to_dict() if res is not None else None
        all_cells.append(CellRecord(
            cell_idx=i + 1,
            cell_id=tlt.cell_id, cell_type=tlt.cell_type,
            xy_seed=(float(seeds_xy[i][0]), float(seeds_xy[i][1])),
            z_center=float(z_centers[i]),
            z_extent=float(row.z_extent_um),
            t_x=tlt.t_x, t_y=tlt.t_y, t_z=tlt.t_z,
            template_xs=np.asarray(row.vertex_x_rel, dtype=np.float32),
            template_ys=np.asarray(row.vertex_y_rel, dtype=np.float32),
            type_resolution=tr,
        ))

    # 6. Nucleus polygons per cell
    nuc_templates_by_id: dict = {}
    for c in all_cells:
        if c.cell_id.startswith("__unobs_"):
            if len(nuc_bank.df) > 0:
                xs_rel, ys_rel = nuc_bank.sample_template(c.cell_type, zone=None, rng=rng)
                nuc_templates_by_id[c.cell_id] = (xs_rel, ys_rel)
        else:
            try:
                xs_rel, ys_rel = nuc_bank.get_observed_template(c.cell_id)
                nuc_templates_by_id[c.cell_id] = (xs_rel, ys_rel)
            except KeyError:
                pass

    owned_cell_ids = {c.cell_id for c in all_cells if not c.cell_id.startswith("__unobs_")}
    z_slices = np.arange(0.0, imaged_depth_um + 1e-3, z_step_um)

    if progress:
        print(f"[scene_first] precompute done: {len(all_cells)} cells, "
              f"{len(nuc_templates_by_id)} nucleus polygons")

    return SharedScene25D(
        all_cells=all_cells,
        nuc_templates_by_id=nuc_templates_by_id,
        owned_cell_ids=owned_cell_ids,
        z_slices_um=z_slices,
        pixel_size_um=psz,
        bundle_path=str(bundle_path),
        scene_bounds_um=tuple(scene_bounds_um),
        annotation_path=str(annotation_path) if annotation_path else None,
        imaged_depth_um=imaged_depth_um,
        emission_backend=emission_backend,
        stpuppeteer_config=stpuppeteer_config,
    )


def render_tile_from_shared(
    model,
    shared: SharedScene25D,
    tile_bounds_um: tuple[float, float, float, float],
    *,
    cell_pad_um: float = 15.0,
    default_mol_per_cell: int = 20,
    rescale_dapi: bool = False,
    rng: np.random.Generator | None = None,
    progress: bool = False,
):
    """Render one tile from the shared bundle state.

    Returns a `Compose25DResult` (without `cell_label_3d` populated to
    keep IPC small; the stitcher doesn't need it).
    """
    from .compose import Compose25DResult
    from .bodies_3d import stack_z_labels, emit_molecules_3d
    from .emit_molecules_priors import emit_molecules_3d_from_priors
    from .render_multi_z import render_multi_z_dapi, _stamp_nucleus_polygons
    from ..mechanistic_scene import MechanisticScene, MechanisticCell

    rng = rng or np.random.default_rng(0)
    xmin, ymin, xmax, ymax = tile_bounds_um
    psz = shared.pixel_size_um
    pad = float(cell_pad_um)

    # 1. Filter shared.all_cells to those whose centroid is in (tile ± pad).
    # Re-index 1..N_local for this tile. Cells "owned" = centroid strictly in tile.
    local_cells: list = []
    cell_id_to_local_idx: dict[str, int] = {}
    owned_local_ids: set[str] = set()
    for c in shared.all_cells:
        cx, cy = c.xy_seed
        if not (xmin - pad <= cx < xmax + pad and ymin - pad <= cy < ymax + pad):
            continue
        local_idx = len(local_cells) + 1
        # Make a shallow copy with re-indexed cell_idx
        from .sdf_tess import CellRecord
        new_c = CellRecord(
            cell_idx=local_idx,
            cell_id=c.cell_id, cell_type=c.cell_type,
            xy_seed=c.xy_seed, z_center=c.z_center, z_extent=c.z_extent,
            t_x=c.t_x, t_y=c.t_y, t_z=c.t_z,
            template_xs=c.template_xs, template_ys=c.template_ys,
            type_resolution=c.type_resolution,
        )
        local_cells.append(new_c)
        cell_id_to_local_idx[c.cell_id] = local_idx
        if (xmin <= cx < xmax and ymin <= cy < ymax) and not c.cell_id.startswith("__unobs_"):
            owned_local_ids.add(c.cell_id)

    if progress: print(f"[scene_first] tile ({xmin:.0f},{ymin:.0f})-({xmax:.0f},{ymax:.0f}) "
                          f"µm: {len(local_cells)} cells ({len(owned_local_ids)} owned)")

    h_tile = int(round((ymax - ymin) / psz))
    w_tile = int(round((xmax - xmin) / psz))
    z_slices = shared.z_slices_um

    # 2. SDF tessellation over the tile area
    stack = stack_z_labels(
        local_cells, z_slices.tolist(),
        tile_origin_um=(xmin, ymin),
        tile_size_px=(h_tile, w_tile), pixel_size_um=psz,
    )

    # 3. Per-cell nucleus templates restricted to local cells
    nuc_templates: dict = {}
    for c in local_cells:
        tmpl = shared.nuc_templates_by_id.get(c.cell_id)
        if tmpl is not None:
            nuc_templates[c.cell_id] = tmpl

    # 3.5 Stamp nuclei across z up-front so nl_3d is in scope for the
    # priors-aware molecule emitter (which needs nucleus voxels for EDT-3D
    # compartment placement). Pre-stamping is what render_multi_z_dapi
    # would do internally — passing pre_stamped avoids duplicate CPU work.
    cells_by_idx = {c.cell_idx: c for c in local_cells}
    cl_3d = stack.astype(np.int32, copy=True)
    nl_3d = np.zeros_like(cl_3d)
    for zi in range(len(z_slices)):
        nl_3d[zi] = _stamp_nucleus_polygons(
            cells_by_idx, float(z_slices[zi]), nuc_templates,
            tile_size_px=(h_tile, w_tile), tile_origin_um=(xmin, ymin),
            pixel_size_um=psz,
            cell_label_at_z=cl_3d[zi],
        )

    # 4. Multi-z DAPI render — 2.5D-novel contribution (z-stack).
    # cell_latents=None for the batched multi-z pass (task 9.AS:
    # label-shift bug). The focal-plane 4-channel render below uses the
    # SETTLED 2D explain_region path per the strict unified-render
    # directive — no parallel focal-render code path.
    dapi_zstack = render_multi_z_dapi(
        model, stack, cells_records=local_cells,
        pixel_size_um=psz, background_mask_sigma=3.0,
        nucleus_templates=nuc_templates,
        z_slices_um=z_slices,
        tile_origin_um=(xmin, ymin),
        cell_latents=None,
        target_p99=(1.0 if rescale_dapi else None),
        pre_stamped=(cl_3d, nl_3d),
    )

    # 5. Focal 2D 4-channel render via explain_region (same path 2D
    # uses). Per feedback_unified_render_path_strict: 2.5D's novel
    # contribution is the z-stack above; the focal-plane render must
    # match 2D verbatim (real-poly nucleus_label, encoded latents from
    # the same code, same background_mask_sigma).
    from ..scene_2d.explain_region import explain_region
    er = explain_region(
        model, shared.bundle_path,
        region_bounds_um=(xmin, ymin, xmax, ymax),
        add_ghosts=False,
        add_transcript_proposed=False,
        sample_molecules=False,
        rng=rng,
        background_mask_sigma=3.0,
    )
    focal_render = er.image    # (C, H, W) float32

    # 7. Molecules from owned cells only. Use priors-aware emitter
    # (per-cell-type negbin counts + EDT-3D compartment placement +
    # real gene panel) when the model carries transcripts_priors;
    # otherwise fall back to the flat-count stub.
    owned_records = [c for c in local_cells if c.cell_id in owned_local_ids]
    if shared.emission_backend == "stpuppeteer":
        from ..emission_stpuppeteer import emit_3d as _emit_3d_stpuppeteer
        molecules = _emit_3d_stpuppeteer(
            cells_records=owned_records,
            cell_label_3d=cl_3d,
            nucleus_label_3d=nl_3d,
            z_slices_um=z_slices.tolist(),
            tile_origin_um=(xmin, ymin),
            pixel_size_um=psz,
            stpuppeteer_config=shared.stpuppeteer_config,
            rng=rng,
        )
    else:
        tx_priors = getattr(model, "transcripts_priors", None)
        if tx_priors is not None:
            molecules = emit_molecules_3d_from_priors(
                owned_records, cl_3d, nl_3d,
                z_slices_um=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                transcripts_priors=tx_priors,
                tx_rate_scale=1.0,
                rng=rng,
            )
        else:
            molecules = emit_molecules_3d(
                owned_records, stack, z_slices=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                default_count_per_cell=default_mol_per_cell, rng=rng,
            )

    # 8. cells_3d table — only the cells we OWN (centroid strictly in tile);
    # the stitcher will dedupe across tiles
    from ..cell_type_resolver import evidence_to_json
    cells_3d_cols = ["cell_id", "cell_idx", "cell_type",
                       "centroid_x", "centroid_y", "z_center_um", "z_extent_um",
                       "t_x", "t_y", "t_z", "is_unobserved",
                       "cell_type_source", "cell_type_confidence",
                       "cell_type_evidence"]
    rows = []
    for c in local_cells:
        if c.cell_id in owned_local_ids or c.cell_id.startswith("__unobs_"):
            # Owned observed OR unobserved that landed in this tile's owner region
            cx, cy = c.xy_seed
            if xmin <= cx < xmax and ymin <= cy < ymax:
                tr = c.type_resolution
                rows.append({
                    "cell_id": c.cell_id, "cell_idx": c.cell_idx,
                    "cell_type": c.cell_type,
                    "centroid_x": cx, "centroid_y": cy,
                    "z_center_um": c.z_center, "z_extent_um": c.z_extent,
                    "t_x": c.t_x, "t_y": c.t_y, "t_z": c.t_z,
                    "is_unobserved": c.cell_id.startswith("__unobs_"),
                    "cell_type_source": (tr["source"] if tr else None),
                    "cell_type_confidence": (float(tr["confidence"])
                                                if tr else 0.0),
                    "cell_type_evidence": (evidence_to_json(tr["evidence"])
                                              if tr else "{}"),
                })
    cells_3d_df = (pd.DataFrame(rows, columns=cells_3d_cols)
                     if rows else pd.DataFrame(columns=cells_3d_cols))

    return Compose25DResult(
        cell_label_3d=np.zeros((1, 1, 1), dtype=np.int32),  # not propagated
        z_slices_um=z_slices,
        dapi_zstack=dapi_zstack,
        focal_2d_render=focal_render,
        molecules=molecules,
        cells_3d=cells_3d_df,
        region_bounds_um=tuple(tile_bounds_um),
        pixel_size_um=psz,
    )


def prepare_tile_25d(
    shared: SharedScene25D,
    tile_bounds_um: tuple[float, float, float, float],
    *,
    cell_pad_um: float = 15.0,
    default_mol_per_cell: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    """CPU-only stage: filter cells, SDF tessellate, stamp nucleus
    polygons, load real_img tile, build mech scenes, emit molecules.

    Returns a dict that `render_prepared_tile_25d` consumes on the GPU.
    All work here is GIL-friendly (numpy + scipy + skimage release the
    GIL during heavy ops) so this runs well in a ThreadPoolExecutor.
    """
    from .bodies_3d import stack_z_labels, emit_molecules_3d
    from .sdf_tess import CellRecord
    from .render_multi_z import _stamp_nucleus_polygons
    from ..mechanistic_scene import MechanisticCell

    rng = rng or np.random.default_rng(0)
    xmin, ymin, xmax, ymax = tile_bounds_um
    psz = shared.pixel_size_um
    pad = float(cell_pad_um)

    # Filter shared.all_cells to (tile ± pad)
    local_cells: list = []
    owned_local_ids: set[str] = set()
    for c in shared.all_cells:
        cx, cy = c.xy_seed
        if not (xmin - pad <= cx < xmax + pad and ymin - pad <= cy < ymax + pad):
            continue
        local_idx = len(local_cells) + 1
        new_c = CellRecord(
            cell_idx=local_idx,
            cell_id=c.cell_id, cell_type=c.cell_type,
            xy_seed=c.xy_seed, z_center=c.z_center, z_extent=c.z_extent,
            t_x=c.t_x, t_y=c.t_y, t_z=c.t_z,
            template_xs=c.template_xs, template_ys=c.template_ys,
            type_resolution=c.type_resolution,
        )
        local_cells.append(new_c)
        if (xmin <= cx < xmax and ymin <= cy < ymax) and not c.cell_id.startswith("__unobs_"):
            owned_local_ids.add(c.cell_id)

    h_tile = int(round((ymax - ymin) / psz))
    w_tile = int(round((xmax - xmin) / psz))
    z_slices = shared.z_slices_um
    n_z = len(z_slices)

    # SDF tessellation
    stack = stack_z_labels(
        local_cells, z_slices.tolist(),
        tile_origin_um=(xmin, ymin),
        tile_size_px=(h_tile, w_tile), pixel_size_um=psz,
    )

    # Nucleus stamping (per-z), mutates cl_3d to include nuclei
    nuc_templates_local: dict = {}
    for c in local_cells:
        tmpl = shared.nuc_templates_by_id.get(c.cell_id)
        if tmpl is not None:
            nuc_templates_local[c.cell_id] = tmpl
    cells_by_idx = {c.cell_idx: c for c in local_cells}
    cl_3d = stack.astype(np.int32, copy=True)
    nl_3d = np.zeros_like(cl_3d)
    for zi in range(n_z):
        nl_3d[zi] = _stamp_nucleus_polygons(
            cells_by_idx, float(z_slices[zi]), nuc_templates_local,
            tile_size_px=(h_tile, w_tile), tile_origin_um=(xmin, ymin),
            pixel_size_um=psz,
            cell_label_at_z=cl_3d[zi],
        )

    # Build mech_cells_full for renders
    mech_cells_full = tuple(MechanisticCell(
        cell_id=c.cell_id, label=c.cell_idx,
        source="observed_anchor", cell_type=c.cell_type,
        nucleus_label=c.cell_idx,
        provenance={"is_ghost": False, "source_tag": "2.5d_sdf"},
    ) for c in local_cells)

    # Load real_img tile for encoder pass (CPU-side via tifffile)
    # Task 9.AV: encoder needs display-LUT-normalized [0,1] input.
    from ..scene_2d.render_tile import real_tile_image, load_model_display_lut
    display_lut = load_model_display_lut(str(model.paths.root))
    real_img = real_tile_image(
        shared.bundle_path, tile_bounds_um=(xmin, xmax, ymin, ymax),
        pixel_size_um=psz, display_lut=display_lut,
    )

    # Build obs_scene_cl for encoder (only observed cells, masked)
    mid_zi = int(n_z // 2)
    focal_cl = stack[mid_zi].astype(np.int32)
    obs_idx = {c.cell_idx for c in local_cells if not c.cell_id.startswith("__unobs_")}
    obs_cl = np.where(np.isin(focal_cl, list(obs_idx)), focal_cl, 0).astype(np.int32)
    mech_obs = tuple(MechanisticCell(
        cell_id=c.cell_id, label=c.cell_idx,
        source="observed_anchor", cell_type=c.cell_type,
        nucleus_label=c.cell_idx,
        provenance={"is_ghost": False},
    ) for c in local_cells if not c.cell_id.startswith("__unobs_"))

    # Emit molecules from owned cells (CPU-only). Priors-aware path:
    # per-type negbin counts + EDT-3D compartment placement + real gene
    # panel. Falls back to flat-count stub when transcripts_priors absent.
    owned_records = [c for c in local_cells if c.cell_id in owned_local_ids]
    if shared.emission_backend == "stpuppeteer":
        from ..emission_stpuppeteer import emit_3d as _emit_3d_stpuppeteer
        molecules = _emit_3d_stpuppeteer(
            cells_records=owned_records,
            cell_label_3d=cl_3d,
            nucleus_label_3d=nl_3d,
            z_slices_um=z_slices.tolist(),
            tile_origin_um=(xmin, ymin),
            pixel_size_um=psz,
            stpuppeteer_config=shared.stpuppeteer_config,
            rng=rng,
        )
    else:
        tx_priors = getattr(model, "transcripts_priors", None)
        if tx_priors is not None:
            from .emit_molecules_priors import emit_molecules_3d_from_priors
            molecules = emit_molecules_3d_from_priors(
                owned_records, cl_3d, nl_3d,
                z_slices_um=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                transcripts_priors=tx_priors,
                tx_rate_scale=1.0,
                rng=rng,
            )
        else:
            molecules = emit_molecules_3d(
                owned_records, stack, z_slices=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                default_count_per_cell=default_mol_per_cell, rng=rng,
            )

    # Build cells_3d_df from owned cells (centroid strictly in tile)
    from ..cell_type_resolver import evidence_to_json
    cells_3d_cols = ["cell_id", "cell_idx", "cell_type",
                       "centroid_x", "centroid_y", "z_center_um", "z_extent_um",
                       "t_x", "t_y", "t_z", "is_unobserved",
                       "cell_type_source", "cell_type_confidence",
                       "cell_type_evidence"]
    rows = []
    for c in local_cells:
        cx, cy = c.xy_seed
        if (c.cell_id in owned_local_ids or c.cell_id.startswith("__unobs_")) \
                and xmin <= cx < xmax and ymin <= cy < ymax:
            tr = c.type_resolution
            rows.append({
                "cell_id": c.cell_id, "cell_idx": c.cell_idx,
                "cell_type": c.cell_type,
                "centroid_x": cx, "centroid_y": cy,
                "z_center_um": c.z_center, "z_extent_um": c.z_extent,
                "t_x": c.t_x, "t_y": c.t_y, "t_z": c.t_z,
                "is_unobserved": c.cell_id.startswith("__unobs_"),
                "cell_type_source": (tr["source"] if tr else None),
                "cell_type_confidence": (float(tr["confidence"])
                                            if tr else 0.0),
                "cell_type_evidence": (evidence_to_json(tr["evidence"])
                                          if tr else "{}"),
            })
    cells_3d_df = (pd.DataFrame(rows, columns=cells_3d_cols)
                     if rows else pd.DataFrame(columns=cells_3d_cols))

    return {
        "tile_bounds_um": tile_bounds_um,
        "local_cells": local_cells,
        "mech_cells_full": mech_cells_full,
        "cl_3d": cl_3d, "nl_3d": nl_3d,
        "focal_cl": focal_cl,
        "obs_cl": obs_cl, "mech_obs": mech_obs,
        "real_img": real_img,
        "h_tile": h_tile, "w_tile": w_tile,
        "z_slices_um": z_slices,
        "psz": psz,
        "molecules": molecules,
        "cells_3d_df": cells_3d_df,
    }


def render_prepared_tile_25d(
    model,
    prepared: dict,
    *,
    rescale_dapi: bool = False,
):
    """GPU-only stage: encoder pass, multi-z DAPI render (z-batched),
    focal 2D render. Consumes a `prepared` dict from `prepare_tile_25d`.

    Returns a `Compose25DResult`.
    """
    from .compose import Compose25DResult
    from .render_multi_z import render_multi_z_dapi
    from ..mechanistic_scene import MechanisticScene

    h_tile = prepared["h_tile"]; w_tile = prepared["w_tile"]
    psz = prepared["psz"]
    xmin, ymin, xmax, ymax = prepared["tile_bounds_um"]
    z_slices = prepared["z_slices_um"]
    local_cells = prepared["local_cells"]
    mech_cells_full = prepared["mech_cells_full"]
    cl_3d = prepared["cl_3d"]
    nl_3d = prepared["nl_3d"]
    focal_cl = prepared["focal_cl"]
    obs_cl = prepared["obs_cl"]
    mech_obs = prepared["mech_obs"]
    real_img = prepared["real_img"]

    # Encoder pass on the tile's real focal-plane image
    encoded_latents: dict[int, np.ndarray] = {}
    if real_img is not None and real_img.size > 0 and mech_obs:
        try:
            obs_scene = MechanisticScene(
                image_shape=(h_tile, w_tile), pixel_size=psz,
                cell_label=obs_cl, nucleus_label=obs_cl,
                cells=mech_obs, scene_id="encode", provenance={},
            )
            n_ch_model = int(model.manifest.get("n_channels", 3))
            real_img_t = real_img[:n_ch_model, :obs_cl.shape[0], :obs_cl.shape[1]]
            encoded_latents = model.encode_real(obs_scene, real_img_t)
        except Exception:
            pass

    # Multi-z DAPI render (uses producer-stamped cl_3d/nl_3d).
    # cell_latents forced None for the batched multi-z pass (task 9.AS);
    # focal_render at the end uses encoded_latents (task 9.AV).
    dapi_zstack = render_multi_z_dapi(
        model, cl_3d, cells_records=local_cells,
        pixel_size_um=psz, background_mask_sigma=3.0,
        nucleus_templates=None,           # already stamped
        z_slices_um=z_slices,
        tile_origin_um=(xmin, ymin),
        cell_latents=None,
        target_p99=(1.0 if rescale_dapi else None),
        pre_stamped=(cl_3d, nl_3d),
    )

    # Focal 2D 4-channel render via explain_region (settled 2D path),
    # per feedback_unified_render_path_strict.
    from ..scene_2d.explain_region import explain_region
    er = explain_region(
        model, shared.bundle_path,
        region_bounds_um=(xmin, ymin, xmax, ymax),
        add_ghosts=False,
        add_transcript_proposed=False,
        sample_molecules=False,
        rng=rng,
        background_mask_sigma=3.0,
    )
    focal_render = er.image

    return Compose25DResult(
        cell_label_3d=np.zeros((1, 1, 1), dtype=np.int32),
        z_slices_um=z_slices,
        dapi_zstack=dapi_zstack,
        focal_2d_render=focal_render,
        molecules=prepared["molecules"],
        cells_3d=prepared["cells_3d_df"],
        region_bounds_um=tuple(prepared["tile_bounds_um"]),
        pixel_size_um=psz,
    )


__all__ = ["SharedScene25D", "precompute_scene_25d",
            "render_tile_from_shared",
            "prepare_tile_25d", "render_prepared_tile_25d"]
