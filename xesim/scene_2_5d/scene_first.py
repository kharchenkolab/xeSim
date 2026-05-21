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

    # Scope the z-fit to the scene bounds (+halo): a sub-region stitch fits
    # only its region; a whole-bundle scene spans the full FOV → fits (+ marks)
    # the whole bundle. A complete cache is used as-is.
    z_attrs = load_or_fit_cells_z(bundle_path, region_um=scene_bounds_um, halo_um=25.0)
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
    # SDF tessellation / nucleus stamping / multi-z render / molecule emission
    # now live in the shared render core (tile_render_core).

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

    z_slices = shared.z_slices_um

    # 2. Per-cell nucleus templates restricted to local cells (from the
    # precomputed shared scene), then the SHARED per-tile render core — the
    # same core compose_region_scene_25d uses (SDF tessellation → nucleus
    # stamp → multi-z DAPI + learned axial profile → focal explain_region →
    # 3D molecules). scene_first's only distinction is the assemble-once-
    # globally scene; the per-tile render must not diverge from compose.
    nuc_templates: dict = {}
    for c in local_cells:
        tmpl = shared.nuc_templates_by_id.get(c.cell_id)
        if tmpl is not None:
            nuc_templates[c.cell_id] = tmpl

    from .tile_render_core import render_25d_tile_core
    core = render_25d_tile_core(
        model, shared.bundle_path, (xmin, ymin, xmax, ymax),
        local_cells, nuc_templates, owned_local_ids,
        z_slices, psz, rng=rng, rescale_dapi=rescale_dapi,
        annotation_path=None, default_mol_per_cell=default_mol_per_cell,
        emission_backend=shared.emission_backend,
        stpuppeteer_config=shared.stpuppeteer_config,
    )
    dapi_zstack = core["dapi_zstack"]
    focal_render = core["focal_render"]
    molecules = core["molecules"]

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


__all__ = ["SharedScene25D", "precompute_scene_25d",
            "render_tile_from_shared"]
