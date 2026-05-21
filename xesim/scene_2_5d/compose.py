"""Top-level driver for the 2.5D Tilted-Template SDF scene composition.

`compose_region_scene_25d(model, bundle_path, region_bounds_um, ...)` is
the analog of `xesim.scene_2d.explain_region` for the 2.5D case. It
produces a multi-z `MechanisticScene` (or rather a list of per-z
`MechanisticScene` objects rendered into one DAPI z-stack), plus 3D
molecules and bundle-compatible 3D ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .sdf_tess import CellRecord


@dataclass
class Compose25DResult:
    cell_label_3d: np.ndarray              # (n_z, H, W) int32
    z_slices_um: np.ndarray                # (n_z,)
    dapi_zstack: np.ndarray                # (n_z, H, W) float32
    focal_2d_render: np.ndarray            # (n_channels, H, W) float32 — non-DAPI channels
    molecules: pd.DataFrame                # x_true, y_true, z_true, true_cell_id, ...
    cells_3d: pd.DataFrame                 # per-cell metadata (z_center, z_extent, tilt vec, ...)
    region_bounds_um: tuple[float, float, float, float]
    pixel_size_um: float


def compose_region_scene_25d(
    model,
    bundle_path: str | Path,
    region_bounds_um: tuple[float, float, float, float],
    *,
    annotation_path: str | Path | None = None,
    add_unobserved: bool = True,
    z_step_um: float = 3.0,
    imaged_depth_um: float = 33.0,
    mrf_sweeps: int = 30,
    mrf_kappa: float = 2.0,
    mrf_anchor_weight: float = 1.0,
    default_mol_per_cell: int = 20,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    rescale_dapi: bool = True,        # per-tile p99 rescaling; OFF in stitch mode
    cell_pad_um: float = 15.0,        # include cells whose centroid is up to this far OUTSIDE the tile (body protrudes in)
    model_dir: str | Path | None = None,  # for nucleus-priors auto-discover in initialize_tilts
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> Compose25DResult:
    """Compose a 2.5D scene over `region_bounds_um` and render multi-z DAPI."""
    from ..xenium import resolve_bundle
    from .z_attrs import load_or_fit_cells_z
    from .templates import TemplateBank, NucleusBank
    from .seeds import sample_unobserved_seeds
    from .tilt import initialize_tilts, mrf_gibbs_sweep
    # SDF tessellation / nucleus stamping / multi-z render / molecule emission
    # now live in the shared render core (tile_render_core).

    rng = rng or np.random.default_rng(0)
    bundle = resolve_bundle(str(bundle_path))
    psz = float(bundle.pixel_size)
    xmin, ymin, xmax, ymax = region_bounds_um

    if progress:
        print(f"[compose_25d] region: ({xmin:.0f},{ymin:.0f})-({xmax:.0f},{ymax:.0f}) µm")

    # 1. Load shared bundle artifacts
    if progress: print("[compose_25d] loading z attrs + template bank + cellAdmix...")
    # Region-scoped z-fit: a monolithic --tile/--region preview fits z only
    # over its region (+halo), not the whole bundle (the whole-bundle fit is a
    # 10-15 min first-touch cost). A complete cache, if present, is used as-is;
    # the scene-first/whole-bundle path (precompute) keeps the full fit.
    z_attrs = load_or_fit_cells_z(bundle_path, region_um=region_bounds_um,
                                  halo_um=cell_pad_um)
    if annotation_path is None:
        # Try default bundle-adjacent annotation location
        cand = Path(bundle_path).parent / "annotations" / "annotation.csv.gz"
        if cand.exists():
            annotation_path = cand
    if annotation_path is not None:
        ann = pd.read_csv(annotation_path, compression="infer")
    else:
        ann = pd.DataFrame({"cell_id": [], "merged_annotation": []})
    cf = model.cell_factor_fractions(only_biology=True, include_annotation=False)
    bank = TemplateBank.from_bundle(bundle_path,
                                       cell_factor_fractions=cf,
                                       cell_annotation_df=ann, rng=rng)
    nuc_bank = NucleusBank.from_bundle(bundle_path,
                                          cell_factor_fractions=cf,
                                          cell_annotation_df=ann, rng=rng)

    # 2. Filter to cells in region (observed) — include a halo of cells
    # whose centroid is just outside the tile but whose body protrudes in.
    # Eliminates the hard cut at tile boundaries when stitching.
    pad = float(cell_pad_um)
    obs_mask = ((bank.df['centroid_x'] >= xmin - pad) &
                 (bank.df['centroid_x'] < xmax + pad) &
                 (bank.df['centroid_y'] >= ymin - pad) &
                 (bank.df['centroid_y'] < ymax + pad))
    obs = bank.df[obs_mask].copy()
    obs = obs.merge(z_attrs[["cell_id", "z_center_um", "z_extent_um"]],
                      on="cell_id", how="left")
    obs["z_center_um"] = obs["z_center_um"].fillna(imaged_depth_um / 2.0)
    obs["z_extent_um"] = obs["z_extent_um"].fillna(15.0)
    # "Owned" cells have centroid strictly inside the tile (no halo).
    # Used to gate molecule emission so cells at tile borders aren't double-counted.
    obs["_owned"] = ((obs['centroid_x'] >= xmin) & (obs['centroid_x'] < xmax) &
                       (obs['centroid_y'] >= ymin) & (obs['centroid_y'] < ymax))
    n_halo = int((~obs["_owned"]).sum())
    if progress: print(f"[compose_25d] {len(obs)} observed cells in region "
                          f"({n_halo} halo cells from pad={pad}µm)")

    # Tier-2..4 retyping for cells the templates_bank left as 'unknown'.
    from .typing_fallback import retype_unknown_observed_cells
    obs = retype_unknown_observed_cells(obs, model, bundle_path, progress=progress)

    # 3. Sample unobserved cells
    unobs_seeds = []
    if add_unobserved and len(obs) > 0:
        unobs_seeds = sample_unobserved_seeds(
            region_bounds_um, obs,
            z_attrs=z_attrs, cell_annotation_df=ann,
            imaged_depth_um=imaged_depth_um,
            rng=rng,
        )
        if progress: print(f"[compose_25d] {len(unobs_seeds)} unobserved seeds sampled")

    # 4. Build cell rows for tilt initialization
    obs_rows = obs[["cell_id", "cell_type", "z_extent_um",
                       "vertex_x_rel", "vertex_y_rel"]].copy()
    # unobserved rows: sample a template per seed
    unobs_rows = []
    unobs_seed_ids = []
    for i, seed in enumerate(unobs_seeds):
        xs_rel, ys_rel = bank.sample_template(seed.cell_type, seed.zone, rng=rng)
        synth_id = f"__unobs_{i:06d}"
        unobs_seed_ids.append(synth_id)
        unobs_rows.append({
            "cell_id": synth_id,
            "cell_type": seed.cell_type,
            "z_extent_um": seed.z_extent,
            "vertex_x_rel": xs_rel,
            "vertex_y_rel": ys_rel,
        })
    if unobs_rows:
        all_rows = pd.concat([obs_rows, pd.DataFrame(unobs_rows)], ignore_index=True)
    else:
        all_rows = obs_rows

    if progress: print(f"[compose_25d] initialize tilts for {len(all_rows)} cells")
    tilts = initialize_tilts(all_rows, rng=rng, model_dir=model_dir)

    # Centroid positions: obs use their centroid; unobs use seed
    obs_pos = obs[["centroid_x", "centroid_y"]].to_numpy()
    if unobs_seeds:
        unobs_pos = np.array([s.xy_seed for s in unobs_seeds])
        positions = np.concatenate([obs_pos, unobs_pos], axis=0)
    else:
        positions = obs_pos
    if progress: print(f"[compose_25d] MRF Gibbs ({mrf_sweeps} sweeps)")
    tilts = mrf_gibbs_sweep(tilts, positions=positions, n_sweeps=mrf_sweeps,
                              kappa=mrf_kappa, anchor_weight=mrf_anchor_weight, rng=rng)

    # 5. Build CellRecord list (one per cell, indexed 1..N)
    cells_records: list[CellRecord] = []
    z_centers = obs["z_center_um"].to_numpy(dtype=np.float32).tolist() + \
                [s.z_center for s in unobs_seeds]
    seeds_xy = obs[["centroid_x", "centroid_y"]].to_numpy().tolist() + \
                [list(s.xy_seed) for s in unobs_seeds]
    for i, (tlt, row) in enumerate(zip(tilts, all_rows.itertuples(index=False))):
        cells_records.append(CellRecord(
            cell_idx=i + 1, cell_id=tlt.cell_id, cell_type=tlt.cell_type,
            xy_seed=(float(seeds_xy[i][0]), float(seeds_xy[i][1])),
            z_center=float(z_centers[i]),
            z_extent=float(row.z_extent_um),
            t_x=tlt.t_x, t_y=tlt.t_y, t_z=tlt.t_z,
            template_xs=np.asarray(row.vertex_x_rel, dtype=np.float32),
            template_ys=np.asarray(row.vertex_y_rel, dtype=np.float32),
        ))

    # 6. Per-cell nucleus templates (real polygon if available, else per-type
    # bank). Built before the shared render core, which takes them as input.
    nuc_templates: dict = {}
    for c in cells_records:
        if c.cell_id.startswith("__unobs_"):
            xs_rel, ys_rel = (nuc_bank.sample_template(c.cell_type, zone=None, rng=rng)
                              if len(nuc_bank.df) > 0 else (None, None))
            if xs_rel is not None:
                nuc_templates[c.cell_id] = (xs_rel, ys_rel)
        else:
            try:
                nuc_templates[c.cell_id] = nuc_bank.get_observed_template(c.cell_id)
            except KeyError:
                pass    # ~8% of cells have no nucleus polygon → round fallback

    # 7. Shared per-tile render core: SDF tessellation → nucleus stamp →
    # multi-z DAPI (+ learned axial profile) → focal explain_region → 3D
    # molecules. The SAME core scene_first.render_tile_from_shared uses, so
    # calibration / axial-profile / render params can't diverge between the
    # monolithic and stitched paths. Owned (for molecule emission) = observed
    # cells whose centroid is in the tile (not halo); unobserved render but
    # don't emit.
    z_slices = np.arange(0.0, imaged_depth_um + 1e-3, z_step_um)
    owned_cell_ids = set(obs.loc[obs["_owned"], "cell_id"].astype(str))
    from .tile_render_core import render_25d_tile_core
    if progress: print(f"[compose_25d] render core: {len(cells_records)} cells, "
                          f"{len(z_slices)} z slices")
    core = render_25d_tile_core(
        model, bundle_path, (xmin, ymin, xmax, ymax),
        cells_records, nuc_templates, owned_cell_ids,
        z_slices, psz, rng=rng, rescale_dapi=rescale_dapi,
        annotation_path=annotation_path, default_mol_per_cell=default_mol_per_cell,
        emission_backend=emission_backend, stpuppeteer_config=stpuppeteer_config,
    )
    stack = core["stack"]
    dapi_zstack = core["dapi_zstack"]
    focal_render = core["focal_render"]
    molecules = core["molecules"]

    # 10. cells_3d table for ground truth
    cells_3d_rows = []
    for c in cells_records:
        cells_3d_rows.append({
            "cell_id": c.cell_id, "cell_idx": c.cell_idx,
            "cell_type": c.cell_type,
            "centroid_x": c.xy_seed[0], "centroid_y": c.xy_seed[1],
            "z_center_um": c.z_center, "z_extent_um": c.z_extent,
            "t_x": c.t_x, "t_y": c.t_y, "t_z": c.t_z,
            "is_unobserved": c.cell_id.startswith("__unobs_"),
        })
    _cells_3d_cols = ["cell_id", "cell_idx", "cell_type",
                        "centroid_x", "centroid_y", "z_center_um", "z_extent_um",
                        "t_x", "t_y", "t_z", "is_unobserved"]
    cells_3d_df = (pd.DataFrame(cells_3d_rows, columns=_cells_3d_cols)
                     if cells_3d_rows else pd.DataFrame(columns=_cells_3d_cols))

    if progress:
        print(f"[compose_25d] done. n_z={len(z_slices)} observed={len(obs)} "
              f"unobserved={len(unobs_seeds)} mols={len(molecules)}")

    return Compose25DResult(
        cell_label_3d=stack,
        z_slices_um=z_slices,
        dapi_zstack=dapi_zstack,
        focal_2d_render=focal_render,
        molecules=molecules,
        cells_3d=cells_3d_df,
        region_bounds_um=tuple(region_bounds_um),
        pixel_size_um=psz,
    )


__all__ = ["Compose25DResult", "compose_region_scene_25d"]
