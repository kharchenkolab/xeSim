"""Shared 2.5D per-tile render core.

``compose_region_scene_25d`` (self-contained, per-region) and
``scene_first.render_tile_from_shared`` (per-tile from a precomputed global
scene) differ only in how they ASSEMBLE the cell list, tilts, and nucleus
templates — per-region vs assemble-once-globally (the latter exists so the tilt
MRF is solved once over the whole bundle, giving a seamless tilt field across
tiles). The per-tile RENDER is identical: SDF tessellation → nucleus stamping →
multi-z DAPI render (+ learned axial profile) → focal-plane ``explain_region``
render → 3D molecule emission.

This module is that single render core, so calibration / axial-profile / render
params live in ONE place and can't diverge between the two paths (which is how
the learned axial profile and the lut_native calibration each had to be patched
twice before this extraction).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def render_25d_tile_core(
    model,
    bundle_path: str | Path,
    region_bounds_um: tuple[float, float, float, float],
    cells_records: list,              # local CellRecords, indexed 1..N
    nuc_templates: dict,              # {cell_id: (xs_rel, ys_rel)}
    owned_cell_ids: set,             # cells whose 3D molecules to emit
    z_slices_um: np.ndarray,
    pixel_size_um: float,
    *,
    rng,
    rescale_dapi: bool = True,
    annotation_path: str | None = None,
    default_mol_per_cell: int = 20,
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> dict[str, Any]:
    """Render one tile/region from an already-assembled cell list.

    Returns ``{stack, cl_3d, nl_3d, dapi_zstack, focal_render, molecules}``.
    The caller owns scene assembly (before) and the cells_3d table (after).
    """
    from .bodies_3d import stack_z_labels, emit_molecules_3d
    from .emit_molecules_priors import emit_molecules_3d_from_priors
    from .render_multi_z import render_multi_z_dapi, _stamp_nucleus_polygons
    from .axial_profile import fit_axial_dapi_profile
    from ..scene_2d.explain_region import explain_region

    xmin, ymin, xmax, ymax = region_bounds_um
    psz = float(pixel_size_um)
    z_slices = np.asarray(z_slices_um)
    h_tile = int(round((ymax - ymin) / psz))
    w_tile = int(round((xmax - xmin) / psz))

    # SDF tessellation → per-z cell labels.
    stack = stack_z_labels(
        cells_records, z_slices.tolist(),
        tile_origin_um=(xmin, ymin),
        tile_size_px=(h_tile, w_tile), pixel_size_um=psz,
    )

    # Stamp nuclei across z up-front (nl_3d needed by the priors-aware molecule
    # emitter's EDT-3D compartment placement; pre_stamped avoids duplicate work
    # inside render_multi_z_dapi).
    cells_by_idx = {c.cell_idx: c for c in cells_records}
    cl_3d = stack.astype(np.int32, copy=True)
    nl_3d = np.zeros_like(cl_3d)
    for zi in range(len(z_slices)):
        nl_3d[zi] = _stamp_nucleus_polygons(
            cells_by_idx, float(z_slices[zi]), nuc_templates,
            tile_size_px=(h_tile, w_tile), tile_origin_um=(xmin, ymin),
            pixel_size_um=psz, cell_label_at_z=cl_3d[zi],
        )

    # Multi-z DAPI render (z-stack) + learned axial DAPI profile (fit once per
    # bundle, cached). cell_latents=None per the strict unified-render directive.
    axial_profile = fit_axial_dapi_profile(bundle_path)
    dapi_zstack = render_multi_z_dapi(
        model, stack, cells_records=cells_records,
        pixel_size_um=psz, background_mask_sigma=3.0,
        nucleus_templates=nuc_templates, z_slices_um=z_slices,
        tile_origin_um=(xmin, ymin), cell_latents=None,
        target_p99=(1.0 if rescale_dapi else None),
        pre_stamped=(cl_3d, nl_3d), axial_profile=axial_profile,
    )

    # Focal-plane 4-channel render via the SETTLED 2D explain_region path
    # (feedback_unified_render_path_strict: no parallel focal-render code path).
    er = explain_region(
        model, str(bundle_path),
        region_bounds_um=(xmin, ymin, xmax, ymax),
        annotation_path=annotation_path,
        add_ghosts=False, add_transcript_proposed=False,
        sample_molecules=False, rng=rng, background_mask_sigma=3.0,
    )
    focal_render = er.image

    # 3D molecule emission — owned cells only (caller decides ownership; halo /
    # non-owned cells render their bodies but don't emit, avoiding double-count
    # across tiles). Priors-aware emitter when the model carries them.
    owned_records = [c for c in cells_records if c.cell_id in owned_cell_ids]
    if emission_backend == "stpuppeteer":
        from ..emission_stpuppeteer import emit_3d as _emit_3d_stpuppeteer
        molecules = _emit_3d_stpuppeteer(
            cells_records=owned_records,
            cell_label_3d=cl_3d, nucleus_label_3d=nl_3d,
            z_slices_um=z_slices.tolist(),
            tile_origin_um=(xmin, ymin), pixel_size_um=psz,
            stpuppeteer_config=stpuppeteer_config, rng=rng,
        )
    else:
        tx_priors = getattr(model, "transcripts_priors", None)
        if tx_priors is not None:
            molecules = emit_molecules_3d_from_priors(
                owned_records, cl_3d, nl_3d, z_slices_um=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                transcripts_priors=tx_priors, tx_rate_scale=1.0, rng=rng,
            )
        else:
            molecules = emit_molecules_3d(
                owned_records, stack, z_slices=z_slices.tolist(),
                tile_origin_um=(xmin, ymin), pixel_size_um=psz,
                default_count_per_cell=default_mol_per_cell, rng=rng,
            )

    return {"stack": stack, "cl_3d": cl_3d, "nl_3d": nl_3d,
            "dapi_zstack": dapi_zstack, "focal_render": focal_render,
            "molecules": molecules}


__all__ = ["render_25d_tile_core"]
