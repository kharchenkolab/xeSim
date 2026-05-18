from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mechanistic_schema import MECHANISTIC_RENDER_TYPE
from .mechanistic_scene import MechanisticParams, MechanisticScene, validate_mechanistic_scene
from .raster import binary_dilation
from .schema import MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class MechanisticRenderOutput:
    images: np.ndarray
    components: dict[str, np.ndarray]
    state: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "type": MECHANISTIC_RENDER_TYPE,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "image_shape": [int(self.images.shape[1]), int(self.images.shape[2])],
            "image_channels": ["dapi", "membrane", "polya"],
            "components": {
                name: {
                    "shape": [int(x) for x in value.shape],
                    "dtype": str(value.dtype),
                    "min": float(np.min(value)) if value.size else 0.0,
                    "max": float(np.max(value)) if value.size else 0.0,
                    "mean": float(np.mean(value)) if value.size else 0.0,
                }
                for name, value in self.components.items()
            },
            "state": self.state,
        }


def render_mechanistic(
    scene: MechanisticScene,
    params: MechanisticParams | None = None,
    seed: int | None = None,
    polya_radial_spectrum: list[float] | None = None,
    transcripts: dict[str, np.ndarray] | None = None,
) -> MechanisticRenderOutput:
    """Render a Plan3 mechanistic DAPI/membrane crop from explicit scene state.

    When ``transcripts`` is provided (dict with keys x_um, y_um from
    ``sample_cell_transcripts``), the polyA cytoplasm field is replaced
    by rasterized point density: each transcript contributes a delta at
    its position scaled by ``polya_per_transcript``. This produces real
    Poisson shot-noise statistics inside cells (closing the B7 std_polya
    gap) and ties polyA exactly to the explicit transcript model
    (Layer T).
    """

    validate_mechanistic_scene(scene)
    p = params or MechanisticParams()
    rng = np.random.default_rng(p.seed if seed is None else seed)
    h, w = scene.image_shape
    cell_label = scene.cell_label.astype(np.int32, copy=False)
    nucleus_label = scene.nucleus_label.astype(np.int32, copy=False)
    cell_mask = cell_label > 0
    nucleus_mask = nucleus_label > 0

    # Build per-pixel parameter fields from per-cell physics; pixels not in any
    # cell fall back to the global MechanisticParams defaults.
    width_field = _per_pixel_param_field(
        scene, cell_label, "membrane_width_px", default=float(p.membrane_width_px),
    )
    dropout_keep_field = _per_pixel_param_field(
        scene, cell_label, "membrane_dropout_keep_prob", default=float(p.membrane_dropout_keep_prob),
    )
    interior_fraction_field = _per_pixel_param_field(
        scene, cell_label, "membrane_interior_fraction", default=float(p.membrane_interior_fraction),
    )
    dapi_texture_field = _per_pixel_param_field(
        scene, cell_label, "dapi_texture_scale", default=float(p.dapi_texture_scale),
    )

    # DAPI nucleus: per-cell independent multi-scale texture field (v35 P6).
    # Real DAPI nuclei show heterochromatin (bright spots) + low-frequency
    # variation (open/closed chromatin domains). A single smooth-noise
    # scale is too uniform (synth intra-CV ~0.15 vs real 0.24). Sample
    # both a coarse field (heterochromatin domains) and a fine field
    # (heterochromatin spots) and combine. Stays cell-source-compliant.
    if bool(getattr(p, "dapi_per_cell_texture", True)):
        bg_field = _smooth_noise(rng, (h, w), sigma=max(1.0, p.dapi_blur_sigma_px * 2.0))
        dapi_texture_field = bg_field.copy()
        for cell in scene.cells:
            if cell.nucleus_label is None:
                continue
            nmask = nucleus_label == int(cell.nucleus_label)
            if not np.any(nmask):
                continue
            coarse = _smooth_noise(rng, (h, w), sigma=max(1.5, p.dapi_blur_sigma_px * 2.5))
            fine = _smooth_noise(rng, (h, w), sigma=max(0.6, p.dapi_blur_sigma_px * 0.7))
            cell_tex = (0.55 * coarse + 0.45 * fine).astype(np.float32)
            dapi_texture_field = np.where(nmask, cell_tex, dapi_texture_field)
    else:
        dapi_texture_field = _smooth_noise(rng, (h, w), sigma=max(1.0, p.dapi_blur_sigma_px * 2.0))
    dapi_amplitude_lut, dapi_texture_lut = _build_dapi_luts(scene, nucleus_label, p)
    nuc_label_clamped = nucleus_label.astype(np.int64, copy=False)
    nuc_label_clamped = np.where(nuc_label_clamped < dapi_amplitude_lut.shape[0], nuc_label_clamped, 0)
    dapi_amp_field = dapi_amplitude_lut[nuc_label_clamped]
    dapi_tex_field = dapi_texture_lut[nuc_label_clamped]
    dapi_nucleus = (dapi_amp_field * (1.0 + dapi_tex_field * dapi_texture_field)).astype(np.float32)
    dapi_nucleus[~nucleus_mask] = 0.0  # only render nuclei
    # Plan3 v35 P6: per-nucleus nucleolus dark spot. Real DAPI nuclei show
    # 1-2 large dark spots (nucleoli, where rRNA is transcribed and DAPI
    # binding is reduced). Synth rendering looked over-uniform without
    # this. Stamp 1 random dark spot per nucleus, radius ~ 15-25% of
    # nucleus radius, dimming factor 0.4-0.6. Cell-source compliant —
    # the spot is a property of the cell's nucleus.
    # Plan3 v35 P7: nuclear envelope brightness. Real nuclei show ~20-40%
    # brighter DAPI signal at the nuclear envelope (heterochromatin
    # adheres to the nuclear membrane). Per-nucleus, multiply DAPI by a
    # radial-distance ramp that boosts edge pixels relative to interior.
    # Cell-source compliant — derived from nucleus geometry.
    envelope_boost = float(getattr(p, "dapi_envelope_boost", 0.0))
    if envelope_boost > 0.0 and nucleus_mask.any():
        # Distance to nucleus boundary (positive inside)
        nuc_inside_dist = np.zeros((h, w), dtype=np.float32)
        # Iterate per-nucleus to compute per-nucleus distance-to-boundary
        for cell in scene.cells:
            if cell.nucleus_label is None:
                continue
            nlbl = int(cell.nucleus_label)
            nmask = nucleus_label == nlbl
            if not np.any(nmask):
                continue
            from scipy.ndimage import distance_transform_edt as _edt
            d_in = _edt(nmask).astype(np.float32)
            d_max = float(d_in.max())
            if d_max < 1.0:
                continue
            # Boost = 1 + envelope_boost * exp(-d/(d_max*0.3)) — falls off into
            # nucleus interior
            ramp = np.exp(-(d_in[nmask] - 0.5) / (d_max * 0.3 + 0.5))
            dapi_nucleus[nmask] = dapi_nucleus[nmask] * (1.0 + envelope_boost * ramp.astype(np.float32))
    nucleolus_amp = float(getattr(p, "dapi_nucleolus_dim_factor", 0.0))
    if nucleolus_amp > 0.0:
        for cell in scene.cells:
            if cell.nucleus_label is None:
                continue
            nlbl = int(cell.nucleus_label)
            nmask = nucleus_label == nlbl
            n_pix = int(nmask.sum())
            if n_pix < 50:  # too small for a nucleolus
                continue
            ys, xs = np.where(nmask)
            cy = int(ys.mean()); cx = int(xs.mean())
            # Nucleolus radius: 15-25% of nucleus equivalent radius
            nuc_radius = float(np.sqrt(n_pix / np.pi))
            r = float(rng.uniform(0.15, 0.25)) * nuc_radius
            # Random offset from centroid (within nucleus)
            offset_y = int(rng.normal(0.0, nuc_radius * 0.2))
            offset_x = int(rng.normal(0.0, nuc_radius * 0.2))
            spot_cy = cy + offset_y; spot_cx = cx + offset_x
            yy, xx = np.indices((h, w))
            dist2 = (yy - spot_cy) ** 2 + (xx - spot_cx) ** 2
            spot = np.exp(-dist2 / (2.0 * max(r, 1.0) ** 2))
            spot_mask = spot > 0.1
            spot_within = spot_mask & nmask
            dapi_nucleus[spot_within] *= (1.0 - nucleolus_amp * spot[spot_within])
    # Plan3 v34: scale per-cell DAPI by anatomy z-projection. A cell at
    # z_position=0 with nucleus fully in the slab contributes the maximum
    # DAPI; a cell with z_position offset such that part of the nucleus is
    # above/below the slab contributes proportionally less. Compute the
    # nucleus-layer integral over the slab per cell and apply at nucleus
    # pixels via cell_label lookup. Normalize by the projection at z=0
    # for a fully-in-slab nucleus so a fully-in-plane cell gets factor=1
    # (preserves v33 absolute intensity baseline).
    if p.per_type_anatomy:
        from .mechanistic_anatomy import project_anatomy_through_slab
        n_lbl = int(cell_label.max()) + 2
        dapi_anatomy_lut = np.ones(n_lbl, dtype=np.float32)
        # Reference "fully in plane" projection per anatomy: z_position=0.
        anatomy_dapi_at_z0: dict[str, float] = {}
        for tname, anat in p.per_type_anatomy.items():
            proj = project_anatomy_through_slab(
                anat, z_position_um=0.0,
                slab_thickness_um=float(p.section_slab_thickness_um),
                channels=("dapi",),
            )
            anatomy_dapi_at_z0[tname] = max(float(proj["dapi"]), 1e-6)
        for cell in scene.cells:
            anat = p.per_type_anatomy.get(cell.cell_type) if cell.cell_type else None
            if anat is None:
                anat = p.per_type_anatomy.get("unknown")
            if anat is None:
                continue
            ref = anatomy_dapi_at_z0.get(cell.cell_type, anatomy_dapi_at_z0.get("unknown", 1.0))
            proj = project_anatomy_through_slab(
                anat, z_position_um=float(cell.z_position_um),
                slab_thickness_um=float(p.section_slab_thickness_um),
                channels=("dapi",),
            )
            scale = float(proj["dapi"]) / ref
            lbl = int(cell.label)
            if 0 < lbl < dapi_anatomy_lut.shape[0]:
                dapi_anatomy_lut[lbl] = scale
        cl_at_nuc = np.where(nucleus_mask, cell_label, 0).astype(np.int64)
        cl_at_nuc = np.clip(cl_at_nuc, 0, dapi_anatomy_lut.shape[0] - 1)
        dapi_anatomy_field = dapi_anatomy_lut[cl_at_nuc]
        dapi_nucleus = dapi_nucleus * dapi_anatomy_field
    dapi_nucleus = np.clip(dapi_nucleus, 0.0, 1.0)
    # Plan3 v35 P8: per-cell cytoplasmic DAPI bleed. Real DAPI shows dim
    # signal in cytoplasm (mitochondrial DNA, chromatin extending out of
    # the nuclear envelope, plus optical bleed from cells above/below
    # via Z-projection). Add a per-cell low-amplitude DAPI contribution
    # across the cytoplasm (cell minus nucleus). Cell-source compliant —
    # each cell contributes its own cytoplasmic dapi at a fixed fraction
    # of its nuclear amplitude.
    cyto_frac = float(getattr(p, "dapi_cytoplasm_fraction", 0.0))
    if cyto_frac > 0.0:
        for cell in scene.cells:
            cmask = cell_label == int(cell.label)
            if not np.any(cmask):
                continue
            cyto_mask = cmask & ~nucleus_mask
            if not np.any(cyto_mask):
                continue
            local_amp = float(p.dapi_base * cell.dapi_efficiency * cell.visibility_fraction) * cyto_frac
            dapi_nucleus[cyto_mask] = local_amp
    dapi_degraded = _gaussian_blur(dapi_nucleus, p.dapi_blur_sigma_px)

    boundary = _label_boundary(cell_label)
    # Use the largest per-pixel width to size the distance transform reach.
    max_width_px = float(max(p.membrane_width_px, float(np.max(width_field))))
    boundary_distance = _distance_from_mask(boundary, max_distance=int(max(8, np.ceil(max_width_px * 6))))
    # Per-pixel Gaussian decay using the local width.
    width_safe = np.maximum(width_field, 1e-3)
    ridge_profile = np.exp(-(boundary_distance**2) / (2.0 * width_safe**2)).astype(np.float32)

    # Plan3 v34: edge-aware membrane rendering. When ``per_type_anatomy`` is
    # configured, render membrane via edge-type maps:
    #   - intercellular edge: one shared thin ridge between two cells,
    #     intensity from the *pair* of per-type efficiencies.
    #   - cell-extracellular edge: asymmetric outward emission only on the
    #     extracellular side, intensity from the cell's per-type anatomy
    #     (basal_membrane / apical_membrane layers projected through the
    #     section slab).
    # Falls back to the v33 symmetric boundary ridge when anatomy is empty.
    if p.per_type_anatomy:
        membrane_boundary = _render_edge_aware_membrane(
            cell_label, cell_mask, scene, p,
        ).astype(np.float32)
    else:
        membrane_boundary = ridge_profile * float(p.membrane_base) * _cell_efficiency_field(scene, "membrane")
    # Plan3 v32: extracellular membrane ridge (per-type basement-membrane
    # sheet emission). Adds to membrane_boundary at extracellular pixels using
    # the SAME boundary-distance Gaussian profile as the intracellular ridge,
    # but with per-type extracellular amplitude and width. Modeling: the same
    # membrane-emission mechanism extends slightly outside the cell as a
    # basement-membrane sheet; per-type amp captures intensity, per-type width
    # captures sheet thickness. Skipped when no per-type extracellular params
    # are configured (older fits) — render then falls back to no contribution
    # in this branch and the cell-blind haze fills the role downstream.
    if p.per_type_extracellular_membrane_amp:
        extracellular_ridge = _render_extracellular_membrane_ridge(
            cell_label,
            cell_mask,
            boundary_distance,
            scene,
            p,
        )
        membrane_boundary = membrane_boundary + extracellular_ridge

    # Per-pixel dropout: convert smooth noise to uniform rank in [0, 1] and
    # keep pixels whose rank exceeds (1 - keep_prob_local). This avoids the
    # global-quantile assumption and lets each cell type have its own
    # dropout density along its boundary.
    dropout_field = _smooth_noise(rng, (h, w), sigma=p.membrane_dropout_scale_px)
    dropout_uniform = _to_uniform_rank(dropout_field)
    membrane_dropout_mask = (dropout_uniform >= (1.0 - dropout_keep_field)).astype(np.float32)
    membrane_boundary *= membrane_dropout_mask
    # Plan3 v35 P10: membrane line texture variation. Real membranes show
    # intensity variation along the boundary (junction density, focal
    # plane crossing, locally bright spots from junction proteins). Adds
    # per-pixel multiplicative jitter to membrane signal — values around
    # 1.0 ± membrane_line_texture_amp. Cell-source compliant — modulates
    # each cell's own boundary.
    line_tex_amp = float(getattr(p, "membrane_line_texture_amp", 0.0))
    if line_tex_amp > 0.0:
        line_tex = _smooth_noise(rng, (h, w), sigma=max(0.5, p.membrane_width_px * 0.5))
        membrane_boundary = membrane_boundary * (1.0 + line_tex_amp * line_tex.astype(np.float32))

    # Plan3 v35: re-enable membrane interior in all anatomy modes. This is
    # the diffuse cytoplasmic membrane signal (Golgi, ER, vesicles) that
    # makes membranes appear *continuous* even where boundary dropout
    # creates gaps. Cell-source compliant — value comes from per-cell LUT.
    # v34's "zero when anatomy" assumption was wrong: anatomy projection
    # contributes to dapi/polya channels, not to membrane.
    interior_lut = _build_membrane_interior_lut(scene, cell_label, p)
    cell_label_int_for_interior = cell_label.astype(np.int64, copy=False)
    interior_clamped = np.where(cell_label_int_for_interior < interior_lut.shape[0], cell_label_int_for_interior, 0)
    membrane_interior = interior_lut[interior_clamped]
    membrane_interior[nucleus_mask] *= 0.35

    # Plan3 v35: re-enable out-of-focus halo in all anatomy modes. Each
    # cell's boundary contributes a blurred outward bleed (PSF + cells
    # just-out-of-plane). Cell-source compliant — derived directly from
    # the cell's own boundary, not a free field.
    membrane_out_of_focus = _gaussian_blur(membrane_boundary, p.membrane_out_of_focus_sigma_px)
    membrane_out_of_focus *= float(p.membrane_out_of_focus_fraction)
    # Plan3 v35 P3: extracellular membrane bleed. Each cell's boundary
    # extends a low-amplitude halo into immediate extracellular space
    # (basement membrane proteins, secreted matrix). Distance-decayed
    # copy of boundary signal, decaying ~exp(-d/extracellular_decay_px).
    # Cell-source compliant. Replaces what `green_tissue_haze` provided
    # in v14 but ties signal to specific cells.
    membrane_extracellular_amp = float(getattr(p, "membrane_extracellular_bleed_amp", 0.0))
    membrane_extracellular_decay = float(getattr(p, "membrane_extracellular_bleed_decay_px", 3.0))
    if membrane_extracellular_amp > 0.0 and membrane_extracellular_decay > 0.0:
        # Distance from cell mask (0 inside, positive outside)
        from scipy.ndimage import distance_transform_edt as _edt
        d_out = _edt(~cell_mask).astype(np.float32)
        # Spread membrane_boundary signal outward via distance-decay weighted
        # blur. Blur the boundary at sigma=decay_px to spread, then mask
        # extracellular only.
        spread = _gaussian_blur(membrane_boundary, membrane_extracellular_decay)
        decay = np.exp(-d_out / membrane_extracellular_decay)
        membrane_extracellular_halo = spread * decay * membrane_extracellular_amp
        membrane_extracellular_halo[cell_mask] = 0.0  # outside-only
        membrane_out_of_focus = membrane_out_of_focus + membrane_extracellular_halo.astype(np.float32)
    # Plan3 v35: keep green_tissue_haze=0 when anatomy is configured — it's
    # the only cell-FREE component and the user has flagged that as
    # off-limits. Tissue context now comes from cell out-of-focus halos
    # and cell-emitted extracellular bleed.
    if p.per_type_anatomy or p.per_type_extracellular_membrane_amp:
        green_tissue_haze = np.zeros((h, w), dtype=np.float32)
    else:
        green_tissue_haze = _smooth_noise(rng, (h, w), sigma=p.haze_scale_px)
        green_tissue_haze = (green_tissue_haze - float(np.min(green_tissue_haze))) / (
            float(np.ptp(green_tissue_haze)) + 1e-6
        )
        green_tissue_haze = (p.haze_level * (0.35 + green_tissue_haze)).astype(np.float32)

    # Extracellular tissue component: continuous low-amplitude membrane signal
    # primarily in tissue regions outside cell masks, fitted from real
    # background pixels. Computed as a tracked component for inspection and
    # synthetic-mode injection, but NOT added to the rough render in observed
    # mode because the latent emission components already fill backgrounds
    # there (and adding both double-counts). The synthetic-mode renderer adds
    # it explicitly when latent emission is absent.
    tissue_noise = _smooth_noise(rng, (h, w), sigma=p.extracellular_tissue_scale_px)
    tissue_noise = (tissue_noise - float(np.min(tissue_noise))) / (
        float(np.ptp(tissue_noise)) + 1e-6
    )
    extracellular_intensity = (
        p.extracellular_tissue_level
        * (1.0 - float(p.extracellular_tissue_texture) + 2.0 * float(p.extracellular_tissue_texture) * tissue_noise)
    ).astype(np.float32)
    cell_dilated = binary_dilation(cell_mask, iterations=2).astype(np.float32)
    cell_softmask = _gaussian_blur(cell_dilated, 6.0)
    extracellular_tissue = (extracellular_intensity * (1.0 - 0.85 * cell_softmask)).astype(np.float32)

    membrane_ideal = (
        membrane_boundary
        + membrane_interior
        + membrane_out_of_focus
        + green_tissue_haze
    )
    membrane_degraded = _gaussian_blur(membrane_ideal, max(0.5, p.membrane_width_px * 0.7))

    illumination = 1.0 + p.illumination_scale * _smooth_noise(rng, (h, w), sigma=max(h, w) / 2.5)
    illumination = np.clip(illumination, 0.75, 1.25).astype(np.float32)
    dapi_degraded *= illumination
    membrane_degraded *= illumination

    # PolyA channel: cytoplasmic mRNA. Per-cell, fill cytoplasm (cell minus
    # nucleus) at base intensity scaled by per-cell polya_efficiency, with a
    # reduced level inside the nucleus (mRNA mostly cytoplasmic). Add a low
    # extracellular polya floor and noise. Per-type efficiencies in
    # ``MechanisticCell.polya_efficiency`` carry the biology (endocrine high,
    # immune low, etc.).
    # PolyA cytoplasm: per-cell independent texture field (v14 style). The
    # v34 single-shared-field optimization gave 12x speed but every cell
    # showed identical texture pattern, only differing in brightness — so
    # all cells looked uniform. Reverting to per-cell sampling: each cell
    # gets its own smooth-noise patch (still cell-source compliant; just
    # uses an independent random field per cell). Fallback to shared field
    # when polya_per_cell_texture is False (e.g., for backward-compat).
    if polya_radial_spectrum is not None and len(polya_radial_spectrum) > 1:
        # Spectrum-matched mode keeps single-field (radial spectrum is
        # tile-level statistic anyway).
        from .polya_spectrum import synthesize_polya_noise

        polya_texture_field = synthesize_polya_noise(
            rng, (h, w),
            radial_power=np.asarray(polya_radial_spectrum, dtype=np.float32),
            n_radial_bins=len(polya_radial_spectrum),
        )
    elif bool(getattr(p, "polya_per_cell_texture", True)):
        # Per-cell texture: each cell gets its own random field, blended
        # together by cell mask. Background uses one shared field.
        bg_texture = _smooth_noise(rng, (h, w), sigma=max(1.0, p.polya_blur_sigma_px * 2.0))
        polya_texture_field = bg_texture.copy()
        for cell in scene.cells:
            cmask = cell_label == int(cell.label)
            if not np.any(cmask):
                continue
            cell_tex = _smooth_noise(rng, (h, w), sigma=max(1.0, p.polya_blur_sigma_px * 2.0))
            polya_texture_field = np.where(cmask, cell_tex, polya_texture_field)
    else:
        polya_texture_field = _smooth_noise(rng, (h, w), sigma=max(1.0, p.polya_blur_sigma_px * 2.0))
    polya_amp_lut = _build_polya_amp_lut(scene, cell_label, p)
    cell_label_int = cell_label.astype(np.int64, copy=False)
    cell_label_clamped = np.where(cell_label_int < polya_amp_lut.shape[0], cell_label_int, 0)
    polya_amp_field = polya_amp_lut[cell_label_clamped]
    # Plan3 v34: scale per-cell polyA by anatomy z-projection. Cytoplasm
    # layers (basal_cytoplasm, supranuclear_cyto, apical_cytoplasm,
    # apical_mucin) all carry polya_factor. A fully-in-plane cell with
    # cytoplasm fully spanning the slab gets factor=1; an apex_above cell
    # gets only the basal cytoplasm contribution. Multiplies polya_amp_field
    # at each cell pixel.
    if p.per_type_anatomy:
        from .mechanistic_anatomy import project_anatomy_through_slab
        n_lbl = polya_amp_lut.shape[0]
        polya_anatomy_lut = np.ones(n_lbl, dtype=np.float32)
        anatomy_polya_at_z0: dict[str, float] = {}
        for tname, anat in p.per_type_anatomy.items():
            proj = project_anatomy_through_slab(
                anat, z_position_um=0.0,
                slab_thickness_um=float(p.section_slab_thickness_um),
                channels=("polya",),
            )
            anatomy_polya_at_z0[tname] = max(float(proj["polya"]), 1e-6)
        for cell in scene.cells:
            anat = p.per_type_anatomy.get(cell.cell_type) if cell.cell_type else None
            if anat is None:
                anat = p.per_type_anatomy.get("unknown")
            if anat is None:
                continue
            ref = anatomy_polya_at_z0.get(cell.cell_type, anatomy_polya_at_z0.get("unknown", 1.0))
            proj = project_anatomy_through_slab(
                anat, z_position_um=float(cell.z_position_um),
                slab_thickness_um=float(p.section_slab_thickness_um),
                channels=("polya",),
            )
            scale = float(proj["polya"]) / ref
            lbl = int(cell.label)
            if 0 < lbl < polya_anatomy_lut.shape[0]:
                polya_anatomy_lut[lbl] = scale
        polya_anatomy_field = polya_anatomy_lut[cell_label_clamped]
        polya_amp_field = polya_amp_field * polya_anatomy_field
    # Plan3 B13: per-cell polya_texture_scale (intra-cell CV) when set on
    # MechanisticCell, else fall back to global p.polya_texture_scale.
    # Closes B7 std_polya overshoot for low-polyA types.
    polya_tex_lut = _build_polya_texture_lut(scene, cell_label, p)
    polya_tex_field = polya_tex_lut[cell_label_clamped]
    polya_cytoplasm = (polya_amp_field * (1.0 + polya_tex_field * polya_texture_field)).astype(np.float32)
    # Plan3 v35 P2: per-cell stochastic polyA puncta. Real Xenium polyA has
    # heavy-tailed pixel distribution (CV>0.7, kurtosis>5) that smooth-noise
    # cytoplasm can't reproduce (CV~0.4, kurtosis~1). Add a per-cell
    # Poisson-density of point sources (no molecule semantics — purely a
    # texture model) at sub-pixel positions, rasterized with Gaussian
    # footprint sigma=0.5 px. Each cell's puncta count scales with cell
    # area * per-cell polyA amplitude, producing higher punctate density
    # in brighter cells. Cell-source compliant: every punctum is owned
    # by a cell. Disabled when polya_puncta_density_per_amp <= 0.
    puncta_density = float(getattr(p, "polya_puncta_density_per_amp", 0.0))
    puncta_amp = float(getattr(p, "polya_puncta_amp", 0.0))
    if puncta_density > 0.0 and puncta_amp > 0.0:
        for cell in scene.cells:
            cmask = cell_label == int(cell.label)
            cell_area = int(cmask.sum())
            if cell_area < 5:
                continue
            cyto_mask = cmask & ~nucleus_mask
            cyto_pixels = np.argwhere(cyto_mask)
            if cyto_pixels.shape[0] == 0:
                continue
            local_amp = float(p.polya_base * cell.polya_efficiency * cell.visibility_fraction)
            n_puncta = int(rng.poisson(cyto_mask.sum() * local_amp * puncta_density))
            if n_puncta <= 0:
                continue
            pick = rng.integers(0, cyto_pixels.shape[0], size=n_puncta)
            ys = cyto_pixels[pick, 0].astype(np.float64) + rng.uniform(-0.5, 0.5, size=n_puncta)
            xs = cyto_pixels[pick, 1].astype(np.float64) + rng.uniform(-0.5, 0.5, size=n_puncta)
            iy = np.clip(np.round(ys).astype(int), 0, h - 1)
            ix = np.clip(np.round(xs).astype(int), 0, w - 1)
            np.add.at(polya_cytoplasm, (iy, ix), float(puncta_amp))
    # Plan3 T-rasterize-polya (negative result): if transcripts are
    # provided, ADD a tiny rasterized-density contribution to the polyA
    # cytoplasm field. Note: Xenium polyA stain is a CONTINUOUS
    # polyA-tagged fluorescence signal, NOT a sum of single-molecule
    # deltas — replacing the smooth-noise model with rasterized points
    # regressed B4 from 0.546 to 0.29 and std_polya from 17 to 127. So
    # we keep the smooth-noise+texture as the polyA cytoplasm and treat
    # transcripts as an auxiliary parquet output (Layer T proper).
    # When ``polya_per_transcript`` > 0 and transcripts are supplied,
    # the rasterized density is ADDED at low amplitude — for now defaults
    # to 0 (disabled).
    if (transcripts is not None and len(transcripts.get("x_um", [])) > 0
            and float(p.polya_per_transcript) > 0.0):
        polya_cytoplasm = polya_cytoplasm + _rasterize_transcript_density(
            transcripts,
            image_shape=(h, w),
            pixel_size_um=float(scene.pixel_size),
            polya_per_transcript=float(p.polya_per_transcript),
        ).astype(np.float32)
    # Plan3 B15: per-pixel Poisson shot noise on polyA. Real polyA pixel
    # intensities are Poisson-distributed counts of mRNA molecules; the
    # smooth_noise field captures spatial correlation but underestimates
    # the per-pixel variance tail. Adding Poisson noise on top closes the
    # B7 std_polya distribution-shape gap (heavy-tail in real cells).
    # shot_lambda controls noise strength; default 50 gives CV ~0.14 at
    # amp=0.3, modest addition to the smooth-noise CV ~0.35.
    if float(p.polya_shot_lambda) > 0.0:
        shot_lambda = float(p.polya_shot_lambda)
        expected_counts = polya_cytoplasm * shot_lambda
        # Clamp to avoid huge Poisson means at clipped 1.0 amp.
        expected_counts = np.clip(expected_counts, 0.0, 1000.0)
        polya_cytoplasm = (rng.poisson(expected_counts).astype(np.float32) / shot_lambda)
    polya_cytoplasm[~cell_mask] = 0.0  # cytoplasm rendered only inside cells
    # Reduce inside nucleus (mRNA depleted there).
    polya_cytoplasm[nucleus_mask] *= float(p.polya_nucleus_fraction)
    polya_extracellular = float(p.polya_extracellular_level)
    polya_ideal = polya_cytoplasm + polya_extracellular * (1.0 - cell_mask.astype(np.float32))
    polya_degraded = _gaussian_blur(polya_ideal, max(0.4, p.polya_blur_sigma_px))
    polya_degraded *= illumination

    if p.noise_sigma > 0:
        dapi_degraded += rng.normal(0.0, p.noise_sigma, size=(h, w)).astype(np.float32)
        membrane_degraded += rng.normal(0.0, p.noise_sigma, size=(h, w)).astype(np.float32)
        polya_degraded += rng.normal(0.0, p.noise_sigma * 0.5, size=(h, w)).astype(np.float32)

    images = np.stack(
        [
            np.clip(dapi_degraded, 0.0, 1.0),
            np.clip(membrane_degraded, 0.0, 1.0),
            np.clip(polya_degraded, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)
    components = {
        "dapi_nucleus": dapi_nucleus.astype(np.float32),
        "dapi_degraded": np.clip(dapi_degraded, 0.0, 1.0).astype(np.float32),
        "cell_boundary": boundary.astype(np.float32),
        "membrane_distance": boundary_distance.astype(np.float32),
        "membrane_boundary": membrane_boundary.astype(np.float32),
        "membrane_dropout_mask": membrane_dropout_mask.astype(np.float32),
        "membrane_cell_interior": membrane_interior.astype(np.float32),
        "membrane_out_of_focus": membrane_out_of_focus.astype(np.float32),
        "green_tissue_haze": green_tissue_haze.astype(np.float32),
        "extracellular_tissue": extracellular_tissue.astype(np.float32),
        "membrane_degraded": np.clip(membrane_degraded, 0.0, 1.0).astype(np.float32),
        "polya_cytoplasm": polya_cytoplasm.astype(np.float32),
        "polya_degraded": np.clip(polya_degraded, 0.0, 1.0).astype(np.float32),
        "illumination_field": illumination.astype(np.float32),
    }
    return MechanisticRenderOutput(
        images=images,
        components=components,
        state={
            "renderer": "mechanistic",
            "params": p.to_dict(),
            "seed": int(p.seed if seed is None else seed),
            "scene_id": scene.scene_id,
            "num_cells": len(scene.cells),
        },
    )


def _label_boundary(label: np.ndarray) -> np.ndarray:
    arr = label.astype(np.int32, copy=False)
    boundary = np.zeros(arr.shape, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.zeros_like(arr)
        if dy == 1:
            shifted[1:] = arr[:-1]
        elif dy == -1:
            shifted[:-1] = arr[1:]
        elif dx == 1:
            shifted[:, 1:] = arr[:, :-1]
        elif dx == -1:
            shifted[:, :-1] = arr[:, 1:]
        boundary |= (arr != shifted) & ((arr > 0) | (shifted > 0))
    return boundary


def _build_edge_maps(cell_label: np.ndarray) -> dict[str, np.ndarray]:
    """Build edge-type maps from a cell_label image.

    Returns a dict with:
      - ``intercellular_edge``: bool mask, True at pixels (in cell A) whose
        4-neighbor lies in a *different* non-zero cell (cell B). The
        same-pair edge contributes from BOTH sides; that's intentional —
        it lets the renderer compute pair-averaged intensity at each pixel
        independently.
      - ``cell_extracellular_edge``: bool mask, True at pixels in any cell
        whose 4-neighbor is background (label 0). Pure free-edge pixels.
      - ``neighbor_label``: int32 array, at each intercellular_edge pixel
        gives the label of the *other* cell (the neighbour whose label
        differs from this pixel's). Zero elsewhere.
      - ``extracellular_distance``: float32 array, distance from the
        nearest cell footprint into extracellular pixels (scipy EDT on
        ``cell_label == 0``). Used for asymmetric outward decay.
      - ``extracellular_nearest_label``: int32 array, label of the nearest
        cell for each extracellular pixel (used to look up per-cell
        anatomy for the cell-extracellular ridge).
    """
    h, w = cell_label.shape
    cl = cell_label.astype(np.int32, copy=False)
    own_nonzero = cl > 0
    intercellular = np.zeros((h, w), dtype=bool)
    free_edge = np.zeros((h, w), dtype=bool)
    neighbor_label = np.zeros((h, w), dtype=np.int32)
    # For each of 4 neighbour directions, build a shifted neighbour map
    # and detect different-label / background-neighbour transitions.
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.zeros_like(cl)
        if dy >= 0:
            ys_src = slice(dy, h); ys_dst = slice(0, h - dy)
        else:
            ys_src = slice(0, h + dy); ys_dst = slice(-dy, h)
        if dx >= 0:
            xs_src = slice(dx, w); xs_dst = slice(0, w - dx)
        else:
            xs_src = slice(0, w + dx); xs_dst = slice(-dx, w)
        shifted[ys_dst, xs_dst] = cl[ys_src, xs_src]
        diff = (shifted != cl) & own_nonzero
        ic = diff & (shifted > 0)
        intercellular |= ic
        new_b = ic & (neighbor_label == 0)
        neighbor_label[new_b] = shifted[new_b]
        free_edge |= diff & (shifted == 0)
    out: dict[str, np.ndarray] = {
        "intercellular_edge": intercellular,
        "cell_extracellular_edge": free_edge,
        "neighbor_label": neighbor_label,
    }
    try:
        from scipy.ndimage import distance_transform_edt as _edt
        extracellular = ~own_nonzero
        if np.any(extracellular):
            dist, indices = _edt(extracellular, return_indices=True)
            nearest = cl[indices[0], indices[1]]
            out["extracellular_distance"] = dist.astype(np.float32, copy=False)
            out["extracellular_nearest_label"] = nearest.astype(np.int32, copy=False)
        else:
            out["extracellular_distance"] = np.zeros((h, w), dtype=np.float32)
            out["extracellular_nearest_label"] = np.zeros((h, w), dtype=np.int32)
    except Exception:  # noqa: BLE001
        out["extracellular_distance"] = np.zeros((h, w), dtype=np.float32)
        out["extracellular_nearest_label"] = np.zeros((h, w), dtype=np.int32)
    return out


def _render_edge_aware_membrane(
    cell_label: np.ndarray,
    cell_mask: np.ndarray,
    scene: MechanisticScene,
    p: MechanisticParams,
) -> np.ndarray:
    """v34 S3 phase 1: edge-aware membrane rendering.

    Replaces the v33 symmetric Gaussian-from-boundary mechanism with two
    separate ridge components:

    1. Intercellular edge ridge: a thin Gaussian (σ ≈ 1 px) along the
       contact line between two cells. Intensity at each contact pixel is
       the AVERAGE of the two cells' per-type membrane efficiencies times
       ``membrane_base``. Renders ONCE per edge — no halo extending into
       either cell interior.
    2. Cell-extracellular edge ridge: asymmetric Gaussian projecting
       *outward* from the cell into extracellular pixels. Intensity is
       the cell's own per-type basal-membrane anatomy emission factor
       (from ``per_type_anatomy[type].layers``) times its
       ``membrane_efficiency`` and ``membrane_base``. The ridge does not
       extend into the cell interior — the cell's own contributions to
       its interior come from anatomy projection in S3 phase 3.

    Both components use a thin ridge width (~1 px); the cell-extracellular
    ridge can extend further outward when basal_membrane anatomy includes
    a wider layer.
    """
    h, w = cell_label.shape
    out = np.zeros((h, w), dtype=np.float32)
    edges = _build_edge_maps(cell_label)
    membrane_base = float(p.membrane_base)

    # Per-cell efficiency LUT (same-shape array, indexed by cell label).
    eff_lut = np.zeros(int(max(cell_label.max(), 1)) + 2, dtype=np.float32)
    type_lut: dict[int, str] = {}
    for cell in scene.cells:
        lbl = int(cell.label)
        if 0 < lbl < eff_lut.shape[0]:
            eff_lut[lbl] = float(cell.membrane_efficiency)
            if cell.cell_type is not None:
                type_lut[lbl] = cell.cell_type

    # 1. Intercellular edge ridge — pair-averaged efficiency, thin Gaussian
    # profile. Use distance-from-mask to get a real Gaussian ridge whose
    # peak intensity at d=0 equals (eff_pair * membrane_base), not a
    # convolution-attenuated version. The ridge falls off both sides of
    # the contact line but with σ small enough that it doesn't paint cell
    # interiors (≤ 1.5 px = 0.32 µm — within plasma membrane width).
    inter = edges["intercellular_edge"]
    if np.any(inter):
        intercell_sigma = float(getattr(p, "intercell_ridge_sigma_px", 0.9))
        intercell_dist = _distance_from_mask(
            inter,
            max_distance=int(np.ceil(intercell_sigma * 6)),
        )
        intercell_profile = np.exp(
            -(intercell_dist**2) / (2.0 * intercell_sigma**2)
        ).astype(np.float32)
        # Pair-averaged efficiency at the contact line. For pixels off the
        # contact, use the closer-cell's efficiency (approximation: use
        # this pixel's own cell label, falling back to the nearest cell
        # via extracellular_nearest_label for extracellular pixels). The
        # ridge intensity is dominated by pixels on or adjacent to the
        # contact, so exact pair lookup off-line matters little.
        eff_at_pix = eff_lut[np.clip(cell_label.astype(np.int64), 0, eff_lut.shape[0] - 1)]
        # Where this pixel sits inside a cell, blend with neighbor at the
        # contact line. Off-line pixels get just eff_at_pix.
        neighbor_idx = np.clip(edges["neighbor_label"].astype(np.int64), 0, eff_lut.shape[0] - 1)
        eff_neighbor = eff_lut[neighbor_idx]
        eff_pair = np.where(inter, 0.5 * (eff_at_pix + eff_neighbor), eff_at_pix).astype(np.float32)
        # Plan3 v35 P9: cell-cell junction intensification. Real cells
        # show brighter membrane at contact junctions (junction proteins).
        # Amplify intercellular signal by junction_boost factor.
        junction_boost = 1.0 + float(getattr(p, "membrane_junction_boost", 0.0))
        intercell_signal = intercell_profile * eff_pair * membrane_base * junction_boost
        out = out + intercell_signal

    # 2. Cell-extracellular edge ridge — asymmetric outward emission.
    # For each extracellular pixel within distance `max_outward_px` of a
    # cell, look up the nearest cell, get its per-type basal_membrane
    # anatomy emission factor, multiply by cell's membrane_efficiency and
    # membrane_base, and apply Gaussian decay with distance.
    ext_dist = edges["extracellular_distance"]
    ext_nearest = edges["extracellular_nearest_label"]
    extracellular = ~cell_mask
    max_outward_px = 5.0
    in_band = extracellular & (ext_dist <= max_outward_px) & (ext_nearest > 0)
    if np.any(in_band):
        # Per-pixel basal-membrane anatomy factor — look up nearest cell's
        # type, then sum its anatomy layer "membrane" emissions over the
        # full z-extent (this is the cell's *outward* membrane signature
        # — basal_membrane + apical_membrane projected to the in-plane
        # footprint). Approximation: use sum of all "membrane"-tagged
        # layers from per_type_anatomy[type].layers.
        anatomy = p.per_type_anatomy
        basal_factor_lut = np.zeros(eff_lut.shape, dtype=np.float32)
        default_for_outward = anatomy.get("unknown")
        for lbl in range(1, basal_factor_lut.shape[0]):
            tname = type_lut.get(lbl)
            anat = anatomy.get(tname) if tname else None
            if anat is None:
                anat = default_for_outward
            if anat is None:
                continue
            layers = anat.get("layers", {})
            # Sum membrane emission across all layers, weighted by layer
            # thickness in z (proportional to (z_hi - z_lo) of each layer).
            total = 0.0
            for layer in layers.values():
                z_lo = float(layer.get("z_lo", 0.0))
                z_hi = float(layer.get("z_hi", 0.0))
                memb = float(layer.get("membrane", 0.0))
                total += memb * max(0.0, z_hi - z_lo)
            basal_factor_lut[lbl] = total
        # Width of the outward ridge: scaled with per-cell membrane width
        # but kept tight (this is the basement-membrane sheet — physically
        # only a few pixels). Use 1.5 × per-type membrane_width (or
        # global default) as σ.
        outward_sigma = max(1.0, float(p.membrane_width_px) * 1.0)
        nearest_idx = np.clip(ext_nearest.astype(np.int64), 0, eff_lut.shape[0] - 1)
        amp_field = basal_factor_lut[nearest_idx] * eff_lut[nearest_idx] * membrane_base
        contrib = amp_field * np.exp(-(ext_dist**2) / (2.0 * outward_sigma**2))
        contrib *= in_band
        out = out + contrib.astype(np.float32)

    # 3. Per-cell interior membrane contribution from anatomy projection.
    # Each cell's apical/basal/cytoplasm-membrane layers, projected through
    # the slab, deposit a low-level membrane signal across the in-plane
    # footprint (out-of-focus apical/basal sheets seen from the top/bottom
    # of the slab). Scale by per-cell membrane efficiency + global
    # membrane_base. The integrated layer projection is small in factor
    # units (~0.04 for cytoplasm-membrane); apply a calibration scale of
    # ``interior_anatomy_scale`` to bring projected intensity into the
    # same range as the legacy v33 interior LUT. Default 6x makes v34
    # mean intensity within ~10% of v33 across mixed-type tiles.
    from .mechanistic_anatomy import project_anatomy_through_slab
    interior_anatomy_scale = 6.0
    interior_lut_v34 = np.zeros(eff_lut.shape, dtype=np.float32)
    default_anat = p.per_type_anatomy.get("unknown")
    for cell in scene.cells:
        anat = p.per_type_anatomy.get(cell.cell_type) if cell.cell_type else None
        if anat is None:
            anat = default_anat
        if anat is None:
            continue
        proj = project_anatomy_through_slab(
            anat,
            z_position_um=float(cell.z_position_um),
            slab_thickness_um=float(p.section_slab_thickness_um),
            channels=("membrane",),
        )
        lbl = int(cell.label)
        if 0 < lbl < interior_lut_v34.shape[0]:
            interior_lut_v34[lbl] = (
                float(proj["membrane"])
                * float(cell.membrane_efficiency)
                * membrane_base
                * interior_anatomy_scale
            )
    cl_idx = np.clip(cell_label.astype(np.int64), 0, interior_lut_v34.shape[0] - 1)
    interior_field = interior_lut_v34[cl_idx]
    # Don't pile interior signal on top of the intercellular ridge — the
    # ridge already paints those pixels with thin-line intensity.
    interior_field[edges["intercellular_edge"]] *= 0.5
    interior_field[~cell_mask] = 0.0
    out = out + interior_field

    return out


def _render_extracellular_membrane_ridge(
    cell_label: np.ndarray,
    cell_mask: np.ndarray,
    boundary_distance: np.ndarray,
    scene: MechanisticScene,
    p: MechanisticParams,
) -> np.ndarray:
    """Cell-aware extracellular membrane emission as basement-membrane sheet.

    For each extracellular pixel, look up the *nearest* cell, and if that
    cell's type has a non-zero per-type extracellular ridge amplitude, add
    ``amp * exp(-d^2 / (2 * sigma^2))`` to the membrane channel — where
    ``d`` is the existing boundary distance map (distance from boundary
    line, valid both inside and outside cells) and ``sigma`` is the per-type
    extracellular ridge width. This models the basement-membrane sheet:
    same Gaussian-from-boundary profile as the intracellular ridge, but
    with per-type emission strength and thickness so that only epithelial
    types (which secrete BM) contribute.

    Cell types without entries in ``per_type_extracellular_membrane_amp``
    contribute zero. Pixels not assigned to any cell (whole-tile background
    if a tile has no cells) also contribute zero.
    """

    h, w = cell_label.shape
    ridge = np.zeros((h, w), dtype=np.float32)
    if not p.per_type_extracellular_membrane_amp:
        return ridge
    extracellular = ~cell_mask
    if not np.any(extracellular):
        return ridge
    try:
        from scipy.ndimage import distance_transform_edt as _edt
    except Exception:  # noqa: BLE001
        return ridge
    # Build per-cell-label LUTs of (extracellular_amp, extracellular_width).
    n = max(int(cell_label.max()) + 1, 2)
    amp_lut = np.zeros(n, dtype=np.float32)
    width_lut = np.full(n, float(p.membrane_width_px), dtype=np.float32)
    for cell in scene.cells:
        if cell.cell_type is None:
            continue
        amp = float(p.per_type_extracellular_membrane_amp.get(cell.cell_type, 0.0))
        if amp <= 0.0:
            continue
        width = float(
            p.per_type_extracellular_membrane_width_px.get(
                cell.cell_type, float(p.membrane_width_px)
            )
        )
        if width <= 0.0:
            continue
        lbl = int(cell.label)
        if 0 <= lbl < n:
            amp_lut[lbl] = amp
            width_lut[lbl] = width
    if not np.any(amp_lut > 0):
        return ridge
    # Find nearest cell label for each pixel via EDT-with-indices on the
    # complement of cell_mask. Pixels inside cells get themselves; pixels
    # outside get their nearest cell-pixel's label.
    inv_cell = ~cell_mask
    _, indices = _edt(inv_cell, return_indices=True)
    nearest_label = cell_label[indices[0], indices[1]].astype(np.int64, copy=False)
    nearest_label = np.clip(nearest_label, 0, n - 1)
    nearest_amp = amp_lut[nearest_label]
    nearest_width = width_lut[nearest_label]
    width_safe = np.maximum(nearest_width, 1e-3)
    contrib = nearest_amp * np.exp(-(boundary_distance**2) / (2.0 * width_safe**2))
    contrib *= extracellular  # restrict to outside-cells; intracellular ridge handled separately
    return contrib.astype(np.float32, copy=False)


def _distance_from_mask(source: np.ndarray, max_distance: int) -> np.ndarray:
    """Distance to nearest True pixel, clipped to ``max_distance``.

    Uses scipy.ndimage.distance_transform_edt when available (orders of
    magnitude faster than the iterative dilation fallback).
    """

    source_bool = source.astype(bool)
    if not np.any(source_bool):
        return np.full(source.shape, float(max_distance), dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        # distance_transform_edt computes distance to nearest 0 pixel; we
        # want distance to nearest True (mask) pixel, so invert.
        dist = distance_transform_edt(~source_bool).astype(np.float32, copy=False)
        return np.minimum(dist, float(max_distance))
    except Exception:  # noqa: BLE001 - fallback to iterative dilation
        distance = np.full(source.shape, float(max_distance), dtype=np.float32)
        distance[source_bool] = 0.0
        current = source_bool.copy()
        for step in range(1, max(1, max_distance) + 1):
            current = binary_dilation(current, iterations=1)
            ring = current & (distance == float(max_distance))
            distance[ring] = float(step)
            if np.all(current):
                break
        return distance


def _build_dapi_luts(scene: MechanisticScene, nucleus_label: np.ndarray, p: MechanisticParams):
    """Build (amplitude_lut, texture_lut) indexed by nucleus_label value.

    amplitude_lut[L] = dapi_base * cell.dapi_efficiency * cell.visibility_fraction
    texture_lut[L] = cell.dapi_texture_scale or p.dapi_texture_scale
    """

    max_label = int(nucleus_label.max()) if nucleus_label.size else 0
    amp = np.zeros(max_label + 1, dtype=np.float32)
    tex = np.full(max_label + 1, float(p.dapi_texture_scale), dtype=np.float32)
    for cell in scene.cells:
        if cell.nucleus_label is None:
            continue
        nl = int(cell.nucleus_label)
        if nl < 0 or nl > max_label:
            continue
        amp[nl] = float(p.dapi_base * cell.dapi_efficiency * cell.visibility_fraction)
        if cell.dapi_texture_scale is not None:
            tex[nl] = float(cell.dapi_texture_scale)
    return amp, tex


def _rasterize_transcript_density(
    transcripts: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    pixel_size_um: float,
    polya_per_transcript: float,
) -> np.ndarray:
    """T-rasterize-polya: turn a transcript point cloud into a polyA density
    image. Each transcript adds ``polya_per_transcript`` to its pixel.
    Output is unblurred — caller's downstream Gaussian blur acts as the
    optical PSF.
    """

    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)
    x_um = np.asarray(transcripts.get("x_um", []), dtype=np.float32)
    y_um = np.asarray(transcripts.get("y_um", []), dtype=np.float32)
    if x_um.size == 0:
        return density
    px = float(pixel_size_um)
    if px <= 0:
        return density
    ix = np.clip(np.round(x_um / px).astype(np.int32), 0, w - 1)
    iy = np.clip(np.round(y_um / px).astype(np.int32), 0, h - 1)
    np.add.at(density, (iy, ix), float(polya_per_transcript))
    return density


def _build_polya_amp_lut(scene: MechanisticScene, cell_label: np.ndarray, p: MechanisticParams) -> np.ndarray:
    max_label = int(cell_label.max()) if cell_label.size else 0
    amp = np.zeros(max_label + 1, dtype=np.float32)
    for cell in scene.cells:
        cl = int(cell.label)
        if cl < 0 or cl > max_label:
            continue
        amp[cl] = float(p.polya_base * cell.polya_efficiency * cell.visibility_fraction)
    return amp


def _build_polya_texture_lut(scene: MechanisticScene, cell_label: np.ndarray, p: MechanisticParams) -> np.ndarray:
    """B13: per-cell polya_texture_scale (intra-cell std/mean) lut.

    Each cell gets ``cell.polya_texture_scale`` if set, else the scene-
    global ``p.polya_texture_scale``. Used in the polya_cytoplasm field
    multiplier so per-type polyA texture amplitude can match real CV.
    """

    max_label = int(cell_label.max()) if cell_label.size else 0
    tex = np.full(max_label + 1, float(p.polya_texture_scale), dtype=np.float32)
    for cell in scene.cells:
        cl = int(cell.label)
        if cl < 0 or cl > max_label:
            continue
        if cell.polya_texture_scale is not None:
            tex[cl] = float(cell.polya_texture_scale)
    return tex


def _build_membrane_interior_lut(scene: MechanisticScene, cell_label: np.ndarray, p: MechanisticParams) -> np.ndarray:
    max_label = int(cell_label.max()) if cell_label.size else 0
    lut = np.zeros(max_label + 1, dtype=np.float32)
    for cell in scene.cells:
        cl = int(cell.label)
        if cl < 0 or cl > max_label:
            continue
        interior_fraction = (
            float(cell.membrane_interior_fraction) if cell.membrane_interior_fraction is not None
            else float(p.membrane_interior_fraction)
        )
        lut[cl] = float(p.membrane_base * interior_fraction * cell.membrane_efficiency)
    return lut


def _per_pixel_param_field(scene: MechanisticScene, cell_label: np.ndarray, attr: str, default: float) -> np.ndarray:
    """Build a per-pixel parameter field by reading per-cell ``attr``.

    Pixels inside cell with label N take the value ``cells[i].attr`` if not None,
    else ``default``. Pixels outside any cell within ~4 px take the nearest
    cell's value (so membrane decay just outside a cell respects that cell's
    parameter). Pixels further away use ``default``.

    Vectorized via scipy ``distance_transform_edt`` with ``return_indices=True``.
    """

    field = np.full(cell_label.shape, float(default), dtype=np.float32)
    if not scene.cells:
        return field
    # Per-label LUT.
    max_label = int(cell_label.max()) if cell_label.size else 0
    lut = np.full(max_label + 1, float(default), dtype=np.float32)
    for cell in scene.cells:
        v = getattr(cell, attr, None)
        cl = int(cell.label)
        if 0 <= cl <= max_label:
            lut[cl] = float(v) if v is not None else float(default)
    cell_label_int = cell_label.astype(np.int64, copy=False)
    field = lut[np.where(cell_label_int <= max_label, cell_label_int, 0)]
    cell_mask = cell_label > 0
    if np.any(cell_mask):
        try:
            from scipy.ndimage import distance_transform_edt
            dist, idx = distance_transform_edt(~cell_mask, return_indices=True)
            within_ring = (dist <= 4.0) & ~cell_mask
            if np.any(within_ring):
                # For each in-ring pixel, copy the nearest cell pixel's value.
                field[within_ring] = field[idx[0][within_ring], idx[1][within_ring]]
        except Exception:  # noqa: BLE001 — fall back to no exterior fill
            pass
    return field


def _to_uniform_rank(values: np.ndarray) -> np.ndarray:
    """Convert a 2D field to per-pixel uniform [0, 1] rank via argsort."""

    flat = values.ravel()
    n = flat.size
    if n == 0:
        return values.astype(np.float32, copy=True)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = (np.arange(n, dtype=np.float32) + 0.5) / float(n)
    return ranks.reshape(values.shape)


def _cell_efficiency_field(scene: MechanisticScene, attr: str) -> np.ndarray:
    field = np.zeros(scene.image_shape, dtype=np.float32)
    for cell in scene.cells:
        value = float(getattr(cell, f"{attr}_efficiency"))
        field[scene.cell_label == int(cell.label)] = value
    background = field == 0
    if np.any(background):
        field[background] = 1.0
    return field


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    return _gaussian_blur(noise, sigma)


# Module-level torch session reused across calls so we don't re-init CUDA per
# image. Initialized lazily on first use; falls back to scipy CPU if torch or
# CUDA is unavailable.
_TORCH_BLUR_DEVICE = None
_TORCH_BLUR_KERNELS: dict[float, "torch.Tensor"] = {}


def _try_init_torch_blur():
    global _TORCH_BLUR_DEVICE
    if _TORCH_BLUR_DEVICE is not None:
        return _TORCH_BLUR_DEVICE
    try:
        import torch  # noqa: F401 — torch is a hard dep but guard anyway
        _TORCH_BLUR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return _TORCH_BLUR_DEVICE
    except Exception:  # noqa: BLE001
        _TORCH_BLUR_DEVICE = "scipy"
        return _TORCH_BLUR_DEVICE


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32, copy=True)
    device = _try_init_torch_blur()
    if device == "cuda":
        return _gaussian_blur_torch(image, float(sigma), device="cuda")
    # CPU path: scipy.ndimage.gaussian_filter is ~50x faster than the manual
    # separable tensordot loop (and was the path used pre-G).
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image.astype(np.float32, copy=False), float(sigma), mode="reflect").astype(
            np.float32, copy=False
        )
    except Exception:  # noqa: BLE001 - last-resort fallback
        kernel = _gaussian_kernel1d(sigma)
        out = _convolve_axis(image.astype(np.float32, copy=False), kernel, axis=0)
        out = _convolve_axis(out, kernel, axis=1)
        return out.astype(np.float32, copy=False)


def _gaussian_blur_torch(image: np.ndarray, sigma: float, device: str = "cuda") -> np.ndarray:
    """GPU separable Gaussian blur via cached 1D kernels and reflect-pad conv.

    For sigmas large enough that the kernel radius exceeds the image dim
    (illumination/haze fields), we fall back to scipy CPU blur — torch's
    reflect padding requires pad < dim, and very-large-sigma blurs are not
    a meaningful speed bottleneck anyway.
    """

    import torch
    import torch.nn.functional as F

    sigma = float(sigma)
    radius = max(1, int(np.ceil(sigma * 3.0)))
    h, w = image.shape
    if radius >= min(h, w):
        # Fall back to scipy for very-wide kernels.
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image.astype(np.float32, copy=False), sigma, mode="reflect").astype(
            np.float32, copy=False
        )
    kernel_key = round(sigma, 4)
    if kernel_key not in _TORCH_BLUR_KERNELS:
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        k = (k / k.sum()).astype(np.float32)
        kt = torch.from_numpy(k).to(device).view(1, 1, -1)
        _TORCH_BLUR_KERNELS[kernel_key] = kt
    kernel = _TORCH_BLUR_KERNELS[kernel_key]
    img = torch.from_numpy(image.astype(np.float32, copy=False)).to(device).unsqueeze(0).unsqueeze(0)
    pad = radius
    img_h = F.pad(img, (0, 0, pad, pad), mode="reflect")
    out_h = F.conv2d(img_h, kernel.unsqueeze(-1))  # vertical
    img_v = F.pad(out_h, (pad, pad, 0, 0), mode="reflect")
    out = F.conv2d(img_v, kernel.unsqueeze(-2))  # horizontal
    return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(float(sigma) * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * float(sigma) ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(image, pad_width, mode="reflect")
    moved = np.moveaxis(padded, axis, 0)
    out_moved = np.empty((image.shape[axis], *moved.shape[1:]), dtype=np.float32)
    for idx in range(image.shape[axis]):
        window = moved[idx : idx + len(kernel)]
        out_moved[idx] = np.tensordot(kernel, window, axes=(0, 0))
    return np.moveaxis(out_moved, 0, axis).astype(np.float32, copy=False)

