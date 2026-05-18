"""Plan3 Stage 5: synthetic mechanistic scene sampler.

Generates fully synthetic scenes (nuclei + Voronoi cell territories +
per-cell type assignments) that share the rendering and refining pipeline
with the observed-mask path. This is the first test of whether the Plan3
realism transfers beyond observed geometry — the simulator question.

The sampler is deliberately minimal: nuclei via inhibited point process,
weighted Voronoi territories from nuclei with random per-cell area weights,
optional smooth elastic boundary warp, per-cell type sampled from real-data
frequencies. No partial off-frame cells in this first version.

Plan3 principle: every synthetic cell is recorded as ``MechanisticCell``
with ``source='synthetic'`` and a ``cell_type``; downstream evaluators
can audit them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mechanistic_render import _gaussian_blur
from .mechanistic_scene import MechanisticCell, MechanisticScene, validate_mechanistic_scene


@dataclass(frozen=True)
class SyntheticSceneConfig:
    image_shape: tuple[int, int] = (302, 302)
    pixel_size: float = 0.2125
    target_num_cells: int = 80
    nucleus_min_radius_um: float = 1.4
    nucleus_max_radius_um: float = 4.0
    cell_area_weight_log_sigma: float = 0.55
    nucleus_aspect_log_sigma: float = 0.30
    boundary_warp_sigma_px: float = 8.0
    boundary_warp_strength_px: float = 2.5
    inhibition_radius_um: float = 3.0
    seed: int = 1
    type_frequencies: dict[str, float] | None = None  # name -> probability
    # Tissue background: fraction of the image that gets no cell coverage.
    # Real Xenium pancreas crops have about 20-35% background-like pixels.
    background_fraction: float = 0.25
    background_field_sigma_px: float = 35.0
    # Per-type spatial clustering: types listed here are sampled in tight
    # spatial clumps (islets, acini) instead of being spread uniformly.
    cluster_types: tuple[str, ...] = (
        "Endocrine", "Exocrine epithelial", "Immune", "Fibroblast / CAF", "Endothelial",
    )
    cluster_sigma_px: float = 18.0
    cluster_size_range: tuple[int, int] = (4, 14)
    # Per-type spatial cluster scale in pixels (assumes pixel_size ~ 0.21 um/px,
    # giving 1um ~ 4.7 px). Approximate biology:
    #   Endocrine    -> islets, 80-200um diameter -> sigma ~ 35-50 px clumps
    #   Exocrine     -> acini, 40-100um -> sigma ~ 18-25 px
    #   Ductal       -> linear ducts, 20-60um -> sigma ~ 10-15 px
    #   Immune       -> small clusters or scattered, sigma ~ 8 px
    #   Endothelial  -> vessels (linear), small clumps, sigma ~ 8 px
    #   Fibroblast   -> diffuse stromal, sigma ~ 15 px
    #   Mural        -> peri-vascular small clumps, sigma ~ 8 px
    cluster_sigma_per_type: dict[str, float] | None = None
    cluster_size_per_type: dict[str, tuple[int, int]] | None = None
    # Plan3 A1v2: tissue-domain-first type assignment. When ``domain_mode``
    # is True, sample coarse domain regions first (random Voronoi
    # tessellation), assign a primary type per domain weighted by
    # ``type_frequencies``, then per-cell type = domain primary with
    # probability ``domain_purity`` else random by frequency. Captures
    # above-cell-scale tissue architecture (acini / islets / vessel beds /
    # stroma) instead of just per-type local clustering.
    domain_mode: bool = False
    domain_mean_radius_px: float = 60.0
    domain_purity: float = 0.85
    # Plan3 B11: scale up target cell area for exemplar stamp by this factor
    # to compensate for first-write-wins overlap clipping + bg_mask erosion.
    # Empirically observed-area/target ~0.55 (real stats: 835 px target ~1000),
    # so 1.5-1.7 overscale lands on real mean.
    exemplar_target_area_overscale: float = 1.5
    # Plan3 Cycle B: optional real-cell-center pattern library. When set,
    # the sampler picks one observed tile's per-cell centers (yi, xi) and
    # uses them for the scene instead of inhibited-Poisson sampling. The
    # observed pattern carries real packing structure (NN distance, regular
    # spacing, close-pair statistics) that hard-core inhibition can't
    # reproduce — real cells pack tightly into tissue with characteristic
    # spacing. Augmentation: random flip, optional translation.
    cell_pattern_library_path: "Path | None" = None
    cell_pattern_jitter_px: float = 0.0
    # Plan3 v37 cleanup: binary-opening + closing iters for nucleus/cell
    # boundary smoothing. Real Xenium perim/sqrt(area) ~ 3.5; raw stamp+
    # warp output ~ 4.5-4.8 (too jagged). 1 iter reduces by ~0.7 toward
    # real without merging adjacent cells. 0 = disabled (legacy path).
    boundary_smooth_iters: int = 1
    # Plan3 v37 variety: per-tile type composition library. When set,
    # `type_frequencies` is sampled per scene from a library of real-tile
    # compositions instead of using a global average. Real Xenium pancreas
    # tiles vary 0-66% Exocrine / 0-31% Ductal / 3-39% Immune per tile;
    # using a global average makes every synth scene look the same
    # (mostly Exocrine). Library is built by `xesim build-tile-compositions`.
    tile_composition_library_path: "Path | None" = None
    # Plan3 v37 + plan3d: when set, replaces exemplar-stamped nucleus
    # shapes with clean ellipses sized + oriented by plan3d's bias-corrected
    # per-type 2D nucleus posterior priors (area + axis_ratio per type).
    # Addresses "jagged/irregular nuclei" complaint by producing smooth
    # ellipses that follow per-type biological priors instead of the
    # exemplar bank's segmentation noise.
    nucleus_prior_path: "Path | None" = None


def _coarse_field_to_log_prior(coarse_field: np.ndarray,
                                  centers: np.ndarray,
                                  image_shape: tuple[int, int],
                                  type_names: list[str]) -> np.ndarray:
    """Sample a (n_cells, n_types) per-cell log-prior from a coarse field.

    coarse_field: (T_field, K, K) — fractions for cell types 1..T_field
      (no background channel). T_field matches len(type_names) - 1 (type
      0 = 'unknown' has no entry in the field).
    centers: (n_cells, 2) cell centroids in pixel coords.
    Returns (n_cells, n_types) log-probabilities, normalized to sum to 1
    per cell (in linear space). The "unknown" / type-0 entry is given
    log_prior = uniform / negligible.
    """
    import torch
    import torch.nn.functional as F
    H, W = image_shape
    T_field, K_h, K_w = coarse_field.shape
    n_types = len(type_names)
    # Upsample to full resolution via bilinear (T_field, H, W)
    cf = torch.from_numpy(coarse_field).unsqueeze(0).float()
    up = F.interpolate(cf, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()
    # Per-cell row by sampling at centroid
    n_cells = int(centers.shape[0])
    out = np.full((n_cells, n_types), 1.0 / n_types, dtype=np.float64)
    for ci, (cy, cx) in enumerate(centers):
        cyi = int(np.clip(round(cy), 0, H - 1))
        cxi = int(np.clip(round(cx), 0, W - 1))
        vals = up[:, cyi, cxi]  # (T_field,)
        # Place at type indices 1..T_field+1. Type 0 is the residual.
        if T_field == n_types - 1:
            # Slot fractions into types 1..n_types-1; type 0 = "unknown" gets residual
            row = np.zeros(n_types, dtype=np.float64)
            row[0] = max(0.0, 1.0 - float(vals.sum()))  # residual = extracellular-ish
            row[1:] = vals
        else:
            row = vals
        # Normalize and convert to log; floor at small epsilon for stability
        row = row / max(row.sum(), 1e-9)
        row = np.clip(row, 1e-6, None)
        out[ci] = np.log(row)
    return out


def sample_synthetic_scene(
    config: SyntheticSceneConfig,
    type_names: list[str] | None = None,
    per_type_efficiencies: dict[str, dict[str, float]] | None = None,
    scene_id: str = "synthetic_scene",
    cell_shape_sampler: "Any" = None,
    tissue_neighborhood: dict[str, "Any"] | None = None,
    sectioning_states: dict[str, "Any"] | None = None,
    cell_shape_exemplar_sampler: "Any" = None,
    coarse_field: np.ndarray | None = None,
    coarse_field_weight: float = 1.0,
    position_log_prior_map: np.ndarray | None = None,
    position_log_prior_weight: float = 1.0,
) -> MechanisticScene:
    """Sample a Plan3-compliant synthetic scene as a ``MechanisticScene``.

    Returns a scene with cell/nucleus label arrays, per-cell records carrying
    ``source='synthetic'``, ``cell_type``, and per-type efficiencies applied
    when ``per_type_efficiencies`` is provided.
    """

    rng = np.random.default_rng(config.seed)
    h, w = config.image_shape
    # Optionally sample per-scene type_frequencies from a real-tile
    # composition library (replaces config.type_frequencies for this scene).
    effective_type_frequencies = config.type_frequencies
    if config.tile_composition_library_path is not None:
        try:
            with open(config.tile_composition_library_path) as f:
                lib = json.load(f)
            per_tile = lib.get("per_tile_freq", [])
            if per_tile:
                effective_type_frequencies = per_tile[rng.integers(len(per_tile))]
        except Exception:
            pass  # fall back to global config.type_frequencies
    # 1a. Sample tissue-background field: smooth Gaussian field that defines
    #     which regions of the image will be left as background (no cells).
    bg_field = _gaussian_blur(
        rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32),
        max(2.0, float(config.background_field_sigma_px)),
    )
    bg_field = (bg_field - float(np.mean(bg_field))) / max(float(np.std(bg_field)), 1e-6)
    # Threshold so the background covers approximately the requested fraction.
    bg_threshold = float(np.quantile(bg_field, 1.0 - max(0.01, min(0.85, config.background_fraction))))
    bg_mask = bg_field > bg_threshold  # True = background pixel
    # 1b. Sample nucleus centers, with rejection inside background regions.
    inhibit_px = max(1.0, config.inhibition_radius_um / max(config.pixel_size, 1e-6))
    centers: np.ndarray
    if config.cell_pattern_library_path is not None:
        # Plan3 Cycle B: clone a real tile's cell centers to inherit real
        # packing patterns (NN distance, close-pair stats, regularity).
        # Skip bg_mask filter — the real pattern already encodes where
        # tissue / cells exist; rebuild bg_mask from the pattern instead
        # so the renderer's tissue-gap regions match the inherited layout.
        centers = _sample_centers_from_real_pattern(
            rng, h, w, config.cell_pattern_library_path,
            target_num=config.target_num_cells,
            jitter_px=float(config.cell_pattern_jitter_px),
            bg_mask=None,
        )
        # Rebuild bg_mask to be everything farther than ~half the local
        # cell-cell spacing from any center — preserves the natural cell
        # gaps the pattern already implies.
        if len(centers) >= 5:
            from scipy.spatial import cKDTree
            yy, xx = np.indices((h, w))
            grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
            tree = cKDTree(centers)
            d_grid, _ = tree.query(grid, k=1)
            d_grid = d_grid.reshape(h, w)
            # Use median NN distance × 1.5 as the tissue radius around each cell.
            d_cells, _ = tree.query(centers, k=2)
            radius = float(np.median(d_cells[:, 1])) * 1.5 if len(d_cells) >= 2 else 30.0
            bg_mask = d_grid > radius
    else:
        centers = _sample_inhibited_points(
            rng, h, w, config.target_num_cells, inhibit_px,
            rejection_mask=bg_mask,
        )
    if centers.size == 0:
        raise RuntimeError("synthetic sampler produced no nucleus centers")
    # 2. Per-cell type sampling, with spatial clustering for the configured types.
    #    Sample types FIRST so per-cell shape can be drawn from the type's
    #    distribution (F3). Per-type spatial scale (F5) reflects pancreas
    #    biology (islets larger than acini, fibroblasts diffuse, immune
    #    scattered, etc.).
    if config.domain_mode and type_names:
        # A1v2 path: tissue-domain-first sampler. Sample coarse domain
        # regions, draw a primary type per domain, then per-cell type =
        # domain primary with probability ``domain_purity``. Captures
        # above-cell-scale architecture (acini / islets / vessels / stroma)
        # the local-cluster Gibbs cannot represent.
        sampled_types = _sample_domain_first_types(
            rng,
            centers,
            type_names,
            effective_type_frequencies,
            mean_domain_radius_px=float(config.domain_mean_radius_px),
            purity=float(config.domain_purity),
            image_shape=(h, w),
        )
        # Optional refinement step: if both domain_mode and tissue_neighborhood
        # are provided, run a few Gibbs sweeps over the domain assignment to
        # restore neighbor-conditional realism within domains.
        if tissue_neighborhood is not None:
            from .tissue_neighborhood import refine_assignment_gibbs

            try:
                sampled_types = refine_assignment_gibbs(
                    centers,
                    initial_types=sampled_types,
                    type_names=list(tissue_neighborhood["type_names"]),
                    type_prior=np.asarray(tissue_neighborhood["type_prior"], dtype=np.float64),
                    neighbor_conditional=np.asarray(tissue_neighborhood["neighbor_conditional"], dtype=np.float64),
                    n_neighbors=int(tissue_neighborhood.get("n_neighbors", 8)),
                    n_iterations=2,
                    seed=int(config.seed) + 7,
                )
            except (ImportError, AttributeError):
                # Fallback: keep domain assignment if refine helper is absent.
                pass
    elif tissue_neighborhood is not None and type_names:
        # A1 path: empirical neighbor-conditional Gibbs sampling. Replaces the
        # heuristic per-type spatial-cluster sampler with a tissue-architecture
        # model fitted from real per-type-pair neighborhoods.
        from .tissue_neighborhood import gibbs_assign_types

        prior = np.asarray(tissue_neighborhood["type_prior"], dtype=np.float64)
        cond = np.asarray(tissue_neighborhood["neighbor_conditional"], dtype=np.float64)
        # Per-cell position prior. Two sources:
        #   1. Explicit `position_log_prior_map` (T, H, W) — slice-guided synth.
        #      Sampled at each cell's centroid → per-cell row.
        #   2. Derived from `coarse_field` (Track A path).
        # If both are given, the explicit map wins.
        tn_type_names = list(tissue_neighborhood["type_names"])
        if position_log_prior_map is not None:
            pmap = np.asarray(position_log_prior_map, dtype=np.float64)
            n_cells_here = int(centers.shape[0])
            pos_log_prior_local = np.zeros((n_cells_here, pmap.shape[0]), dtype=np.float64)
            for ci, (cy, cx) in enumerate(centers):
                cyi = int(np.clip(round(cy), 0, pmap.shape[1] - 1))
                cxi = int(np.clip(round(cx), 0, pmap.shape[2] - 1))
                pos_log_prior_local[ci] = pmap[:, cyi, cxi]
            pos_log_prior_weight_local = float(position_log_prior_weight)
        elif coarse_field is not None:
            pos_log_prior_local = _coarse_field_to_log_prior(
                coarse_field, centers, (h, w), tn_type_names)
            pos_log_prior_weight_local = float(coarse_field_weight)
        else:
            pos_log_prior_local = None
            pos_log_prior_weight_local = 1.0
        sampled_types = gibbs_assign_types(
            centers,
            type_names=tn_type_names,
            type_prior=prior,
            neighbor_conditional=cond,
            n_neighbors=int(tissue_neighborhood.get("n_neighbors", 8)),
            n_iterations=5,
            seed=int(config.seed),
            type_frequencies=effective_type_frequencies,
            position_log_prior=pos_log_prior_local,
            position_prior_weight=pos_log_prior_weight_local,
            # Skip rebalance when position prior provides spatial structure
            # we want to preserve (the rebalance would relabel cells to
            # match the marginal regardless of position).
            rebalance_to_prior=(position_log_prior_map is None),
        )
    else:
        sigma_per_type = dict(config.cluster_sigma_per_type or _DEFAULT_CLUSTER_SIGMA_PER_TYPE_PX)
        size_per_type = dict(config.cluster_size_per_type or _DEFAULT_CLUSTER_SIZE_PER_TYPE)
        sampled_types = _sample_clustered_types(
            rng,
            centers,
            type_names,
            effective_type_frequencies,
            cluster_types=tuple(config.cluster_types or ()),
            cluster_sigma_px=float(config.cluster_sigma_px),
            cluster_size_range=tuple(config.cluster_size_range or (4, 14)),
            sigma_per_type_px=sigma_per_type,
            size_per_type=size_per_type,
        )
    # 3. Per-cell shape sampling: for cells with type-aware shape moments in
    #    ``per_type_efficiencies``, draw cell area, aspect ratio, and
    #    orientation from the type's distribution. Otherwise fall back to the
    #    legacy Uniform(min, max) radius sampling.
    cell_axes_a, cell_axes_b, cell_thetas = _sample_per_type_shape(
        rng,
        sampled_types,
        per_type_efficiencies or {},
        config,
    )
    # 3b. Plan3 D1: sample sectioning state per cell ONCE so the exemplar
    #     bucket and the per-cell construction loop see the same state.
    #     Without this they double-sample, with state mismatch dropping
    #     correctly-stamped nuclei (regressing nucleus_to_cell_ratio).
    sampled_section_states: list[str | None] = [None] * len(centers)
    sampled_state_params: list[dict[str, "Any"] | None] = [None] * len(centers)
    if sectioning_states is not None:
        from .sectioning_states import sample_state_for_type as _sample_state

        for idx, type_name in enumerate(sampled_types):
            if type_name is None:
                continue
            state, params = _sample_state(rng, type_name, sectioning_states)
            sampled_section_states[idx] = state
            sampled_state_params[idx] = params
    # 4. Build cell territories. Three modes:
    #    - exemplar (A2-exemplar): stamp real (cell_mask, nucleus_mask) pairs
    #      from a per-type real-mask library. Joint cell+nucleus geometry
    #      directly from real Xenium segmentations. Highest realism.
    #    - VAE (A2): decode per-cell mask from a per-type cell-shape VAE,
    #      then SDF-Voronoi assign. Cell-mask only; nucleus rasterized
    #      independently.
    #    - default: oriented-ellipse SDF Voronoi.
    nucleus_label_from_exemplar: np.ndarray | None = None
    if cell_shape_exemplar_sampler is not None:
        # Compute target areas per cell (px): use per-type cell_area_um2_mean
        # × random log-normal scatter, converted to pixels.
        target_areas_px: list[float] = []
        # Plan3 B11: stamp clipping (first-write-wins overlap) and bg_mask
        # boundary erosion shrink the rendered cell area below the target.
        # Empirically real-vs-synth cell area ratio is ~1.5x → over-target
        # by 1.5x so the rendered mean ends up at the per-type real mean.
        target_area_overscale = float(config.exemplar_target_area_overscale)
        for idx, type_name in enumerate(sampled_types):
            eff = (per_type_efficiencies or {}).get(type_name or "", {})
            mean_log = float(np.log(max(float(eff.get("cell_area_um2_mean", 50.0)), 1.0)))
            std = float(eff.get("cell_area_um2_log_std", 0.5))
            sample_um2 = float(np.exp(rng.normal(mean_log, std))) * target_area_overscale
            target_areas_px.append(sample_um2 / max(float(config.pixel_size) ** 2, 1e-6))
        # Load plan3d nucleus priors when provided.
        nucleus_priors = None
        if config.nucleus_prior_path is not None:
            try:
                with open(config.nucleus_prior_path) as f:
                    np_payload = json.load(f)
                nucleus_priors = np_payload.get("priors", {})
            except Exception:
                nucleus_priors = None
        cell_label, nucleus_label_from_exemplar = _exemplar_stamp_voronoi(
            h, w, centers, sampled_types, sampled_section_states,
            cell_shape_exemplar_sampler, target_areas_px, rng,
            nucleus_priors=nucleus_priors,
            pixel_size_um=float(config.pixel_size),
        )
    elif cell_shape_sampler is not None:
        cell_label = _vae_mask_voronoi(
            h, w, centers, cell_axes_a, cell_axes_b, cell_thetas,
            sampled_types, cell_shape_sampler, rng,
        )
    else:
        cell_label = _ellipse_sdf_voronoi(h, w, centers, cell_axes_a, cell_axes_b, cell_thetas)
    cell_label[bg_mask] = 0
    # 5. Per-cell nucleus shapes. With exemplar mode we already have a
    #    paired nucleus_label from the stamp; otherwise sample radii +
    #    aspects and rasterize independent ellipses.
    if nucleus_label_from_exemplar is not None:
        nucleus_label = nucleus_label_from_exemplar
        nucleus_label[bg_mask] = 0
    else:
        nuc_radii_px = _sample_nucleus_radii(
            rng, sampled_types, per_type_efficiencies or {}, config, cell_axes_a, cell_axes_b
        )
        nuc_aspects = np.exp(rng.normal(0.0, config.nucleus_aspect_log_sigma, size=len(centers)))
        nuc_thetas = cell_thetas  # nuclei share cell orientation
        nucleus_label = _rasterize_ellipses(h, w, centers, nuc_radii_px, nuc_aspects, nuc_thetas)
    nucleus_label[cell_label == 0] = 0
    # Plan3 v34 forward-mode bug fix: nucleus rasterization (both ellipse
    # and exemplar paths) can leave nucleus pixels that bleed across
    # adjacent cell boundaries when cells are tightly packed. Since both
    # nucleus_label and cell_label use the same 1-based idx, nucleus
    # pixels are valid only where ``nucleus_label == cell_label`` —
    # mismatch indicates cross-cell spillover that creates DAPI bridge
    # artifacts. Applies to both the exemplar-stamped path and the
    # ellipse-rasterized path.
    nucleus_label = np.where(nucleus_label == cell_label, nucleus_label, 0).astype(np.int32)
    # Plan3 v34 forward-mode bug fix: filter slivers / sub-cell-area cells
    # produced by tight packing. A cell with area < 30 px (~6 µm²) is
    # smaller than a real nucleus — these aren't real cells, they're
    # leftover Voronoi remainders. Audit on 8 synth scenes showed 12% of
    # all rasterized cells fell below this threshold. Drop them.
    min_cell_area_px = 30
    for lab in np.unique(cell_label):
        if int(lab) == 0:
            continue
        cell_pix = cell_label == int(lab)
        if int(cell_pix.sum()) < min_cell_area_px:
            cell_label[cell_pix] = 0
            nucleus_label[nucleus_label == int(lab)] = 0
    # 6. Optional elastic warp.
    if config.boundary_warp_strength_px > 0.0:
        cell_label = _elastic_warp_label(rng, cell_label, config.boundary_warp_sigma_px, config.boundary_warp_strength_px)
        nucleus_label = _elastic_warp_label(rng, nucleus_label, config.boundary_warp_sigma_px, config.boundary_warp_strength_px)
        # Re-align: independent warps of cell_label and nucleus_label can
        # offset them slightly, putting nucleus pixels outside their own
        # cell again. Re-mask to enforce nucleus ⊆ same-id cell.
        nucleus_label = np.where(nucleus_label == cell_label, nucleus_label, 0).astype(np.int32)
        # Sliver filter must also run post-warp: warp can create new
        # tiny disconnected fragments at cell boundaries.
        for lab in np.unique(cell_label):
            if int(lab) == 0:
                continue
            cp_w = cell_label == int(lab)
            if int(cp_w.sum()) < 30:
                cell_label[cp_w] = 0
                nucleus_label[nucleus_label == int(lab)] = 0
    # 6b. Smooth boundaries to match real nucleus/cell jaggedness. Real
    # Xenium segmentation perimeter/sqrt(area) ~ 3.5 (cell) / 3.5 (nuc);
    # exemplar stamp + elastic warp produces ~4.8 / 4.2 — too crinkly.
    # Single iteration of binary opening + closing per-label flattens
    # boundary stipple without changing area much. Per-label so adjacent
    # cells don't merge.
    if config.boundary_smooth_iters > 0:
        from scipy.ndimage import binary_opening, binary_closing
        for arr in (cell_label, nucleus_label):
            for lab in np.unique(arr):
                if int(lab) == 0:
                    continue
                m = arr == int(lab)
                if m.sum() < 30:
                    continue
                m2 = binary_closing(binary_opening(m, iterations=config.boundary_smooth_iters),
                                     iterations=config.boundary_smooth_iters)
                arr[m & ~m2] = 0  # remove jagged-out pixels
                # don't add new pixels (would risk overwriting neighbors)
        # Re-align nucleus ⊆ cell after smoothing.
        nucleus_label = np.where(nucleus_label == cell_label, nucleus_label, 0).astype(np.int32)
    cells: list[MechanisticCell] = []
    used_labels: set[int] = set()
    # Plan3 D1: cells whose sampled sectioning state lacks a nucleus get
    # their nucleus pixels stripped from `nucleus_label` post-loop. We
    # collect the labels here so the strip is one masked numpy op below.
    drop_nucleus_labels: list[int] = []
    for idx in range(len(centers)):
        label_value = int(idx + 1)
        if not np.any(cell_label == label_value):
            continue
        nucleus_drawn = bool(np.any((nucleus_label == label_value) & (cell_label == label_value)))
        type_name = sampled_types[idx] if sampled_types else None
        eff = (per_type_efficiencies or {}).get(type_name or "", {})
        d_eff = float(eff.get("dapi_efficiency", 1.0))
        m_eff = float(eff.get("membrane_efficiency", 1.0))
        p_eff = float(eff.get("polya_efficiency", 1.0))
        # Plan3 B8: per-type intensity log-normal scatter. Real per-cell
        # mean intensity has substantial within-type variance (B7 surfaced
        # synth std ~3× narrower than real). Multiply per-cell efficiency
        # by exp(N(0, log_std)). Defaults to 0 when log_std missing so
        # legacy params keep deterministic per-type behavior.
        d_log_std = float(eff.get("dapi_efficiency_log_std", 0.0))
        m_log_std = float(eff.get("membrane_efficiency_log_std", 0.0))
        p_log_std = float(eff.get("polya_efficiency_log_std", 0.0))
        if d_log_std > 0.0:
            d_eff *= float(np.exp(rng.normal(0.0, d_log_std)))
        if m_log_std > 0.0:
            m_eff *= float(np.exp(rng.normal(0.0, m_log_std)))
        if p_log_std > 0.0:
            p_eff *= float(np.exp(rng.normal(0.0, p_log_std)))
        memb_width = eff.get("membrane_width_px")
        memb_dropout = eff.get("membrane_dropout_keep_prob")
        memb_interior = eff.get("membrane_interior_fraction")
        dapi_texture = eff.get("dapi_texture_scale")
        polya_texture = eff.get("polya_texture_scale")
        # Plan3 D1: re-use the sectioning state we already sampled for this
        # cell at step 3b (shared with the exemplar bucket lookup). Falls
        # back to the legacy binary full/nucleus_poor based on whether
        # geometry placed a nucleus.
        pre_state = sampled_section_states[idx] if idx < len(sampled_section_states) else None
        pre_params = sampled_state_params[idx] if idx < len(sampled_state_params) else None
        if pre_state is not None and pre_params is not None:
            state = pre_state
            nucleus_present = bool(pre_params.get("nucleus_present", True)) and nucleus_drawn
            visibility_fraction = float(pre_params.get("visibility_fraction", 1.0))
        else:
            state = "full" if nucleus_drawn else "nucleus_poor"
            nucleus_present = nucleus_drawn
            visibility_fraction = 1.0
        if not nucleus_present:
            drop_nucleus_labels.append(label_value)
        cells.append(
            MechanisticCell(
                cell_id=f"synthetic_{idx:05d}",
                label=label_value,
                source="synthetic",
                nucleus_label=label_value if nucleus_present else None,
                sectioning_state=state,
                visibility_fraction=visibility_fraction,
                cell_type=type_name,
                dapi_efficiency=d_eff,
                membrane_efficiency=m_eff,
                polya_efficiency=p_eff,
                membrane_width_px=float(memb_width) if memb_width is not None else None,
                membrane_dropout_keep_prob=float(memb_dropout) if memb_dropout is not None else None,
                membrane_interior_fraction=float(memb_interior) if memb_interior is not None else None,
                dapi_texture_scale=float(dapi_texture) if dapi_texture is not None else None,
                polya_texture_scale=float(polya_texture) if polya_texture is not None else None,
            )
        )
        used_labels.add(label_value)
    if drop_nucleus_labels:
        drop_set = set(drop_nucleus_labels)
        nucleus_label = np.where(
            np.isin(nucleus_label, list(drop_set)),
            0,
            nucleus_label,
        ).astype(np.int32)
    # Drop any nucleus-only artifacts (from warp) whose cell label disappeared.
    nucleus_label = np.where(np.isin(nucleus_label, list(used_labels) + [0]), nucleus_label, 0).astype(np.int32)
    cell_label = np.where(np.isin(cell_label, list(used_labels) + [0]), cell_label, 0).astype(np.int32)
    scene = MechanisticScene(
        image_shape=(h, w),
        pixel_size=float(config.pixel_size),
        cell_label=cell_label,
        nucleus_label=nucleus_label,
        cells=tuple(cells),
        scene_id=scene_id,
        provenance={
            "sampler": "synthetic_voronoi_v0",
            "config": {
                "target_num_cells": int(config.target_num_cells),
                "inhibition_radius_um": float(config.inhibition_radius_um),
                "nucleus_radius_um_range": [float(config.nucleus_min_radius_um), float(config.nucleus_max_radius_um)],
                "boundary_warp_sigma_px": float(config.boundary_warp_sigma_px),
                "boundary_warp_strength_px": float(config.boundary_warp_strength_px),
                "seed": int(config.seed),
            },
        },
    )
    validate_mechanistic_scene(scene)
    return scene


_DEFAULT_CLUSTER_SIGMA_PER_TYPE_PX = {
    "Endocrine": 45.0,
    "Exocrine epithelial": 22.0,
    "Ductal/tumor epithelial": 14.0,
    "Immune": 8.0,
    "Endothelial": 8.0,
    "Fibroblast / CAF": 15.0,
    "Mural / pericyte": 8.0,
}

_DEFAULT_CLUSTER_SIZE_PER_TYPE = {
    "Endocrine": (8, 25),       # tight islet clusters
    "Exocrine epithelial": (4, 12),
    "Ductal/tumor epithelial": (3, 10),
    "Immune": (2, 8),
    "Endothelial": (2, 6),
    "Fibroblast / CAF": (2, 10),
    "Mural / pericyte": (2, 5),
}


def _sample_per_type_shape(
    rng: np.random.Generator,
    sampled_types: list[str],
    per_type_efficiencies: dict[str, dict[str, float]],
    config: SyntheticSceneConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-cell ellipse semi-axes (a, b, in pixels) and orientation from per-type
    shape moments. Falls back to the legacy uniform radius range when no type info
    is available. ``a`` is the long-axis, ``b`` the short-axis.
    """

    n = len(sampled_types)
    a_axes = np.zeros(n, dtype=np.float32)
    b_axes = np.zeros(n, dtype=np.float32)
    thetas = np.zeros(n, dtype=np.float32)
    px = max(float(config.pixel_size), 1e-6)
    for i, type_name in enumerate(sampled_types):
        record = per_type_efficiencies.get(type_name or "", {})
        # Cell area in µm², log-normal.
        if "cell_area_um2_mean" in record:
            log_mu = math.log(max(float(record["cell_area_um2_mean"]), 1.0))
            log_sd = float(record.get("cell_area_um2_log_std", 0.30))
            area_um2 = float(np.exp(rng.normal(log_mu, log_sd)))
            area_um2 = max(15.0, min(area_um2, 800.0))
        else:
            r_um = float(rng.uniform(config.nucleus_min_radius_um * 2.5, config.nucleus_max_radius_um * 2.5))
            area_um2 = math.pi * r_um * r_um
        if "cell_aspect_log_mean" in record:
            asp_mu = float(record["cell_aspect_log_mean"])
            asp_sd = float(record.get("cell_aspect_log_std", 0.30))
            aspect = float(np.exp(rng.normal(asp_mu, asp_sd)))
            aspect = max(1.0, min(aspect, 5.0))
        else:
            aspect = float(np.exp(rng.normal(0.0, config.cell_area_weight_log_sigma)))
        # Convert area + aspect ratio into semi-axes (a >= b) in pixels.
        # area = pi * a * b; aspect = a / b => b = sqrt(area / (pi * aspect)),
        # a = aspect * b.
        b_um = math.sqrt(area_um2 / (math.pi * aspect))
        a_um = aspect * b_um
        a_axes[i] = float(a_um / px)
        b_axes[i] = float(b_um / px)
        thetas[i] = float(rng.uniform(0.0, math.pi))
    return a_axes, b_axes, thetas


def _ellipse_mask(shape: tuple[int, int], cy: float, cx: float,
                   radius_y: float, radius_x: float, theta: float) -> np.ndarray:
    """Filled-ellipse boolean mask at (cy, cx) with given radii and rotation."""
    h, w = shape
    yy, xx = np.indices((h, w))
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    dy = yy - cy; dx = xx - cx
    yr = dy * cos_t + dx * sin_t
    xr = -dy * sin_t + dx * cos_t
    return (yr / max(radius_y, 0.5))**2 + (xr / max(radius_x, 0.5))**2 <= 1.0


def _exemplar_stamp_voronoi(
    h: int, w: int,
    centers: np.ndarray,
    sampled_types: list[str],
    exemplar_states: list[str | None],
    exemplar_sampler: "Any",
    target_areas_px: list[float],
    rng: np.random.Generator,
    nucleus_priors: dict | None = None,
    pixel_size_um: float = 0.2125,
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp real-mask exemplars at synthetic centers, producing paired
    cell + nucleus label arrays. First-stamped cell wins on overlapping
    pixels; the per-cell exemplar nucleus is added only inside that cell.
    """

    from scipy.ndimage import zoom

    cell_label = np.zeros((h, w), dtype=np.int32)
    nucleus_label = np.zeros((h, w), dtype=np.int32)
    n_centers = len(centers)
    if n_centers == 0:
        return cell_label, nucleus_label
    # Stamp in random order so no spatial bias from iteration order.
    order = list(rng.permutation(n_centers))
    for idx in order:
        cy, cx = centers[idx]
        type_name = sampled_types[idx] if sampled_types else None
        state = exemplar_states[idx] if exemplar_states else None
        if type_name is None:
            continue
        cell_em, nuc_em, _info = exemplar_sampler.sample(rng, type_name, state)
        em_area = max(int(cell_em.sum()), 1)
        target_area = max(float(target_areas_px[idx]), 16.0)
        scale = float(np.sqrt(target_area / em_area))
        # Clamp scale to a reasonable range so noisy area moments don't
        # produce a single 1-px or 1000-px stamp.
        scale = float(np.clip(scale, 0.4, 2.5))
        # Resize via nearest-neighbor zoom.
        cell_scaled = zoom(cell_em, zoom=(scale, scale), order=0)
        nuc_scaled = zoom(nuc_em, zoom=(scale, scale), order=0)
        sh, sw = cell_scaled.shape
        # Center the stamp on (cy, cx).
        y0 = int(round(float(cy))) - sh // 2
        x0 = int(round(float(cx))) - sw // 2
        y1 = y0 + sh
        x1 = x0 + sw
        # Clip to image.
        ay0 = max(0, -y0)
        ax0 = max(0, -x0)
        by0 = max(0, y0)
        bx0 = max(0, x0)
        by1 = min(h, y1)
        bx1 = min(w, x1)
        if by1 <= by0 or bx1 <= bx0:
            continue
        cell_clip = cell_scaled[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)]
        nuc_clip = nuc_scaled[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)]
        # First-write-wins: only stamp where currently free.
        region = cell_label[by0:by1, bx0:bx1]
        free = (region == 0) & (cell_clip > 0)
        region[free] = int(idx + 1)
        cell_label[by0:by1, bx0:bx1] = region
        # Nucleus: plan3d-prior-driven ellipse if priors available,
        # otherwise the exemplar's paired nucleus.
        if nucleus_priors is not None and type_name in nucleus_priors:
            prior = nucleus_priors[type_name]
            # Sample area + axis_ratio from per-type prior quantiles.
            # Log-normal on radius -> area = pi * r^2; we sample log-area
            # via mean/std of the area distribution.
            area_p10 = float(prior["target_area_quantiles"]["p10"])
            area_p50 = float(prior["target_area_quantiles"]["p50"])
            area_p90 = float(prior["target_area_quantiles"]["p90"])
            # Fit log-normal: ln_p50 ~ mean, (ln_p90 - ln_p10) / 2.563 ~ std
            log_mean = float(np.log(max(area_p50, 1.0)))
            log_std = max(float((np.log(max(area_p90, 1.0)) - np.log(max(area_p10, 1.0))) / 2.563), 0.05)
            area_um2 = float(np.exp(rng.normal(log_mean, log_std)))
            ar_p10 = float(prior["target_axis_ratio_quantiles"]["p10"])
            ar_p50 = float(prior["target_axis_ratio_quantiles"]["p50"])
            ar_p90 = float(prior["target_axis_ratio_quantiles"]["p90"])
            log_ar_mean = float(np.log(max(ar_p50, 1.0)))
            log_ar_std = max(float((np.log(max(ar_p90, 1.0)) - np.log(max(ar_p10, 1.0))) / 2.563), 0.05)
            axis_ratio = float(np.clip(np.exp(rng.normal(log_ar_mean, log_ar_std)), 1.0, 3.0))
            # Convert area in um^2 to pixel radius: area_px = area_um2 / px^2,
            # area = pi*r_a*r_b, axis_ratio = r_a/r_b -> r_b = sqrt(area/(pi*ar))
            area_px = area_um2 / max(pixel_size_um ** 2, 1e-6)
            r_b = float(np.sqrt(max(area_px / max(np.pi * axis_ratio, 1e-3), 1.0)))
            r_a = float(axis_ratio * r_b)
            theta = float(rng.uniform(0, np.pi))
            mask = _ellipse_mask((h, w), float(cy), float(cx), r_a, r_b, theta)
            nuc_in_my_cell = mask & (cell_label == int(idx + 1))
            nucleus_label[nuc_in_my_cell] = int(idx + 1)
        else:
            # Fallback: exemplar-stamped nucleus.
            nuc_region = nucleus_label[by0:by1, bx0:bx1]
            nuc_in_my_cell = (cell_label[by0:by1, bx0:bx1] == int(idx + 1)) & (nuc_clip > 0)
            nuc_region[nuc_in_my_cell] = int(idx + 1)
            nucleus_label[by0:by1, bx0:bx1] = nuc_region
    # Plan3 B12: stamp clipping (first-write-wins overlap) drops a fraction
    # of intended-nucleus pixels when the exemplar nucleus falls in a
    # neighboring cell's territory. Result: many cells with intended nuclei
    # ended up nucleus-less, biasing per-cell N/C ratio low. Salvage these
    # by giving each cell a fallback nucleus = its 2-px erosion when the
    # exemplar nucleus area dropped below 30% of the per-cell expected
    # area. This restores intended-nucleus cells without re-running the
    # stamp loop.
    from scipy.ndimage import binary_erosion

    expected_nuc_frac_of_cell = 0.40
    for idx in range(n_centers):
        label_value = int(idx + 1)
        cell_mask_idx = cell_label == label_value
        cell_area = int(cell_mask_idx.sum())
        if cell_area < 16:
            continue
        nuc_mask_idx = nucleus_label == label_value
        nuc_area = int(nuc_mask_idx.sum())
        # Salvage if intended (got stamped at all in nuc_label) AND lost
        # most of it to clipping. We detect "intended" by checking whether
        # ANY of the original exemplar nucleus pixels (nuc_clip) ended up
        # written for this cell; the post-stamp signal is `nuc_area > 0`.
        if nuc_area > 0 and nuc_area >= int(cell_area * expected_nuc_frac_of_cell * 0.5):
            continue  # nucleus is fine
        if nuc_area > 0:
            # Some nucleus survived; expand it to ~expected area via
            # iterative dilation toward the cell interior.
            from scipy.ndimage import binary_dilation

            target_nuc_area = int(cell_area * expected_nuc_frac_of_cell)
            current = nuc_mask_idx.copy()
            for _ in range(10):
                if int(current.sum()) >= target_nuc_area:
                    break
                grown = binary_dilation(current) & cell_mask_idx
                if int(grown.sum()) <= int(current.sum()):
                    break
                current = grown
            nucleus_label[current & ~nuc_mask_idx] = label_value
        # else: nuc_area == 0. Could be intentional (nucleus_poor /
        # membrane_only / sliver state) — leave alone.
    return cell_label, nucleus_label


def _vae_mask_voronoi(
    h: int, w: int,
    centers: np.ndarray,
    a_axes: np.ndarray, b_axes: np.ndarray, thetas: np.ndarray,
    sampled_types: list[str],
    sampler: "Any",
    rng: np.random.Generator,
) -> np.ndarray:
    """Cell territories from per-cell VAE-decoded masks centered + scaled.

    For each cell, sample a 64x64 binary mask from the VAE conditioned on
    its type, scale it to match the cell's fitted area, place at the cell's
    centroid. Overlapping pixels are assigned to the cell whose centroid is
    closest (preserves Voronoi-style non-overlap while honoring per-cell
    organic shape).
    """

    n = int(centers.shape[0])
    if n == 0:
        return np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    label_img = np.zeros((h, w), dtype=np.int32)
    best_sq = np.full((h, w), np.inf, dtype=np.float32)
    for idx in range(n):
        cy, cx = float(centers[idx, 0]), float(centers[idx, 1])
        type_name = sampled_types[idx] if idx < len(sampled_types) else None
        # Decode mask (64x64 binary).
        try:
            mask_64 = sampler.sample(type_name, rng=rng)
        except Exception:  # noqa: BLE001 - fall back to ellipse for this cell
            mask_64 = None
        # Target area: pi * a * b (the fitted ellipse area).
        target_area = float(np.pi * float(a_axes[idx]) * float(b_axes[idx]))
        if mask_64 is None or mask_64.sum() < 4:
            # Fall back to ellipse for this cell.
            theta = float(thetas[idx])
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dy = yy - cy
            dx = xx - cx
            rot_y = cos_t * dy + sin_t * dx
            rot_x = -sin_t * dy + cos_t * dx
            mask_yx = (rot_y / max(float(a_axes[idx]), 1e-3)) ** 2 + (rot_x / max(float(b_axes[idx]), 1e-3)) ** 2 <= 1.0
        else:
            # Scale the 64x64 mask so its area matches target_area.
            current_area = float(mask_64.sum())
            if current_area <= 0:
                continue
            scale = float(np.sqrt(target_area / current_area))
            new_h = max(8, int(round(64 * scale)))
            new_w = new_h
            ys = np.linspace(0, 63, new_h).astype(np.int64)
            xs = np.linspace(0, 63, new_w).astype(np.int64)
            mask_scaled = mask_64[ys[:, None], xs[None, :]] > 0.5
            # Place at center.
            half_h = new_h // 2
            half_w = new_w // 2
            iy0 = int(round(cy)) - half_h
            ix0 = int(round(cx)) - half_w
            iy1 = iy0 + new_h
            ix1 = ix0 + new_w
            mask_yx = np.zeros((h, w), dtype=bool)
            sy0 = max(0, -iy0)
            sx0 = max(0, -ix0)
            sy1 = mask_scaled.shape[0] - max(0, iy1 - h)
            sx1 = mask_scaled.shape[1] - max(0, ix1 - w)
            ty0 = max(0, iy0)
            tx0 = max(0, ix0)
            ty1 = min(h, iy1)
            tx1 = min(w, ix1)
            if sy1 > sy0 and sx1 > sx0 and ty1 > ty0 and tx1 > tx0:
                mask_yx[ty0:ty1, tx0:tx1] = mask_scaled[sy0:sy1, sx0:sx1]
        # Distance-from-center tiebreaker: cells closer to a covered pixel win.
        dy_field = yy - cy
        dx_field = xx - cx
        dist_sq = dy_field * dy_field + dx_field * dx_field
        update = mask_yx & (dist_sq < best_sq)
        best_sq[update] = dist_sq[update]
        label_img[update] = int(idx + 1)
    return label_img


def _ellipse_sdf_voronoi(
    h: int, w: int,
    centers: np.ndarray,
    a_axes: np.ndarray,
    b_axes: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    """Cell territory assignment by minimum signed-distance-to-ellipse.

    Each cell is an oriented ellipse. Per-pixel cell label = argmin(SDF + small
    proximity penalty), so cells naturally tile space respecting their fitted
    sizes and orientations rather than a uniform Voronoi tessellation.
    """

    n = int(centers.shape[0])
    if n == 0:
        return np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    best_score = np.full((h, w), np.inf, dtype=np.float32)
    label_img = np.zeros((h, w), dtype=np.int32)
    for idx in range(n):
        cy, cx = float(centers[idx, 0]), float(centers[idx, 1])
        a = max(float(a_axes[idx]), 1.0)
        b = max(float(b_axes[idx]), 1.0)
        theta = float(thetas[idx])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dy = yy - cy
        dx = xx - cx
        rot_y = cos_t * dy + sin_t * dx
        rot_x = -sin_t * dy + cos_t * dx
        # Approximate SDF: < 0 inside ellipse, > 0 outside, scaled in pixels.
        radial = np.sqrt((rot_y / a) ** 2 + (rot_x / b) ** 2)
        sdf = (radial - 1.0) * float(min(a, b))
        # Tiny tie-breaker by physical distance so pixels far from any ellipse
        # still get a cell assignment without producing huge "swallowed" cells.
        score = sdf + 0.01 * np.sqrt(dy * dy + dx * dx)
        update = score < best_score
        best_score[update] = score[update]
        label_img[update] = int(idx + 1)
    # Pixels too far from any ellipse (SDF much greater than 0) become
    # background. Threshold at 4× cell radius equivalent.
    far_threshold = float(np.median(np.maximum(a_axes, b_axes)) * 1.5)
    label_img[best_score > far_threshold] = 0
    return label_img


def _sample_nucleus_radii(
    rng: np.random.Generator,
    sampled_types: list[str],
    per_type_efficiencies: dict[str, dict[str, float]],
    config: SyntheticSceneConfig,
    cell_axes_a: np.ndarray,
    cell_axes_b: np.ndarray,
) -> np.ndarray:
    """Per-cell nucleus radius (px). Uses per-type nucleus/cell area ratio when
    available, scaling to the cell's own size.
    """

    n = len(sampled_types)
    radii = np.zeros(n, dtype=np.float32)
    px = max(float(config.pixel_size), 1e-6)
    for i, type_name in enumerate(sampled_types):
        record = per_type_efficiencies.get(type_name or "", {})
        if "nucleus_to_cell_ratio" in record:
            cell_area_px = math.pi * float(cell_axes_a[i]) * float(cell_axes_b[i])
            ratio = float(record["nucleus_to_cell_ratio"])
            ratio = max(0.05, min(ratio, 0.85))
            jitter = float(np.exp(rng.normal(0.0, 0.20)))
            nuc_area_px = cell_area_px * ratio * jitter
            nuc_radius_px = math.sqrt(max(nuc_area_px, 4.0) / math.pi)
        else:
            nuc_radius_um = float(rng.uniform(config.nucleus_min_radius_um, config.nucleus_max_radius_um))
            nuc_radius_px = nuc_radius_um / px
        radii[i] = float(nuc_radius_px)
    return radii


def _sample_centers_from_real_pattern(
    rng: np.random.Generator,
    h: int,
    w: int,
    pattern_library_path: "Path",
    target_num: int,
    jitter_px: float = 0.0,
    bg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample cell centers by cloning one observed tile's center pattern.

    Loads the saved cell-pattern library (object array of (N_i, 2) center
    arrays per real tile), picks one at random, optionally flips/jitters,
    crops to the tile shape, and returns at most ``target_num`` centers.
    Centers in ``bg_mask=True`` regions are dropped to keep tissue gaps.
    """

    data = np.load(str(pattern_library_path), allow_pickle=True)
    patterns = data["patterns"]
    if len(patterns) == 0:
        raise RuntimeError(f"empty cell pattern library: {pattern_library_path}")
    pat_h = int(data.get("h", h))
    pat_w = int(data.get("w", w))
    idx = int(rng.integers(0, len(patterns)))
    centers = np.asarray(patterns[idx], dtype=np.float32).copy()
    # Optional augmentation: random flip in y/x.
    if rng.random() < 0.5:
        centers[:, 0] = pat_h - 1 - centers[:, 0]
    if rng.random() < 0.5:
        centers[:, 1] = pat_w - 1 - centers[:, 1]
    # Per-cell jitter for slight variation.
    if jitter_px > 0:
        centers += rng.normal(0.0, jitter_px, size=centers.shape).astype(np.float32)
    # Scale from pattern dimensions to scene dimensions.
    if (pat_h, pat_w) != (h, w):
        centers[:, 0] *= h / max(pat_h, 1)
        centers[:, 1] *= w / max(pat_w, 1)
    # Clamp into bounds and de-duplicate.
    centers[:, 0] = np.clip(centers[:, 0], 0, h - 1)
    centers[:, 1] = np.clip(centers[:, 1], 0, w - 1)
    if bg_mask is not None:
        ys = np.round(centers[:, 0]).astype(int)
        xs = np.round(centers[:, 1]).astype(int)
        keep = ~bg_mask[ys, xs]
        centers = centers[keep]
    # Drop pairs closer than ~10 px (artifact from jitter/clip; real has
    # only ~0.6% of cells with NN<10 vs synth's earlier 10.8%).
    if len(centers) >= 2:
        from scipy.spatial import cKDTree
        keep = np.ones(len(centers), dtype=bool)
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r=10.0, output_type="ndarray")
        for a, b in pairs:
            if keep[a] and keep[b]:
                keep[b] = False
        centers = centers[keep]
    if len(centers) > target_num:
        sel = rng.choice(len(centers), target_num, replace=False)
        centers = centers[sel]
    return centers


def _sample_inhibited_points(rng: np.random.Generator, h: int, w: int, target: int, inhibit_px: float,
                              rejection_mask: np.ndarray | None = None) -> np.ndarray:
    """Dart-throwing inhibited point process. Returns ``(N, 2)`` (y, x) centers.

    When ``rejection_mask`` is provided (boolean ``(h, w)`` array, True =
    forbidden), candidates that fall on True pixels are rejected so cells
    avoid background regions.
    """

    accepted: list[tuple[float, float]] = []
    attempts_per_cell = 120
    max_attempts = max(target * attempts_per_cell, 6000)
    inhibit_sq = float(inhibit_px) ** 2
    for _ in range(max_attempts):
        if len(accepted) >= target:
            break
        y = float(rng.uniform(2.0, h - 2.0))
        x = float(rng.uniform(2.0, w - 2.0))
        if rejection_mask is not None and rejection_mask[int(y), int(x)]:
            continue
        ok = True
        for ay, ax in accepted:
            if (ay - y) ** 2 + (ax - x) ** 2 < inhibit_sq:
                ok = False
                break
        if ok:
            accepted.append((y, x))
    return np.asarray(accepted, dtype=np.float32) if accepted else np.zeros((0, 2), dtype=np.float32)


def _sample_domain_first_types(
    rng: np.random.Generator,
    centers: np.ndarray,
    type_names: list[str] | None,
    type_frequencies: dict[str, float] | None,
    mean_domain_radius_px: float = 60.0,
    purity: float = 0.85,
    image_shape: tuple[int, int] = (302, 302),
) -> list[str]:
    """A1v2: tissue-domain-first type assignment.

    Sample N domain centers via Voronoi tessellation. Each domain gets a
    primary type drawn from ``type_frequencies``. Cell type = domain
    primary type with probability ``purity``, else a random type weighted
    by ``type_frequencies``. Captures coarse tissue architecture
    (acini, islets, vessel beds, stromal regions) the Gibbs neighborhood
    sampler cannot directly represent.

    Domain radius drives the spatial scale: smaller radius → more, smaller
    domains → finer-grained tissue mosaic; larger radius → fewer, larger
    domains → islet/acini-scale organization. Default 60 px ≈ 12 µm at
    0.21 µm/px, matching pancreatic acinus / small islet scale.
    """

    n = int(len(centers))
    if not type_names or n == 0:
        return [""] * n
    candidate_names = [name for name in type_names if name and name != "unknown"]
    if not candidate_names:
        return [""] * n
    if type_frequencies:
        weights = np.asarray([float(type_frequencies.get(name, 0.0)) for name in candidate_names], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
    else:
        weights = np.ones(len(candidate_names), dtype=np.float64)
    weights /= weights.sum()

    h, w = image_shape
    image_area = float(h) * float(w)
    domain_area = float(np.pi) * float(mean_domain_radius_px) ** 2
    n_domains = max(2, int(round(image_area / max(domain_area, 1.0))))

    # Sample domain centers uniformly (Poisson-disk would be nicer but uniform
    # gives a reasonable mosaic at this density).
    domain_y = rng.uniform(0.0, float(h), size=n_domains).astype(np.float32)
    domain_x = rng.uniform(0.0, float(w), size=n_domains).astype(np.float32)
    domain_types = list(rng.choice(candidate_names, size=n_domains, p=weights))

    centers_yx = centers.astype(np.float32)
    # Pairwise distances cell -> domain.
    diff_y = centers_yx[:, 0:1] - domain_y[None, :]
    diff_x = centers_yx[:, 1:2] - domain_x[None, :]
    dist_sq = diff_y * diff_y + diff_x * diff_x
    nearest_domain = np.argmin(dist_sq, axis=1)

    out: list[str] = []
    purity = float(np.clip(purity, 0.0, 1.0))
    for i in range(n):
        if rng.random() < purity:
            out.append(str(domain_types[int(nearest_domain[i])]))
        else:
            out.append(str(rng.choice(candidate_names, p=weights)))
    return out


def _sample_clustered_types(
    rng: np.random.Generator,
    centers: np.ndarray,
    type_names: list[str] | None,
    type_frequencies: dict[str, float] | None,
    cluster_types: tuple[str, ...],
    cluster_sigma_px: float,
    cluster_size_range: tuple[int, int],
    sigma_per_type_px: dict[str, float] | None = None,
    size_per_type: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Per-type spatial-cluster type sampler. Each cluster_type uses its own
    sigma_per_type_px (Plan3 F5: pancreas islets bigger than immune scatter).
    Falls back to ``cluster_sigma_px`` and ``cluster_size_range`` when no
    per-type override is provided.
    """

    n = len(centers)
    if not type_names or n == 0:
        return [""] * n
    candidate_names = [name for name in type_names if name and name != "unknown"]
    if not candidate_names:
        return [""] * n

    if type_frequencies:
        weights = np.asarray([float(type_frequencies.get(name, 0.0)) for name in candidate_names], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
    else:
        weights = np.ones(len(candidate_names), dtype=np.float64)
    weights /= weights.sum()

    assignments: list[str | None] = [None] * n
    centers_xy = centers[:, [1, 0]].astype(np.float32)

    name_to_freq = {name: weights[i] for i, name in enumerate(candidate_names)}
    sigma_per_type_px = sigma_per_type_px or {}
    size_per_type = size_per_type or {}
    for type_name in cluster_types:
        if type_name not in candidate_names:
            continue
        target_count = int(round(float(name_to_freq.get(type_name, 0.0)) * n))
        if target_count <= 0:
            continue
        type_sigma = float(sigma_per_type_px.get(type_name, cluster_sigma_px))
        type_size_range = size_per_type.get(type_name, cluster_size_range)
        cluster_sigma_sq = type_sigma * type_sigma
        n_seeds = max(1, target_count // max(1, (type_size_range[0] + type_size_range[1]) // 2))
        seeds = []
        for _ in range(n_seeds):
            sy = float(rng.uniform(0, centers[:, 0].max() if n > 0 else 1.0))
            sx = float(rng.uniform(0, centers[:, 1].max() if n > 0 else 1.0))
            seeds.append((sx, sy))
        for sx, sy in seeds:
            dx = centers_xy[:, 0] - sx
            dy = centers_xy[:, 1] - sy
            dist_sq = dx * dx + dy * dy
            unassigned = [i for i in range(n) if assignments[i] is None]
            if not unassigned:
                break
            cluster_size = int(rng.integers(type_size_range[0], type_size_range[1] + 1))
            sorted_unassigned = sorted(unassigned, key=lambda i: float(dist_sq[i]))
            chosen = sorted_unassigned[: min(cluster_size, len(sorted_unassigned))]
            for ci in chosen:
                if float(dist_sq[ci]) > 9.0 * cluster_sigma_sq:
                    continue
                assignments[ci] = type_name
    remaining_names = [n_ for n_ in candidate_names if n_ not in cluster_types]
    if remaining_names:
        remaining_weights = np.asarray([name_to_freq.get(n_, 0.0) for n_ in remaining_names], dtype=np.float64)
        if remaining_weights.sum() <= 0:
            remaining_weights = np.ones_like(remaining_weights)
        remaining_weights /= remaining_weights.sum()
    else:
        remaining_names = candidate_names
        remaining_weights = weights
    for i in range(n):
        if assignments[i] is None:
            assignments[i] = remaining_names[int(rng.choice(len(remaining_names), p=remaining_weights))]
    return [assignments[i] or candidate_names[0] for i in range(n)]


def _rasterize_ellipses(h: int, w: int, centers: np.ndarray, radii: np.ndarray, aspects: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Rasterize a set of ellipses into a label image (1-based)."""

    label_img = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    for idx, (cy, cx) in enumerate(centers):
        a = float(radii[idx]) * float(aspects[idx])
        b = float(radii[idx]) / max(float(aspects[idx]), 1e-3)
        theta = float(thetas[idx])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dy = yy - float(cy)
        dx = xx - float(cx)
        rot_y = cos_t * dy + sin_t * dx
        rot_x = -sin_t * dy + cos_t * dx
        mask = (rot_y / max(a, 1e-3)) ** 2 + (rot_x / max(b, 1e-3)) ** 2 <= 1.0
        label_img[mask & (label_img == 0)] = int(idx + 1)
    return label_img


def _weighted_voronoi(h: int, w: int, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a weighted-Voronoi label image (1-based). Weights act as area scales."""

    if centers.size == 0:
        return np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    best_score = np.full((h, w), np.inf, dtype=np.float32)
    label_img = np.zeros((h, w), dtype=np.int32)
    for idx, (cy, cx) in enumerate(centers):
        # Power diagram: dist^2 - weight (smaller = closer).
        d2 = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2
        score = d2 - float(weights[idx]) * 200.0
        update = score < best_score
        best_score[update] = score[update]
        label_img[update] = int(idx + 1)
    return label_img


def _elastic_warp_label(rng: np.random.Generator, label: np.ndarray, sigma_px: float, strength_px: float) -> np.ndarray:
    """Apply a smooth random vector field warp to a label image (nearest neighbor)."""

    h, w = label.shape
    dx = _gaussian_blur(rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32), float(sigma_px))
    dy = _gaussian_blur(rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32), float(sigma_px))
    # Normalize by the std of the smoothed field to make strength meaningful.
    dx_norm = dx / (np.std(dx) + 1e-6) * float(strength_px)
    dy_norm = dy / (np.std(dy) + 1e-6) * float(strength_px)
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = np.clip(np.round(yy + dy_norm).astype(np.int32), 0, h - 1)
    src_x = np.clip(np.round(xx + dx_norm).astype(np.int32), 0, w - 1)
    return label[src_y, src_x].astype(np.int32, copy=False)


def _sample_types(rng: np.random.Generator, n_cells: int, type_names: list[str] | None,
                  type_frequencies: dict[str, float] | None) -> list[str]:
    if not type_names:
        return [""] * n_cells
    candidate_names = [name for name in type_names if name and name != "unknown"]
    if not candidate_names:
        return [""] * n_cells
    if type_frequencies:
        weights = np.asarray([float(type_frequencies.get(name, 0.0)) for name in candidate_names], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
    else:
        weights = np.ones(len(candidate_names), dtype=np.float64)
    weights /= weights.sum()
    return [candidate_names[int(rng.choice(len(candidate_names), p=weights))] for _ in range(n_cells)]
