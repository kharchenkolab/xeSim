"""Slice-guided scene sampler.

Takes a SliceGuide extracted from a real tile and samples a new
MechanisticScene that matches the guide on slide/domain-level structure
(per-region type composition, density) while resampling all fine details
(cell positions, exact type assignments, shapes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mechanistic_sampler import (
    SyntheticSceneConfig,
    sample_synthetic_scene,
    _sample_inhibited_points,
)
from .slice_guide import SliceGuide, multiscale_log_prior_map


def density_to_rejection_mask(density_field: np.ndarray,
                                target_h: int,
                                target_w: int,
                                density_threshold: float = 0.10) -> np.ndarray:
    """Upsample density field to (H, W) and threshold to get a rejection mask
    where True = background (no cells should be sampled there)."""
    import torch
    import torch.nn.functional as F
    df = torch.from_numpy(density_field).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(df, size=(target_h, target_w),
                        mode="bilinear", align_corners=False).squeeze().numpy()
    return up < float(density_threshold)


def sample_slice_guided_scene(
    guide: SliceGuide,
    *,
    seed: int = 1,
    type_names: list[str] | None = None,
    tissue_neighborhood: dict[str, Any] | None = None,
    cell_shape_sampler: Any = None,
    cell_shape_exemplar_sampler: Any = None,
    per_type_efficiencies: dict[str, dict[str, float]] | None = None,
    sectioning_states: dict[str, Any] | None = None,
    scene_id: str = "slice_guided",
    # Guide-following knobs
    guide_weights: dict[int, float] | None = None,
    type_marginal_weight: float = 0.5,
    type_jitter_p: float = 0.0,
    n_cells_jitter_frac: float = 0.10,
    density_threshold: float = 0.10,
    pixel_size: float = 0.2125,
    nucleus_min_radius_um: float = 1.8,
    nucleus_max_radius_um: float = 5.2,
    nucleus_dropout_fraction: float | None = None,  # if None, use 1 - guide.has_nucleus_fraction
    nucleus_to_cell_ratio_boost: float = 1.3,
    mechanistic_params_path: str | None = None,
):
    """Sample a scene that follows the guide's slide-level composition
    while resampling cells locally.

    Args:
      guide: SliceGuide from extract_guide(real_scene).
      type_names: full type list (including 'unknown' at index 0). If None,
        derived from guide.type_names + ['unknown' prepended].
      tissue_neighborhood: optional Gibbs neighbor-conditional artifact.
      cell_shape_sampler / cell_shape_exemplar_sampler: as in
        sample_synthetic_scene.
      guide_weights: per-scale strength {scale: weight}. Default
        {coarsest: 3.0, mid: 2.0, finest: 1.0} — coarser scales lead.
      type_marginal_weight: how strongly the global type marginal pulls
        types toward the guide's slide-level proportions.
      type_jitter_p: after Gibbs, with this probability per cell, swap
        the type to a different one drawn from the marginal (adds local
        diversity without changing global composition).
      n_cells_jitter_frac: target_num_cells = guide.expected_n_cells × (1 ± frac).
      density_threshold: pixels with density below this become background
        (rejection mask for nucleus sampling).

    Returns: MechanisticScene built via the existing sample_synthetic_scene
      pipeline, just with the guide-injected priors and centers.
    """
    if type_names is None:
        type_names = ["unknown"] + list(guide.type_names)
    # Auto-load per_type_efficiencies (with fitted nucleus_to_cell_ratio
    # per type) if not provided — gives realistic nucleus sizes by type.
    if per_type_efficiencies is None and mechanistic_params_path is not None:
        try:
            import json as _json
            with open(mechanistic_params_path) as _f:
                _mp = _json.load(_f)
            per_type_efficiencies = _mp.get("per_type_efficiencies", None)
        except Exception:
            per_type_efficiencies = None
    # Optional boost on nucleus_to_cell_ratio — compensates for ellipse
    # rasterization + cell-mask intersection shrinkage (~20-30%).
    if per_type_efficiencies is not None and nucleus_to_cell_ratio_boost != 1.0:
        boosted = {}
        for t, v in per_type_efficiencies.items():
            nv = dict(v)
            if "nucleus_to_cell_ratio" in nv:
                nv["nucleus_to_cell_ratio"] = min(0.85,
                    float(nv["nucleus_to_cell_ratio"]) * float(nucleus_to_cell_ratio_boost))
            boosted[t] = nv
        per_type_efficiencies = boosted
    # Per-guide dropout rate
    if nucleus_dropout_fraction is None:
        nucleus_dropout_fraction = max(0.0, 1.0 - float(guide.has_nucleus_fraction))
    H, W = guide.tile_shape
    rng = np.random.default_rng(int(seed))

    # 1. Sample cell positions: inhibited Poisson respecting density mask.
    bg_mask = density_to_rejection_mask(guide.density_field, H, W, density_threshold)
    n_target = max(1, int(round(guide.expected_n_cells *
                                   (1.0 + (rng.random() * 2 - 1) * n_cells_jitter_frac))))
    # 3 µm at 0.21 µm/px ≈ 14 px — matches the original SyntheticSceneConfig
    # default (was incorrectly 6 px, which packed cells 2× too tight).
    inhibit_px = 3.0 / max(pixel_size, 1e-6)
    centers = _sample_inhibited_points(rng, H, W, n_target, inhibit_px,
                                          rejection_mask=bg_mask)
    if centers.size == 0:
        # Fall back: allow whole tile if rejection was too aggressive
        centers = _sample_inhibited_points(rng, H, W, n_target, inhibit_px,
                                              rejection_mask=None)
    n_target = int(centers.shape[0])

    # 2. Build the per-cell multi-scale position prior.
    if guide_weights is None:
        sorted_scales = sorted(guide.coarse_fields.keys())  # coarsest first
        guide_weights = {}
        for i, s in enumerate(sorted_scales):
            guide_weights[s] = max(0.5, 3.0 - i)  # 3, 2, 1, 0.5, ...
    pos_log_prior_map = multiscale_log_prior_map(
        guide, H, W,
        guide_weights=guide_weights,
        include_type_marginal=True,
        type_marginal_weight=type_marginal_weight,
    )

    # 3. Reuse the existing sampler entry point. It runs Gibbs (with our
    # extension that accepts position_log_prior) and builds the full
    # MechanisticScene including shapes, efficiencies, etc.
    # Pass pre-built centers via the cell_pattern_library_path codepath:
    # save centers to a temp .npz exactly in the format
    # _sample_centers_from_real_pattern expects.
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    patterns = np.empty(1, dtype=object)
    patterns[0] = centers.astype(np.float32)
    np.savez(tmp.name, patterns=patterns, h=H, w=W)
    tmp.close()
    config = SyntheticSceneConfig(
        image_shape=(H, W),
        target_num_cells=n_target,
        seed=int(seed),
        pixel_size=pixel_size,
        background_fraction=float((bg_mask).mean()),
        cell_pattern_library_path=tmp.name,
        cell_pattern_jitter_px=0.0,
        nucleus_min_radius_um=float(nucleus_min_radius_um),
        nucleus_max_radius_um=float(nucleus_max_radius_um),
    )
    scene = sample_synthetic_scene(
        config,
        type_names=type_names,
        per_type_efficiencies=per_type_efficiencies,
        scene_id=scene_id,
        cell_shape_sampler=cell_shape_sampler,
        tissue_neighborhood=tissue_neighborhood,
        sectioning_states=sectioning_states,
        cell_shape_exemplar_sampler=cell_shape_exemplar_sampler,
        position_log_prior_map=pos_log_prior_map,
        position_log_prior_weight=1.0,
    )
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass

    # 4. Optional type jitter (post-Gibbs). For each cell, with prob p,
    # swap its type to a different one sampled from the type marginal.
    if type_jitter_p > 0 and scene.cells:
        import dataclasses
        marg = np.asarray(guide.type_marginal, dtype=np.float64)
        marg = marg / max(marg.sum(), 1e-9)
        new_cells = []
        for c in scene.cells:
            if rng.random() < type_jitter_p:
                new_t_idx = int(rng.choice(len(marg), p=marg))  # 0..T_field-1
                new_t_name = guide.type_names[new_t_idx]
                if new_t_name != c.cell_type:
                    new_cells.append(dataclasses.replace(c, cell_type=new_t_name))
                    continue
            new_cells.append(c)
        scene = dataclasses.replace(scene, cells=tuple(new_cells))

    # 5. Optional nucleus dropout: real Xenium tiles have ~15% of cells with
    # no visible nucleus (partial sections, out-of-plane). The default sampler
    # gives ~98% of cells a nucleus, biasing renders toward dense bright DAPI.
    # Randomly drop nuclei to match real frequency.
    if nucleus_dropout_fraction > 0 and scene.cells:
        import dataclasses
        new_cells = []
        new_nuc = scene.nucleus_label.copy()
        for c in scene.cells:
            if c.nucleus_label is not None and rng.random() < nucleus_dropout_fraction:
                new_nuc[new_nuc == int(c.label)] = 0
                new_cells.append(dataclasses.replace(c, nucleus_label=None))
            else:
                new_cells.append(c)
        scene = dataclasses.replace(scene, cells=tuple(new_cells), nucleus_label=new_nuc)

    return scene
