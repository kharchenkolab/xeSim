"""Per-cell-type negbin + EDT-3D compartment-aware molecule emission.

Mirrors `xesim.transcripts.sample_scene_transcripts` in 3D. Reads the same
`model.transcripts_priors` dict (factor_signatures, per_type_alpha,
per_type_count_negbin, per_gene_compartment_posterior, compartment_model)
and produces molecules with realistic per-cell-type counts and gene
identities — the 2.5D pipeline previously emitted a flat 20 mol/cell with
gene name "synth".

The 2D sampler's positional logic is tangled with 2D-specific bbox / EDT /
visible-fraction code; replicating the relevant ~40 LOC here keeps the 2D
path completely untouched. See misc/plan_25d_molecule_port.md.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


_OUT_COLS = (
    "x_true", "y_true", "z_true",
    "true_cell_idx", "true_cell_id", "true_cell_type",
    "source", "gene", "factor_label", "qv", "is_ghost",
)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in _OUT_COLS})


def emit_molecules_3d_from_priors(
    cells_records: Sequence,            # CellRecord list (cell_idx, cell_id, cell_type)
    cell_label_3d: np.ndarray,          # (n_z, H, W) int32
    nucleus_label_3d: np.ndarray,       # (n_z, H, W) int32
    *,
    z_slices_um: Sequence[float],       # n_z absolute z positions (µm)
    tile_origin_um: tuple[float, float],
    pixel_size_um: float,
    transcripts_priors: dict,
    tx_rate_scale: float = 1.0,
    alpha_concentration: float = 50.0,
    synthetic_qv: float = 40.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample 3D molecules per cell using priors.

    Output columns: x_true, y_true, z_true, true_cell_idx, true_cell_id,
    true_cell_type, source, gene, factor_label, qv, is_ghost.
    """
    rng = rng or np.random.default_rng(0)

    # --- Priors unpack ---------------------------------------------------
    fs = transcripts_priors["factor_signatures"]
    bio = [int(f) for f in fs["biology_factors"]]
    sig_empirical = fs.get("data_empirical")
    sig = np.asarray(sig_empirical if sig_empirical else fs["data"],
                       dtype=np.float64)
    sig = sig / sig.sum(axis=1, keepdims=True).clip(min=1e-12)
    sig_bio = sig[[f - 1 for f in bio], :]              # K_bio × G

    gene_names = list(transcripts_priors["panel"]["gene_names"])
    n_genes = len(gene_names)

    per_type_alpha = dict(transcripts_priors.get("per_type_alpha", {}))
    per_type_negbin = transcripts_priors.get("per_type_count_negbin", {}) or {}
    default_alpha = transcripts_priors.get("default_alpha")
    if not default_alpha and per_type_alpha:
        default_alpha = np.mean(
            np.array(list(per_type_alpha.values())), axis=0).tolist()
    default_negbin = None
    if per_type_negbin:
        means = [v["mean"] for v in per_type_negbin.values()]
        lstds = [v["log_std"] for v in per_type_negbin.values()]
        default_negbin = {"mean": float(np.mean(means)),
                            "log_std": float(np.mean(lstds))}

    # Compartments — per-gene-type → per-gene → global prior fallback
    pgt_post = transcripts_priors.get(
        "per_gene_type_compartment_posterior", {}) or {}
    pgp_post = transcripts_priors.get(
        "per_gene_compartment_posterior", {}) or {}
    comp_model = transcripts_priors.get("compartment_model") or {}
    nuclear_frac = np.asarray(
        transcripts_priors.get("per_gene_nuclear_fraction",
                                  [0.5] * n_genes),
        dtype=np.float64)
    global_comp_prior = np.asarray(
        comp_model.get("global_prior") or
        [float(np.mean(nuclear_frac)),
          (1 - float(np.mean(nuclear_frac))) * 0.5,
          (1 - float(np.mean(nuclear_frac))) * 0.3,
          (1 - float(np.mean(nuclear_frac))) * 0.2],
        dtype=np.float64,
    )
    comp_boundaries = comp_model.get("boundaries_um", [0.0, 1.0, 3.0])
    b1, b2, b3 = float(comp_boundaries[0]), float(comp_boundaries[1]), \
                   float(comp_boundaries[2])
    use_compartments = bool(pgp_post) or bool(comp_model)

    # --- 3D geometry ----------------------------------------------------
    if cell_label_3d.size == 0:
        return _empty_df()
    z_arr = np.asarray(z_slices_um, dtype=np.float64)
    z_spacing = float(z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 1.0
    tx0, ty0 = float(tile_origin_um[0]), float(tile_origin_um[1])
    psz = float(pixel_size_um)

    from scipy.ndimage import find_objects, distance_transform_edt
    max_lab = int(cell_label_3d.max())
    if max_lab <= 0:
        return _empty_df()
    bboxes = find_objects(cell_label_3d, max_label=max_lab)

    # --- Per-cell loop --------------------------------------------------
    chunks: list[dict] = []                          # collect per-cell rows

    for c in cells_records:
        lab = int(c.cell_idx)
        if lab <= 0 or lab > len(bboxes):
            continue
        bbox = bboxes[lab - 1]
        if bbox is None:
            continue
        sz, sy, sx = bbox
        cell_mask = cell_label_3d[sz, sy, sx] == lab
        if not cell_mask.any():
            continue
        nuc_mask = (nucleus_label_3d[sz, sy, sx] == lab) & cell_mask
        has_nuc = bool(nuc_mask.any())

        # Resolve alpha + negbin
        t = c.cell_type or ""
        alpha = per_type_alpha.get(t)
        if alpha is None:
            if not default_alpha:
                continue
            alpha = default_alpha
        # Log-normal Poisson mixture: drawing n ~ Poisson(mean) directly is
        # under-dispersed vs real (B6 max ~88 vs real ~644 on pancreas; same
        # defect as the 2D sampler before the parallel fix). The priors
        # store both mean and log_std (std of log1p of real per-cell
        # counts), so the principled draw is:
        #   lambda_cell ~ LogNormal(log(mean) - log_std^2/2, log_std^2)
        #   n_mol ~ Poisson(lambda_cell)
        # which preserves mean = base_count AND matches the heavy tail.
        nb = per_type_negbin.get(t) or default_negbin
        base_count = float(nb["mean"]) if nb else 20.0
        log_std = float(nb.get("log_std", 0.0)) if nb else 0.0
        scale = float(tx_rate_scale)
        if log_std > 0.0:
            mu = np.log(max(base_count * scale, 1e-9)) - 0.5 * log_std ** 2
            lam = float(np.exp(mu + log_std * rng.standard_normal()))
        else:
            lam = base_count * scale
        n_mol = int(rng.poisson(lam))
        if n_mol == 0:
            continue

        # Dirichlet factor mix
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        h_mix = rng.dirichlet(alpha_concentration * alpha_arr + 1e-3)

        # 4 compartments: 0 nuclear, 1 perinuc, 2 cyto, 3 distal
        # EDT-3D from nucleus boundary in µm (anisotropic sampling so z
        # spacing is respected; output is straight-up µm).
        if use_compartments and has_nuc:
            dist_um = distance_transform_edt(
                ~nuc_mask, sampling=(z_spacing, psz, psz))
            outside = cell_mask & ~nuc_mask
            comp_masks = [
                nuc_mask,
                outside & (dist_um > b1) & (dist_um <= b2),
                outside & (dist_um > b2) & (dist_um <= b3),
                outside & (dist_um > b3),
            ]
        else:
            empty = np.zeros_like(cell_mask)
            comp_masks = [empty, cell_mask, empty, empty]

        z_off, y_off, x_off = sz.start, sy.start, sx.start
        offset = np.array([z_off, y_off, x_off], dtype=np.int64)
        comp_coords = []
        for m in comp_masks:
            local = np.argwhere(m)
            if local.size:
                local = local + offset
            comp_coords.append(local)
        all_coords = np.argwhere(cell_mask) + offset    # full-cell fallback

        # Sample factors → genes vectorized
        k_assign = rng.choice(len(bio), size=n_mol, p=h_mix)
        all_g = np.empty(n_mol, dtype=np.int64)
        all_factor = np.empty(n_mol, dtype=np.int64)
        all_comp = np.empty(n_mol, dtype=np.int64)
        wi = 0
        for k in range(len(bio)):
            sel = k_assign == k
            n_k = int(sel.sum())
            if n_k == 0:
                continue
            g_idx = rng.choice(n_genes, size=n_k, p=sig_bio[k])
            if use_compartments:
                posts = np.empty((n_k, 4), dtype=np.float64)
                for i, gi in enumerate(g_idx):
                    gname = gene_names[int(gi)]
                    p = pgt_post.get(f"{gname}|{t}")
                    if p is None:
                        p = pgp_post.get(gname)
                    if p is None:
                        p = global_comp_prior
                    posts[i] = np.asarray(p, dtype=np.float64)
                csum = np.cumsum(posts, axis=1)
                u = rng.random(n_k)
                comps = (u[:, None] >= csum).sum(axis=1).clip(max=3)
            else:
                nf = nuclear_frac[g_idx]
                comps = np.where(
                    (rng.random(n_k) < nf) & has_nuc, 0, 1)
            all_g[wi:wi + n_k] = g_idx
            all_factor[wi:wi + n_k] = bio[k]
            all_comp[wi:wi + n_k] = comps
            wi += n_k

        # Compartment fallback chain (mirrors 2D sampler)
        def _resolve(ci: int) -> np.ndarray:
            cs = comp_coords[ci]
            if cs.shape[0] > 0:
                return cs
            order = (1, 0, 2, 3) if ci != 1 else (0, 2, 3, 1)
            for cf in order:
                if comp_coords[cf].shape[0] > 0:
                    return comp_coords[cf]
            return all_coords

        # Position-sample per compartment
        zs_px = np.empty(n_mol, dtype=np.float64)
        ys_px = np.empty(n_mol, dtype=np.float64)
        xs_px = np.empty(n_mol, dtype=np.float64)
        for ci in range(4):
            sel = all_comp == ci
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            coords = _resolve(ci)
            pis = rng.integers(0, coords.shape[0], size=n_sel)
            picked = coords[pis]                        # (n_sel, 3) zi,y,x
            zs_px[sel] = picked[:, 0].astype(np.float64)
            ys_px[sel] = picked[:, 1].astype(np.float64)
            xs_px[sel] = picked[:, 2].astype(np.float64)

        jit_z = rng.uniform(-z_spacing / 2.0, z_spacing / 2.0, size=n_mol)
        jit_y = rng.uniform(-psz / 2.0, psz / 2.0, size=n_mol)
        jit_x = rng.uniform(-psz / 2.0, psz / 2.0, size=n_mol)
        z_um = z_arr[zs_px.astype(np.int64)] + jit_z
        y_um = ty0 + ys_px * psz + jit_y
        x_um = tx0 + xs_px * psz + jit_x

        chunks.append({
            "x_true": x_um.astype(np.float32),
            "y_true": y_um.astype(np.float32),
            "z_true": z_um.astype(np.float32),
            "true_cell_idx": np.full(n_mol, lab, dtype=np.int64),
            "true_cell_id": np.full(n_mol, c.cell_id, dtype=object),
            "true_cell_type": np.full(n_mol, t, dtype=object),
            "source": np.full(n_mol, "body", dtype=object),
            "gene": np.asarray([gene_names[int(g)] for g in all_g], dtype=object),
            "factor_label": all_factor.astype(np.int64),
            "qv": np.full(n_mol, float(synthetic_qv), dtype=np.float32),
            "is_ghost": np.zeros(n_mol, dtype=bool),
        })

    if not chunks:
        return _empty_df()
    return pd.DataFrame({
        col: np.concatenate([ch[col] for ch in chunks]) for col in _OUT_COLS
    })


__all__ = ["emit_molecules_3d_from_priors"]
