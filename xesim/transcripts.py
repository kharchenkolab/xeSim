"""Transcripts layer for xeSim — Phase 1: NMF fit via cellAdmix-core.

Adapter around `celladmix.CellAdmix(...).fit(...)` that produces the
artifacts an xeSim model directory needs for downstream explain /
generate use of transcript evidence.

Outputs into ``MODEL_DIR/priors/``:
  transcripts_nmf.json   — schema-validated metadata + factor signatures
                            + classification + per-type priors
  cell_factors.parquet   — per-cell H matrix (cells × biology factors)

The `celladmix` import is deferred to call time so xesim still imports
on boxes that don't have it installed; an ImportError raised here
points the user at the install steps in misc/transcripts_v2.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .mechanistic_scene import MechanisticScene

TRANSCRIPTS_SCHEMA_VERSION = "xesim.transcripts_nmf.v2"
# v2 (2026-05-13): factor identifiers everywhere in priors JSON are now
# 1-based to align with cellAdmix-core's public Python API (factor: 1..K,
# factor_label: 'F1..FK', column names 'factor_K_fraction'). Internally
# `factor_signatures.data` is still a K × G numpy array indexed positionally
# 0..K-1; consumers convert 1-based factor IDs to array indices via
# `sig[factor_idx - 1]`. Pre-v2 priors are not forward-compatible.


def _require_celladmix():
    try:
        import celladmix  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "cellAdmix-core is required for transcript priors. Install per "
            "misc/transcripts_v2.md: apt deps + Apache Arrow apt repo + "
            "pip install -e /path/to/cellAdmix-core/python --no-build-isolation"
        ) from e
    import celladmix
    return celladmix


def fit_transcripts_nmf(
    bundle_path: str | Path,
    annotation_path: str | Path,
    model_dir: str | Path,
    *,
    annotation_col: str = "merged_annotation",
    cell_id_col: str = "cell_id",
    min_qv: float = 20.0,
    rank: int | None = None,
    rank_multiplier: float = 1.2,
    rank_cap: int = 30,
    nmf_variant: str = "invsqrt_kl",
    num_threads: int | None = None,
    rules_p_thresh: float = 0.1,
    score_membrane: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    celladmix_dir: str | Path | None = None,
) -> Path:
    """Fit cellAdmix NMF on a Xenium bundle and write priors into model_dir.

    Returns the path to the written `priors/transcripts_nmf.json`.

    Parameters mostly mirror cellAdmix; defaults match its production
    recipe (invsqrt_kl, rank = ceil(1.2 * n_types)).

    If `score_membrane=False`, biology/admixture classification falls
    back to the factor-source score alone (no membrane image required).
    Useful for bundles without morphology channels.

    If ``celladmix_dir`` is supplied, xeSim points cellAdmix at that
    directory instead of the default ``<bundle_parent>/_xesim_celladmix/``.
    cellAdmix's internal caching means an already-fit run at that path
    is reused (no re-fit) — the "supply an existing cellAdmix result"
    integration mode.
    """
    ca = _require_celladmix()
    model_dir = Path(model_dir)
    priors_dir = model_dir / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    out_json = priors_dir / "transcripts_nmf.json"
    out_parquet = priors_dir / "cell_factors.parquet"

    if out_json.exists() and not overwrite:
        if verbose:
            print(f"[transcripts] using cached {out_json}")
        return out_json

    t0 = time.time()
    # Phase 2.G: cellAdmix runs/store are bundle-derived; write to a
    # bundle-level location so multiple xeSim models trained on the same
    # bundle share the fit. See `bundle_celladmix_dir`. Caller can override
    # via `celladmix_dir` to point at a user-supplied existing fit.
    if celladmix_dir is not None:
        ca_dir = Path(celladmix_dir)
        if verbose:
            print(f"[transcripts] using supplied cellAdmix dir: {ca_dir}")
    else:
        ca_dir = bundle_celladmix_dir(bundle_path)
    if verbose:
        print(f"[transcripts] cellAdmix.CellAdmix from {bundle_path}")
        print(f"[transcripts] output_dir: {ca_dir} (bundle-level)")
    ds = ca.CellAdmix(
        source=str(bundle_path),
        output_dir=str(ca_dir),
        format="xenium",
        annotation=str(annotation_path),
        annotation_col=annotation_col,
        cell_id_col=cell_id_col,
        min_qv=float(min_qv),
        keep_unassigned=False,
        num_threads=num_threads,
    )
    ds.ensure_store(force=False, materialize_molecules=True, verbose=verbose)

    if verbose:
        print(f"[transcripts] fitting NMF ({nmf_variant}, rank={rank or 'auto'})")
    fit = ds.fit(
        nmf_variant=nmf_variant,
        rank=rank, rank_multiplier=rank_multiplier, rank_cap=rank_cap,
        nmf_n_runs=None,            # auto: == threads
        molecule_scoring="gene_loadings",
        overwrite=overwrite,
        verbose=verbose,
    )

    # Pull artifacts out. cellAdmix conventions:
    #   factor_loadings: rows indexed by gene; columns 'F1'..'FK'.
    #   cells: cols include cell_id, transcript_count, dominant_factor,
    #          factor_1_fraction..factor_K_fraction (1-indexed in name).
    #   source_annot: cols factor (1-indexed int), source_cell_type, score, margin.
    loadings = fit.factor_loadings()
    cells_df = fit.cells()
    if verbose:
        print(f"[transcripts] factor_loadings shape: {loadings.shape}; "
              f"columns: {list(loadings.columns)[:5]}...")

    # Factor columns in loadings: 'F1', 'F2', ...
    factor_cols = [c for c in loadings.columns if isinstance(c, str)
                    and c.startswith("F") and c[1:].isdigit()]
    factor_cols.sort(key=lambda c: int(c[1:]))
    K = len(factor_cols)
    if K == 0:
        raise RuntimeError(f"could not find factor columns in factor_loadings: "
                              f"{list(loadings.columns)}")

    # Factor signatures: K × n_genes, row-normalized
    # cellAdmix's factor_loadings sets index.name='gene' with str entries;
    # be defensive across versions.
    if loadings.index.name == "gene" or loadings.index.dtype == object:
        gene_names = loadings.index.astype(str).tolist()
    elif "gene" in loadings.columns:
        gene_names = loadings["gene"].astype(str).tolist()
    else:
        # Fall back to the cellAdmix manifest's gene list
        gene_names = list(fit.manifest.get("genes", []))
    if len(gene_names) != len(loadings):
        raise RuntimeError(
            f"gene_names length {len(gene_names)} != loadings rows {len(loadings)}"
        )
    # Sanity: gene names should be strings without scientific notation
    if any(g.replace(".", "").replace("-", "").replace("e", "").replace("E", "").isdigit()
            for g in gene_names[:5]):
        # Numeric — wrong column; try the manifest
        manifest_genes = list(fit.manifest.get("genes", []))
        if len(manifest_genes) == len(loadings):
            gene_names = [str(g) for g in manifest_genes]
    sig = loadings[factor_cols].to_numpy(dtype=np.float64).T   # K × n_genes
    sig_rowsum = sig.sum(axis=1, keepdims=True).clip(min=1e-12)
    sig_norm = sig / sig_rowsum

    # Source score → factor → annotated source type
    if verbose:
        print("[transcripts] scoring factor sources")
    source_annot = fit.score_factor_sources().annotation()

    # Score membrane scoring → admixture rules
    biology_factors: list[int] = []
    admixture_factors: list[dict[str, Any]] = []
    rules = pd.DataFrame()
    if score_membrane:
        try:
            if verbose:
                print("[transcripts] scoring membrane")
            membrane = fit.score_membrane(verbose=verbose)
            rules = membrane.rules(p_thresh=rules_p_thresh)
        except Exception as exc:
            if verbose:
                print(f"[transcripts] membrane scoring failed ({exc}); "
                      f"falling back to source-only classification")

    if not rules.empty and "factor" in rules.columns:
        for _, row in rules.iterrows():
            f_id = int(row["factor"])      # cellAdmix is already 1-based
            admixture_factors.append({
                "factor": f_id,
                "source": str(row.get("source_cell_type", "unknown")),
                "target": str(row.get("target_cell_type", "unknown")),
                "membrane_neg_log10_p": float(row.get("neg_log10_p", 0.0)),
            })

    # Per-factor source labels (one per factor with a known native source type).
    # Keys are 1-based factor IDs as strings (matching cellAdmix public API).
    factor_source_labels: dict[str, dict[str, Any]] = {}
    for _, row in source_annot.iterrows():
        f_id = int(row["factor"])
        factor_source_labels[str(f_id)] = {
            "type": str(row["source_cell_type"]),
            "score": float(row.get("score", 0.0)),
            "margin": float(row.get("margin", 0.0)),
        }

    # Biology factors: those with a known native source type from
    # score_factor_sources. Admixture rules above describe per-(target, factor)
    # spillover, but each factor remains biology in its native source type.
    # All factor IDs are 1-based.
    biology_factors = sorted(int(k) for k in factor_source_labels.keys())
    if not biology_factors:
        # Fallback: treat all factors as biology if no source labels resolved.
        biology_factors = list(range(1, K + 1))

    # Per-cell factor fractions: cols 'factor_1_fraction' .. 'factor_K_fraction'
    # (cellAdmix's 1-based column naming — matches our public schema directly).
    cf_factor_cols: list[str] = []
    for k in range(1, K + 1):
        col = f"factor_{k}_fraction"
        if col in cells_df.columns:
            cf_factor_cols.append(col)
    if len(cf_factor_cols) != K and verbose:
        print(f"[transcripts] warning: found {len(cf_factor_cols)}/{K} factor "
              f"fraction columns in cells_df")

    # Join cell annotation onto cells_df so we can aggregate by type.
    ann = pd.read_csv(annotation_path)
    ann_keep = ann[[cell_id_col, annotation_col]].copy()
    ann_keep.columns = ["cell_id", "_cell_type"]
    cells_typed = cells_df.merge(ann_keep, on="cell_id", how="left")

    # Per-type admixture-factor sets (factor F admixes INTO target T when
    # there is a rule (factor=F, target=T) — F is *native* to some OTHER type).
    # All IDs 1-based.
    per_type_admixture_factor_indices: dict[str, list[int]] = {}
    for rule in admixture_factors:
        t = str(rule.get("target"))
        per_type_admixture_factor_indices.setdefault(t, [])
        if int(rule["factor"]) not in per_type_admixture_factor_indices[t]:
            per_type_admixture_factor_indices[t].append(int(rule["factor"]))

    per_type_alpha: dict[str, list[float]] = {}
    per_type_admix_rate: dict[str, float] = {}
    per_type_count_negbin: dict[str, dict[str, float]] = {}
    count_col = "transcript_count" if "transcript_count" in cells_typed.columns else None
    for tname, g in cells_typed.groupby("_cell_type", dropna=True):
        if pd.isna(tname): continue
        arr = g[cf_factor_cols].to_numpy(dtype=np.float64)
        row_sums = arr.sum(axis=1, keepdims=True).clip(min=1e-12)
        arr_n = arr / row_sums                                   # rows sum to 1
        mean_weights = arr_n.mean(axis=0)                        # length K, positions 0..K-1
        # biology_factors are 1-based IDs; convert to array positions
        bio_alpha = [float(mean_weights[fid - 1]) for fid in biology_factors]
        per_type_alpha[str(tname)] = bio_alpha
        admix_for_this = set(per_type_admixture_factor_indices.get(str(tname), []))
        per_type_admix_rate[str(tname)] = float(sum(
            mean_weights[fid - 1] for fid in admix_for_this
        ))
        if count_col is not None:
            n = g[count_col].to_numpy(dtype=np.float64)
            n_valid = n[n > 0]
            if len(n_valid) > 0:
                per_type_count_negbin[str(tname)] = {
                    "mean": float(n_valid.mean()),
                    "log_std": float(np.std(np.log1p(n_valid))),
                    "n_cells": int(len(n_valid)),
                }

    # Per-gene nuclear fraction + compartment posteriors + empirical
    # factor-gene rates — read cellAdmix's molecules() once and reuse the
    # dataframe for all three.
    per_gene_nuclear_fraction: list[float] = [0.5] * len(gene_names)
    compartment_priors: dict[str, Any] = {}
    factor_gene_rates_empirical: list[list[float]] | None = None
    try:
        mol_df = fit.molecules(
            columns=["gene_idx", "cell_idx", "factor",
                       "overlaps_nucleus", "nucleus_distance"],
        ) if hasattr(fit, "molecules") else None
        if mol_df is not None:
            # Empirical per-factor gene rates — replaces the W-derived
            # signature for synthesis sampling. The W matrix is in the
            # invsqrt_kl-reweighted space and systematically under-emits
            # high-prevalence markers (e.g. INS, AMY2A, GCG); the empirical
            # rates count molecules cellAdmix actually attributed to each
            # factor and give the unbiased per-factor gene distribution.
            mol_for_factors = mol_df[mol_df["factor"].notna()]
            counts = (mol_for_factors.groupby(["factor", "gene_idx"])
                                       .size().reset_index(name="count"))
            G = len(gene_names)
            factor_gene_rates_empirical = []
            for f_id in factor_cols:    # 'F1', 'F2', ...
                f_int = int(f_id[1:])  # 1-based factor index
                sub = counts[counts["factor"] == f_int]
                rate = np.zeros(G, dtype=np.float64)
                for _, row in sub.iterrows():
                    gi = int(row["gene_idx"])
                    if 0 <= gi < G:
                        rate[gi] = float(row["count"])
                s = rate.sum()
                if s > 0:
                    rate /= s
                factor_gene_rates_empirical.append(rate.tolist())
            if verbose:
                # Sanity-check on a known marker
                ins_idx = gene_names.index("INS") if "INS" in gene_names else None
                if ins_idx is not None and len(factor_gene_rates_empirical) >= 8:
                    sig_w = sig_norm[7, ins_idx] if sig_norm.shape[0] >= 8 else 0
                    sig_e = factor_gene_rates_empirical[7][ins_idx]
                    print(f"[transcripts] empirical-vs-W signature[F8, INS]: "
                          f"empirical={sig_e:.4f}  W-derived={sig_w:.4f}  "
                          f"ratio={sig_e/max(sig_w,1e-12):.2f}x")
            # Per-gene nuclear fraction (kept for backwards compat — also the
            # nuclear column of per_gene_compartment_posterior)
            agg = (mol_df.groupby("gene_idx")["overlaps_nucleus"]
                          .agg(["mean", "count"]).reset_index())
            gene_idx_to_frac = {int(row["gene_idx"]): float(row["mean"])
                                  for _, row in agg.iterrows()
                                  if row["count"] >= 10}
            per_gene_nuclear_fraction = [
                gene_idx_to_frac.get(i, 0.5) for i in range(len(gene_names))
            ]
            # Compartment posterior: join cell_idx → cell_id → cell_type,
            # gene_idx → gene name, then fit the EM + Dirichlet shrinkage model.
            mol_for_comp = mol_df.merge(
                cells_df[["cell_idx", "cell_id"]], on="cell_idx", how="left",
            ).merge(
                ann_keep, on="cell_id", how="left",
            )
            mol_for_comp["gene"] = mol_for_comp["gene_idx"].astype(int).map(
                {i: g for i, g in enumerate(gene_names)})
            compartment_priors = _fit_compartment_model(
                mol_for_comp[["gene", "_cell_type",
                                  "overlaps_nucleus", "nucleus_distance"]],
                verbose=verbose,
            )
    except Exception as exc:
        if verbose:
            print(f"[transcripts] per-gene compartment / nuclear fraction skipped: {exc}")
            import traceback; traceback.print_exc()

    # Per-type typical cell area + default alpha, derived from canonical
    # crops (so sample-time can scale n_mol by per-cell visible_fraction
    # and handle untyped cells gracefully). See `sample_scene_transcripts`.
    per_type_typical_area_um2, default_typical_area_um2 = \
        _compute_typical_cell_area_um2(Path(model_dir), cells_typed, verbose=verbose)
    default_alpha = _compute_default_alpha(per_type_alpha, cells_typed)

    # Stability per factor — not always exposed; leave blank
    factor_stability: list[float] = []

    elapsed = time.time() - t0

    # Resolve the run dir cellAdmix wrote into so accessors can find it.
    # Path is now bundle-side; record relative to bundle_celladmix_dir so
    # the model can be relocated without breaking the link.
    try:
        run_dir_rel = str(Path(fit.run_path).relative_to(ca_dir))
    except (AttributeError, ValueError):
        try:
            run_dir_rel = str(Path(fit.run_path).relative_to(model_dir))
        except (AttributeError, ValueError):
            run_dir_rel = None

    priors: dict[str, Any] = {
        "schema_version": TRANSCRIPTS_SCHEMA_VERSION,
        "celladmix_version": ca.__version__,
        "celladmix_run_dir": run_dir_rel,
        "nmf_variant": nmf_variant,
        "rank_K": int(K),
        "panel": {"gene_names": gene_names, "n_genes": int(len(gene_names))},
        "factor_signatures": {
            "data": sig_norm.tolist(),
            "data_empirical": factor_gene_rates_empirical,
            "biology_factors": [int(i) for i in biology_factors],
            "admixture_factors": admixture_factors,
        },
        "factor_source_labels": factor_source_labels,
        "per_type_alpha": per_type_alpha,
        "per_type_admix_rate": per_type_admix_rate,
        "per_type_admixture_factor_indices": per_type_admixture_factor_indices,
        "per_type_count_negbin": per_type_count_negbin,
        "per_type_typical_area_um2": per_type_typical_area_um2,
        "default_typical_area_um2": float(default_typical_area_um2),
        "default_alpha": default_alpha,
        "per_gene_nuclear_fraction": per_gene_nuclear_fraction,
        "compartment_model": compartment_priors.get("compartment_model"),
        "per_gene_compartment_posterior":
            compartment_priors.get("per_gene_compartment_posterior", {}),
        "per_gene_type_compartment_posterior":
            compartment_priors.get("per_gene_type_compartment_posterior", {}),
        "qv_threshold": float(min_qv),
        "rules_p_thresh": float(rules_p_thresh),
        "fit_stats": {
            "n_cells_total": int(len(cells_df)),
            "wall_time_seconds": float(elapsed),
            "factor_stability": factor_stability,
        },
    }

    out_json.write_text(json.dumps(priors, indent=2))

    # Sibling per-cell factors parquet (joins type in for downstream)
    keep_cols = ["cell_id", "_cell_type", "transcript_count"] + cf_factor_cols
    keep_cols = [c for c in keep_cols if c in cells_typed.columns]
    cells_typed[keep_cols].to_parquet(out_parquet, index=False)

    if verbose:
        print(f"[transcripts] wrote {out_json} ({elapsed:.1f}s, K={K}, "
              f"{len(biology_factors)} biology / {len(admixture_factors)} admixture rules)")
    return out_json


def _fit_compartment_model(
    mol_df: "pd.DataFrame",
    *,
    n_iter: int = 10,
    concentration: float = 50.0,
    min_count_per_type: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit a hierarchical compartment-distribution posterior over molecules.

    Observation model:
      Z (latent compartment) ∈ {0=nuclear, 1=perinuc, 2=cyto, 3=distal}.
      overlaps_nucleus is observed deterministically as [Z == 0].
      For outside-nucleus molecules (overlaps_nucleus = 0):
          nucleus_distance | Z=c ~ Normal(μ_c, σ_c²)  for c ∈ {1, 2, 3}.

    Global EM fits π_overall (4-vec mixture proportions) and (μ_c, σ_c) for
    the 3 outside-nucleus components. Per-(gene, type) posteriors are
    Dirichlet posteriors with the global mixture as the prior (concentration
    `concentration`).

    Returns a dict suitable for the priors JSON:
      compartment_model: {centers_um, scales_um, global_prior, ...}
      per_gene: dict gene → 4-vec
      per_gene_type: dict "gene|type" → 4-vec   (only N_{g,t} ≥ min_count_per_type)
    """
    if verbose:
        print(f"[compartment] fitting on {len(mol_df):,} molecules")
    # Required columns
    needed = {"gene", "_cell_type", "overlaps_nucleus", "nucleus_distance"}
    missing = needed - set(mol_df.columns)
    if missing:
        raise ValueError(f"compartment fit missing columns: {missing}")
    df = mol_df[mol_df["_cell_type"].notna()].copy()
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(int)
    if verbose:
        print(f"[compartment] {len(df):,} molecules after dropping untyped cells")

    nuclear_mask = df["overlaps_nucleus"].to_numpy() == 1
    n_total = len(df)
    p_nuclear_global = float(nuclear_mask.mean())
    if verbose:
        print(f"[compartment] global nuclear fraction: {p_nuclear_global:.3f}")

    # ---- Global EM on the outside-nucleus distances ----------------------
    d_out = df.loc[~nuclear_mask, "nucleus_distance"].to_numpy(dtype=np.float64)
    # Init: well-spaced means in the empirically-seen range
    centers = np.array([0.4, 1.5, 4.0], dtype=np.float64)
    scales = np.array([0.4, 1.0, 2.5], dtype=np.float64)
    pi_out = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    log_lik_history = []
    for it in range(n_iter):
        # E-step
        log_pdf = np.empty((d_out.size, 3), dtype=np.float64)
        for c in range(3):
            log_pdf[:, c] = (-0.5 * ((d_out - centers[c]) / scales[c]) ** 2
                              - np.log(scales[c]) - 0.5 * np.log(2 * np.pi))
        log_pi = np.log(pi_out.clip(min=1e-12))
        log_post = log_pdf + log_pi
        max_lp = log_post.max(axis=1, keepdims=True)
        log_norm = max_lp.ravel() + np.log(
            np.exp(log_post - max_lp).sum(axis=1).clip(min=1e-300))
        log_lik = float(log_norm.sum())
        log_lik_history.append(log_lik)
        resp = np.exp(log_post - log_norm[:, None])
        # M-step
        n_c = resp.sum(axis=0).clip(min=1.0)
        new_pi = n_c / n_c.sum()
        new_centers = (resp * d_out[:, None]).sum(axis=0) / n_c
        new_scales = np.sqrt(((resp * (d_out[:, None] - new_centers) ** 2)
                                .sum(axis=0) / n_c).clip(min=1e-6))
        # Enforce ordering so components stay interpretable (perinuc < cyto < distal)
        order = np.argsort(new_centers)
        pi_out = new_pi[order]; centers = new_centers[order]; scales = new_scales[order]
        if verbose and (it == 0 or it == n_iter - 1):
            print(f"[compartment] EM iter {it+1}: log_lik={log_lik:,.0f}  "
                  f"π_out={pi_out.round(3).tolist()}  "
                  f"centers={centers.round(2).tolist()}  "
                  f"scales={scales.round(2).tolist()}")

    # Global 4-vec mixture proportions
    alpha0 = np.array([p_nuclear_global,
                          (1 - p_nuclear_global) * pi_out[0],
                          (1 - p_nuclear_global) * pi_out[1],
                          (1 - p_nuclear_global) * pi_out[2]], dtype=np.float64)

    # ---- One final E-step to get per-molecule responsibilities -----------
    # For outside-nucleus molecules: compute responsibilities for 3 components.
    log_pdf = np.empty((d_out.size, 3), dtype=np.float64)
    for c in range(3):
        log_pdf[:, c] = (-0.5 * ((d_out - centers[c]) / scales[c]) ** 2
                          - np.log(scales[c]) - 0.5 * np.log(2 * np.pi))
    log_pi = np.log(pi_out.clip(min=1e-12))
    log_post = log_pdf + log_pi
    max_lp = log_post.max(axis=1, keepdims=True)
    log_norm_const = max_lp.ravel() + np.log(
        np.exp(log_post - max_lp).sum(axis=1).clip(min=1e-300))
    resp_out = np.exp(log_post - log_norm_const[:, None])    # (n_out, 3)

    # Construct full per-molecule responsibility matrix (n_total, 4)
    resp = np.zeros((n_total, 4), dtype=np.float64)
    resp[nuclear_mask, 0] = 1.0
    resp[~nuclear_mask, 1:] = resp_out

    # ---- Per-(gene, type) and per-gene posteriors with Dirichlet shrinkage
    df["_gene"] = df["gene"].astype(str)
    df["_type"] = df["_cell_type"].astype(str)
    # Vectorized: factorize (gene, type) into a single key
    gt_keys = df["_gene"] + "|" + df["_type"]
    g_keys = df["_gene"].to_numpy()
    t_keys = df["_type"].to_numpy()

    # Aggregate responsibilities by (gene, type) and by gene
    df_index = df.reset_index(drop=True)
    gt_unique, gt_inverse = np.unique(gt_keys.to_numpy(), return_inverse=True)
    g_unique, g_inverse = np.unique(g_keys, return_inverse=True)
    # Sum responsibilities per (g, t) and per g
    n_gt = gt_unique.size; n_g = g_unique.size
    soft_gt = np.zeros((n_gt, 4), dtype=np.float64)
    soft_g = np.zeros((n_g, 4), dtype=np.float64)
    for c in range(4):
        np.add.at(soft_gt[:, c], gt_inverse, resp[:, c])
        np.add.at(soft_g[:, c], g_inverse, resp[:, c])
    counts_gt = soft_gt.sum(axis=1)
    counts_g = soft_g.sum(axis=1)

    # Dirichlet posterior with global α₀ prior
    c_param = float(concentration)
    posterior_gt = (c_param * alpha0[None, :] + soft_gt) / (c_param + counts_gt[:, None])
    posterior_g = (c_param * alpha0[None, :] + soft_g) / (c_param + counts_g[:, None])

    per_gene_posterior = {str(g): posterior_g[i].tolist()
                          for i, g in enumerate(g_unique)}
    per_gene_type_posterior = {}
    for i, key in enumerate(gt_unique):
        if counts_gt[i] >= float(min_count_per_type):
            per_gene_type_posterior[str(key)] = posterior_gt[i].tolist()

    if verbose:
        print(f"[compartment] {len(per_gene_posterior)} genes × "
              f"{len(per_gene_type_posterior)} (gene, type) posteriors "
              f"(N_{{g,t}} >= {min_count_per_type})")

    return {
        "compartment_model": {
            "boundaries_um": [0.0, 1.0, 3.0],   # boundaries between perinuc/cyto/distal at sample-time
            "em_centers_um": centers.tolist(),
            "em_scales_um": scales.tolist(),
            "em_pi_out": pi_out.tolist(),
            "global_prior": alpha0.tolist(),
            "p_nuclear_global": p_nuclear_global,
            "concentration": c_param,
            "fit_iterations": n_iter,
            "log_lik_history": log_lik_history,
            "n_molecules": int(n_total),
            "min_count_per_type": int(min_count_per_type),
        },
        "per_gene_compartment_posterior": per_gene_posterior,
        "per_gene_type_compartment_posterior": per_gene_type_posterior,
    }


def _compute_typical_cell_area_um2(
    model_dir: Path,
    cells_typed: "pd.DataFrame",
    *,
    verbose: bool = True,
) -> tuple[dict[str, float], float]:
    """Estimate per-type median cell area (µm²) from canonical crops.

    For each canonical crop's npz, read cell_label and count pixels per
    cell label. Only count cells whose pixel mass touches the interior
    (not the boundary) so we don't over-count partial cells. Convert
    pixels → µm² via the crop's pixel_size (crop_size_um / image_size).
    Returns (per_type_dict, fallback_value_for_unknown).
    """
    canonical_dir = model_dir / "canonical"
    manifest_path = canonical_dir / "manifest.json"
    if not manifest_path.exists():
        if verbose:
            print(f"[transcripts] no canonical manifest at {manifest_path}; "
                  f"typical-area fallback to {300.0}")
        return {}, 300.0
    manifest = json.loads(manifest_path.read_text())
    cell_to_type = dict(zip(cells_typed["cell_id"], cells_typed["_cell_type"]))
    per_type_areas: dict[str, list[float]] = {}
    n_crops_processed = 0
    for crop in manifest.get("crops", []):
        npz_path = canonical_dir / crop["npz_path"]
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                cell_label = np.asarray(d["cell_label"])[:256, :256].astype(np.int32)
                cell_ids = [str(v) for v in d["cell_ids"].tolist()] \
                            if "cell_ids" in d.files else []
        except Exception:
            continue
        if not cell_ids:
            continue
        # Pixel size for this crop
        bx = crop["crop_box"]
        crop_size_um = float(bx["xmax"]) - float(bx["xmin"])
        pixel_size_um = crop_size_um / 256.0
        px_area_um2 = pixel_size_um * pixel_size_um
        # Per-label pixel count; mapping label_value → cell_id is positional
        # in the cell_ids list (sorted unique non-zero labels), matching
        # scene_io.scene_from_canonical_crop's convention.
        unique, counts = np.unique(cell_label, return_counts=True)
        nz = unique[unique > 0]
        for idx_pos, lab in enumerate(nz):
            n_px = int(counts[unique == lab][0])
            # Touch top/left/right/bottom boundary? → partial; skip
            if (cell_label[0, :] == lab).any() or (cell_label[-1, :] == lab).any() or \
               (cell_label[:, 0] == lab).any() or (cell_label[:, -1] == lab).any():
                continue
            if idx_pos >= len(cell_ids):
                continue
            cid = cell_ids[idx_pos]
            t = cell_to_type.get(cid)
            if t is None or (isinstance(t, float) and np.isnan(t)):
                continue
            area_um2 = n_px * px_area_um2
            per_type_areas.setdefault(str(t), []).append(area_um2)
        n_crops_processed += 1
    out: dict[str, float] = {}
    all_areas = []
    for t, areas in per_type_areas.items():
        arr = np.array(areas, dtype=np.float64)
        out[t] = float(np.median(arr))
        all_areas.extend(areas)
    fallback = float(np.median(all_areas)) if all_areas else 300.0
    if verbose:
        print(f"[transcripts] typical cell area from {n_crops_processed} canonical crops:")
        for t, a in out.items():
            print(f"             {t:30s}  median = {a:.1f} µm² (n_cells={len(per_type_areas[t])})")
        print(f"             default (unknown):           {fallback:.1f} µm²")
    return out, fallback


def _compute_default_alpha(
    per_type_alpha: dict[str, list[float]],
    cells_typed: "pd.DataFrame",
) -> list[float]:
    """Weighted average of per_type_alpha by cell-count, for cells with
    unknown / unrecognized types."""
    if not per_type_alpha:
        return []
    alpha_len = len(next(iter(per_type_alpha.values())))
    type_counts = cells_typed.groupby("_cell_type").size().to_dict()
    total = 0.0
    acc = np.zeros(alpha_len, dtype=np.float64)
    for t, a in per_type_alpha.items():
        w = float(type_counts.get(t, 1))
        acc += w * np.asarray(a, dtype=np.float64)
        total += w
    if total > 0:
        acc /= total
    return acc.tolist()


def load_transcripts_priors(model_dir: str | Path) -> dict[str, Any] | None:
    """Load transcripts_nmf.json from a model dir, returning None if absent."""
    p = Path(model_dir) / "priors" / "transcripts_nmf.json"
    if not p.exists():
        return None
    priors = json.loads(p.read_text())
    # Stale-checkpoint guard: older models were fit before
    # _compute_typical_cell_area_um2 existed, so per_type_typical_area_um2
    # comes back empty and default_typical_area_um2 falls back to 300 µm² —
    # which is ~5× larger than real Xenium cells and silently scales every
    # cell's molecule count down by ~5× via visible_fraction. Warn loudly.
    if (not priors.get("per_type_typical_area_um2")
            and float(priors.get("default_typical_area_um2", 0.0)) > 200.0):
        import warnings as _w
        _w.warn(
            f"transcripts_nmf.json at {p} has empty per_type_typical_area_um2 "
            f"and default_typical_area_um2={priors.get('default_typical_area_um2')} "
            f"(typical real cells ~50µm²). Per-cell molecule counts will be "
            f"scaled DOWN by visible_fraction=cell_area/default; expect ~5× "
            f"undercount of synth transcripts. Re-fit the model or patch "
            f"per_type_typical_area_um2 with realistic per-type medians.",
            RuntimeWarning, stacklevel=2,
        )
    return priors


# ---------------------------------------------------------------------------
# Phase 2 — accessors over cellAdmix's existing per-cell / per-molecule data
# ---------------------------------------------------------------------------


def bundle_celladmix_dir(bundle_path: str | Path) -> Path:
    """Canonical location for cellAdmix runs/data derived from this bundle.

    Pattern: `<bundle_parent>/_xesim_celladmix/`. Multiple xeSim models trained
    on the same bundle share this directory — cellAdmix runs are bundle-level
    artifacts (derived from `transcripts.parquet`), not model-level. Storing
    them at the bundle level avoids per-model copies/symlinks.

    Returns the directory path; existence is NOT verified here.
    """
    return Path(bundle_path).resolve().parent / "_xesim_celladmix"


def _pick_celladmix_run(runs_root: Path, priors: dict[str, Any]) -> Path | None:
    """Return the run directory under `runs_root` matching the priors metadata,
    or None if no run.json is found."""
    if not runs_root.exists():
        return None
    runs = list(runs_root.glob("*/run.json"))
    if not runs:
        return None
    if len(runs) > 1:
        variant = priors.get("nmf_variant", "")
        rank = priors.get("rank_K")
        wanted = f"fit_rank{rank}_{variant}"
        for r in runs:
            if r.parent.name == wanted:
                return r.parent
    return runs[0].parent


def _celladmix_run_dir(model_dir: Path, priors: dict[str, Any],
                        *, bundle_path: str | Path | None = None) -> Path:
    """Resolve the cellAdmix run output dir given the priors metadata.

    Resolution prioritizes priors' RECORDED path over globs — factor labels in
    `priors` are tied to a specific fit, so we must honor that link rather
    than silently switching to whatever fit lives at the bundle level.

    Order:
      1. ``priors["celladmix_run_dir"]`` if absolute and exists.
      2. (New layout) bundle_celladmix_dir / recorded_rel — if recorded path
         is relative and resolves there.
      3. (New layout with old-style prefix) bundle_celladmix_dir /
         strip_legacy_prefix(recorded_rel) — handles priors that recorded
         "_celladmix/runs/..." but the data lives at "<bundle>/_xesim_celladmix/runs/..."
      4. (Legacy) model_dir / recorded_rel — back-compat for priors that
         pointed model-relative and the data is still in model_dir/_celladmix/.
      5. Glob fallback when no recorded path exists: prefer bundle-side, then
         model-side.

    Kept for direct parquet reads. For molecule operations prefer
    ``_resolve_celladmix_fit`` which uses cellAdmix's
    ``CellAdmixFit.load`` reattachment API.
    """
    rel = priors.get("celladmix_run_dir")

    if rel:
        p = Path(rel)
        # 1. Absolute
        if p.is_absolute() and p.exists():
            return p
        # 2. Bundle-relative
        if not p.is_absolute() and bundle_path is not None:
            cand = bundle_celladmix_dir(bundle_path) / p
            if cand.exists():
                return cand
            # 3. Strip legacy "_celladmix/" prefix, try bundle-side
            if p.parts and p.parts[0] == "_celladmix":
                cand_strip = bundle_celladmix_dir(bundle_path) / Path(*p.parts[1:])
                if cand_strip.exists():
                    return cand_strip
        # 4. Model-relative (legacy)
        if not p.is_absolute():
            cand = Path(model_dir) / p
            if cand.exists():
                return cand
        # Recorded path doesn't resolve anywhere — fall through to glob.

    # 5a. Bundle-side glob (only used when no recorded path or all candidates missing)
    if bundle_path is not None:
        picked = _pick_celladmix_run(bundle_celladmix_dir(bundle_path) / "runs", priors)
        if picked is not None:
            return picked

    # 5b. Legacy model-side glob
    picked = _pick_celladmix_run(Path(model_dir) / "_celladmix" / "runs", priors)
    if picked is not None:
        return picked

    raise FileNotFoundError(
        f"no cellAdmix run found.\n"
        f"  Expected (new): "
        f"{bundle_celladmix_dir(bundle_path) / 'runs/' if bundle_path else '<bundle>/_xesim_celladmix/runs/'}\n"
        f"  Legacy fallback: {Path(model_dir) / '_celladmix/runs/'}\n"
        f"Run `xesim fit-model --with-transcripts` (or its standalone form) "
        f"on the bundle."
    )


def _resolve_celladmix_fit(
    model_dir: Path,
    priors: dict[str, Any],
    *,
    bundle_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
):
    """Reattach to the cellAdmix fit via the new `CellAdmixFit.load` API."""
    ca = _require_celladmix()
    run_dir = _celladmix_run_dir(Path(model_dir), priors, bundle_path=bundle_path)
    return ca.CellAdmixFit.load(
        str(run_dir),
        source=str(bundle_path) if bundle_path else None,
        annotation=str(annotation_path) if annotation_path else None,
    )


def load_cell_factor_fractions(
    model_dir: str | Path,
    priors: dict[str, Any] | None = None,
    *,
    only_biology: bool = False,
    include_annotation: bool = False,
    bundle_path: str | Path | None = None,
) -> "pd.DataFrame":
    """Read per-cell factor fractions from the cellAdmix fit's cells.parquet.

    Column naming is 1-based throughout (matches cellAdmix's public API):
    `factor_1_fraction..factor_K_fraction`, `dominant_factor ∈ 1..K`. Returns:
      cell_id, transcript_count, dominant_factor, dominant_fraction,
      factor_1_fraction .. factor_K_fraction.
    With `only_biology=True`, non-biology factor columns are dropped.
    With `include_annotation=True`, `_cell_type` is joined from
    `priors/cell_factors.parquet`.
    """
    model_dir = Path(model_dir)
    if priors is None:
        priors = load_transcripts_priors(model_dir)
        if priors is None:
            raise RuntimeError(f"no transcripts priors at {model_dir}/priors/")
    K = int(priors["rank_K"])
    bio = set(int(f) for f in priors["factor_signatures"]["biology_factors"])

    run_dir = _celladmix_run_dir(model_dir, priors, bundle_path=bundle_path)
    cells = pd.read_parquet(run_dir / "cells.parquet")
    keep_cols = ["cell_id", "transcript_count", "dominant_factor", "dominant_fraction"]
    fac_cols = [f"factor_{k}_fraction" for k in range(1, K + 1)
                  if (not only_biology) or k in bio]
    keep_cols = [c for c in keep_cols if c in cells.columns] + fac_cols
    out = cells[keep_cols].copy()
    if include_annotation:
        ann_pq = Path(model_dir) / "priors" / "cell_factors.parquet"
        if ann_pq.exists():
            ann = pd.read_parquet(ann_pq, columns=["cell_id", "_cell_type"])
            out = out.merge(ann, on="cell_id", how="left")
    return out


def classify_cells_by_centroids(
    priors: dict[str, Any],
    cell_h: dict[str, np.ndarray],
    *,
    softmax_temperature: float = 8.0,
    no_signal_weight: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Assign each cell a transcript-implied type by cosine similarity to
    per-type alpha centroids in the priors.

    `cell_h` values must be vectors over biology factors only (i.e. length
    `len(priors['factor_signatures']['biology_factors'])`), in factor-index
    order. Cells with zero-norm `h` get a null classification.

    Returns dict cell_id → {
      cell_type_transcripts: str | None,
      type_uncertainty:      float | None  # Q3-monotonic: see below
      type_uncertainty_margin: float | None  # legacy 1 - cosine_margin
      cosine_top:            float | None,
      type_scores:           dict[type → cosine_score],
    }

    The headline `type_uncertainty` is **softmax-entropy of the type
    cosines**, normalized to [0, 1] by `log(n_types)`, optionally
    multiplied by `(1 + no_signal_weight * (1 - cosine_top))` to penalize
    cells with no clear top-type signal (those previously confused the
    old margin-only metric — they'd get high margin-U by accident even
    though the prediction was essentially random).
    """
    per_type_alpha = priors.get("per_type_alpha", {})
    types = list(per_type_alpha.keys())
    if not types:
        return {cid: {"cell_type_transcripts": None,
                      "type_uncertainty": None,
                      "type_uncertainty_margin": None,
                      "cosine_top": None,
                      "type_scores": None} for cid in cell_h}
    centroids = np.asarray([per_type_alpha[t] for t in types], dtype=np.float64)
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    C = centroids / c_norms      # T × K_bio
    log_T = np.log(len(types))

    out: dict[str, dict[str, Any]] = {}
    for cid, h in cell_h.items():
        h = np.asarray(h, dtype=np.float64).ravel()
        nh = np.linalg.norm(h)
        if nh < 1e-12 or h.shape[0] != C.shape[1]:
            out[cid] = {"cell_type_transcripts": None,
                          "type_uncertainty": None,
                          "type_uncertainty_margin": None,
                          "cosine_top": None,
                          "type_scores": None}
            continue
        cos = C @ (h / nh)
        order = np.argsort(-cos)
        top = int(order[0])
        if len(cos) > 1:
            margin = float(cos[order[0]] - cos[order[1]])
        else:
            margin = float(cos[order[0]])
        # Entropy-based uncertainty
        scaled = softmax_temperature * (cos - cos.max())
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        norm_entropy = entropy / log_T if log_T > 0 else 0.0
        # No-signal weight: cells with low cosine_top get up-weighted
        ns = float(np.clip(1.0 - cos[top], 0.0, 1.0))
        type_u = float(norm_entropy * (1.0 + no_signal_weight * ns))
        out[cid] = {
            "cell_type_transcripts": types[top],
            "type_uncertainty": type_u,
            "type_uncertainty_margin": float(max(0.0, 1.0 - margin)),
            "cosine_top": float(cos[top]),
            "type_scores": {types[int(i)]: float(cos[int(i)]) for i in order},
        }
    return out


def tile_aware_per_type_alpha(
    priors: dict[str, Any],
    cf: "pd.DataFrame",
    *,
    guide_cell_ids: list[str] | None = None,
    smoothing: float = 0.3,
) -> dict[str, list[float]]:
    """Build per-type Dirichlet centers averaging biology-factor fractions
    over a *subset* of cells (e.g. the cells visible in a guide tile).

    Returns dict cell_type → length-K_bio vector. For types missing from the
    guide, falls back to the global per_type_alpha. The result is a convex
    combination `(1-smoothing)*tile_alpha + smoothing*global_alpha` to avoid
    overfitting when the guide tile has very few cells of a type.
    """
    bio = list(priors["factor_signatures"]["biology_factors"])
    fac_cols = [f"factor_{k}_fraction" for k in bio]
    global_alpha = priors.get("per_type_alpha", {})
    sub = cf
    if guide_cell_ids is not None:
        sub = cf[cf["cell_id"].isin(guide_cell_ids)]
    if "_cell_type" not in sub.columns:
        return dict(global_alpha)
    out: dict[str, list[float]] = dict(global_alpha)
    for t, g in sub.groupby("_cell_type", dropna=True):
        if len(g) == 0 or t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        tile_mean = g[fac_cols].mean(axis=0).to_numpy(dtype=np.float64)
        s = tile_mean.sum()
        if s <= 0:
            continue
        tile_mean = tile_mean / s
        glob = np.asarray(global_alpha.get(str(t), tile_mean), dtype=np.float64)
        mixed = (1.0 - smoothing) * tile_mean + smoothing * glob
        out[str(t)] = mixed.tolist()
    return out


def apply_admixture_rate(
    priors: dict[str, Any],
    rate_multiplier: float,
) -> dict[str, list[float]]:
    """Build per-type alpha vectors that boost (or attenuate) the admixture
    factors' weights by `rate_multiplier`.

    rate_multiplier = 1.0 returns the empirical per_type_alpha (observed
    bundle admixture). 0.0 removes all admixture-flagged mass (pure-biology
    cells). > 1.0 injects extra admixture, capped so total weight on
    admixture factors stays ≤ 0.95 per type.

    Used by `model.generate(admixture_rate_multiplier=...)` for Phase 8
    controlled-admixture forward synth.
    """
    bio = list(priors["factor_signatures"]["biology_factors"])
    per_type_alpha = priors.get("per_type_alpha", {})
    per_type_admix = priors.get("per_type_admixture_factor_indices", {})
    factor_to_pos = {int(f): i for i, f in enumerate(bio)}
    out: dict[str, list[float]] = {}
    for t, alpha in per_type_alpha.items():
        a = np.asarray(alpha, dtype=np.float64)
        admix_factor_set = set(int(f) for f in per_type_admix.get(t, []))
        if not admix_factor_set or rate_multiplier == 1.0:
            out[t] = a.tolist()
            continue
        admix_pos = np.array([factor_to_pos[f] for f in admix_factor_set
                                if f in factor_to_pos], dtype=np.int64)
        if admix_pos.size == 0:
            out[t] = a.tolist()
            continue
        native_mass = float(a.sum() - a[admix_pos].sum())
        admix_mass = float(a[admix_pos].sum())
        new_admix_mass = min(0.95, max(0.0, admix_mass * rate_multiplier))
        # Scale admixture positions to the new mass; rescale native to fill.
        if admix_mass > 1e-12:
            a[admix_pos] *= (new_admix_mass / admix_mass)
        if native_mass > 1e-12:
            native_factor_count = a.shape[0] - admix_pos.size
            mask = np.ones(a.shape[0], dtype=bool); mask[admix_pos] = False
            new_native_total = 1.0 - new_admix_mass
            a[mask] *= (new_native_total / native_mass)
        a = a / a.sum().clip(min=1e-12)
        out[t] = a.tolist()
    return out


def sample_scene_transcripts(
    priors: dict[str, Any],
    scene: "MechanisticScene",
    *,
    rng: np.random.Generator | None = None,
    pixel_size_um: float | None = None,
    alpha_concentration: float = 50.0,
    synthetic_qv: float = 40.0,
    nuclear_jitter_um: float = 0.1,
    per_type_alpha_override: dict[str, list[float]] | None = None,
    tx_rate_scale: float = 1.0,
) -> "pd.DataFrame":
    """Sample synthetic transcripts for a generated scene using NMF priors.

    Per cell:
      1. h_c ~ Dirichlet(`alpha_concentration` * per_type_alpha[type] + ε)
      2. n_molecules ~ NegBin (parameterized by per_type_count_negbin[type])
      3. For each molecule:
         - factor k ~ Categorical(h_c)  (over biology factors only)
         - gene g ~ Categorical(factor_signatures[k])
         - position: uniform over the cell mask; with probability
           `per_gene_nuclear_fraction[g]`, restrict to the nucleus pixels
           (or the central core if no nucleus is segmented)
         - x, y → scene-local µm via the scene's pixel_size

    Returns a DataFrame with columns:
      cell_id, gene, x, y, factor_label (0-indexed), qv, source_cell_type.
    """
    rng = rng or np.random.default_rng(0)
    pixel_size_um = pixel_size_um or float(getattr(scene, "pixel_size", 0.25))

    bio = [int(f) for f in priors["factor_signatures"]["biology_factors"]]
    # Prefer empirical per-factor gene rates (computed at fit time from
    # cellAdmix's molecule-level factor labels) over the W-derived
    # signatures. The W matrix is in invsqrt_kl-reweighted space and
    # systematically under-emits high-prevalence markers.
    sig_empirical = priors["factor_signatures"].get("data_empirical")
    if sig_empirical:
        sig = np.asarray(sig_empirical, dtype=np.float64)
    else:
        sig = np.asarray(priors["factor_signatures"]["data"], dtype=np.float64)
    sig = sig / sig.sum(axis=1, keepdims=True).clip(min=1e-12)
    # sig is K × G indexed positionally; biology factors are 1-based IDs
    sig_bio = sig[[f - 1 for f in bio], :]   # K_bio × G
    gene_names = priors["panel"]["gene_names"]
    nuclear_frac = np.asarray(priors.get("per_gene_nuclear_fraction",
                                            [0.5] * len(gene_names)),
                                dtype=np.float64)
    per_type_alpha = dict(priors.get("per_type_alpha", {}))
    if per_type_alpha_override is not None:
        per_type_alpha.update(per_type_alpha_override)
    per_type_negbin = priors.get("per_type_count_negbin", {})
    per_type_area = priors.get("per_type_typical_area_um2", {})
    default_area = float(priors.get("default_typical_area_um2", 300.0))
    default_alpha = priors.get("default_alpha")

    # Compartment posterior (per-gene-type → per-gene → global α₀).
    pgt_post = priors.get("per_gene_type_compartment_posterior", {}) or {}
    pgp_post = priors.get("per_gene_compartment_posterior", {}) or {}
    comp_model = priors.get("compartment_model") or {}
    global_comp_prior = np.asarray(
        comp_model.get("global_prior") or
        [float(np.mean(nuclear_frac)),
          (1 - float(np.mean(nuclear_frac))) * 0.5,
          (1 - float(np.mean(nuclear_frac))) * 0.3,
          (1 - float(np.mean(nuclear_frac))) * 0.2],
        dtype=np.float64,
    )
    comp_boundaries = comp_model.get("boundaries_um", [0.0, 1.0, 3.0])
    use_compartments = bool(pgp_post) or bool(comp_model)

    # Defaults for untyped cells: weighted mean alpha across types,
    # and weighted mean negbin params.
    if default_alpha is None or len(default_alpha) == 0:
        # Fall back: simple mean over per_type_alpha
        if per_type_alpha:
            default_alpha = np.mean(np.array(list(per_type_alpha.values())), axis=0).tolist()
    default_negbin = None
    if per_type_negbin:
        means = [v["mean"] for v in per_type_negbin.values()]
        lstds = [v["log_std"] for v in per_type_negbin.values()]
        default_negbin = {"mean": float(np.mean(means)),
                            "log_std": float(np.mean(lstds))}

    px_to_um2 = pixel_size_um * pixel_size_um
    cell_label = scene.cell_label
    nuc_label = scene.nucleus_label
    h, w = cell_label.shape

    cell_ids: list[str] = []
    genes_out: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    factor_labels: list[int] = []
    qvs: list[float] = []
    src_types: list[str] = []

    # Pre-compute per-label bounding boxes once. scipy.ndimage.find_objects
    # returns a list indexed by label-1 of (slice_y, slice_x) or None.
    # Used to crop cell_label/nuc_label to a small bbox per cell so the
    # per-cell np.nonzero / EDT work scales with cell size, not tile size.
    # On dense 64um tiles this cuts the per-cell ops by ~50-100×.
    from scipy.ndimage import find_objects, distance_transform_edt
    max_lab = int(cell_label.max()) if cell_label.size else 0
    bboxes = find_objects(cell_label, max_label=max_lab) if max_lab > 0 else []

    for c in scene.cells:
        lab = int(c.label)
        t_raw = c.cell_type or ""
        alpha = per_type_alpha.get(t_raw)
        if alpha is None:
            # No per-type alpha available — caller should have classified
            # this cell via transcripts/stains before calling. Fall back
            # to default_alpha (weighted mean) rather than skipping, so
            # we don't silently lose molecules. (Skipping caused ~10-15%
            # of bundle cells to emit zero, visible as undercount.)
            if default_alpha is None or len(default_alpha) == 0:
                continue
            alpha = default_alpha
        t = t_raw
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        h_dirichlet = rng.dirichlet(alpha_concentration * alpha_arr + 1e-3)

        # Lookup bbox for this label
        bbox = bboxes[lab - 1] if 0 < lab <= len(bboxes) else None
        if bbox is None:
            continue
        sy, sx = bbox
        # Work on small bbox-local slices (typical ~30² pixels vs full ~302²)
        cell_label_local = cell_label[sy, sx]
        cell_pix_mask = cell_label_local == lab
        ys_local, xs_local = np.nonzero(cell_pix_mask)
        if ys_local.size == 0:
            continue
        # Offset back to tile coordinates for output positions later
        y_off, x_off = sy.start, sx.start
        ys_px = ys_local + y_off
        xs_px = xs_local + x_off
        cell_area_um2 = float(ys_local.size) * px_to_um2
        typical_area = float(per_type_area.get(t, default_area))
        if typical_area <= 0:
            typical_area = default_area
        # visible_fraction only scales DOWN when the cell is clipped at
        # the tile boundary. Check the original tile-coord extrema.
        touches_boundary = (
            (ys_px == 0).any() or (ys_px == h - 1).any() or
            (xs_px == 0).any() or (xs_px == w - 1).any()
        )
        if touches_boundary:
            visible_fraction = float(np.clip(cell_area_um2 / typical_area, 0.01, 1.0))
        else:
            visible_fraction = 1.0

        nb = per_type_negbin.get(t) or default_negbin
        if nb is None:
            base_count = 20.0
            log_std = 0.0
        else:
            base_count = float(nb["mean"])
            log_std = float(nb.get("log_std", 0.0))
        # Optional per-type rate scaling (default 1.0). The cellAdmix-fit
        # per-type negbin mean tends to be ~10-15% below the real per-
        # cell rate (cellAdmix only counts assigned tx); combined with
        # boundary visible_fraction the synth tx/cell ends up ~75% of
        # real on pancreas. tx_rate_scale boosts the mean uniformly.
        # See misc/per_gene_divergence_findings.md for the analysis.
        #
        # Log-normal Poisson mixture (Cox process): drawing n ~ Poisson(mean)
        # is under-dispersed vs real (synth max ~88 vs real max ~644 on
        # pancreas, B6 diagnostic). The priors store both mean and log_std
        # (std of log1p of real per-cell counts), so the principled draw is
        #   lambda_cell ~ LogNormal(log(mean) - log_std^2/2, log_std^2)
        #   n_mol ~ Poisson(lambda_cell)
        # which preserves mean = base_count AND matches the real heavy tail.
        scale = float(visible_fraction) * float(tx_rate_scale)
        if log_std > 0.0:
            mu = np.log(max(base_count * scale, 1e-9)) - 0.5 * log_std ** 2
            lam = float(np.exp(mu + log_std * rng.standard_normal()))
        else:
            lam = base_count * scale
        n_mol = max(0, int(round(rng.poisson(lam))))
        if n_mol == 0:
            continue
        nuc_label_local = nuc_label[sy, sx]
        nuc_mask_cell = (nuc_label_local == lab) & cell_pix_mask
        has_nuc = bool(nuc_mask_cell.any())

        # ---- Compartment masks within this cell (bbox-local) -----------
        # Compartment 0 = nuclear; 1 = perinuc; 2 = cyto; 3 = distal.
        # Distances are µm from nucleus boundary. EDT is now on the
        # bbox-local array (typically ~30² vs ~302² tile-wide).
        if use_compartments and has_nuc:
            dist_px = distance_transform_edt(~nuc_mask_cell)
            dist_um = dist_px * pixel_size_um
            b1, b2, b3 = comp_boundaries[0], comp_boundaries[1], comp_boundaries[2]
            out_mask = cell_pix_mask & ~nuc_mask_cell
            comp_masks = [
                nuc_mask_cell,
                out_mask & (dist_um > b1) & (dist_um <= b2),
                out_mask & (dist_um > b2) & (dist_um <= b3),
                out_mask & (dist_um > b3),
            ]
        else:
            # No nucleus → use the whole cell as compartment 1 (perinuc-ish)
            # and leave other compartments empty (sampler will fall back).
            comp_masks = [
                np.zeros_like(cell_pix_mask),
                cell_pix_mask,
                np.zeros_like(cell_pix_mask),
                np.zeros_like(cell_pix_mask),
            ]
        # Convert bbox-local coords back to tile coords by adding (y_off, x_off)
        comp_coords: list[np.ndarray] = []
        for m in comp_masks:
            local = np.argwhere(m)
            if local.size:
                local = local + np.array([y_off, x_off], dtype=local.dtype)
            comp_coords.append(local)
        cell_pix_coords = np.column_stack([ys_px, xs_px])

        # Sample n_mol factors and genes vectorized.
        k_assign = rng.choice(len(bio), size=n_mol, p=h_dirichlet)
        # Collect per-molecule factor/gene/compartment arrays across all
        # factors first, then position-sample in one vectorized pass per
        # compartment. Avoids the n_mol-deep Python loop that was 28% of
        # explain CPU time on a typical bundle.
        all_g_idx = np.empty(n_mol, dtype=np.int64)
        all_factors = np.empty(n_mol, dtype=np.int64)
        all_compartments = np.empty(n_mol, dtype=np.int64)
        write_at = 0
        for k in range(len(bio)):
            mols_k = k_assign == k
            n_k = int(mols_k.sum())
            if n_k == 0:
                continue
            g_idx = rng.choice(sig_bio.shape[1], size=n_k, p=sig_bio[k])
            # Look up compartment posteriors for each gene
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
                compartments = (u[:, None] >= csum).sum(axis=1).clip(max=3)
            else:
                nf = nuclear_frac[g_idx]
                compartments = np.where((rng.random(n_k) < nf) & has_nuc, 0, 1)
            all_g_idx[write_at:write_at + n_k] = g_idx
            all_factors[write_at:write_at + n_k] = bio[k]
            all_compartments[write_at:write_at + n_k] = compartments
            write_at += n_k
        # Trim to actual emitted count (defensive — should equal n_mol)
        all_g_idx = all_g_idx[:write_at]
        all_factors = all_factors[:write_at]
        all_compartments = all_compartments[:write_at]
        if write_at == 0:
            continue

        # Resolve fallback per cell (not per molecule): for each compartment
        # index, decide which non-empty compartment's coords array to actually
        # sample from. Falls through nucleus → cell → anywhere chain.
        def _resolve_fallback(ci: int) -> np.ndarray:
            coords = comp_coords[ci]
            if coords.shape[0] > 0:
                return coords
            fallback_order = (1, 0, 2, 3) if ci != 1 else (0, 2, 3, 1)
            for ci_fb in fallback_order:
                if comp_coords[ci_fb].shape[0] > 0:
                    return comp_coords[ci_fb]
            return cell_pix_coords

        # Batch-sample positions per compartment then assemble output
        ys_arr = np.empty(write_at, dtype=np.float64)
        xs_arr = np.empty(write_at, dtype=np.float64)
        for ci in range(4):
            sel = all_compartments == ci
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            coords = _resolve_fallback(ci)
            pis = rng.integers(0, coords.shape[0], size=n_sel)
            picked = coords[pis]                            # (n_sel, 2): (y, x) px
            ys_arr[sel] = picked[:, 0].astype(np.float64) * pixel_size_um
            xs_arr[sel] = picked[:, 1].astype(np.float64) * pixel_size_um
        # Jitter all molecules in one shot
        ys_arr += rng.uniform(-nuclear_jitter_um, nuclear_jitter_um, size=write_at)
        xs_arr += rng.uniform(-nuclear_jitter_um, nuclear_jitter_um, size=write_at)

        # Append batched outputs (one Python-level extend per cell vs
        # n_mol per-molecule appends).
        cell_ids.extend([c.cell_id] * write_at)
        genes_out.extend(gene_names[int(g)] for g in all_g_idx)
        xs.extend(xs_arr.tolist())
        ys.extend(ys_arr.tolist())
        factor_labels.extend(int(f) for f in all_factors)
        qvs.extend([float(synthetic_qv)] * write_at)
        src_types.extend([t] * write_at)

    return pd.DataFrame({
        "cell_id": cell_ids, "gene": genes_out,
        "x": xs, "y": ys,
        "factor_label": factor_labels, "qv": qvs,
        "source_cell_type": src_types,
    })


def cell_h_dict_from_factor_fractions(
    priors: dict[str, Any],
    cf: "pd.DataFrame",
) -> dict[str, np.ndarray]:
    """Extract per-cell biology-factor vectors from a cell_factor_fractions DataFrame.

    `biology_factors` are 1-based IDs; column names match
    `factor_K_fraction` directly (no shift)."""
    bio = list(priors["factor_signatures"]["biology_factors"])
    fac_cols = [f"factor_{int(k)}_fraction" for k in bio]
    return dict(zip(cf["cell_id"].tolist(),
                      cf[fac_cols].to_numpy(dtype=np.float64)))


def load_molecule_factor_assignments(
    model_dir: str | Path,
    priors: dict[str, Any] | None = None,
    *,
    columns: list[str] | None = None,
    with_gene_names: bool = True,
    with_cell_ids: bool = True,
    bundle_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
    include_unassigned: bool = False,
    unassigned_min_qv: float = 20.0,
) -> "pd.DataFrame":
    """Read per-molecule factor assignments via cellAdmix's public API.

    Uses `fit.molecules()` which exposes 1-based `factor` (1..K) and
    `factor_label` ('F1..FK'). The on-disk parquet's raw 0-based encoding is
    not surfaced here (use cellAdmix's `fit.molecules(raw=True)` for that).

    Default columns include `factor`, `factor_label`, `factor_margin`, plus
    `cell_id`, `gene`, `x`, `y`, `qv`, `overlaps_nucleus`,
    `nucleus_distance`. Pass `columns` to restrict.

    `include_unassigned=True` augments the result with the bundle's
    `cell_id == 'UNASSIGNED'` transcripts (with `qv >= unassigned_min_qv`),
    each labeled with a `factor` chosen as the argmax over per-gene loadings
    in the cellAdmix `factors.parquet`. `factor_margin` is the normalized
    gap between top and runner-up loadings. Synthetic `cell_idx = -1`,
    `cell_id = 'UNASSIGNED'`. Required by transcript-based cell proposers,
    which can only find missed cells from orphan transcripts.
    """
    model_dir = Path(model_dir)
    if priors is None:
        priors = load_transcripts_priors(model_dir)
        if priors is None:
            raise RuntimeError(f"no transcripts priors at {model_dir}/priors/")
    fit = _resolve_celladmix_fit(model_dir, priors,
                                    bundle_path=bundle_path,
                                    annotation_path=annotation_path)
    requested = columns
    if requested is None:
        requested = ["x", "y", "gene_idx", "cell_idx", "factor", "factor_label",
                       "factor_margin", "qv", "overlaps_nucleus", "nucleus_distance"]
    df = fit.molecules(columns=requested)

    # input_store is bundle-side under the new layout, model-side under legacy.
    if bundle_path is not None:
        bca_store = bundle_celladmix_dir(bundle_path) / "input_store"
        store_dir = bca_store if bca_store.exists() else Path(model_dir) / "_celladmix" / "input_store"
    else:
        store_dir = Path(model_dir) / "_celladmix" / "input_store"
    if with_cell_ids and "cell_idx" in df.columns and "cell_id" not in df.columns:
        cells = pd.read_parquet(store_dir / "cells.parquet",
                                  columns=["cell_idx", "cell_id"])
        df = df.merge(cells, on="cell_idx", how="left")
    if with_gene_names and "gene_idx" in df.columns and "gene" not in df.columns:
        genes = pd.read_parquet(store_dir / "genes.parquet")
        gene_col = "gene" if "gene" in genes.columns else genes.columns[-1]
        df = df.merge(genes[["gene_idx", gene_col]].rename(columns={gene_col: "gene"}),
                       on="gene_idx", how="left")

    if include_unassigned:
        if bundle_path is None:
            raise RuntimeError("include_unassigned=True requires bundle_path")
        orphan_df = _load_unassigned_with_factors(
            bundle_path=bundle_path,
            priors=priors,
            store_dir=store_dir,
            min_qv=unassigned_min_qv,
            requested_columns=requested,
            with_gene_names=with_gene_names,
            with_cell_ids=with_cell_ids,
        )
        if len(orphan_df) > 0:
            for col in df.columns:
                if col not in orphan_df.columns:
                    orphan_df[col] = None
            df = pd.concat([df, orphan_df[df.columns]], ignore_index=True)
    return df


def _load_unassigned_with_factors(
    *,
    bundle_path: str | Path,
    priors: dict[str, Any],
    store_dir: Path,
    min_qv: float,
    requested_columns: list[str],
    with_gene_names: bool,
    with_cell_ids: bool,
) -> "pd.DataFrame":
    """Build a DataFrame of bundle-level UNASSIGNED transcripts with per-gene
    argmax-factor labels from the cellAdmix factors.parquet."""
    run_dir = _celladmix_run_dir(Path("/dev/null"), priors, bundle_path=bundle_path)
    factors = pd.read_parquet(run_dir / "factors.parquet")

    # Pivot to gene_idx × factor_id matrix of loadings.
    loadings = factors.pivot_table(index="gene_idx", columns="factor_id",
                                       values="loading", fill_value=0.0)
    loadings = loadings.sort_index()
    L = loadings.to_numpy(dtype=np.float32)
    factor_ids = np.asarray(loadings.columns, dtype=np.int32)
    # argmax + runner-up margin per gene
    order = np.argsort(-L, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1] if L.shape[1] > 1 else order[:, 0]
    top_val = L[np.arange(L.shape[0]), top_idx]
    second_val = L[np.arange(L.shape[0]), second_idx]
    denom = top_val + second_val + 1e-12
    gene_to_factor = factor_ids[top_idx]
    gene_to_margin = (top_val - second_val) / denom
    gene_idx_arr = np.asarray(loadings.index, dtype=np.int64)
    gene_factor_map = dict(zip(gene_idx_arr, gene_to_factor))
    gene_margin_map = dict(zip(gene_idx_arr, gene_to_margin))

    # gene name -> gene_idx (cellAdmix's encoding)
    genes_pq = pd.read_parquet(store_dir / "genes.parquet")
    gene_name_col = "gene" if "gene" in genes_pq.columns else genes_pq.columns[-1]
    name_to_idx = dict(zip(genes_pq[gene_name_col].astype(str), genes_pq["gene_idx"].astype(np.int64)))

    raw = pd.read_parquet(Path(bundle_path) / "transcripts.parquet",
                            columns=["cell_id", "feature_name", "x_location",
                                     "y_location", "z_location", "qv",
                                     "overlaps_nucleus", "nucleus_distance"])
    raw = raw[(raw["cell_id"] == "UNASSIGNED") & (raw["qv"] >= float(min_qv))]
    if len(raw) == 0:
        return pd.DataFrame(columns=requested_columns)
    raw = raw.assign(gene_idx=raw["feature_name"].astype(str).map(name_to_idx))
    raw = raw[raw["gene_idx"].notna()].copy()
    raw["gene_idx"] = raw["gene_idx"].astype(np.int64)
    raw["factor"] = raw["gene_idx"].map(gene_factor_map).astype(np.int32)
    raw["factor_margin"] = raw["gene_idx"].map(gene_margin_map).astype(np.float32)
    raw["factor_label"] = "F" + raw["factor"].astype(str)
    out = pd.DataFrame({
        "x": raw["x_location"].to_numpy(dtype=np.float32),
        "y": raw["y_location"].to_numpy(dtype=np.float32),
        "z": raw["z_location"].to_numpy(dtype=np.float32) if "z_location" in raw.columns else 0.0,
        "gene_idx": raw["gene_idx"].to_numpy(dtype=np.int64),
        "cell_idx": np.full(len(raw), -1, dtype=np.int64),
        "factor": raw["factor"].to_numpy(dtype=np.int32),
        "factor_label": raw["factor_label"].to_numpy(),
        "factor_margin": raw["factor_margin"].to_numpy(dtype=np.float32),
        "qv": raw["qv"].to_numpy(dtype=np.float32),
        "overlaps_nucleus": raw["overlaps_nucleus"].to_numpy() if "overlaps_nucleus" in raw.columns else False,
        "nucleus_distance": raw["nucleus_distance"].to_numpy(dtype=np.float32) if "nucleus_distance" in raw.columns else 0.0,
    })
    if with_cell_ids:
        out["cell_id"] = "UNASSIGNED"
    if with_gene_names:
        out = out.merge(genes_pq[["gene_idx", gene_name_col]].rename(
            columns={gene_name_col: "gene"}), on="gene_idx", how="left")
    return out
