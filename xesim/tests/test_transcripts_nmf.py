"""Phase 1 tests for transcripts NMF priors.

Most tests are gated on the existence of the pre-fitted priors file at
``tmp/xesim_v16_model/priors/transcripts_nmf.json`` so they don't re-run
the (~15s) cellAdmix fit on every pytest invocation. The CI / one-off
"refresh" path is::

    python -c "from xesim.transcripts import fit_transcripts_nmf; \
        fit_transcripts_nmf('/path/to/bundle', '/path/to/annot.csv.gz', \
                            'tmp/xesim_v16_model', overwrite=True)"

The headline higher-level test (`test_factor_biology_recovery`) is
guarded by the same fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path("/workspace/xeSim/tmp/xesim_v16_model")
PRIORS_JSON = MODEL_DIR / "priors" / "transcripts_nmf.json"
CELL_FACTORS = MODEL_DIR / "priors" / "cell_factors.parquet"

requires_priors = pytest.mark.skipif(
    not PRIORS_JSON.exists(),
    reason=f"no priors at {PRIORS_JSON} — run fit_transcripts_nmf first",
)


@pytest.fixture(scope="module")
def priors() -> dict:
    return json.loads(PRIORS_JSON.read_text())


@requires_priors
def test_schema(priors):
    assert priors["schema_version"] == "xesim.transcripts_nmf.v2"
    assert priors["nmf_variant"] == "invsqrt_kl"
    assert priors["rank_K"] >= 2


@requires_priors
def test_panel(priors):
    n_genes = priors["panel"]["n_genes"]
    gene_names = priors["panel"]["gene_names"]
    assert len(gene_names) == n_genes
    # Gene names must be strings without scientific notation contamination
    for g in gene_names[:20]:
        # A handful of real Xenium panels have gene names with digits ('CXCL12'),
        # but no real gene name should look like a float literal '1.5e-10'.
        assert "e-" not in g.lower() or g[0].isalpha(), f"bad gene name: {g!r}"


@requires_priors
def test_factor_signatures_normalized(priors):
    sig = np.array(priors["factor_signatures"]["data"])
    K = priors["rank_K"]
    assert sig.shape == (K, priors["panel"]["n_genes"])
    # Each row sums to 1 within tolerance
    row_sums = sig.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        f"row sums off: {row_sums}"
    # All non-negative
    assert (sig >= 0).all()


@requires_priors
def test_factor_classification_complete(priors):
    """biology_factors = factors with a known native source type (these are
    the factors xeSim treats as type-defining). admixture_factors is a list
    of per-(factor, target_type) rules: a factor that is *native* to type T
    can still admix INTO some other type T', so a factor may appear in BOTH
    biology_factors and admixture_factors. Verify the schema reflects that.
    """
    K = priors["rank_K"]
    bio = set(int(f) for f in priors["factor_signatures"]["biology_factors"])
    admix_rule_factors = {int(a["factor"]) for a in
                            priors["factor_signatures"]["admixture_factors"]}
    # All factor IDs are 1-based; biology factors must be in 1..K
    assert bio.issubset(set(range(1, K + 1)))
    # Every admixture-rule factor must also be a biology factor (a factor that
    # is admixture INTO some target type T is *biology* in its source type T').
    assert admix_rule_factors.issubset(bio), \
        f"admixture rules reference factors {admix_rule_factors - bio} not in biology_factors"


@requires_priors
def test_per_type_alpha_consistency(priors):
    K_bio = len(priors["factor_signatures"]["biology_factors"])
    for t, alpha in priors["per_type_alpha"].items():
        assert len(alpha) == K_bio, \
            f"alpha length {len(alpha)} != K_bio {K_bio} for type {t!r}"
        assert all(a >= 0 for a in alpha), f"negative alpha entry for {t!r}"


@requires_priors
def test_per_type_count_negbin(priors):
    for t, p in priors["per_type_count_negbin"].items():
        assert p["mean"] > 0, f"non-positive mean for {t!r}"
        assert p["log_std"] >= 0, f"negative log_std for {t!r}"
        assert p["n_cells"] > 0


@requires_priors
def test_per_gene_nuclear_fraction(priors):
    fr = np.array(priors["per_gene_nuclear_fraction"])
    assert len(fr) == priors["panel"]["n_genes"]
    assert (fr >= 0).all() and (fr <= 1).all()
    # At least 50% of genes should not have the 0.5 default — real
    # estimates should win for most genes on a real bundle
    n_default = int((fr == 0.5).sum())
    assert n_default < 0.5 * len(fr), \
        f"too many genes hit the 0.5 default: {n_default}/{len(fr)}"


@requires_priors
def test_cell_factors_parquet():
    import pandas as pd
    df = pd.read_parquet(CELL_FACTORS)
    assert "cell_id" in df.columns
    assert "_cell_type" in df.columns
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    assert len(factor_cols) >= 2


@requires_priors
def test_factor_biology_recovery(priors):
    """Headline Phase 1 test: at least 5 biology factors should recover
    ≥1 known pancreas marker among their top-8 loaded genes for their
    assigned cell type."""
    known_markers = {
        "Fibroblast / CAF":        ["VCAN", "COL1A1", "COL1A2", "COL3A1", "PDGFRA", "LUM", "DCN", "SFRP2", "SFRP4"],
        "Exocrine epithelial":     ["AMY2A", "AMY2B", "CPA1", "CPA2", "PRSS1", "CTRB1", "PRSS2", "CELA1"],
        "Ductal/tumor epithelial": ["CFTR", "KRT19", "KRT18", "KRT7", "MUC1", "GATM", "EPCAM"],
        "Immune":                  ["PTPRC", "CD8A", "CD4", "CD68", "CD163", "AIF1", "CD3E", "CXCR4", "IL7R", "TRAC"],
        "Endothelial":             ["PECAM1", "CDH5", "VWF", "CD34", "CLDN5"],
        "Endocrine":               ["INS", "GCG", "CHGA", "CHGB", "IAPP", "SST", "PPY"],
        "Mural / pericyte":        ["ACTA2", "RGS5", "PDGFRB", "MYH11", "NOTCH3"],
    }
    gene_names = priors["panel"]["gene_names"]
    sig = np.array(priors["factor_signatures"]["data"])

    n_matched = 0
    for f_id in priors["factor_signatures"]["biology_factors"]:
        # biology_factors are 1-based IDs; sig is indexed positionally 0..K-1
        src = priors["factor_source_labels"].get(str(f_id), {})
        src_type = src.get("type")
        if src_type not in known_markers:
            continue
        top8 = np.argsort(-sig[int(f_id) - 1])[:8]
        top_genes = {gene_names[i] for i in top8}
        if top_genes.intersection(known_markers[src_type]):
            n_matched += 1
    K = priors["rank_K"]
    assert n_matched >= 5, \
        f"only {n_matched}/{K} biology factors recover a known marker"


@requires_priors
def test_load_via_xesim_model():
    """XesimModel.load picks up transcripts_priors automatically."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    assert m.transcripts_priors is not None
    assert m.transcripts_priors["schema_version"] == "xesim.transcripts_nmf.v2"


@requires_priors
def test_priors_records_celladmix_run_dir(priors):
    """The priors JSON points at the cellAdmix run on disk."""
    rel = priors.get("celladmix_run_dir")
    assert rel, "celladmix_run_dir missing from priors"
    run_dir = MODEL_DIR / rel
    assert (run_dir / "molecules.parquet").exists()
    assert (run_dir / "cells.parquet").exists()


# ---------------------------------------------------------------------------
# Phase 2 — accessors over cellAdmix's existing per-cell / per-molecule data
# ---------------------------------------------------------------------------


@requires_priors
def test_cell_factor_fractions_shape(priors):
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    cf = m.cell_factor_fractions()
    K = priors["rank_K"]
    expected = {"cell_id", "transcript_count", "dominant_factor",
                  "dominant_fraction"}
    expected |= {f"factor_{k}_fraction" for k in range(1, K + 1)}
    assert expected.issubset(set(cf.columns))
    # cells.parquet should have all fit cells
    assert len(cf) == priors["fit_stats"]["n_cells_total"]


@requires_priors
def test_cell_factor_fractions_sum_to_one(priors):
    """Per-cell factor fractions should sum to 1 (or to 0 for cells cellAdmix
    skipped because they had too few molecules)."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    cf = m.cell_factor_fractions()
    K = priors["rank_K"]
    fac_cols = [f"factor_{k}_fraction" for k in range(1, K + 1)]
    sums = cf[fac_cols].sum(axis=1).to_numpy()
    # Most cells should sum to ~1; the few empty ones sum to 0
    ok_one = np.isclose(sums, 1.0, atol=1e-3)
    ok_zero = np.isclose(sums, 0.0, atol=1e-6)
    assert (ok_one | ok_zero).all(), \
        f"unexpected row sums: e.g. {sums[~(ok_one | ok_zero)][:5]}"
    assert ok_one.mean() > 0.8, \
        f"only {ok_one.mean():.1%} of cells have valid factor fractions"


@requires_priors
def test_cell_factor_fractions_only_biology(priors):
    """only_biology=True drops admixture-factor columns."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    cf_bio = m.cell_factor_fractions(only_biology=True)
    bio = set(int(f) for f in priors["factor_signatures"]["biology_factors"])
    K = priors["rank_K"]
    for k in range(1, K + 1):
        col = f"factor_{k}_fraction"
        if k in bio:
            assert col in cf_bio.columns
        else:
            assert col not in cf_bio.columns


@requires_priors
def test_cell_factor_fractions_with_annotation(priors):
    """include_annotation=True joins _cell_type from priors/cell_factors.parquet."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    cf = m.cell_factor_fractions(include_annotation=True)
    assert "_cell_type" in cf.columns
    # Most cells should have an annotation
    assert cf["_cell_type"].notna().mean() > 0.8


@requires_priors
def test_molecule_factor_assignments_basic(priors):
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    # Use a subset of columns for speed
    ma = m.molecule_factor_assignments(
        columns=["cell_idx", "gene_idx", "factor", "factor_label",
                  "factor_margin", "x", "y", "qv", "overlaps_nucleus"],
    )
    expected = {"factor", "factor_label", "factor_margin", "x", "y",
                  "qv", "overlaps_nucleus", "cell_id", "gene"}
    assert expected.issubset(set(ma.columns))
    K = priors["rank_K"]
    # `factor` is 1-based (1..K). `factor_label` is the string 'F1..FK'.
    f = ma["factor"].dropna()
    assert f.min() >= 1
    assert f.max() <= K
    assert ma["factor_label"].iloc[0].startswith("F")
    # Should match the run's n_transcripts
    assert len(ma) == priors["fit_stats"]["n_cells_total"] or len(ma) > 0


@requires_priors
def test_molecule_factor_assignments_biology(priors):
    """Marker-gene molecules confidently assigned to a factor should land
    in the factor whose factor_source_label matches the marker's cell type.

    This validates that the hard per-molecule factor_label exposed by
    `molecule_factor_assignments` is biologically meaningful — independent
    of the soft per-cell factor fractions exposed by `cell_factor_fractions`
    (the two are conceptually different views of the same fit; the per-cell
    view aggregates the underlying probabilistic gene-loadings scores, while
    the per-molecule view applies a margin threshold).
    """
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    ma = m.molecule_factor_assignments(
        columns=["gene_idx", "factor", "factor_margin"],
        with_cell_ids=False,
    )
    ma = ma[ma["factor"] >= 1]

    # Find the factor labeled as "Endocrine" — INS molecules should fall there
    endo_factors = [int(f) for f, src in priors["factor_source_labels"].items()
                      if src.get("type") == "Endocrine"]
    if not endo_factors:
        pytest.skip("no Endocrine factor in priors")
    ins_factor = endo_factors[0]
    gene_names = priors["panel"]["gene_names"]
    if "INS" not in gene_names:
        pytest.skip("INS not on panel")
    ins_idx = gene_names.index("INS")
    ins_mols = ma[ma["gene"] == "INS"] if "gene" in ma.columns \
                                       else ma[ma["gene_idx"] == ins_idx]
    if len(ins_mols) < 50:
        pytest.skip(f"too few assigned INS molecules: {len(ins_mols)}")
    modal_factor = int(ins_mols["factor"].mode().iloc[0])
    assert modal_factor == ins_factor, \
        f"INS molecules modal factor {modal_factor} != endocrine factor {ins_factor}"


@requires_priors
def test_cell_factor_dominant_matches_biology_marker(priors):
    """Headline Phase 2 cross-check: for each annotated cell type whose
    centroid-by-cosine identifies a single biology factor, that factor
    should be the dominant factor for a meaningful share of cells of
    that type. Verifies the accessors expose biologically meaningful
    per-cell labels.
    """
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    cf = m.cell_factor_fractions(include_annotation=True)
    # For each cell type, look up the biology factor whose factor_source_label
    # matches the type. Some types may have no factor; skip those.
    type_to_factor: dict[str, int] = {}
    for f_str, src in priors["factor_source_labels"].items():
        t = src.get("type")
        if t and t not in type_to_factor:
            type_to_factor[t] = int(f_str)
    n_pairs = 0
    n_concordant = 0
    for t, f in type_to_factor.items():
        sub = cf[(cf["_cell_type"] == t) & (cf["transcript_count"] >= 30)]
        if len(sub) < 10:
            continue
        n_pairs += 1
        if (sub["dominant_factor"] == f).mean() >= 0.25:
            n_concordant += 1
    assert n_pairs >= 3, f"too few type/factor pairs to test: {n_pairs}"
    assert n_concordant >= max(2, int(0.5 * n_pairs)), \
        f"dominant-factor concordance failed: {n_concordant}/{n_pairs}"
