"""Tests for the 2.5D 3D-priors fitting code.

Covers:
- nucleus_prior: expected_section_quantiles is monotone in log_radius_mean
  and axis_ratio_log_kappa; end-to-end fit + JSON round-trip.
- fit_priors: bundle-native per-cell polygon stats and annotation merge.

Carried over from plan3d as the keepable subset of test_plan3d_section.py
— the posterior/HMC/audit tests that depended on canonical crops are
dropped (those research paths are not on plan3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_nucleus_prior_quantile_consistency():
    """Larger log_radius_mean should produce larger expected nucleus
    section area; larger axis kappa should produce larger expected
    axis ratio."""
    from xesim.scene_2_5d.nucleus_prior import expected_nucleus_section_quantiles
    rng = np.random.default_rng(0)
    q1 = expected_nucleus_section_quantiles(0.5, 0.1, (0.0, 0.0, 0.0),
                                              n_samples=2000, rng=rng)
    q2 = expected_nucleus_section_quantiles(1.5, 0.1, (0.0, 0.0, 0.0),
                                              n_samples=2000, rng=rng)
    assert q2["area"]["median"] > q1["area"]["median"]
    # Larger anisotropy -> larger axis ratio
    q3 = expected_nucleus_section_quantiles(1.0, 0.1, (0.5, -0.5, 0.0),
                                              n_samples=2000, rng=rng)
    assert q3["axis_ratio"]["median"] > q1["axis_ratio"]["median"]


def test_fit_priors_per_cell_polygon_stats_matches_naive():
    """The vectorized per_cell_polygon_stats reduceat path produces the same
    area / axis ratio as a naive Python-loop reference on a small synthetic
    bundle of two cells (a circle and a 3:1 ellipse)."""
    import pandas as pd
    from xesim.scene_2_5d.fit_priors import per_cell_polygon_stats

    # Cell A: ~circle, 60 vertices on r=5 µm.
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    x_a, y_a = 5.0 * np.cos(theta), 5.0 * np.sin(theta)
    # Cell B: 3:1 ellipse, 60 vertices on (a, b) = (6, 2) µm, centered at (100, 50).
    x_b = 100.0 + 6.0 * np.cos(theta)
    y_b = 50.0 + 2.0 * np.sin(theta)
    df = pd.DataFrame({
        "cell_id": ["A"] * 60 + ["B"] * 60,
        "vertex_x": np.concatenate([x_a, x_b]),
        "vertex_y": np.concatenate([y_a, y_b]),
    })
    stats = per_cell_polygon_stats(df)
    by_id = dict(zip(stats["cell_id"], range(len(stats))))
    # Circle area ≈ π * 25
    assert abs(stats.loc[by_id["A"], "area_um2"] - np.pi * 25.0) < 0.5
    assert abs(stats.loc[by_id["A"], "axis_ratio"] - 1.0) < 0.05
    # Ellipse area ≈ π * 6 * 2 = 12π
    assert abs(stats.loc[by_id["B"], "area_um2"] - np.pi * 12.0) < 0.5
    # Axis ratio of a 3:1 ellipse (from vertex covariance) ≈ 3
    assert abs(stats.loc[by_id["B"], "axis_ratio"] - 3.0) < 0.2


def test_fit_priors_attach_cell_types(tmp_path):
    """attach_cell_types reads gzip CSV and merges by cell_id."""
    import pandas as pd
    from xesim.scene_2_5d.fit_priors import attach_cell_types

    ann_path = tmp_path / "annotation.csv.gz"
    pd.DataFrame({
        "cell_id": ["A", "B", "C"],
        "merged_annotation": ["t1", "t2", "t1"],
    }).to_csv(ann_path, index=False, compression="gzip")
    stats = pd.DataFrame({"cell_id": ["A", "B", "D"], "area_um2": [1.0, 2.0, 3.0]})
    merged = attach_cell_types(stats, ann_path, type_col="merged_annotation")
    assert merged.loc[merged["cell_id"] == "A", "cell_type"].iloc[0] == "t1"
    assert merged.loc[merged["cell_id"] == "B", "cell_type"].iloc[0] == "t2"
    # D has no annotation -> NaN, not crash
    assert pd.isna(merged.loc[merged["cell_id"] == "D", "cell_type"].iloc[0])


def test_nucleus_prior_fit_and_load(tmp_path):
    """End-to-end smoke: fit a nucleus prior to synthetic target quantiles,
    write JSON, reload."""
    from xesim.scene_2_5d.nucleus_prior import (
        fit_nucleus_type_prior_to_quantiles,
        load_nucleus_priors,
        NucleusTypePrior,
    )
    target_a = {"p10": 8.0, "p25": 12.0, "p50": 18.0, "p75": 25.0, "p90": 30.0}
    target_ar = {"p10": 1.05, "p25": 1.10, "p50": 1.20, "p75": 1.50, "p90": 1.80}
    prior = fit_nucleus_type_prior_to_quantiles(
        target_a, target_ar, n_train=100, type_name="test",
        n_iter=20, n_samples=2000, seed=0,
    )
    assert prior.name == "test"
    assert 0.0 < prior.log_radius_mean < 3.0
    assert prior.log_radius_std >= 0.05
    # Round-trip via JSON
    out_path = tmp_path / "nuc_priors.json"
    out_path.write_text(json.dumps({
        "type": "xesim.nucleus_prior.v0",
        "schema_version": "0.1.0",
        "kind": "nucleus",
        "priors": {"test": prior.to_dict()},
    }, indent=2))
    loaded = load_nucleus_priors(out_path)
    assert "test" in loaded
    assert isinstance(loaded["test"], NucleusTypePrior)
    assert abs(loaded["test"].log_radius_mean - prior.log_radius_mean) < 1e-6


def test_tilt_load_nucleus_priors_table_auto_discover(tmp_path):
    """The tilt-side auto-discover loader picks up
    <model_dir>/priors_3d/nucleus_priors.json and converts each entry to
    {vol_um3, elong}."""
    from xesim.scene_2_5d.tilt import (
        load_nucleus_priors_table, per_type_c_along_t, DEFAULT_NUCLEUS_PRIOR,
    )

    # Missing dir -> empty table; per_type falls back to single-median default.
    assert load_nucleus_priors_table(model_dir=tmp_path / "no_such") == {}
    c_default = per_type_c_along_t("anything", priors_table={})
    assert c_default > 0

    # Fitted-prior JSON in the expected location.
    priors_dir = tmp_path / "priors_3d"
    priors_dir.mkdir()
    (priors_dir / "nucleus_priors.json").write_text(json.dumps({
        "type": "xesim.nucleus_prior.v0",
        "schema_version": "0.1.0",
        "kind": "nucleus",
        "priors": {
            "T1": {
                "name": "T1", "n_train": 100,
                "log_radius_mean": 1.0,
                "log_radius_std": 0.2,
                "axis_ratio_log_kappa": [0.5, -0.25, -0.25],
                "target_area_quantiles": {},
                "target_axis_ratio_quantiles": {},
                "fit_diagnostics": {},
            }
        },
    }))
    table = load_nucleus_priors_table(model_dir=tmp_path)
    assert "T1" in table
    assert table["T1"]["vol_um3"] > 0
    assert table["T1"]["elong"] > 1.0  # zero-sum kappas not all equal
    # Per-type lookup uses the loaded entry; unknown types fall back.
    c_t1 = per_type_c_along_t("T1", priors_table=table)
    c_unk = per_type_c_along_t("UNKNOWN", priors_table=table)
    assert c_t1 != c_unk  # different parameters -> different long-axis length
