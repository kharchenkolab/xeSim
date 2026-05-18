"""3D shape prior calibration for plan3d.

Cell-type-conditioned 3D ellipsoid prior `p(S | c)`. The 2D observed area
of a triaxial ellipsoid sectioned uniformly through orientation and depth
has a known distribution; we calibrate the population mean axes and axis
ratio so that the marginal 2D distribution of section area matches the real
per-type candidate area distribution.

Approach:
- For each cell type, compute the empirical 2D candidate area distribution
  (only using fully-inside-crop cells to avoid truncation bias).
- Choose a 3D ellipsoid model with mean equivalent radius r_eq and
  axis-ratio (kappa_a, kappa_b, kappa_c) constrained to kappa_a*kappa_b*kappa_c
  = 1 (so r_eq is the volume-equivalent radius in um). Parameterize as
  log-normal radius and Dirichlet axis ratios.
- Use the closed-form Cauchy formula relating 3D ellipsoid surface and the
  expected projected area: E[2D section area | 3D ellipsoid] for a uniform
  random orientation and uniform random depth in [-c, c]. We use a Monte
  Carlo simulation over orientations and depths to compute the expected
  2D-area distribution.
- Fit parameters by matching the empirical (median, p25, p75, p10, p90) of
  the 2D-area distribution per type, by gradient descent (or simply grid
  search since we have few parameters).

Outputs: per-type {r_eq_mean_um, r_eq_logsigma, kappa_axes,
nucleus_radius_ratio, nucleus_axis_ratios}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
import math
import numpy as np


@dataclass
class CellTypePrior:
    name: str
    n_train: int
    log_radius_mean: float          # mean log r_eq (um)
    log_radius_std: float           # sd of log r_eq
    axis_ratio_log_kappa: tuple[float, float, float]
    nucleus_radius_ratio: float
    nucleus_axis_log_kappa: tuple[float, float, float]
    membrane_efficiency_mean: float
    membrane_efficiency_std: float
    dapi_efficiency_mean: float
    dapi_efficiency_std: float
    polya_efficiency_mean: float
    polya_efficiency_std: float
    target_area_quantiles: dict[str, float] = field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_train": self.n_train,
            "log_radius_mean": float(self.log_radius_mean),
            "log_radius_std": float(self.log_radius_std),
            "axis_ratio_log_kappa": [float(x) for x in self.axis_ratio_log_kappa],
            "nucleus_radius_ratio": float(self.nucleus_radius_ratio),
            "nucleus_axis_log_kappa": [float(x) for x in self.nucleus_axis_log_kappa],
            "membrane_efficiency_mean": float(self.membrane_efficiency_mean),
            "membrane_efficiency_std": float(self.membrane_efficiency_std),
            "dapi_efficiency_mean": float(self.dapi_efficiency_mean),
            "dapi_efficiency_std": float(self.dapi_efficiency_std),
            "polya_efficiency_mean": float(self.polya_efficiency_mean),
            "polya_efficiency_std": float(self.polya_efficiency_std),
            "target_area_quantiles": dict(self.target_area_quantiles),
            "fit_diagnostics": dict(self.fit_diagnostics),
        }


def _sample_random_rotation(rng: np.random.Generator, n: int) -> np.ndarray:
    """n x 3 x 3 uniform-random rotation matrices via QR."""
    A = rng.standard_normal((n, 3, 3))
    Q, R = np.linalg.qr(A)
    sgn = np.sign(np.linalg.det(Q))[:, None, None]
    return Q * np.concatenate([sgn, sgn, sgn], axis=-1)


def expected_section_area_quantiles(
    log_radius_mean: float,
    log_radius_std: float,
    axis_ratio_log_kappa: tuple[float, float, float],
    n_samples: int = 4000,
    n_depths: int = 1,
    rng: np.random.Generator | None = None,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.5, 0.75, 0.90),
) -> dict[str, float]:
    """Monte-carlo estimate of the marginal 2D section area distribution.

    For each sample:
      r ~ logN(log_radius_mean, log_radius_std)
      a, b, c = r * kappa  (kappa normalized to volume-preserving with
                            log_kappa minus its mean)
      pick a uniform-random orientation R
      pick a uniform depth t in [-1, 1] (relative to slab in body frame)
      compute the 2D ellipse formed by intersecting plane z = t * c_section
      with the ellipsoid; its area is pi * a' * b' where (a', b') are the
      semiaxes of that intersection. We pick the slab cut perpendicular to
      the lab z-axis: in the body frame it's plane n^T q = d where n = R^T e_z
      and d ~ U(-1, 1) * |n . axes|.

    Returns: dict of quantiles {p10, p25, ..., median, mean, std}.

    NOTE: because we sample uniformly in *body-frame depth fraction*, the
    distribution is biased toward the equator (you're more likely to land
    in a region with higher cross section because we don't rescale by |dn/dt|).
    For now we accept this; a more careful sampler weights by the projected
    z-extent of the ellipsoid.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    log_kappa = np.array(axis_ratio_log_kappa)
    log_kappa = log_kappa - log_kappa.mean()  # volume-preserving
    kappa = np.exp(log_kappa)  # 3 multipliers

    r = np.exp(rng.normal(log_radius_mean, log_radius_std, size=n_samples))
    axes = r[:, None] * kappa[None, :]  # (N, 3)

    R = _sample_random_rotation(rng, n_samples)  # (N, 3, 3)
    # Section plane normal is e_z in lab frame; in body frame it's
    # R^T e_z = R[:, 2, :]^T?  Actually: q = R^T (p - c). For p = (u, v, z) the
    # plane z = const becomes plane: e_z^T (R q + c) = const, i.e.
    # (R^T e_z)^T q = const - e_z^T c. So in body frame plane normal is
    # n = R^T e_z = R[:, :, 2]^T = R[:, 2, :] interpreted as the 3rd row of R.
    n = R[:, 2, :]  # (N, 3)
    # The ellipsoid in body frame is q^T M q <= 1 with M = diag(1/axes^2).
    # Plane n^T q = d intersects ellipsoid when d^2 <= n^T M^{-1} n =
    # sum_i n_i^2 axes_i^2. Range of d for non-empty intersection is
    # [-sqrt(S), sqrt(S)] where S = sum n_i^2 axes_i^2. Sample d uniformly
    # in that interval (this is uniform in body-frame depth along the
    # plane normal).
    S = np.sum(n * n * axes * axes, axis=1)  # (N,)
    sqrtS = np.sqrt(S)
    d = (rng.random(n_samples) * 2 - 1) * sqrtS

    # The intersection ellipse area: A = pi * det(M)^{-1/2} * (1 - d^2/S) /
    # |n|_M where ... actually a cleaner formula:
    # The ellipsoid intersects plane n^T q = d in an ellipse whose area is
    #    A = pi * det(P M^{-1} P)^{1/2} * (1 - d^2 / S),
    # where P = I - n n^T / |n|^2 is the projection onto the plane perp to n.
    # We compute it numerically per sample (small loop).
    # For perf on 4000 samples we vectorize:
    inv_axes_sq = 1.0 / (axes ** 2)  # (N, 3)
    # M^{-1} = diag(axes^2). M restricted to plane perp to n:
    # area = pi * sqrt( |M^{-1}| * (n^T M^{-1} n)^{-1}^? ) * (1 - d^2/S)?
    # Simpler: use known fact for ellipsoid intersection plane through center
    #   A0 = pi * a * b * c / sqrt(sum n_i^2 axes_i^2)?  Not quite either.
    # Use the formula:
    #   A(d) = pi * abc / sqrt( sum_i (n_i^2 axes_i^2) ) * (1 - d^2 / S)
    # Reference: cross-section of a triaxial ellipsoid through plane
    # (n . q = d) has area pi * |a*b*c| / sqrt(sum n_i^2 axes_i^2)
    # at d=0 (the central section), scaling by (1 - d^2/S) for off-center.
    # This assumes the central section's area is pi*abc / sqrt(S).
    abc = axes.prod(axis=1)
    central_area = math.pi * abc / np.sqrt(S + 1e-12)
    rel = 1.0 - d * d / (S + 1e-12)
    area = central_area * np.maximum(rel, 0.0)

    out = {}
    for q in quantiles:
        out[f"p{int(q * 100):02d}"] = float(np.quantile(area, q))
    out["median"] = float(np.median(area))
    out["mean"] = float(np.mean(area))
    out["std"] = float(np.std(area))
    return out


def fit_type_prior_to_quantiles(
    target_quantiles: dict[str, float],
    n_train: int,
    type_name: str,
    n_iter: int = 60,
    n_samples: int = 4000,
    seed: int = 0,
) -> CellTypePrior:
    """Fit a log-normal radius + axis-ratio prior so the simulated 2D-area
    quantiles match the empirical ones.

    We use a simple coordinate-descent-like search over four parameters:
      log_radius_mean, log_radius_std, log_kappa_aspect (1D simplification:
      anisotropy along dominant axis), and a secondary kappa along
      perpendicular axis. We parameterize as (log mu, log sigma, eta1, eta2)
      with kappa = exp(eta1, eta2, -(eta1+eta2)).
    """
    rng = np.random.default_rng(seed)
    target = np.array([target_quantiles["p10"], target_quantiles["p25"],
                        target_quantiles["p50"], target_quantiles["p75"],
                        target_quantiles["p90"]])

    def loss(params: np.ndarray) -> tuple[float, dict[str, float]]:
        lm, ls, e1, e2 = params
        kappa_log = (e1, e2, -(e1 + e2))
        ls = max(0.05, ls)
        q = expected_section_area_quantiles(lm, ls, kappa_log, n_samples=n_samples, rng=rng)
        sim = np.array([q["p10"], q["p25"], q["median"], q["p75"], q["p90"]])
        # Use log-space loss to be scale invariant (areas range over decades)
        L = float(np.mean((np.log(sim + 1e-3) - np.log(target + 1e-3)) ** 2))
        return L, q

    # Initial guess: mean from median, std from spread
    init_lm = math.log(max(1.0, math.sqrt(target_quantiles["p50"] / math.pi)) * 1.4)
    init_ls = 0.35
    params = np.array([init_lm, init_ls, 0.15, 0.0])
    best_loss, best_q = loss(params)
    step_sizes = np.array([0.10, 0.05, 0.10, 0.10])
    for it in range(n_iter):
        # try random perturbation
        cand = params + rng.normal(0, 1, size=4) * step_sizes
        cand_loss, cand_q = loss(cand)
        if cand_loss < best_loss:
            params = cand
            best_loss = cand_loss
            best_q = cand_q
            step_sizes *= 1.05
        else:
            step_sizes *= 0.96
        step_sizes = np.clip(step_sizes, 0.005, 0.5)

    lm, ls, e1, e2 = params
    return CellTypePrior(
        name=type_name,
        n_train=n_train,
        log_radius_mean=float(lm),
        log_radius_std=float(max(0.05, ls)),
        axis_ratio_log_kappa=(float(e1), float(e2), float(-(e1 + e2))),
        nucleus_radius_ratio=0.55,
        nucleus_axis_log_kappa=(0.0, 0.0, 0.0),
        membrane_efficiency_mean=0.6,
        membrane_efficiency_std=0.15,
        dapi_efficiency_mean=0.6,
        dapi_efficiency_std=0.15,
        polya_efficiency_mean=0.55,
        polya_efficiency_std=0.18,
        target_area_quantiles=dict(target_quantiles),
        fit_diagnostics={
            "final_loss": float(best_loss),
            "final_quantiles": best_q,
            "n_iter": n_iter,
            "n_samples": n_samples,
            "init_lm": float(init_lm),
            "init_ls": float(init_ls),
        },
    )


def fit_priors_from_audit(
    audit_path: str,
    splits_path: str,
    out_path: str,
    quantile_levels: tuple[float, ...] = (0.10, 0.25, 0.5, 0.75, 0.90),
    use_split: str = "train",
    only_complete: bool = True,
) -> dict[str, Any]:
    """Fit a per-type 3D-ellipsoid prior from the audit JSON, restricted to
    the train split and only complete (no edge-touching) cells.
    """
    audit = json.loads(open(audit_path).read())
    splits = json.loads(open(splits_path).read())["splits"]
    train_set = set(splits.get(use_split, []))
    type_names = audit.get("type_names", []) or []
    cells = audit["cells"]
    # group by cell_type
    by_type: dict[str, list[float]] = {n: [] for n in type_names}
    for r in cells:
        if r.get("cell_type") is None:
            continue
        if r["crop_id"] not in train_set:
            continue
        if only_complete and r["touches_crop_edge"]:
            continue
        a = r["area_um2"]
        if a is None or not (a > 0):
            continue
        by_type[r["cell_type"]].append(float(a))
    priors: dict[str, dict[str, Any]] = {}
    for name, vals in by_type.items():
        if len(vals) < 30:
            continue
        arr = np.array(vals)
        target = {f"p{int(q*100):02d}": float(np.quantile(arr, q)) for q in quantile_levels}
        target["p50"] = target.get("p50", float(np.median(arr)))
        prior = fit_type_prior_to_quantiles(target, n_train=len(arr), type_name=name)
        priors[name] = prior.to_dict()
    out = {
        "type": "xesim.section_prior.v0",
        "schema_version": "0.1.0",
        "use_split": use_split,
        "only_complete": only_complete,
        "priors": priors,
    }
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out
