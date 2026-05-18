"""3D nucleus shape prior calibration for plan3d.

Parallel to `section_prior.py` but for nuclei. The cell-level prior is
already in place and well validated; this module provides per-type
3D-nucleus shape priors (log-radius mean/std, axis-ratio log_kappa) fit
to the empirical 2D-nucleus distribution observed in real Xenium crops.

Approach:
- Re-open the canonical npz files for each crop in the train split.
- For each cell that is fully inside its crop AND has an associated
  nucleus mask, fit a 2D ellipse to the nucleus mask (regionprops
  major/minor axis lengths). Compute nucleus equivalent radius and
  major/minor ratio.
- Pool per cell type and compute empirical quantiles of:
    nucleus_area_um2  -> p10/p25/p50/p75/p90
    nucleus_axis_ratio (= major/minor of the 2D fitted ellipse)
- Run the same Monte-Carlo-over-orientations-and-depths fit as
  `section_prior.expected_section_area_quantiles`, parameterized by
  log-radius mean+std and a 3D axis-ratio log_kappa.
- Add an axis-ratio quantile match: per sample (random orientation,
  depth), compute the 2D ellipse axis ratio implied by the slab cut
  through the 3D ellipsoid. Match the empirical 2D axis ratio
  quantiles too.
- Output: `nucleus_priors.json` with the same per-type schema as
  `section_priors.json` but with `kind: nucleus`.

This module shares idiomatic structure with `section_prior.py`. It is
deliberately additive — it does NOT modify any existing prior file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import math
import numpy as np


@dataclass
class NucleusTypePrior:
    name: str
    n_train: int
    log_radius_mean: float          # mean log r_eq (um) for the 3D nucleus
    log_radius_std: float           # sd of log r_eq
    axis_ratio_log_kappa: tuple[float, float, float]  # 3D axis kappas
    target_area_quantiles: dict[str, float] = field(default_factory=dict)
    target_axis_ratio_quantiles: dict[str, float] = field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_train": self.n_train,
            "log_radius_mean": float(self.log_radius_mean),
            "log_radius_std": float(self.log_radius_std),
            "axis_ratio_log_kappa": [float(x) for x in self.axis_ratio_log_kappa],
            "target_area_quantiles": dict(self.target_area_quantiles),
            "target_axis_ratio_quantiles": dict(self.target_axis_ratio_quantiles),
            "fit_diagnostics": dict(self.fit_diagnostics),
        }


def _sample_random_rotation(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.standard_normal((n, 3, 3))
    Q, R = np.linalg.qr(A)
    sgn = np.sign(np.linalg.det(Q))[:, None, None]
    return Q * np.concatenate([sgn, sgn, sgn], axis=-1)


def expected_nucleus_section_quantiles(
    log_radius_mean: float,
    log_radius_std: float,
    axis_ratio_log_kappa: tuple[float, float, float],
    n_samples: int = 4000,
    rng: np.random.Generator | None = None,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.5, 0.75, 0.90),
) -> dict[str, dict[str, float]]:
    """Monte-Carlo over orientation x depth.

    For each sample: pick a 3D ellipsoid (radius, kappa), uniform random
    orientation, uniform random depth in feasible range; compute the
    2D ellipse formed by the slab cut, and record:
      - section area
      - axis ratio (major / minor of the 2D ellipse)

    Return dict with two sub-dicts: 'area' and 'axis_ratio', each with
    quantile keys p10/p25/p50/p75/p90 and median/mean/std.

    The 2D ellipse from a triaxial-ellipsoid plane cut has axes derived
    from the conic section of the quadric Q = R diag(1/axes^2) R^T
    restricted to the plane perpendicular to the slab normal. We
    compute it numerically per sample (vectorized).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    log_kappa = np.array(axis_ratio_log_kappa, dtype=float)
    log_kappa = log_kappa - log_kappa.mean()
    kappa = np.exp(log_kappa)

    r = np.exp(rng.normal(log_radius_mean, log_radius_std, size=n_samples))
    axes = r[:, None] * kappa[None, :]  # (N, 3)

    R = _sample_random_rotation(rng, n_samples)  # (N, 3, 3)
    n = R[:, 2, :]  # body-frame plane normal = R^T e_z

    # Range of feasible d
    S = np.sum(n * n * axes * axes, axis=1)
    sqrtS = np.sqrt(S)
    d = (rng.random(n_samples) * 2 - 1) * sqrtS

    abc = axes.prod(axis=1)
    central_area = math.pi * abc / np.sqrt(S + 1e-12)
    rel = 1.0 - d * d / (S + 1e-12)
    area = central_area * np.maximum(rel, 0.0)

    # Per-sample axis ratio of the section ellipse.
    # Ellipsoid in body frame: q^T M q <= 1, M = diag(1/axes^2).
    # Plane: n^T q = d. Restrict M to plane: project onto orthonormal basis
    # of the plane. Build per-sample two orthogonal unit vectors u, v
    # spanning the plane and form M_plane = U^T M U where U = [u v] (3x2).
    # The 2D ellipse equation in (alpha, beta) coords: (q - q0)^T M (q - q0)
    # = 1 - d^2/S where q0 = d * n / |n|^2 (since |n| = 1, q0 = d * n).
    # Eigenvalues of M_plane give 1/r_major^2 and 1/r_minor^2 scaled by
    # (1 - d^2/S).
    # Build u, v: orthogonal to n.
    # Pick e = e_x (or e_y if n is parallel to e_x), Gram-Schmidt
    e = np.zeros_like(n)
    e[:, 0] = 1.0
    parallel_x = np.abs(n[:, 0]) > 0.9
    e[parallel_x, 0] = 0.0
    e[parallel_x, 1] = 1.0
    u = e - (n * (n * e).sum(axis=1, keepdims=True))  # remove n component
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(n, u)
    # M = diag(1/axes^2). M_plane[i,j] = u_i^T M u_j etc.
    inv_axes_sq = 1.0 / (axes ** 2)  # (N, 3)
    Muu = (u * u * inv_axes_sq).sum(axis=1)  # (N,)
    Mvv = (v * v * inv_axes_sq).sum(axis=1)
    Muv = (u * v * inv_axes_sq).sum(axis=1)
    # Eigenvalues of [[Muu, Muv], [Muv, Mvv]]
    tr = Muu + Mvv
    det = Muu * Mvv - Muv * Muv
    disc_eig = np.maximum(tr * tr / 4 - det, 0.0)
    sqrtd = np.sqrt(disc_eig)
    lam1 = tr / 2 + sqrtd  # larger eigval -> smaller axis
    lam2 = tr / 2 - sqrtd  # smaller eigval -> larger axis
    lam2 = np.maximum(lam2, 1e-12)
    # Axes of 2D ellipse: r_axis^2 = (1 - d^2/S) / lam_i
    rel_pos = np.maximum(rel, 1e-9)
    r_minor = np.sqrt(rel_pos / lam1)
    r_major = np.sqrt(rel_pos / lam2)
    axis_ratio = r_major / np.maximum(r_minor, 1e-9)
    # Filter degenerate small/empty sections (rel near zero -> ratio noisy)
    valid = rel > 1e-3
    if not np.any(valid):
        valid = np.ones_like(valid, dtype=bool)
    area_v = area[valid]
    ratio_v = axis_ratio[valid]

    def _quantile_dict(arr: np.ndarray) -> dict[str, float]:
        out = {}
        for q in quantiles:
            out[f"p{int(q * 100):02d}"] = float(np.quantile(arr, q))
        out["median"] = float(np.median(arr))
        out["mean"] = float(np.mean(arr))
        out["std"] = float(np.std(arr))
        return out

    return {"area": _quantile_dict(area_v), "axis_ratio": _quantile_dict(ratio_v)}


def _precompute_section_samples(
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Precompute the parameter-INdependent Monte-Carlo samples used by
    every ``loss(params)`` call inside ``fit_nucleus_type_prior_to_quantiles``.

    Rotations, plane normals and unit-depth samples don't depend on the
    optimization parameters (log_radius_mean, log_radius_std, log_kappa),
    so we generate them once per fit and reuse. This cuts the per-iter
    cost roughly in half (the QR of 4000 random 3x3 matrices was the
    single biggest cost inside the loss function).
    """
    # Normal samples for log-radius scaling (reused via affine transform).
    z = rng.standard_normal(n_samples)
    R = _sample_random_rotation(rng, n_samples)
    n = R[:, 2, :]
    u01 = rng.random(n_samples) * 2 - 1  # uniform [-1, 1]
    # Plane orthonormal basis u, v perpendicular to n.
    e = np.zeros_like(n)
    e[:, 0] = 1.0
    parallel_x = np.abs(n[:, 0]) > 0.9
    e[parallel_x, 0] = 0.0
    e[parallel_x, 1] = 1.0
    u = e - n * (n * e).sum(axis=1, keepdims=True)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(n, u)
    return {"z": z, "n": n, "u": u, "v": v, "u01": u01}


def _section_quantiles_with_pool(
    log_radius_mean: float,
    log_radius_std: float,
    axis_ratio_log_kappa: tuple[float, float, float],
    pool: dict[str, np.ndarray],
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.5, 0.75, 0.90),
) -> dict[str, dict[str, float]]:
    """Same as expected_nucleus_section_quantiles but uses a precomputed
    pool of parameter-independent samples."""
    log_kappa = np.array(axis_ratio_log_kappa, dtype=float)
    log_kappa = log_kappa - log_kappa.mean()
    kappa = np.exp(log_kappa)
    r = np.exp(log_radius_mean + log_radius_std * pool["z"])
    axes = r[:, None] * kappa[None, :]
    n = pool["n"]
    S = np.sum(n * n * axes * axes, axis=1)
    sqrtS = np.sqrt(S)
    d = pool["u01"] * sqrtS
    abc = axes.prod(axis=1)
    central_area = math.pi * abc / np.sqrt(S + 1e-12)
    rel = 1.0 - d * d / (S + 1e-12)
    area = central_area * np.maximum(rel, 0.0)
    u = pool["u"]
    v = pool["v"]
    inv_axes_sq = 1.0 / (axes ** 2)
    Muu = (u * u * inv_axes_sq).sum(axis=1)
    Mvv = (v * v * inv_axes_sq).sum(axis=1)
    Muv = (u * v * inv_axes_sq).sum(axis=1)
    tr = Muu + Mvv
    det_ = Muu * Mvv - Muv * Muv
    disc_eig = np.maximum(tr * tr / 4 - det_, 0.0)
    sqrtd = np.sqrt(disc_eig)
    lam1 = tr / 2 + sqrtd
    lam2 = np.maximum(tr / 2 - sqrtd, 1e-12)
    rel_pos = np.maximum(rel, 1e-9)
    r_minor = np.sqrt(rel_pos / lam1)
    r_major = np.sqrt(rel_pos / lam2)
    axis_ratio = r_major / np.maximum(r_minor, 1e-9)
    valid = rel > 1e-3
    if not np.any(valid):
        valid = np.ones_like(valid, dtype=bool)
    area_v = area[valid]
    ratio_v = axis_ratio[valid]

    def _qd(arr: np.ndarray) -> dict[str, float]:
        out = {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in quantiles}
        out["median"] = float(np.median(arr))
        out["mean"] = float(np.mean(arr))
        out["std"] = float(np.std(arr))
        return out

    return {"area": _qd(area_v), "axis_ratio": _qd(ratio_v)}


def fit_nucleus_type_prior_to_quantiles(
    target_area: dict[str, float],
    target_axis_ratio: dict[str, float],
    n_train: int,
    type_name: str,
    n_iter: int = 80,
    n_samples: int = 4000,
    seed: int = 0,
    area_weight: float = 1.0,
    ratio_weight: float = 0.6,
) -> NucleusTypePrior:
    """Fit log_radius_mean, log_radius_std, axis_ratio_log_kappa so that
    simulated quantiles match the empirical ones.

    Parameters: (log_mu, log_sigma, eta1, eta2) with kappa_log =
    (eta1, eta2, -(eta1+eta2)). Random-search coordinate descent like
    `fit_type_prior_to_quantiles`.

    Internally uses a precomputed sample pool — see
    :func:`_precompute_section_samples` — so the parameter-independent
    Monte-Carlo work (random rotations, plane bases, depth samples) is
    done once per fit rather than ``n_iter + 1`` times.
    """
    rng = np.random.default_rng(seed)
    pool = _precompute_section_samples(n_samples, rng)
    target_a = np.array([target_area["p10"], target_area["p25"],
                          target_area["p50"], target_area["p75"],
                          target_area["p90"]])
    target_ar = np.array([target_axis_ratio["p10"], target_axis_ratio["p25"],
                          target_axis_ratio["p50"], target_axis_ratio["p75"],
                          target_axis_ratio["p90"]])

    def loss(params: np.ndarray) -> tuple[float, dict[str, dict[str, float]]]:
        lm, ls, e1, e2 = params
        ls = max(0.05, ls)
        kappa_log = (e1, e2, -(e1 + e2))
        q = _section_quantiles_with_pool(lm, ls, kappa_log, pool)
        sim_a = np.array([q["area"]["p10"], q["area"]["p25"], q["area"]["median"],
                          q["area"]["p75"], q["area"]["p90"]])
        sim_ar = np.array([q["axis_ratio"]["p10"], q["axis_ratio"]["p25"],
                            q["axis_ratio"]["median"], q["axis_ratio"]["p75"],
                            q["axis_ratio"]["p90"]])
        L_area = float(np.mean((np.log(sim_a + 1e-3) - np.log(target_a + 1e-3)) ** 2))
        L_ratio = float(np.mean((np.log(sim_ar + 1e-3) - np.log(target_ar + 1e-3)) ** 2))
        return area_weight * L_area + ratio_weight * L_ratio, q

    # Initial guess: median area -> log_mu via 3D-radius from 2D radius
    # heuristic. Log-sigma initial 0.3 is generous.
    init_lm = math.log(max(0.5, math.sqrt(target_area["p50"] / math.pi)) * 1.4)
    init_ls = 0.3
    # Use median axis ratio to seed log_kappa
    init_eta1 = float(0.5 * math.log(max(1.0, target_axis_ratio["p50"])))
    init_eta2 = 0.0
    params = np.array([init_lm, init_ls, init_eta1, init_eta2])
    best_loss, best_q = loss(params)
    step_sizes = np.array([0.10, 0.05, 0.10, 0.10])
    for it in range(n_iter):
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
    return NucleusTypePrior(
        name=type_name,
        n_train=n_train,
        log_radius_mean=float(lm),
        log_radius_std=float(max(0.05, ls)),
        axis_ratio_log_kappa=(float(e1), float(e2), float(-(e1 + e2))),
        target_area_quantiles=dict(target_area),
        target_axis_ratio_quantiles=dict(target_axis_ratio),
        fit_diagnostics={
            "final_loss": float(best_loss),
            "final_quantiles": best_q,
            "n_iter": int(n_iter),
            "n_samples": int(n_samples),
            "init_lm": float(init_lm),
            "init_ls": float(init_ls),
            "init_eta1": float(init_eta1),
            "init_eta2": float(init_eta2),
        },
    )


def load_nucleus_priors(path: str | Path) -> dict[str, NucleusTypePrior]:
    """Load nucleus priors from disk into NucleusTypePrior objects keyed by
    cell-type name.
    """
    raw = json.loads(Path(path).read_text())
    out: dict[str, NucleusTypePrior] = {}
    for name, d in raw.get("priors", {}).items():
        out[name] = NucleusTypePrior(
            name=d["name"],
            n_train=int(d["n_train"]),
            log_radius_mean=float(d["log_radius_mean"]),
            log_radius_std=float(d["log_radius_std"]),
            axis_ratio_log_kappa=tuple(float(x) for x in d["axis_ratio_log_kappa"]),
            target_area_quantiles=dict(d.get("target_area_quantiles", {})),
            target_axis_ratio_quantiles=dict(d.get("target_axis_ratio_quantiles", {})),
            fit_diagnostics=dict(d.get("fit_diagnostics", {})),
        )
    return out
