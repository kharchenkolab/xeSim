"""Per-cell tilt vector with MRF-coordinated direction across neighbors.

Each cell gets a 3D unit vector `t = (t_x, t_y, t_z)` with `t_z > 0`
representing the cell's long-axis orientation. Magnitude (i.e. how far
from vertical) is determined by the cell's observed z_extent + per-type
3D-elongation prior:

    c_along_t = 2 * elong * ((3 * V) / (4π * elong))^(1/3)
              = (6 * V * elong² / π)^(1/3)

For an observed cell with fitted `z_extent`:

    t_z = clip(z_extent / c_along_t, 0.1, 1.0)

Direction `(t_x, t_y)` (with `t_x² + t_y² = 1 - t_z²`) is determined by
Gibbs sampling under an MRF prior:

    E_pair(i, j) = -κ · |cos(angle(t_i_xy, t_j_xy))|   for neighbors (i, j)
    E_anchor(i)  = -λ_i · |cos(angle(t_i_xy, PCA_xy_long_axis_i))|
                   for observed cells with non-trivial 2D PCA axis ratio

Per-type 3D nucleus priors are loaded from
``<MODEL_DIR>/priors_3d/nucleus_priors.json`` (produced by
``xesim.scene_2_5d.fit_priors``). If no model dir is supplied or the file
is absent, every cell gets a single-median default
(vol_um3=130, elong=3.6) — no hardcoded per-type table.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Single-median fallback when no fitted priors are available. Deliberately
# NOT per-type — see project guideline "no tissue-specific defaults".
DEFAULT_NUCLEUS_PRIOR = {"vol_um3": 130.0, "elong": 3.6}


def _prior_from_json_entry(d: dict) -> dict:
    """Convert a NucleusTypePrior JSON dict into the tilt-side
    (vol_um3, elong) representation used by per_type_c_along_t.

    The fitted prior stores log_radius_mean (volume-equivalent r_eq in µm)
    and axis_ratio_log_kappa (zero-sum triplet). Elongation is the
    longest/shortest axis ratio = exp(max(kappa) - min(kappa)).
    """
    r_eq = math.exp(float(d["log_radius_mean"]))
    vol_um3 = (4.0 / 3.0) * math.pi * r_eq ** 3
    log_k = [float(x) for x in d["axis_ratio_log_kappa"]]
    elong = math.exp(max(log_k) - min(log_k))
    return {"vol_um3": vol_um3, "elong": elong}


def load_nucleus_priors_table(
    model_dir: str | Path | None = None,
    *,
    priors_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Auto-discover and load a per-type {vol_um3, elong} table from a
    fitted nucleus-priors JSON.

    Search order:
      1. explicit ``priors_path`` argument
      2. ``<model_dir>/priors_3d/nucleus_priors.json``
      3. return empty dict (callers fall back to DEFAULT_NUCLEUS_PRIOR)
    """
    candidates: list[Path] = []
    if priors_path is not None:
        candidates.append(Path(priors_path))
    if model_dir is not None:
        candidates.append(Path(model_dir) / "priors_3d" / "nucleus_priors.json")
    for p in candidates:
        if p.exists():
            raw = json.loads(p.read_text())
            return {name: _prior_from_json_entry(d)
                    for name, d in raw.get("priors", {}).items()}
    return {}


def per_type_c_along_t(
    cell_type: str,
    priors_table: dict[str, dict[str, float]] | None = None,
) -> float:
    """Length of the nucleus's full long axis (µm).

    Uses the per-type entry from ``priors_table`` if present, else the
    single-median DEFAULT_NUCLEUS_PRIOR.
    """
    if priors_table is None:
        priors_table = {}
    p = priors_table.get(cell_type, DEFAULT_NUCLEUS_PRIOR)
    V = float(p["vol_um3"]); elong = float(p["elong"])
    a_short = (3.0 * V / (4.0 * np.pi * elong)) ** (1.0 / 3.0)
    c_along_t = 2.0 * elong * a_short
    return float(c_along_t)


def compute_tz(cell_type: str, z_extent_um: float, *,
                priors_table: dict[str, dict[str, float]] | None = None,
                tz_floor: float = 0.1, tz_ceil: float = 1.0) -> float:
    """t_z = clip(z_extent / c_along_t, tz_floor, tz_ceil)."""
    if not np.isfinite(z_extent_um) or z_extent_um <= 0:
        return float(tz_ceil)
    c = per_type_c_along_t(cell_type, priors_table=priors_table)
    return float(np.clip(z_extent_um / c, tz_floor, tz_ceil))


def _contour_pca_axis(xs_rel: np.ndarray, ys_rel: np.ndarray
                        ) -> tuple[np.ndarray, float]:
    """Compute the 2D PCA long-axis direction (unit vector) + axis-ratio
    (long/short) for a contour centered at origin. Returns (direction,
    axis_ratio). axis_ratio close to 1.0 means weak preferred direction."""
    pts = np.stack([xs_rel, ys_rel], axis=1)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    # vecs[:, -1] is principal direction
    long_idx = int(np.argmax(vals))
    short_idx = 1 - long_idx
    direction = vecs[:, long_idx].astype(np.float32)
    long_eig = float(vals[long_idx]); short_eig = float(vals[short_idx])
    ratio = float(np.sqrt(long_eig / max(short_eig, 1e-6)))
    # Normalize and ensure direction has positive x (sign-ambiguous anyway)
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    if direction[0] < 0:
        direction = -direction
    return direction, ratio


@dataclass
class CellTilt:
    cell_id: str
    t_x: float
    t_y: float
    t_z: float
    cell_type: str
    anchor_ratio: float = 1.0   # 2D PCA axis ratio (≥1; 1=circular)
    anchor_dir: tuple[float, float] = (1.0, 0.0)


def initialize_tilts(
    cell_rows: pd.DataFrame,
    *,
    model_dir: str | Path | None = None,
    priors_path: str | Path | None = None,
    priors_table: dict[str, dict[str, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> list[CellTilt]:
    """Build per-cell `CellTilt` initialized with:
      - `t_z` from per-type c_along_t + z_extent_um
      - `(t_x, t_y)` initial direction: PCA long axis if available (else random)

    Required columns of `cell_rows`:
      cell_id, cell_type, z_extent_um
      OPTIONAL: vertex_x_rel, vertex_y_rel (list, centered at origin) for
      observed cells — used as PCA anchor.

    Per-type 3D priors are auto-discovered from
    ``<model_dir>/priors_3d/nucleus_priors.json`` unless a
    ``priors_table`` is passed in directly. Missing priors fall back to
    a single-median default (DEFAULT_NUCLEUS_PRIOR) — no hardcoded
    per-type table.
    """
    rng = rng or np.random.default_rng(0)
    if priors_table is None:
        priors_table = load_nucleus_priors_table(
            model_dir=model_dir, priors_path=priors_path)
    tilts: list[CellTilt] = []
    have_vertex = ("vertex_x_rel" in cell_rows.columns
                    and "vertex_y_rel" in cell_rows.columns)
    for _, row in cell_rows.iterrows():
        ct = str(row.get("cell_type", "unknown"))
        ze = float(row.get("z_extent_um", np.nan))
        tz = compute_tz(ct, ze, priors_table=priors_table)
        if have_vertex and isinstance(row["vertex_x_rel"], (list, np.ndarray)) \
                and len(row["vertex_x_rel"]) >= 3:
            xs = np.asarray(row["vertex_x_rel"], dtype=np.float32)
            ys = np.asarray(row["vertex_y_rel"], dtype=np.float32)
            direction, ratio = _contour_pca_axis(xs, ys)
            t_x_init = float(direction[0])
            t_y_init = float(direction[1])
            anchor = (t_x_init, t_y_init)
        else:
            # Random azimuth
            theta = rng.uniform(0, 2 * np.pi)
            t_x_init = float(np.cos(theta))
            t_y_init = float(np.sin(theta))
            ratio = 1.0
            anchor = (t_x_init, t_y_init)
        # Scale (t_x, t_y) so that |t| = 1, i.e. t_x² + t_y² + t_z² = 1
        target_xy_mag = float(np.sqrt(max(0.0, 1.0 - tz * tz)))
        norm = float(np.hypot(t_x_init, t_y_init)) + 1e-6
        tilts.append(CellTilt(
            cell_id=str(row["cell_id"]),
            t_x=t_x_init / norm * target_xy_mag,
            t_y=t_y_init / norm * target_xy_mag,
            t_z=tz,
            cell_type=ct,
            anchor_ratio=ratio,
            anchor_dir=anchor,
        ))
    return tilts


def mrf_gibbs_sweep(
    tilts: list[CellTilt],
    *,
    positions: np.ndarray,       # (N, 2) xy seeds
    neighbor_radius_um: float = 30.0,
    kappa: float = 2.0,
    anchor_weight: float = 1.0,
    n_sweeps: int = 30,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    device: str = "auto",        # "auto" → cuda if available, else cpu
) -> list[CellTilt]:
    """Gibbs-style MRF sampler for tilt azimuths.

    For each cell, given its neighbors' current azimuths and its own
    PCA anchor, sample a new azimuth from the conditional. The energy
    is:

        E(θ_i | θ_neighbors) = -κ Σ_j |cos(θ_i - θ_j)|
                                -λ_i |cos(θ_i - θ_anchor_i)|

    where λ_i = anchor_weight * (anchor_ratio_i - 1) (stronger anchor
    for cells with strongly elongated 2D contours).

    Direction is sign-ambiguous (axial; θ and θ+π are the same), so the
    |cos| construction puts the energy on a [0, π] period rather than
    [0, 2π].

    Returns the same tilts list (mutated) with updated (t_x, t_y).
    """
    rng = rng or np.random.default_rng(0)
    from scipy.spatial import cKDTree

    N = len(tilts)
    if N == 0:
        return tilts
    tree = cKDTree(positions)
    # Per-cell neighbor lists (indices)
    nbr_lists = tree.query_ball_point(positions, r=neighbor_radius_um)

    # Current azimuth per cell
    theta = np.array([np.arctan2(t.t_y, t.t_x) for t in tilts], dtype=np.float64)
    target_xy_mag = np.array([np.hypot(t.t_x, t.t_y) for t in tilts])
    anchor_theta = np.array([np.arctan2(t.anchor_dir[1], t.anchor_dir[0])
                                  for t in tilts])
    anchor_lambda = np.array(
        [anchor_weight * max(0.0, t.anchor_ratio - 1.0) for t in tilts])

    # Candidate angles (a small grid for the conditional)
    n_candidates = 36
    candidate_dtheta = np.linspace(0.0, np.pi, n_candidates, endpoint=False)

    # Vectorized Jacobi-style MRF on torch tensors. Runs on GPU when
    # available — whole-bundle (N~200k) finishes in ~10s vs minutes on CPU.
    import torch
    n_lens = np.fromiter((len(l) for l in nbr_lists), dtype=np.int32, count=N)
    max_nbr = int(n_lens.max())
    if max_nbr == 0:
        if verbose: print("  MRF: no neighbors, skipping sweeps")
        n_sweeps = 0

    if n_sweeps > 0:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose: print(f"  MRF device={device}, max_nbr={max_nbr}")
        nbr_idx_np = np.full((N, max_nbr), -1, dtype=np.int64)
        for i, lst in enumerate(nbr_lists):
            if lst:
                nbr_idx_np[i, :len(lst)] = lst
        nbr_mask_np = (nbr_idx_np >= 0)
        nbr_idx_safe_np = np.where(nbr_mask_np, nbr_idx_np, 0)

        # Move to torch
        nbr_idx_t = torch.from_numpy(nbr_idx_safe_np).to(device)
        nbr_mask_t = torch.from_numpy(nbr_mask_np).to(device).to(torch.float32)
        cand_theta_t = torch.from_numpy(candidate_dtheta).to(device).to(torch.float32)
        theta_t = torch.from_numpy(theta).to(device).to(torch.float32)
        anchor_theta_t = torch.from_numpy(anchor_theta).to(device).to(torch.float32)
        anchor_lambda_t = torch.from_numpy(anchor_lambda).to(device).to(torch.float32)

        # Chunk to bound memory: (chunk, C, max_nbr) intermediate.
        # 200k * 36 * 100 * 4 (fp32) = 2.9 GB if all-at-once; chunk 50k → 720 MB.
        bytes_per_row = 36 * max_nbr * 4
        target_bytes = 1_000_000_000  # ~1 GB intermediate budget
        chunk = max(1024, min(N, int(target_bytes / max(1, bytes_per_row))))

        for sweep in range(n_sweeps):
            new_theta_t = torch.empty(N, dtype=torch.float32, device=device)
            for c0 in range(0, N, chunk):
                c1 = min(N, c0 + chunk)
                # theta_nbr: (cs, max_nbr)
                theta_nbr = theta_t[nbr_idx_t[c0:c1]]
                valid = nbr_mask_t[c0:c1]                                # (cs, max_nbr)
                # cos_nbr: (cs, C, max_nbr)
                cos_nbr = torch.abs(torch.cos(
                    cand_theta_t[None, :, None] - theta_nbr[:, None, :]))
                cos_nbr = cos_nbr * valid[:, None, :]
                E_nbr = -kappa * cos_nbr.sum(dim=2)                       # (cs, C)
                E_anc = -anchor_lambda_t[c0:c1, None] * torch.abs(
                    torch.cos(cand_theta_t[None, :] - anchor_theta_t[c0:c1, None]))
                E_total = E_nbr + E_anc
                log_p = -E_total
                log_p = log_p - log_p.max(dim=1, keepdim=True).values
                p = torch.exp(log_p)
                p = p / p.sum(dim=1, keepdim=True)
                # Sample by inverse-CDF (broadcast u against cdf)
                cdf = torch.cumsum(p, dim=1)
                u = torch.rand(c1 - c0, dtype=torch.float32, device=device)
                new_idx_block = (cdf > u[:, None]).int().argmax(dim=1)
                new_theta_t[c0:c1] = cand_theta_t[new_idx_block]

            d = torch.abs(((new_theta_t - theta_t) + np.pi / 2) % np.pi - np.pi / 2)
            max_diff = float(d.max().item())
            theta_t = new_theta_t
            if verbose:
                print(f"  sweep {sweep+1}/{n_sweeps}: max_dtheta={max_diff:.3f} rad")

        theta = theta_t.cpu().numpy().astype(np.float64)

    # Write back to tilts
    for i, t in enumerate(tilts):
        new_tx = float(np.cos(theta[i]) * target_xy_mag[i])
        new_ty = float(np.sin(theta[i]) * target_xy_mag[i])
        # Replace (mutate dataclass via new assignment — dataclass is frozen=False here)
        tilts[i] = CellTilt(
            cell_id=t.cell_id, t_x=new_tx, t_y=new_ty, t_z=t.t_z,
            cell_type=t.cell_type, anchor_ratio=t.anchor_ratio,
            anchor_dir=t.anchor_dir,
        )
    return tilts


__all__ = [
    "DEFAULT_NUCLEUS_PRIOR",
    "load_nucleus_priors_table",
    "per_type_c_along_t",
    "compute_tz",
    "CellTilt",
    "initialize_tilts",
    "mrf_gibbs_sweep",
]
