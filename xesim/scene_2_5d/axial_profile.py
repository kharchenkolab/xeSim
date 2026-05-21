"""Learned axial DAPI profile for the 2.5D z-stack.

The synth z-stack renders each plane independently, so a nucleus appears only
within its hard geometric z-extent and the off-focus planes are dark/sparse —
unlike a real DAPI z-stack, where each nucleus's signal extends through z as a
broad, smoothly-tapering envelope (combined optical axial response + 3D
chromatin structure). Empirically that envelope is **size-, type-, and
latent-invariant** (a uniform system response), so a *single* curve `f(Δz)`
fit from the source bundle captures it.

This module fits `f(Δz)` from the real bundle at explain time (tilt-corrected
core-tracking — like the other priors learned from the source bundle) and
applies it to the rendered volume: each in-focus pixel's signal is propagated
across z by `f(z − z_focus)` with a small lateral spread on the tail, plus a
diffuse background floor. See misc/followup_zstack_defocus_blur.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Module cache: f(Δz) is a per-bundle global property — fit once.
_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def fit_axial_dapi_profile(
    bundle_path: str | Path,
    *,
    region_um: float = 700.0,
    max_nuclei: int = 500,
    max_dz_um: float = 15.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit the global axial DAPI profile ``f(Δz)`` from the real bundle's 3D
    DAPI z-stack (``morphology.ome.tif``) + nucleus centroids.

    Tilt-corrected: each nucleus's core is tracked through z (intensity-weighted
    centroid from the peak plane outward) so tilt-induced lateral drift doesn't
    bias the envelope. Returns ``(dz_um_grid, f_values)`` normalized to peak=1,
    or ``None`` if unavailable. Cached per bundle.
    """
    key = str(bundle_path)
    if key in _CACHE:
        return _CACHE[key]
    try:
        import tifffile
        import zarr
        import pandas as pd
        bp = Path(bundle_path)
        with tifffile.TiffFile(bp / "morphology.ome.tif") as tf:
            za = zarr.open(tf.series[0].aszarr(), mode="r")["0"]   # (z, H, W)
            nz, Hf, Wf = za.shape
            # central tissue region
            psz = 0.2125
            try:
                import json
                psz = float(json.loads((bp / "experiment.xenium").read_text())
                            .get("pixel_size", psz))
            except Exception:
                pass
            rpx = int(region_um / psz)
            cy0, cx0 = Hf // 2 - rpx // 2, Wf // 2 - rpx // 2
            cy0, cx0 = max(0, cy0), max(0, cx0)
            Z = np.asarray(za[:, cy0:cy0 + rpx, cx0:cx0 + rpx]).astype(np.float32)
        H, W = Z.shape[1:]
        cells = pd.read_parquet(bp / "cells.parquet")
        x0u, y0u = cx0 * psz, cy0 * psz
        m = ((cells.x_centroid >= x0u) & (cells.x_centroid < x0u + region_um) &
             (cells.y_centroid >= y0u) & (cells.y_centroid < y0u + region_um) &
             (cells.nucleus_area > 0))
        cells = cells[m]
        if len(cells) > max_nuclei:
            cells = cells.sample(max_nuclei, random_state=0)

        def disk_mean(img, cx, cy, r):
            x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
            y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
            if x1 <= x0 or y1 <= y0:
                return 0.0
            yy, xx = np.mgrid[y0:y1, x0:x1]
            v = img[y0:y1, x0:x1][((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r]
            return float(v.mean()) if v.size else 0.0

        def wcentroid(img, cx, cy, rwin):
            x0, x1 = max(0, int(cx - rwin)), min(W, int(cx + rwin) + 1)
            y0, y1 = max(0, int(cy - rwin)), min(H, int(cy + rwin) + 1)
            if x1 <= x0 or y1 <= y0:
                return cx, cy
            yy, xx = np.mgrid[y0:y1, x0:x1]
            sub = img[y0:y1, x0:x1]
            w = np.where(((xx - cx) ** 2 + (yy - cy) ** 2) <= rwin * rwin, sub, 0.0)
            s = w.sum()
            if s <= 0:
                return cx, cy
            return float((w * xx).sum() / s), float((w * yy).sum() / s)

        max_dp = int(round(max_dz_um / 3.0))
        grid = np.arange(-max_dp, max_dp + 1)
        acc = np.zeros(len(grid)); cnt = np.zeros(len(grid))
        for _, c in cells.iterrows():
            cx = (c.x_centroid - x0u) / psz; cy = (c.y_centroid - y0u) / psz
            r = 0.6 * np.sqrt(c.nucleus_area / np.pi) / psz
            if r < 1.5:
                continue
            base = np.array([disk_mean(Z[zi], cx, cy, r) for zi in range(nz)])
            if base.max() <= 0:
                continue
            pk = int(np.argmax(base))
            prof = np.full(nz, np.nan); prof[pk] = base[pk]
            for d in (+1, -1):                       # track core outward from peak
                ccx, ccy = cx, cy; z = pk + d
                while 0 <= z < nz:
                    ncx, ncy = wcentroid(Z[z], ccx, ccy, 1.5 * r)
                    prof[z] = disk_mean(Z[z], ncx, ncy, r); ccx, ccy = ncx, ncy; z += d
            pn = prof / prof[pk]
            for zi in range(nz):
                d = zi - pk
                if -max_dp <= d <= max_dp and not np.isnan(pn[zi]):
                    acc[d + max_dp] += pn[zi]; cnt[d + max_dp] += 1
        keep = cnt >= 20
        if keep.sum() < 3:
            return None
        f = (acc / np.maximum(cnt, 1))[keep]
        dz_um = grid[keep] * 3.0
        f = np.clip(f / f.max(), 0.0, 1.0)
        _CACHE[key] = (dz_um.astype(np.float32), f.astype(np.float32))
        return _CACHE[key]
    except Exception:
        return None


def apply_axial_dapi_profile(
    zstack: np.ndarray,
    z_slices_um: np.ndarray,
    profile: tuple[np.ndarray, np.ndarray],
    pixel_size_um: float,
    *,
    lateral_um_per_um: float = 0.5,
    floor_frac: float = 0.15,
) -> np.ndarray:
    """Re-synthesize the z-stack's axial structure from the learned profile.

    Take each pixel's in-focus value (max over z) at its in-focus depth
    (argmax over z), then propagate it across planes weighted by ``f(z −
    z_focus)`` with a small lateral spread that grows with axial distance (the
    off-focus tail is diffuse). Add a smooth diffuse background floor. This
    fills the off-focus planes the way real DAPI z-stacks look, without
    distorting the in-focus plane (Δz=0 → f=1, no lateral spread).
    """
    from scipy.ndimage import gaussian_filter
    dz_grid, f_vals = profile
    nz = zstack.shape[0]
    z_step = float(z_slices_um[1] - z_slices_um[0]) if len(z_slices_um) > 1 else 3.0

    def f_interp(dz_um):
        return float(np.interp(dz_um, dz_grid, f_vals, left=f_vals[0], right=f_vals[-1]))

    P = zstack.max(axis=0)                 # in-focus projection
    ZC = zstack.argmax(axis=0)             # in-focus depth (z index) per pixel
    out = np.zeros_like(zstack)
    for d in range(nz):                    # bin pixels by their in-focus depth
        src = np.where(ZC == d, P, 0.0)
        if src.max() <= 0:
            continue
        for z in range(nz):
            w = f_interp((z - d) * z_step)
            if w <= 1e-4:
                continue
            sig = lateral_um_per_um * abs(z - d) * z_step / pixel_size_um
            out[z] += w * (src if sig < 0.3 else gaussian_filter(src, sig))
    if floor_frac > 0:
        out += (floor_frac * gaussian_filter(P, 12.0))[None]
    return out


__all__ = ["fit_axial_dapi_profile", "apply_axial_dapi_profile"]
