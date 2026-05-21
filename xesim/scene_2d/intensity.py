"""Per-channel intensity calibration: match synth uint16 stats to real bundle.

The renderer outputs float values in a roughly [0, 1] range. The naive approach
of mapping p99.5 → 4095 fills the full 12-bit range, which makes synth images
~2.5× brighter than real Xenium images and saturates the top 0.5% of pixels.

This module provides:

  * ``auto_tune_intensity_stats(bundle_path, channel_names)`` — sample the real
    bundle's morphology_focus pyramid (a low level for speed) and return per-
    channel target ``(p50, p99_5)`` statistics.

  * ``calibrate_to_uint16(arr, target_stats=None)`` — apply a 2-anchor linear
    remap that puts ``synth_p50 → target_p50`` and ``synth_p99_5 → target_p99_5``,
    preserving the relative dynamic range while matching the real bundle's
    absolute brightness. Falls back to the original p99.5 → 4095 scheme when
    ``target_stats`` is None.

Real Xenium pancreas bundle (4-channel) target stats live in the OME-XML of
``morphology_focus_*.ome.tif`` — we read them straight from there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def auto_tune_intensity_stats(
    bundle_path: str | Path,
    channel_names: list[str],
    *,
    pyramid_level: int = 5,
) -> dict[str, tuple[float, float]]:
    """Compute per-channel ``(p50, p99_5)`` from the real bundle's
    morphology_focus image.

    Uses pyramid level ``pyramid_level`` (default 5 → ~430×1067 for the pancreas
    bundle) so the sample is fast but still has ~450k pixels per channel —
    plenty for stable p99.5 estimation. Falls back to level 0 if the requested
    level is missing.

    Returns ``{channel_name: (p50, p99_5)}``. Channel order in the file is
    matched to ``channel_names`` by OME-XML Channel/Name when possible, else
    by position.
    """
    import tifffile

    bundle = Path(bundle_path)
    # Standard Xenium layout: bundle/morphology_focus/morphology_focus_0000.ome.tif
    candidates = [
        bundle / "morphology_focus" / "morphology_focus_0000.ome.tif",
        bundle / "data" / "morphology_focus" / "morphology_focus_0000.ome.tif",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"morphology_focus_0000.ome.tif not found under {bundle}")

    with tifffile.TiffFile(src) as tf:
        series = tf.series[0]
        # Pick the lowest-resolution level that still has >=64k pixels per channel
        levels = series.levels
        idx = min(pyramid_level, len(levels) - 1)
        arr = levels[idx].asarray()
        # Read channel names from OME-XML
        import re
        meta = tf.ome_metadata or ""
        ome_names = re.findall(r'Channel ID="[^"]+" Name="([^"]+)"', meta)

    if arr.ndim != 3:
        raise ValueError(
            f"morphology_focus has unexpected shape {arr.shape} (expected CYX)")

    # Map channel position → name (prefer OME-XML, fall back to caller list)
    n_ch = arr.shape[0]
    if len(ome_names) >= n_ch:
        positional_names = ome_names[:n_ch]
    else:
        positional_names = list(channel_names[:n_ch])

    out: dict[str, tuple[float, float]] = {}
    for c, name in enumerate(positional_names):
        ch = arr[c]
        p50 = float(np.percentile(ch, 50))
        p99_5 = float(np.percentile(ch, 99.5))
        out[name] = (p50, p99_5)
    return out


def calibrate_to_uint16(
    arr: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    target_stats: dict[str, tuple[float, float]] | None = None,
    target_quantiles: dict[str, np.ndarray] | None = None,
    mode: str = "scale",
    p99: float = 99.5,
    max_val: int = 65535,
    fallback_max_val: int = 4095,
    display_lut: dict | None = None,
) -> np.ndarray:
    """Convert a CYX (or HW) float array to uint16.

    Four modes:

    * ``"scale"`` (default, ``target_stats`` required): per-channel multiplier
      ``target_p99_5 / synth_p99_5`` applied uniformly. Origin-preserving —
      no offset, so the renderer's relative texture is preserved exactly.
      Same shape as the original ``p99.5 → 4095`` behavior, just with target
      = real bundle's ``p99.5`` instead of 4095.

    * ``"match2"`` (opt-in, ``target_stats`` required): the 2-anchor linear
      remap ``(sp50 → tp50, sp99_5 → tp99_5)``. Matches both median and high
      percentile but DISTORTS texture when the synth's local distribution
      shape differs from the target's bundle-wide shape (e.g. sparse vs dense
      regions). Available for users who care more about absolute median than
      local texture.

    * ``"histmatch"`` (opt-in, ``target_quantiles`` required): full quantile-
      to-value remap. ``target_quantiles[ch]`` should be a length-K array of
      real-bundle values at evenly-spaced quantiles (e.g. K=257 → 0, 1/256,
      2/256, ..., 1). Each synth pixel is replaced with the target value at
      its empirical quantile in the synth channel.

    * ``"none"``: truly no calibration — a fixed ``round(arr)*fallback_max_val``
      scale, with NO per-channel stats, display LUT, or target matching. For
      inspecting raw renderer output and unstained bundles.

    ``lut_native`` is the default at the CLI. ``arr`` already-uint16 inputs are
    returned unchanged (idempotent).
    """
    if arr.dtype == np.uint16:
        return arr
    arr = np.clip(arr.astype(np.float32), 0, None)
    if mode == "none":
        # No calibration: fixed linear scale, no per-channel/LUT/stat work.
        return np.clip(np.round(arr * fallback_max_val), 0, max_val).astype(np.uint16)
    if arr.ndim == 2:
        p = float(np.percentile(arr, p99))
        return (np.clip(arr / max(p, 1e-6), 0, 1) * fallback_max_val).astype(np.uint16)

    # "lut_native" mode: the renderer was trained against LUT-normalized
    # real images, so its native output range maps directly back to real
    # intensity via the LUT's per-channel (lo, hi). This preserves the
    # renderer's natural channel ratios AND matches real's absolute scale.
    # It's the right default whenever a display_lut is available — much
    # better than "off" mode (per-channel p99→4095) which destroys channel
    # ratios.
    if display_lut is not None and (mode in ("lut_native", "lut_zerofloor") or
                                         (target_stats is None and target_quantiles is None)):
        n_ch = arr.shape[0]
        out = np.zeros_like(arr, dtype=np.uint16)
        lut_channels = display_lut.get("channels", []) if display_lut else []
        # lut_native: synth ∈ [0,1] → real uint16 ∈ [lo, hi]. Faithful inverse
        # of (real - lo) / (hi - lo). lut_zerofloor: synth ∈ [0,1] → uint16
        # ∈ [0, hi]. Both modes also add Gaussian noise calibrated from the
        # real bundle's per-channel dark-pixel stdev when noise_stats are
        # in the display_lut — this matches real's sensor noise floor (real
        # dark std ≈ 3-8) which the deterministic renderer doesn't produce.
        zerofloor = (mode == "lut_zerofloor")
        noise_stats = display_lut.get("noise_stats", []) if display_lut else []
        rng = np.random.default_rng(0)  # deterministic seed for bundle reproducibility
        for c in range(n_ch):
            lo = float(lut_channels[c].get("lo", 0.0)) if c < len(lut_channels) else 0.0
            hi = float(lut_channels[c].get("hi", fallback_max_val)) if c < len(lut_channels) else fallback_max_val
            # If we have a measured bundle-wide background mean, use it as the
            # synth floor — usually below LUT's lo (training crops were dense
            # tissue). This makes synth's dark mean match real instead of
            # sitting at lo, which the renderer was trained to map to s=0.
            if c < len(noise_stats) and "dark_mean" in noise_stats[c]:
                lo = float(noise_stats[c]["dark_mean"])
            if zerofloor:
                scaled = arr[c] * hi
            else:
                scaled = lo + arr[c] * (hi - lo)
            if c < len(noise_stats):
                # Gaussian read-noise component: constant std across all pixels.
                # Poisson shot-noise component scales with sqrt(signal).
                ns = noise_stats[c]
                read_std = float(ns.get("read_std", 0.0))
                shot_k = float(ns.get("shot_k", 0.0))
                if read_std > 0 or shot_k > 0:
                    poisson_std = shot_k * np.sqrt(np.maximum(scaled, 0.0))
                    total_std = np.sqrt(read_std * read_std + poisson_std * poisson_std)
                    scaled = scaled + rng.standard_normal(scaled.shape).astype(np.float32) * total_std
            out[c] = np.clip(scaled, 0, max_val).astype(np.uint16)
        return out

    n_ch = arr.shape[0]
    out = np.zeros_like(arr, dtype=np.uint16)
    effective_mode = mode
    if mode != "off" and target_stats is None and target_quantiles is None:
        effective_mode = "off"

    for c in range(n_ch):
        ch = arr[c]
        name = channel_names[c] if channel_names and c < len(channel_names) else None

        if effective_mode == "histmatch" and target_quantiles is not None and name in target_quantiles:
            scaled = _quantile_remap(ch, target_quantiles[name])
            cap = max_val
        elif effective_mode in ("scale", "match2") and target_stats is not None and name in target_stats:
            sp99 = float(np.percentile(ch, p99))
            tp50, tp99 = float(target_stats[name][0]), float(target_stats[name][1])
            if effective_mode == "scale":
                # Origin-preserving multiplier: synth p99.5 → target p99.5
                scaled = ch * (tp99 / max(sp99, 1e-6))
            else:  # match2
                sp50 = float(np.percentile(ch, 50))
                denom = max(sp99 - sp50, 1e-6)
                slope = (tp99 - tp50) / denom
                offset = tp50 - slope * sp50
                scaled = ch * slope + offset
            cap = max_val
        else:
            sp99 = float(np.percentile(ch, p99))
            scaled = ch / max(sp99, 1e-6) * fallback_max_val
            cap = fallback_max_val
        out[c] = np.clip(scaled, 0, cap).astype(np.uint16)
    return out


def _quantile_remap(ch: np.ndarray, target_quantile_values: np.ndarray) -> np.ndarray:
    """Map each synth pixel to the target value at the synth's empirical
    quantile. ``target_quantile_values`` is a sorted array of length K with
    the target value at quantile k/(K-1).
    """
    flat = ch.ravel()
    ranks = np.argsort(np.argsort(flat))                     # 0..N-1
    quantiles = ranks / max(len(flat) - 1, 1)                # 0..1
    k = len(target_quantile_values)
    idx_f = quantiles * (k - 1)
    idx_lo = np.floor(idx_f).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, k - 1)
    frac = idx_f - idx_lo
    mapped = (target_quantile_values[idx_lo] * (1 - frac) +
              target_quantile_values[idx_hi] * frac)
    return mapped.reshape(ch.shape)


def auto_tune_intensity_quantiles(
    bundle_path: str | Path,
    channel_names: list[str],
    *,
    pyramid_level: int = 5,
    n_quantiles: int = 257,
) -> dict[str, np.ndarray]:
    """Sample per-channel quantile-value arrays from the real bundle (for
    ``histmatch`` mode). Returns ``{channel_name: array[length n_quantiles]}``
    holding the real-bundle value at quantile ``k/(n_quantiles-1)``.
    """
    import tifffile

    bundle = Path(bundle_path)
    candidates = [
        bundle / "morphology_focus" / "morphology_focus_0000.ome.tif",
        bundle / "data" / "morphology_focus" / "morphology_focus_0000.ome.tif",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"morphology_focus_0000.ome.tif not found under {bundle}")
    with tifffile.TiffFile(src) as tf:
        levels = tf.series[0].levels
        idx = min(pyramid_level, len(levels) - 1)
        arr = levels[idx].asarray()
        import re
        meta = tf.ome_metadata or ""
        ome_names = re.findall(r'Channel ID="[^"]+" Name="([^"]+)"', meta)
    if arr.ndim != 3:
        raise ValueError(f"morphology_focus has unexpected shape {arr.shape}")

    positional_names = (ome_names[:arr.shape[0]]
                        if len(ome_names) >= arr.shape[0]
                        else list(channel_names[:arr.shape[0]]))
    qs = np.linspace(0, 1, n_quantiles)
    out: dict[str, np.ndarray] = {}
    for c, name in enumerate(positional_names):
        out[name] = np.quantile(arr[c].ravel(), qs).astype(np.float32)
    return out


def calibrate_noise_stats(
    bundle_path: str | Path,
    *,
    pyramid_level: int = 4,
    display_lut: dict | None = None,
) -> list[dict]:
    """Measure per-channel sensor-noise parameters from the real bundle.

    Decomposes pixel variance into a Gaussian read-noise component (signal-
    independent, ``read_std``) and a Poisson shot-noise component
    (signal-scaling, ``shot_k``) via:

        total_var(signal) = read_std² + shot_k² · signal

    ``read_std`` is estimated from dark-pixel stdev (bottom-5% pixels) and
    ``shot_k`` from the residual variance at mid-signal (around p50).

    Returns ``[{read_std, shot_k}, ...]`` per channel, ready to drop into
    ``display_lut["noise_stats"]`` for use by ``calibrate_to_uint16``.
    """
    import tifffile

    bundle = Path(bundle_path)
    candidates = [
        bundle / "morphology_focus" / "morphology_focus_0000.ome.tif",
        bundle / "data" / "morphology_focus" / "morphology_focus_0000.ome.tif",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"morphology_focus_0000.ome.tif not found under {bundle}")
    with tifffile.TiffFile(src) as tf:
        levels = tf.series[0].levels
        idx = min(pyramid_level, len(levels) - 1)
        arr = levels[idx].asarray()
    if arr.ndim != 3:
        raise ValueError(f"morphology_focus unexpected shape {arr.shape}")

    lut_channels = display_lut.get("channels", []) if display_lut else []
    out: list[dict] = []
    for c in range(arr.shape[0]):
        ch = arr[c].astype(np.float32)
        # Restrict to TISSUE pixels (above the LUT's lo, which is the p1
        # of training crops). The Xenium bundle has a large empty surround
        # outside tissue that drives bundle-wide percentiles down — sampling
        # there gives dark_mean ~0, which would over-darken the synth output.
        # If no LUT, fall back to all non-zero pixels.
        if c < len(lut_channels):
            lo_lut = float(lut_channels[c].get("lo", 0.0))
            tissue = ch[ch >= lo_lut]
        else:
            tissue = ch[ch > 0]
        if tissue.size < 1000:
            out.append({"read_std": 0.0, "shot_k": 0.0, "dark_mean": 0.0})
            continue
        p5 = float(np.percentile(tissue, 5))
        p50 = float(np.percentile(tissue, 50))
        nonzero = tissue  # downstream uses 'nonzero'
        # Dark-pixel std (signal-independent read noise) + mean (true
        # background floor — usually < LUT lo, since lo is p1 of dense
        # training crops while bundle-wide background is lower).
        dark = nonzero[nonzero < p5 + 1]
        read_std = float(np.std(dark)) if dark.size > 100 else 0.0
        dark_mean = float(np.mean(dark)) if dark.size > 100 else 0.0
        # Mid-signal std → infer shot_k
        mid_lo, mid_hi = max(p50 - 30, 1.0), p50 + 30
        mid = nonzero[(nonzero >= mid_lo) & (nonzero <= mid_hi)]
        if mid.size > 1000 and p50 > 0:
            mid_var = float(np.var(mid))
            excess_var = max(mid_var - read_std * read_std, 0.0)
            shot_k = float(np.sqrt(excess_var / p50))
        else:
            shot_k = 0.0
        out.append({"read_std": read_std, "shot_k": shot_k,
                       "dark_mean": dark_mean})
    return out


__all__ = ["auto_tune_intensity_stats", "auto_tune_intensity_quantiles",
           "calibrate_to_uint16", "calibrate_noise_stats"]
