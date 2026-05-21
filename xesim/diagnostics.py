"""Standard diagnostic plots produced by ``--diagnostic`` on fit-model
and explain.

Each function takes its inputs explicitly and writes a PNG (or small
set) into ``out_dir``. Errors are caught per function so one bad plot
doesn't take down the whole diagnostic step. Adding a new diagnostic
means dropping a function in here and adding it to the orchestrator.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Path resolution


def resolve_diagnostic_dir(
    diagnostic_arg: str | None, out_dir: str | Path,
) -> Path | None:
    """Translate the parsed --diagnostic argument into an output directory.

    ``None`` → diagnostics disabled, return None.
    ``"auto"`` (the const used by ``nargs='?'`` with no value) →
    ``<out_dir>/diagnostics/``.
    Anything else → that path verbatim.
    """
    if diagnostic_arg is None:
        return None
    if diagnostic_arg == "auto":
        diag = Path(out_dir) / "diagnostics"
    else:
        diag = Path(diagnostic_arg)
    diag.mkdir(parents=True, exist_ok=True)
    return diag


def _safe(fn, *args, **kwargs):
    """Run a plot function, swallow failures, return the path it wrote
    (or None on failure). Errors print but don't abort."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[diagnostics] {fn.__name__} failed: {e}", file=sys.stderr)
        traceback.print_exc(limit=2, file=sys.stderr)
        return None


def _skip(panel: str, reason: str):
    """Log a panel skip with its reason and return None. A panel must NEVER
    silently vanish or emit a blank figure — empty selection, a missing
    helper/input/annotation, or out-of-scope all log here and return None."""
    print(f"[diagnostics] {panel} skipped: {reason}", file=sys.stderr)
    return None


def _scope_label(scene_bounds) -> str:
    """Human-readable scope tag for panel titles, e.g.
    'region 1500×1000µm @ (5000,500)' or 'whole bundle'."""
    if scene_bounds is None:
        return "whole bundle"
    x0, y0, x1, y1 = scene_bounds
    return f"region {x1-x0:.0f}×{y1-y0:.0f}µm @ ({x0:.0f},{y0:.0f})"


def _is_region(scene_bounds, full_bounds_xy) -> bool:
    """True if scene_bounds is a strict sub-window of the real bundle's
    full extent (so titles can say 'region' vs 'whole bundle')."""
    if scene_bounds is None or full_bounds_xy is None:
        return False
    x0, y0, x1, y1 = scene_bounds
    fx0, fy0, fx1, fy1 = full_bounds_xy
    return (x0 > fx0 + 1 or y0 > fy0 + 1 or x1 < fx1 - 1 or y1 < fy1 - 1)


# ---------------------------------------------------------------------------
# Channel palette (matches scene_2d.realism_panel.CHANNEL_STYLE)

CHANNEL_HUE = {
    "DAPI":                       ((1.00, 0.20, 0.20), 1.0),
    "ATP1A1/CD45/E-Cadherin":     ((0.20, 1.00, 0.30), 1.0),
    "18S":                        ((0.30, 0.50, 1.00), 0.55),
    "alphaSMA/Vimentin":          ((0.95, 0.40, 0.85), 0.45),
}


def _composite_rgb(arr_chw: np.ndarray, channel_names) -> np.ndarray:
    """4-channel intensity stack → RGB display via the canonical hue mix."""
    rgb = np.zeros((arr_chw.shape[1], arr_chw.shape[2], 3), dtype=np.float32)
    for ci in range(min(len(channel_names), arr_chw.shape[0])):
        hue, w = CHANNEL_HUE.get(channel_names[ci], ((0.8, 0.8, 0.8), 0.5))
        intensity = np.clip(arr_chw[ci], 0, 1) * w
        for k in range(3):
            rgb[..., k] += intensity * hue[k]
    p = float(np.percentile(rgb, 99))
    if p > 0:
        rgb = rgb / max(p, 1.0)
    return np.clip(rgb, 0, 1)


def _lut_normalize(arr_chw: np.ndarray, display_lut: dict) -> np.ndarray:
    """Apply display LUT per channel — output in [0, 1]."""
    out = np.zeros_like(arr_chw, dtype=np.float32)
    chans = display_lut.get("channels", [])
    for ci in range(arr_chw.shape[0]):
        if ci < len(chans):
            lo = float(chans[ci].get("lo", 0.0))
            hi = float(chans[ci].get("hi", 1.0))
            out[ci] = (arr_chw[ci] - lo) / max(hi - lo, 1e-6)
        else:
            p99 = float(np.percentile(arr_chw[ci][arr_chw[ci] > 0], 99)) \
                if (arr_chw[ci] > 0).any() else 1.0
            out[ci] = arr_chw[ci] / max(p99, 1e-6)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# fit-model diagnostics


def fit_model_diagnostics(model_dir: str | Path, bundle_path: str | Path,
                              out_dir: Path) -> list[Path]:
    """Standard fit-model diagnostic set."""
    written: list[Path] = []
    model_dir = Path(model_dir)
    p_priors = model_dir / "priors_3d" / "nucleus_priors.json"
    if p_priors.exists():
        p = _safe(_plot_nucleus_prior_fit, p_priors, out_dir)
        if p: written.append(p)
    p_train = model_dir / "training.log"
    if p_train.exists():
        p = _safe(_plot_training_loss, p_train, out_dir)
        if p: written.append(p)
    return written


def fit_priors_diagnostics(priors_path: str | Path,
                              out_dir: Path) -> list[Path]:
    p = _safe(_plot_nucleus_prior_fit, Path(priors_path), out_dir)
    return [p] if p else []


def _plot_nucleus_prior_fit(priors_path: Path, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mf = json.loads(priors_path.read_text())
    priors = mf.get("priors", {})
    out = Path(out_dir) / "nucleus_prior_fit.png"
    if not priors:
        return out

    q_levels = ["p10", "p25", "p50", "p75", "p90"]
    qx = [10, 25, 50, 75, 90]
    type_names = list(priors.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(type_names):
        prior = priors[name]
        emp_area = prior.get("target_area_quantiles", {})
        emp_ar   = prior.get("target_axis_ratio_quantiles", {})
        fd = prior.get("fit_diagnostics", {}).get("final_quantiles", {})
        pred_area = fd.get("area", {})
        pred_ar   = fd.get("axis_ratio", {})
        ea  = [emp_area.get(q, np.nan) for q in q_levels]
        pa  = [pred_area.get(q, np.nan) for q in q_levels]
        ear = [emp_ar.get(q, np.nan) for q in q_levels]
        par = [pred_ar.get(q, np.nan) for q in q_levels]
        color = cmap(i % 10)
        n = prior.get("n_train", 0)
        axes[0].plot(qx, ea, "o-", color=color, lw=1.0, ms=4,
                       label=f"{name[:20]} (n={n})")
        axes[0].plot(qx, pa, "x--", color=color, lw=1.0, ms=5, alpha=0.7)
        axes[1].plot(qx, ear, "o-", color=color, lw=1.0, ms=4)
        axes[1].plot(qx, par, "x--", color=color, lw=1.0, ms=5, alpha=0.7)
    axes[0].set_xlabel("quantile (%)"); axes[0].set_ylabel("2D area (µm²)")
    axes[0].set_title("Area: solid = empirical, dashed = fitted", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("quantile (%)"); axes[1].set_ylabel("axis ratio")
    axes[1].set_title("Axis ratio: solid = empirical, dashed = fitted", fontsize=9)
    fig.suptitle(
        f"3D nucleus prior fit quality — {mf.get('n_records_used', 0):,} "
        f"cells, {len(type_names)} types",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _plot_training_loss(training_log: Path, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re
    rows: list[dict] = []
    line_re = re.compile(r"^step\s+(\d+)\s+(.+)$")
    kv_re   = re.compile(r"(\w+)\s+(-?[\d.eE+-]+)")
    with training_log.open() as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m: continue
            kvs = dict((k, float(v)) for k, v in kv_re.findall(m.group(2)))
            kvs["step"] = int(m.group(1))
            rows.append(kvs)
    out = Path(out_dir) / "training_loss.png"
    if not rows: return out
    steps = np.array([r["step"] for r in rows])
    keys  = [k for k in rows[0].keys() if k != "step"]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(keys):
        vals = np.array([r.get(k, np.nan) for r in rows])
        ax.plot(steps, vals, label=k, color=cmap(i % 10), lw=1.0)
    ax.set_xlabel("training step"); ax.set_ylabel("loss / metric")
    ax.set_title("Renderer training trajectory", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# explain diagnostics — orchestrator


# Tile-scale 3-panel bench regions (real | m.render | saved bundle)
STANDARD_REGIONS: list[tuple[tuple[float, float, float, float], str]] = [
    ((1750, 1300, 2240, 1790), "ductal_mixed"),
    ((6289, 2018, 6779, 2508), "endocrine_islet"),
]


def explain_diagnostics(bundle_path: str | Path, synth_dir: str | Path,
                            model_dir: str | Path, out_dir: Path,
                            regions: list = None) -> list[Path]:
    """Produce the standard explain diagnostic set:

      A1.  Whole-bundle thumbnail (real vs synth, low-res morphology pyramid).
      A2.  Three mid-scale regions across the bundle (real vs synth).
      A3.  Tile-scale 3-panel (real | m.render | saved bundle) at each
           standard bench region.
      A4.  Cell-level grid: hero cells per cell-type, real vs synth.
      B5.  Per-cell-type breakdown (synth vs real cell counts + tx means).
      B6.  Per-cell transcript-count distribution (synth vs real).
      B7.  Per-cell area distribution (synth vs real).
      C8.  Per-gene total-count scatter (synth vs real, log-log).
      D10. Per-channel intensity histograms (synth vs real).

    Each diagnostic runs in its own try/except so one failure doesn't
    abort the rest.
    """
    bundle_path = Path(bundle_path)
    synth_dir   = Path(synth_dir)
    model_dir   = Path(model_dir)
    out_dir     = Path(out_dir)
    regions     = regions or STANDARD_REGIONS

    # Scope is a first-class input: read the synth bundle's rendered extent
    # ONCE (single source of truth — bundle_bounds_um) and hand it to every
    # panel, which gates example-selection against it and annotates its
    # title with the bounds it used. None => whole bundle (no restriction).
    from .scene_2d.render_tile import bundle_bounds_um
    scene_bounds = bundle_bounds_um(synth_dir)
    print(f"[diagnostics] scope: {_scope_label(scene_bounds)}", file=sys.stderr)

    # Restrict A3 bench regions to the rendered extent (region bundles only
    # cover part of the slide). Keep those that fit; if none do, synthesize
    # in-bounds bench regions centered in the rendered area.
    if scene_bounds is not None:
        x0b, y0b, x1b, y1b = scene_bounds
        kept = [(b, lab) for (b, lab) in regions
                if b[0] >= x0b and b[1] >= y0b and b[2] <= x1b and b[3] <= y1b]
        if not kept:
            cx, cy = 0.5 * (x0b + x1b), 0.5 * (y0b + y1b)
            for s, lab in ((300.0, "region_center"), (490.0, "region_wide")):
                h = s / 2
                bx0, by0 = max(x0b, cx - h), max(y0b, cy - h)
                kept.append(((bx0, by0, min(x1b, bx0 + s), min(y1b, by0 + s)), lab))
        regions = kept

    written: list[Path] = []
    def _add(p):
        if p is not None: written.append(p)

    # A. Real vs render across scales (scope-aware)
    _add(_safe(_plot_whole_bundle_thumbnail, bundle_path, synth_dir,
                  model_dir, out_dir, scene_bounds))
    for p in _safe(_plot_midscale_regions, bundle_path, synth_dir,
                       model_dir, out_dir, scene_bounds) or []:
        _add(p)
    for p in _safe(_plot_tile_3panels, bundle_path, synth_dir, model_dir,
                       out_dir, regions) or []:
        _add(p)
    _add(_safe(_plot_cell_level_grid, bundle_path, synth_dir, model_dir,
                  out_dir, scene_bounds=scene_bounds))

    # B. Population stats
    _add(_safe(_plot_celltype_breakdown, bundle_path, synth_dir, out_dir))
    _add(_safe(_plot_per_cell_distributions, bundle_path, synth_dir, out_dir))

    # T. Cell-type resolution provenance (skipped silently on bundles
    # written by pre-resolver-refactor xesim that don't have the
    # cell_type_source column).
    _add(_safe(_plot_type_resolution_breakdown, bundle_path, synth_dir, out_dir))

    # C. Transcript level
    _add(_safe(_plot_per_gene_scatter, bundle_path, synth_dir, out_dir))

    # D. Intensity
    _add(_safe(_plot_intensity_histograms, bundle_path, synth_dir,
                  model_dir, out_dir))

    return written


# ---------------------------------------------------------------------------
# A. Real vs render across scales


def _read_morph_lowres(bundle_dir: Path, pyramid_level: int = 5) -> np.ndarray:
    """Read morphology_focus at a downsampled pyramid level. Returns
    (C, Y, X) float32."""
    import tifffile
    p0 = bundle_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
    if not p0.exists():
        return None
    with tifffile.TiffFile(p0) as tf:
        series = tf.series[0]
        levels = series.levels
        idx = min(pyramid_level, len(levels) - 1)
        arr = levels[idx].asarray()
    if arr.ndim == 3 and arr.shape[0] >= 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32)
    return None


def _read_morph_synth_lowres(synth_dir: Path, pyramid_level: int = 5,
                                 n_ch: int = 4) -> np.ndarray | None:
    """Read synth morphology at a downsampled pyramid level.

    For Xenium-style multi-channel OME bundles, file 0 already returns the
    full (C, Y, X) stack at this level via OME cross-references — same
    code path as the real reader. For older single-channel-per-file
    bundles, falls back to reading each file's plane. If the synth bundle
    was written with a shallower pyramid than requested, in-memory block-
    mean downsamples by 2^deficit so callers comparing real-vs-synth at
    the same `pyramid_level` see matched spatial scales.
    """
    import tifffile
    p0 = synth_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
    if not p0.exists():
        return None
    with tifffile.TiffFile(p0) as tf:
        series = tf.series[0]
        levels = series.levels
        idx = min(pyramid_level, len(levels) - 1)
        deficit = max(0, pyramid_level - idx)
        arr = levels[idx].asarray()
    if arr.ndim == 3 and arr.shape[0] >= 1:
        arr = arr.astype(np.float32)
        return _block_mean_downsample(arr, 2 ** deficit) if deficit else arr
    # Legacy bundles: per-channel files (single-plane each).
    arrs = [arr.astype(np.float32)]
    for ci in range(1, n_ch):
        p = synth_dir / "morphology_focus" / f"morphology_focus_{ci:04d}.ome.tif"
        if not p.exists():
            break
        with tifffile.TiffFile(p) as tf:
            series = tf.series[0]
            idx_c = min(pyramid_level, len(series.levels) - 1)
            arrs.append(series.levels[idx_c].asarray().astype(np.float32))
    h_min = min(a.shape[0] for a in arrs); w_min = min(a.shape[1] for a in arrs)
    arrs = [a[:h_min, :w_min] for a in arrs]
    stacked = np.stack(arrs, axis=0)
    return _block_mean_downsample(stacked, 2 ** deficit) if deficit else stacked


def _block_mean_downsample(arr_chw: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample (C, H, W) by `factor` along H and W.
    Trims to a multiple of `factor` first, then averages."""
    if factor <= 1:
        return arr_chw
    C, H, W = arr_chw.shape
    H2, W2 = (H // factor) * factor, (W // factor) * factor
    return arr_chw[:, :H2, :W2].reshape(
        C, H2 // factor, factor, W2 // factor, factor).mean(axis=(2, 4))


def _load_display_lut(model_dir: Path) -> dict | None:
    from .scene_2d.render_tile import load_model_display_lut
    return load_model_display_lut(str(model_dir))


def _plot_whole_bundle_thumbnail(bundle_path: Path, synth_dir: Path,
                                     model_dir: Path, out_dir: Path,
                                     scene_bounds=None) -> Path:
    """A1: morphology thumbnail, real vs synth, side by side. For a region
    bundle the real panel is cropped to the same rendered extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "scale_A1_whole_bundle.png"
    lut = _load_display_lut(model_dir)
    LVL = 5
    real = _read_morph_lowres(bundle_path, pyramid_level=LVL)
    synth = _read_morph_synth_lowres(synth_dir, pyramid_level=LVL,
                                          n_ch=real.shape[0] if real is not None else 4)
    if real is None:
        return _skip("A1", f"real morphology not readable ({bundle_path})")
    if synth is None:
        return _skip("A1", f"synth morphology not readable ({synth_dir})")
    # For a region synth bundle, crop the real plane to the same rendered
    # extent so the two panels show the SAME area (else real-whole vs
    # synth-region are misaligned). scene_bounds is None for a full bundle.
    rb = scene_bounds
    psz = 0.2125
    try:
        import json as _j
        psz = float(_j.load(open(model_dir / "manifest.json")).get("pixel_size", 0.2125))
    except Exception:
        pass
    if rb is not None:
        sc = psz * (2 ** LVL)
        ry0, ry1 = int(rb[1] / sc), int(round(rb[3] / sc))
        rx0, rx1 = int(rb[0] / sc), int(round(rb[2] / sc))
        ry1 = max(ry1, ry0 + 1); rx1 = max(rx1, rx0 + 1)
        real = real[:, ry0:min(ry1, real.shape[1]), rx0:min(rx1, real.shape[2])]
    # Match sizes
    h = min(real.shape[1], synth.shape[1]); w = min(real.shape[2], synth.shape[2])
    real = real[:, :h, :w]; synth = synth[:, :h, :w]
    if lut:
        real_n = _lut_normalize(real, lut)
        synth_n = _lut_normalize(synth, lut)
    else:
        real_n = real / max(real.max(), 1.0)
        synth_n = synth / max(synth.max(), 1.0)
    ch_names = (lut and [c.get("name", n) for c, n in zip(
        lut.get("channels", []), list(CHANNEL_HUE.keys()))]) or list(CHANNEL_HUE.keys())
    rgb_real  = _composite_rgb(real_n, ch_names)
    rgb_synth = _composite_rgb(synth_n, ch_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white",
                              gridspec_kw={"wspace": 0.02})
    scope = _scope_label(scene_bounds)
    axes[0].imshow(rgb_real)
    axes[0].set_title(f"real — {scope}", fontsize=11, fontweight="bold")
    axes[1].imshow(rgb_synth)
    axes[1].set_title(f"synth — {scope}", fontsize=11, fontweight="bold")
    for a in axes: a.axis("off")
    plt.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05,
                  facecolor="white")
    plt.close(fig)
    return out


def _plot_midscale_regions(bundle_path: Path, synth_dir: Path,
                                model_dir: Path, out_dir: Path,
                                scene_bounds=None,
                                size_um: float = 1500.0) -> list[Path]:
    """A2: up to 3 mid-scale regions (~1.5 mm, smaller if the scene is) —
    real vs saved bundle at each."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    lut = _load_display_lut(model_dir)
    # Pick regions by cell-density: top 3 non-overlapping windows.
    cells_real = pd.read_parquet(bundle_path / "cells.parquet",
                                      columns=["x_centroid", "y_centroid"])
    xmin, xmax = cells_real["x_centroid"].min(), cells_real["x_centroid"].max()
    ymin, ymax = cells_real["y_centroid"].min(), cells_real["y_centroid"].max()
    # Clamp the scan to the rendered extent so region bundles only pick
    # in-region windows (windows must fully fit). scene_bounds=None => full.
    if scene_bounds is not None:
        xmin = max(xmin, scene_bounds[0]); ymin = max(ymin, scene_bounds[1])
        xmax = min(xmax, scene_bounds[2]); ymax = min(ymax, scene_bounds[3])
    # Adapt the window to the available extent so small regions still get a
    # panel (use ~80% of the smaller side, capped at the default 1500 µm).
    avail = min(xmax - xmin, ymax - ymin)
    if avail <= 0:
        return _skip("A2", f"no in-scope area for {_scope_label(scene_bounds)}")
    size_um = float(min(size_um, 0.8 * avail))
    # Coarse grid scan
    step = max(size_um / 2, 1.0)
    candidates = []
    for x0 in np.arange(xmin, xmax - size_um + 1, step):
        for y0 in np.arange(ymin, ymax - size_um + 1, step):
            in_box = ((cells_real.x_centroid >= x0) &
                        (cells_real.x_centroid < x0 + size_um) &
                        (cells_real.y_centroid >= y0) &
                        (cells_real.y_centroid < y0 + size_um))
            candidates.append((int(in_box.sum()), x0, y0))
    candidates.sort(reverse=True)
    chosen = []
    for n, x0, y0 in candidates:
        if all(abs(x0 - c[1]) > size_um * 0.5 or abs(y0 - c[2]) > size_um * 0.5
                  for c in chosen):
            chosen.append((n, x0, y0))
        if len(chosen) >= 3:
            break
    if not chosen or chosen[0][0] == 0:
        return _skip("A2", f"no in-bounds cell-dense window for "
                     f"{_scope_label(scene_bounds)}")

    written = []
    psz = 0.2125
    try:
        import json as _j
        m = _j.load(open(model_dir / "manifest.json"))
        psz = float(m.get("pixel_size", 0.2125))
    except Exception:
        pass
    # Bounded reads via real_tile_image — handles tiled OME-TIFF via
    # ImageStackReader without loading the full multi-GB morphology into
    # RAM (the breast 5K bundle is 24.5 GB; float32 load would OOM).
    from .scene_2d.render_tile import real_tile_image

    ch_names = list(CHANNEL_HUE.keys())
    for k, (n_cells, x0, y0) in enumerate(chosen):
        x1, y1 = x0 + size_um, y0 + size_um
        real_c = real_tile_image(str(bundle_path),
            tile_bounds_um=(float(x0), float(x1), float(y0), float(y1)),
            pixel_size_um=psz)
        synth_c = real_tile_image(str(synth_dir),
            tile_bounds_um=(float(x0), float(x1), float(y0), float(y1)),
            pixel_size_um=psz)
        if real_c is None or synth_c is None:
            continue
        h = min(real_c.shape[1], synth_c.shape[1])
        w = min(real_c.shape[2], synth_c.shape[2])
        real_c = real_c[:, :h, :w]; synth_c = synth_c[:, :h, :w]
        if lut:
            real_n = _lut_normalize(real_c, lut); synth_n = _lut_normalize(synth_c, lut)
        else:
            real_n = real_c / max(real_c.max(), 1.0)
            synth_n = synth_c / max(synth_c.max(), 1.0)
        rgb_real = _composite_rgb(real_n, ch_names)
        rgb_synth = _composite_rgb(synth_n, ch_names)
        # 4× higher resolution than the previous (13, 6.6 @ dpi=130) output —
        # mid-scale (~1500 µm) panels are useless at small size, but at 4× the
        # cell detail is visible. ~6.8k × 3.5k px per pair.
        fig, axes = plt.subplots(1, 2, figsize=(26, 13.2), facecolor="white",
                                   gridspec_kw={"wspace": 0.01})
        axes[0].imshow(rgb_real)
        axes[0].set_title(f"real — region {k+1}\n{int(size_um)}×{int(size_um)} µm, "
                            f"{n_cells} cells", fontsize=14, fontweight="bold")
        axes[1].imshow(rgb_synth)
        axes[1].set_title(f"synth — region {k+1}", fontsize=14, fontweight="bold")
        for a in axes: a.axis("off")
        out = Path(out_dir) / f"scale_A2_region{k+1}.png"
        plt.savefig(out, dpi=260, bbox_inches="tight", pad_inches=0.05,
                      facecolor="white")
        plt.close(fig)
        written.append(out)
    return written


def _plot_tile_3panels(bundle_path: Path, synth_dir: Path, model_dir: Path,
                          out_dir: Path, regions) -> list[Path]:
    """A3: small bench-scale 3-panel real | m.render | saved bundle.

    Runs the packaged region-3panel module as a subprocess (``python -m
    xesim.scene_2d.region_3panel``) so the model load + CUDA context that
    panel 2 (m.render) needs live in their own process and are freed after —
    keeping the GPU dependency out of the parent diagnostics process."""
    import subprocess
    if not regions:
        _skip("A3", "no in-bounds bench regions for this scope")
        return []
    written = []
    for bounds, label in regions:
        bounds_str = ",".join(f"{v:.0f}" for v in bounds)
        out = Path(out_dir) / f"scale_A3_{label}.png"
        try:
            subprocess.run(
                [sys.executable, "-m", "xesim.scene_2d.region_3panel",
                 "--bundle", str(bundle_path),
                 "--synth", str(synth_dir),
                 "--model", str(model_dir),
                 "--region", bounds_str,
                 "--out", str(out)],
                check=True, capture_output=True, text=True)
            written.append(out)
        except subprocess.CalledProcessError as e:
            _skip("A3", f"region {label} render failed: "
                  f"{e.stderr.strip()[-200:]}")
    return written


# Per-cell-type contour palette for A4. Matches the merged_annotation
# vocabulary seen across pancreas + breast 5K models. Unknown types
# fall back to TYPE_DEFAULT_COLOR.
TYPE_PALETTE = {
    "Exocrine epithelial":      "#e41a1c",
    "Ductal/tumor epithelial":  "#377eb8",
    "Fibroblast / CAF":         "#4daf4a",
    "Immune":                   "#984ea3",
    "Endothelial":              "#ff7f00",
    "Mural / pericyte":         "#ffff33",
    "Endocrine":                "#a65628",
    "Epithelial":               "#e41a1c",
    "T / NK":                   "#f781bf",
    "Myeloid":                  "#984ea3",
    "B / plasma":               "#fb8072",
    "Ambiguous / low-quality":  "#999999",
}
TYPE_DEFAULT_COLOR = "#cccccc"


def _find_annotation(bundle_path: Path, model_dir: Path):
    """Locate the real-bundle cell-type annotation CSV.

    Tries the standard locations in order; returns the DataFrame
    (cols ``cell_id`` + ``merged_annotation``) or None if not found."""
    import pandas as pd
    cands = [
        bundle_path / "annotations" / "annotation.csv.gz",
        bundle_path.parent / "annotations" / "annotation.csv.gz",
        model_dir / "annotations" / "annotation.csv.gz",
        bundle_path / "annotations" / "annotation.csv",
        bundle_path.parent / "annotations" / "annotation.csv",
        model_dir / "annotations" / "annotation.csv",
    ]
    for p in cands:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "cell_id" in df.columns and "merged_annotation" in df.columns:
                    return df
            except Exception:
                continue
    return None


def _plot_cell_level_grid(bundle_path: Path, synth_dir: Path,
                              model_dir: Path, out_dir: Path,
                              crop_um: float = 64.0,
                              n_regions: int = 12,
                              composition_window_um: float = 200.0,
                              min_spatial_dist_um: float = 300.0,
                              cells_per_type: int | None = None,
                              scene_bounds=None) -> Path:
    """A4: compositionally distinct cell-level regions (3-column layout).

    For up to ``n_regions`` cell types, pick a representative ~``crop_um``
    window whose ``composition_window_um`` neighborhood is enriched for
    that type. Each row shows real | rendered | rendered+contours+tx with
    cell contours colored per type and a per-type legend below.

    Region selection is anchored to the REAL bundle's cells + annotation
    (NOT the synth bundle's cells_synth.parquet) so that the SAME real
    bundle always yields the SAME region picks, regardless of which model
    rendered the synth_dir. This makes density-vs-stratified diagnostic
    comparisons apples-to-apples.

    Replaces the older per-type 3-cell-per-row 2-column layout.
    The ``cells_per_type`` argument is accepted for API compatibility
    and ignored.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "scale_A4_cell_grid.png"

    # Anchor picks to the REAL bundle (deterministic across synth bundles)
    real_cells_path = bundle_path / "cells.parquet"
    if not real_cells_path.exists():
        return _skip("A4", f"real cells.parquet not found ({bundle_path})")
    real_cells = pd.read_parquet(
        real_cells_path, columns=["cell_id", "x_centroid", "y_centroid"])
    ann_df = _find_annotation(bundle_path, model_dir)
    if ann_df is None:
        return _skip("A4", "no cell-type annotation (annotation.csv[.gz]) found")
    cell_to_type = dict(zip(ann_df["cell_id"].astype(str),
                              ann_df["merged_annotation"].astype(str)))
    real_cells["cell_type"] = real_cells["cell_id"].astype(str).map(cell_to_type)
    gt = real_cells.dropna(subset=["cell_type"]).rename(
        columns={"x_centroid": "centroid_x", "y_centroid": "centroid_y"})
    # Restrict candidate cells to the rendered extent so a region bundle
    # picks in-region examples (crop fits) instead of empty out-of-region
    # windows. Anchored to the real-cell ordering, so the same region/full
    # always selects the same examples where they overlap. scene_bounds is
    # shrunk by crop_um/2 so the crop window fits; None => full bundle.
    if scene_bounds is not None:
        m = crop_um / 2
        x0b, y0b, x1b, y1b = (scene_bounds[0] + m, scene_bounds[1] + m,
                              scene_bounds[2] - m, scene_bounds[3] - m)
        gt = gt[(gt.centroid_x >= x0b) & (gt.centroid_x < x1b) &
                (gt.centroid_y >= y0b) & (gt.centroid_y < y1b)]
        if len(gt) == 0:
            return _skip("A4", f"no in-bounds annotated cells for "
                         f"{_scope_label(scene_bounds)}")
    types_used = [t for t in sorted(gt["cell_type"].dropna().unique())
                    if t.lower() != "unknown" and t.lower() != "ambiguous / low-quality"]
    if not types_used:
        return _skip("A4", "no usable cell types in scope")

    # For each candidate type, find up to K spatially-distinct windows
    # whose composition_window_um neighborhood is most enriched for that
    # type. K = ceil(n_regions / n_types) so we cover the diversity of
    # cell types while still picking n_regions total.
    import math as _math
    picks_per_type = max(1, _math.ceil(n_regions / max(len(types_used), 1)))

    half_comp = composition_window_um / 2
    cells_xy = gt[["cell_id", "cell_type",
                       "centroid_x", "centroid_y"]].copy()
    cells_xy_arr = cells_xy[["centroid_x", "centroid_y"]].to_numpy()
    types_arr = cells_xy["cell_type"].to_numpy()
    picks: list[tuple[str, float, float, int, int]] = []
    for t in types_used:
        cand = cells_xy[cells_xy.cell_type == t]
        if len(cand) < 5:
            continue
        # Subsample candidates for speed (max 200), then score each by
        # the same-type fraction in its composition window. Sort by score
        # and greedily pick top-K subject to min_spatial_dist_um from
        # already-picked spots of any type.
        sample = cand.sample(min(200, len(cand)), random_state=42)
        scored: list[tuple[float, float, float, int, int]] = []
        for _, row in sample.iterrows():
            cx, cy = float(row.centroid_x), float(row.centroid_y)
            in_win = ((cells_xy_arr[:, 0] >= cx - half_comp) &
                        (cells_xy_arr[:, 0] <= cx + half_comp) &
                        (cells_xy_arr[:, 1] >= cy - half_comp) &
                        (cells_xy_arr[:, 1] <= cy + half_comp))
            n = int(in_win.sum())
            if n < 5:
                continue
            n_match = int(((types_arr == t) & in_win).sum())
            frac = n_match / n
            scored.append((frac, cx, cy, n_match, n))
        scored.sort(key=lambda r: -r[0])
        type_picks_taken = 0
        for frac, cx, cy, n_match, n_tot in scored:
            if all((cx - px)**2 + (cy - py)**2 >= min_spatial_dist_um**2
                       for _, px, py, _, _ in picks):
                picks.append((t, cx, cy, n_match, n_tot))
                type_picks_taken += 1
                if type_picks_taken >= picks_per_type:
                    break
        if len(picks) >= n_regions:
            break
    picks = picks[:n_regions]
    if not picks:
        return _skip("A4", f"no enriched in-bounds cell windows for "
                     f"{_scope_label(scene_bounds)}")

    psz = 0.2125
    try:
        import json as _j
        m = _j.load(open(model_dir / "manifest.json"))
        psz = float(m.get("pixel_size", 0.2125))
    except Exception:
        pass

    from .scene_2d.render_tile import real_tile_image
    lut = _load_display_lut(model_dir)
    ch_names = list(CHANNEL_HUE.keys())

    poly = pd.read_parquet(synth_dir / "cell_boundaries.parquet",
                              columns=["cell_id", "vertex_x", "vertex_y"])
    tx_path = synth_dir / "transcripts.parquet"
    tx = (pd.read_parquet(tx_path, columns=["x_location", "y_location"])
            if tx_path.exists() else None)

    cell_to_color = dict(zip(gt.cell_id.astype(str),
                                gt.cell_type.map(
                                    lambda t: TYPE_PALETTE.get(t, TYPE_DEFAULT_COLOR))))

    n = len(picks)
    half = crop_um / 2
    import math as _math
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.3 * n), facecolor="white",
                              gridspec_kw={"wspace": 0.0, "hspace": 0.04})
    if n == 1:
        axes = axes[None, :]

    for i, (t, cx, cy, n_match, n_tot) in enumerate(picks):
        bounds = (float(cx - half), float(cx + half),
                    float(cy - half), float(cy + half))
        real_c = real_tile_image(str(bundle_path),
            tile_bounds_um=bounds, pixel_size_um=psz)
        synth_c = real_tile_image(str(synth_dir),
            tile_bounds_um=bounds, pixel_size_um=psz)
        if real_c is None or synth_c is None:
            continue
        h = min(real_c.shape[1], synth_c.shape[1])
        w = min(real_c.shape[2], synth_c.shape[2])
        real_c = real_c[:, :h, :w]
        synth_c = synth_c[:, :h, :w]
        x0_um = _math.floor(bounds[0] / psz) * psz
        y0_um = _math.floor(bounds[2] / psz) * psz
        x1_um = x0_um + w * psz
        y1_um = y0_um + h * psz
        ext = (x0_um, x1_um, y1_um, y0_um)
        real_n = _lut_normalize(real_c, lut) if lut else \
                    real_c / max(real_c.max(), 1.0)
        synth_n = _lut_normalize(synth_c, lut) if lut else \
                    synth_c / max(synth_c.max(), 1.0)
        axes[i, 0].imshow(_composite_rgb(real_n,  ch_names), extent=ext)
        axes[i, 1].imshow(_composite_rgb(synth_n, ch_names), extent=ext)
        axes[i, 2].imshow(_composite_rgb(synth_n, ch_names), extent=ext)
        local_p = poly[(poly.vertex_x >= x0_um) & (poly.vertex_x < x1_um)
                          & (poly.vertex_y >= y0_um) & (poly.vertex_y < y1_um)]
        for cid, g in local_p.groupby("cell_id"):
            color = cell_to_color.get(str(cid), TYPE_DEFAULT_COLOR)
            axes[i, 2].plot(g.vertex_x.values, g.vertex_y.values,
                                color=color, linewidth=0.7, alpha=0.95)
        if tx is not None:
            local_tx = tx[(tx.x_location >= x0_um) & (tx.x_location < x1_um)
                              & (tx.y_location >= y0_um) & (tx.y_location < y1_um)]
            axes[i, 2].scatter(local_tx.x_location, local_tx.y_location,
                                  s=0.75, c="white", alpha=0.7,
                                  marker=".", linewidths=0)
        for a in axes[i]:
            a.set_xlim(x0_um, x1_um); a.set_ylim(y1_um, y0_um)
            a.set_xticks([]); a.set_yticks([])
        axes[i, 0].set_ylabel(f"{t[:18]}\n{n_match}/{n_tot} of type",
                                  fontsize=9, rotation=0, ha="right",
                                  va="center", labelpad=4)

    axes[0, 0].set_title("real", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("rendered", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("rendered + cells (typed) + tx",
                            fontsize=12, fontweight="bold")

    # Per-type color legend
    present_types = sorted(set(gt["cell_type"].dropna().astype(str)))
    legend_handles = [plt.Line2D([0], [0],
                                       color=TYPE_PALETTE.get(t, TYPE_DEFAULT_COLOR),
                                       lw=3, label=t)
                          for t in present_types]
    fig.legend(handles=legend_handles, loc="lower center",
                  ncol=min(4, len(legend_handles)),
                  fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"A4. Composition-distinct regions — "
                    f"{int(crop_um)}×{int(crop_um)} µm cell-scale crops "
                    f"[{_scope_label(scene_bounds)}]",
                    fontsize=11, fontweight="bold", y=0.998)
    plt.subplots_adjust(left=0.13, right=1.0, top=0.97, bottom=0.06)
    plt.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.02,
                  facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# B. Population stats


def _plot_celltype_breakdown(bundle_path: Path, synth_dir: Path,
                                  out_dir: Path) -> Path:
    """B5: synth vs real cell counts + mean tx per type (horizontal bars)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "stats_B5_celltype_breakdown.png"
    gt = pd.read_parquet(synth_dir / "ground_truth" / "cells_synth.parquet")
    if "cell_type" not in gt.columns:
        return out
    synth_counts = gt.groupby("cell_type").size().sort_values(ascending=False)
    tx_synth = pd.read_parquet(synth_dir / "transcripts.parquet",
                                    columns=["cell_id"])
    cell2type = dict(zip(gt["cell_id"].astype(str), gt["cell_type"]))
    tx_synth["cell_type"] = tx_synth["cell_id"].astype(str).map(cell2type)
    synth_tx_per_type = tx_synth.groupby("cell_type").size()

    # Real: look up annotation
    real_ann = None
    cand = [bundle_path.parent / "annotations" / "annotation.csv.gz",
              bundle_path / "annotations" / "annotation.csv.gz"]
    for c in cand:
        if c.exists():
            real_ann = pd.read_csv(c, compression="infer"); break
    real_counts = pd.Series(dtype=int)
    real_tx_per_type = pd.Series(dtype=int)
    if real_ann is not None:
        col_type = ("merged_annotation" if "merged_annotation" in real_ann.columns
                       else real_ann.columns[-1])
        real_counts = real_ann.groupby(col_type).size()
        # tx classification: use cell_id → type lookup (only for assigned)
        real_id2type = dict(zip(real_ann["cell_id"].astype(str), real_ann[col_type]))
        try:
            tx_r = pd.read_parquet(bundle_path / "transcripts.parquet",
                                        columns=["cell_id", "qv"])
            tx_r = tx_r[tx_r.qv >= 20]
            tx_r["cell_type"] = tx_r["cell_id"].astype(str).map(real_id2type)
            real_tx_per_type = tx_r.groupby("cell_type").size()
        except Exception:
            pass

    types = list(synth_counts.index)
    if not types: return out

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(types) + 1.5)),
                               facecolor="white", gridspec_kw={"wspace": 0.32})
    y = np.arange(len(types))
    sc = [synth_counts.get(t, 0) for t in types]
    rc = [real_counts.get(t, 0)  for t in types]
    axes[0].barh(y - 0.20, rc, 0.40, color="#aa5555", label="real", alpha=0.85)
    axes[0].barh(y + 0.20, sc, 0.40, color="#3a6fb0", label="synth", alpha=0.85)
    axes[0].set_yticks(y); axes[0].set_yticklabels([t[:24] for t in types],
                                                          fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("cell count"); axes[0].set_title(
        "Cells per type", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    s_tx = [synth_tx_per_type.get(t, 0) / max(synth_counts.get(t, 1), 1)
              for t in types]
    r_tx = [real_tx_per_type.get(t, 0) / max(real_counts.get(t, 1), 1)
              for t in types]
    axes[1].barh(y - 0.20, r_tx, 0.40, color="#aa5555", label="real", alpha=0.85)
    axes[1].barh(y + 0.20, s_tx, 0.40, color="#3a6fb0", label="synth", alpha=0.85)
    axes[1].set_yticks(y); axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean transcripts / cell")
    axes[1].set_title("Mean tx per cell", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle(f"B5. Population breakdown — {gt.shape[0]:,} synth cells",
                   fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _plot_per_cell_distributions(bundle_path: Path, synth_dir: Path,
                                       out_dir: Path) -> Path:
    """B6 + B7: per-cell transcript-count histogram and per-cell area
    histogram, both real vs synth on the same axes (log-y)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "stats_B6_B7_per_cell_distributions.png"
    tx_s = pd.read_parquet(synth_dir / "transcripts.parquet",
                                columns=["cell_id"])
    s_per_cell = tx_s[tx_s.cell_id != "UNASSIGNED"].groupby("cell_id").size()
    cells_s = pd.read_parquet(synth_dir / "cells.parquet",
                                  columns=["cell_id", "cell_area"])
    s_area = cells_s["cell_area"]

    tx_r = pd.read_parquet(bundle_path / "transcripts.parquet",
                                columns=["cell_id", "qv"])
    tx_r = tx_r[tx_r.qv >= 20]
    r_per_cell = tx_r[tx_r.cell_id != "UNASSIGNED"].groupby("cell_id").size()
    try:
        cells_r = pd.read_parquet(bundle_path / "cells.parquet",
                                       columns=["cell_id", "cell_area"])
        r_area = cells_r["cell_area"]
    except Exception:
        r_area = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white",
                               gridspec_kw={"wspace": 0.27})
    # B6: tx per cell
    bins = np.logspace(0, np.log10(max(r_per_cell.max(), s_per_cell.max())+1), 60)
    axes[0].hist(r_per_cell, bins=bins, alpha=0.55, color="#aa5555",
                   label=f"real (n={len(r_per_cell):,})", log=True, density=True)
    axes[0].hist(s_per_cell, bins=bins, alpha=0.55, color="#3a6fb0",
                   label=f"synth (n={len(s_per_cell):,})", log=True, density=True)
    axes[0].set_xscale("log"); axes[0].set_xlabel("transcripts per cell")
    axes[0].set_ylabel("density (log)")
    axes[0].set_title(
        f"B6. Per-cell tx: synth max={s_per_cell.max()}, real max={r_per_cell.max()}",
        fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    # B7: cell area
    if r_area is not None:
        bins = np.linspace(0, max(r_area.quantile(0.99),
                                       s_area.quantile(0.99)), 60)
        axes[1].hist(r_area, bins=bins, alpha=0.55, color="#aa5555",
                       label="real", log=True, density=True)
        axes[1].hist(s_area, bins=bins, alpha=0.55, color="#3a6fb0",
                       label="synth", log=True, density=True)
    else:
        bins = np.linspace(0, s_area.quantile(0.99), 60)
        axes[1].hist(s_area, bins=bins, alpha=0.55, color="#3a6fb0",
                       label="synth", log=True, density=True)
    axes[1].set_xlabel("cell area (µm²)"); axes[1].set_ylabel("density (log)")
    axes[1].set_title("B7. Per-cell area", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight",
                                          facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# C. Transcript level


def _plot_per_gene_scatter(bundle_path: Path, synth_dir: Path,
                                out_dir: Path) -> Path:
    """C8: per-gene total count, synth vs real, log-log scatter."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "stats_C8_per_gene_scatter.png"
    tx_s = pd.read_parquet(synth_dir / "transcripts.parquet",
                                columns=["feature_name"]) \
        if "feature_name" in pd.read_parquet(synth_dir / "transcripts.parquet").columns \
        else pd.read_parquet(synth_dir / "transcripts.parquet",
                                columns=["gene"])
    gene_col_s = "feature_name" if "feature_name" in tx_s.columns else "gene"
    gene_s = tx_s[gene_col_s].value_counts()

    tx_r = pd.read_parquet(bundle_path / "transcripts.parquet",
                                columns=["feature_name", "qv"]) \
        if "feature_name" in pd.read_parquet(bundle_path / "transcripts.parquet").columns \
        else pd.read_parquet(bundle_path / "transcripts.parquet",
                                columns=["gene", "qv"])
    gene_col_r = "feature_name" if "feature_name" in tx_r.columns else "gene"
    tx_r = tx_r[tx_r.qv >= 20]
    gene_r = tx_r[gene_col_r].value_counts()

    genes = sorted(set(gene_s.index) | set(gene_r.index))
    rs = np.array([gene_r.get(g, 0) for g in genes], dtype=float)
    ss = np.array([gene_s.get(g, 0) for g in genes], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    mask = (rs > 0) & (ss > 0)
    ax.scatter(rs[mask], ss[mask], s=8, alpha=0.65, color="#3a6fb0",
                 edgecolor="none")
    lo = 1.0
    hi = max(rs.max(), ss.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("real total count per gene (log)")
    ax.set_ylabel("synth total count per gene (log)")
    ax.set_title(f"C8. Per-gene counts: {mask.sum()} genes "
                   f"(real {rs.sum():,.0f} tx, synth {ss.sum():,.0f} tx)",
                   fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight",
                                          facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# D. Intensity


def _plot_type_resolution_breakdown(bundle_path: Path, synth_dir: Path,
                                          out_dir: Path) -> Path | None:
    """T1: per-cell-type stacked bar of which resolver tier produced the
    type call. Reveals which cell types lean on which classifiers and
    where curation gaps are (e.g., a row dominated by stain_knn means
    that type has weak annotation + transcript coverage).
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "stats_T1_type_resolution_breakdown.png"
    cs_path = synth_dir / "ground_truth" / "cells_synth.parquet"
    if not cs_path.exists():
        return None
    gt = pd.read_parquet(cs_path)
    if "cell_type_source" not in gt.columns:
        return None    # legacy bundle, no resolver provenance

    # Stacked bar: rows = cell types (sorted by count), stacks = source tier.
    # Six possible sources; lock the order so panels are comparable across runs.
    tier_order = ["annotation", "transcripts", "training", "stain_knn",
                   "ghost_prior", "tx_proposer"]
    tier_colors = {
        "annotation":  "#2c7a3b",
        "transcripts": "#3a6fb0",
        "training":    "#7c5dd1",
        "stain_knn":   "#d18a3a",
        "ghost_prior": "#888888",
        "tx_proposer": "#bb4477",
    }
    # cell-type ordering: by total count, descending. Skip empty rows.
    type_totals = gt.groupby("cell_type").size().sort_values(ascending=False)
    types_sorted = [t for t in type_totals.index if type_totals[t] > 0]
    if not types_sorted:
        return None
    pivot = (gt.groupby(["cell_type", "cell_type_source"]).size()
                  .unstack(fill_value=0)
                  .reindex(index=types_sorted, columns=tier_order, fill_value=0))

    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(types_sorted) + 1.8)),
                              facecolor="white")
    y = np.arange(len(types_sorted))
    left = np.zeros(len(types_sorted))
    for tier in tier_order:
        vals = pivot[tier].to_numpy()
        if vals.sum() == 0:
            continue
        ax.barh(y, vals, left=left, color=tier_colors[tier],
                  edgecolor="white", linewidth=0.5, label=tier, alpha=0.95)
        # In-bar labels for tiers contributing ≥5% of that row.
        for yi, v in enumerate(vals):
            if v >= 0.05 * type_totals.iloc[yi] and v > 0:
                ax.text(left[yi] + v/2, yi, f"{int(v)}",
                          ha="center", va="center", fontsize=7,
                          color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t[:24]}  (n={int(type_totals[t]):,})"
                          for t in types_sorted], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cells")
    ax.set_title("T1. Cell-type resolution: per-tier provenance per type",
                   fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    # Confidence summary line in the figure caption.
    conf_mean = float(gt["cell_type_confidence"].mean())
    high_conf_pct = 100.0 * (gt["cell_type_confidence"] >= 0.5).sum() / len(gt)
    fig.text(0.5, 0.005,
              f"Mean per-cell confidence: {conf_mean:.2f}.  "
              f"{high_conf_pct:.1f}% of cells at confidence ≥ 0.5.",
              ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _plot_intensity_histograms(bundle_path: Path, synth_dir: Path,
                                     model_dir: Path, out_dir: Path,
                                     region: tuple = (1750, 1300, 2240, 1790)
                                     ) -> Path:
    """D10: per-channel pixel-intensity histograms, real vs synth, log-y.
    Sampled at a single tile-scale region (the ductal bench)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    import json as _j

    out = Path(out_dir) / "stats_D10_intensity_histograms.png"
    psz = 0.2125
    try:
        psz = float(_j.load(open(model_dir / "manifest.json")).get(
            "pixel_size", 0.2125))
    except Exception:
        pass
    xmin, ymin, xmax, ymax = region
    # Bounded reads — avoid loading multi-GB full morphology to RAM
    # (24.5 GB on the breast 5K bundle; full load OOMs).
    from .scene_2d.render_tile import real_tile_image
    bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
    real = real_tile_image(str(bundle_path),
        tile_bounds_um=bounds, pixel_size_um=psz)
    synth = real_tile_image(str(synth_dir),
        tile_bounds_um=bounds, pixel_size_um=psz)
    if real is None or synth is None:
        return out
    h = min(real.shape[1], synth.shape[1]); w = min(real.shape[2], synth.shape[2])
    real = real[:, :h, :w]; synth = synth[:, :h, :w]

    n_ch = real.shape[0]
    names = list(CHANNEL_HUE.keys())[:n_ch]
    fig, axes = plt.subplots(1, n_ch, figsize=(4.5 * n_ch, 4),
                               facecolor="white")
    if n_ch == 1: axes = [axes]
    for ci in range(n_ch):
        rf = real[ci].ravel(); sf = synth[ci].ravel()
        upper = max(rf.max(), sf.max(), 10)
        bins = np.logspace(0, np.log10(upper), 80)
        axes[ci].hist(rf, bins=bins, alpha=0.55, color="#aa5555",
                        label="real", log=True, density=True)
        axes[ci].hist(sf, bins=bins, alpha=0.55, color="#3a6fb0",
                        label="synth", log=True, density=True)
        # Quantified p50 / p99 lines so the gap is readable, not just visual.
        rp50, rp99 = np.percentile(rf, 50), np.percentile(rf, 99)
        sp50, sp99 = np.percentile(sf, 50), np.percentile(sf, 99)
        axes[ci].axvline(rp50, color="#aa5555", ls="--", lw=1.0, alpha=0.8)
        axes[ci].axvline(rp99, color="#aa5555", ls=":", lw=1.0, alpha=0.8)
        axes[ci].axvline(sp50, color="#3a6fb0", ls="--", lw=1.0, alpha=0.8)
        axes[ci].axvline(sp99, color="#3a6fb0", ls=":", lw=1.0, alpha=0.8)
        axes[ci].set_xscale("log"); axes[ci].set_title(
            f"{names[ci]}\np50 real={rp50:.0f} synth={sp50:.0f} | "
            f"p99 real={rp99:.0f} synth={sp99:.0f}", fontsize=9)
        axes[ci].set_xlabel("intensity (uint16)")
        if ci == 0: axes[ci].set_ylabel("density (log)")
        axes[ci].legend(fontsize=8); axes[ci].grid(alpha=0.25, which="both")
    fig.suptitle(
        f"D10. Per-channel intensity histograms — ductal bench region "
        f"({int(xmax-xmin)}×{int(ymax-ymin)} µm)",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


__all__ = [
    "resolve_diagnostic_dir",
    "fit_model_diagnostics",
    "fit_priors_diagnostics",
    "explain_diagnostics",
    "STANDARD_REGIONS",
]
