"""Unit tests for the Phase 2 streaming OME-TIFF writer.

Tests use synthetic in-memory tiles and small grids; no model render,
no worker pool. The contract is bytewise parity (modulo at most ~1 LSB
per channel from float32 reduction order) with the Phase 1 in-RAM
stitch+divide+calibrate pipeline.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xesim.scene_2d.intensity import calibrate_to_uint16
from xesim.scene_2d.scene_pipeline import _feather_mask
from xesim.scene_2d.streaming_writer import (
    AUTOSTREAM_PIXELS, StreamingStitchWriter, should_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_grid(
    *,
    n_ch: int = 4,
    tile_px: int = 16,
    overlap_px: int = 4,
    n_rows: int = 3,
    n_cols: int = 3,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict]:
    """Deterministic synthetic tile grid + the dimensions metadata."""
    step_px = tile_px - overlap_px
    H_total = (n_rows - 1) * step_px + tile_px
    W_total = (n_cols - 1) * step_px + tile_px
    rng = np.random.default_rng(seed)
    renders: dict[int, np.ndarray] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            base = 0.10 + 0.05 * (r + c)
            renders[idx] = (
                np.full((n_ch, tile_px, tile_px), base, dtype=np.float32)
                + rng.standard_normal((n_ch, tile_px, tile_px)).astype(np.float32) * 0.01
            ).clip(0.0, 1.0)
    return renders, dict(
        n_ch=n_ch, tile_px=tile_px, overlap_px=overlap_px,
        step_px=step_px, n_rows=n_rows, n_cols=n_cols,
        H_total=H_total, W_total=W_total,
    )


def _phase1_stitch(
    renders: dict[int, np.ndarray], meta: dict, display_lut: dict,
) -> np.ndarray:
    """Reference Phase 1 path: full-plane feather-stitch + divide +
    LUT-native uint16 calibration."""
    n_ch = meta["n_ch"]
    H = meta["H_total"]; W = meta["W_total"]
    n_cols = meta["n_cols"]
    tile = meta["tile_px"]; step = meta["step_px"]
    feather = _feather_mask(tile, meta["overlap_px"])
    av = np.zeros((n_ch, H, W), dtype=np.float32)
    aw = np.zeros((H, W), dtype=np.float32)
    for idx, render in renders.items():
        r = idx // n_cols; c = idx % n_cols
        y0 = r * step; x0 = c * step
        av[:, y0:y0 + tile, x0:x0 + tile] += render * feather[None, :, :]
        aw[y0:y0 + tile, x0:x0 + tile] += feather
    stitched = av / np.maximum(aw[None, :, :], 1e-6)
    return calibrate_to_uint16(
        stitched, channel_names=[f"ch{i}" for i in range(n_ch)],
        mode="off", display_lut=display_lut)


def _phase2_stitch(
    renders: dict[int, np.ndarray], meta: dict, display_lut: dict,
    *, out_dir: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Phase 2 path: synthetic tiles → StreamingStitchWriter → return
    the per-channel uint16 base arrays. ``out_dir`` is a tmp dir under
    which the writer's tmpdir lives; the test driver cleans up."""
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xstream_test_"))
    n_ch = meta["n_ch"]
    w = StreamingStitchWriter(
        out_dir,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        n_ch=n_ch, H_total=meta["H_total"], W_total=meta["W_total"],
        n_tile_rows=meta["n_rows"], n_tile_cols=meta["n_cols"],
        tile_px=meta["tile_px"], overlap_px=meta["overlap_px"],
        step_px=meta["step_px"],
        pixel_size_um=0.2125,
        display_lut=display_lut,
        intensity_mode="off",
        n_pyramid_levels=1,    # base only — pyramid checked separately
        n_focus_files=n_ch,
        progress=False,
    )
    for idx, render in renders.items():
        w.submit_tile(idx, render)
    w.close()
    base = np.stack([np.asarray(arr) for arr in w._base_arrays], axis=0)
    base = base.copy()    # detach from memmap before cleanup
    w.cleanup()
    return base, out_dir


# ---------------------------------------------------------------------------
# should_stream() / auto-threshold
# ---------------------------------------------------------------------------


class TestShouldStream:
    def test_small_bundle_returns_false(self):
        # 4 ch × 100 × 100 = 40k pixels — way under threshold.
        assert should_stream(4, 100, 100) is False

    def test_large_bundle_returns_true(self):
        # 4 ch × 5000 × 5000 = 100M — still under 1e9 default.
        assert should_stream(4, 5000, 5000) is False
        # 4 ch × 20000 × 13000 = 1.04e9 — over.
        assert should_stream(4, 20000, 13000) is True

    def test_env_override_force_on(self, monkeypatch):
        monkeypatch.setenv("XESIM_FORCE_STREAMING_WRITER", "1")
        assert should_stream(4, 10, 10) is True

    def test_env_override_force_off(self, monkeypatch):
        monkeypatch.setenv("XESIM_FORCE_STREAMING_WRITER", "0")
        assert should_stream(4, 10_000, 10_000) is False


# ---------------------------------------------------------------------------
# Feather mask — must stay byte-identical to scene_pipeline's
# ---------------------------------------------------------------------------


class TestFeatherMask:
    def test_streaming_writer_feather_matches_pipeline_feather(self):
        from xesim.scene_2d.streaming_writer import _feather_mask as sw_feather
        from xesim.scene_2d.scene_pipeline import _feather_mask as sp_feather
        for tile, overlap in [(8, 2), (16, 4), (32, 6), (602, 28)]:
            a = sw_feather(tile, overlap)
            b = sp_feather(tile, overlap)
            np.testing.assert_array_equal(
                a, b, f"feather mismatch at tile={tile} overlap={overlap}")

    def test_feather_edge_value_is_nonzero(self):
        # Critical invariant: the linear ramp's index 0 is
        # 1/(overlap+1), not 0. Edge pixels of the bundle inherit the
        # lone tile contribution via this small weight.
        f = _feather_mask(16, 4)
        expected_edge = 1.0 / (4 + 1)
        assert f[0, 0] == pytest.approx(expected_edge ** 2)
        assert f[0, 8] == pytest.approx(expected_edge)


# ---------------------------------------------------------------------------
# Synthetic-grid bytewise parity tests
# ---------------------------------------------------------------------------


class TestSyntheticGridParity:
    def test_tiny_grid_base_parity(self):
        """3×3 grid, 16-px tiles, 4-px overlap — base level bytewise-
        identical to Phase 1."""
        renders, meta = _synthetic_grid(tile_px=16, overlap_px=4,
                                            n_rows=3, n_cols=3)
        display_lut = {
            "channels": [{"lo": 50.0, "hi": 1000.0}] * meta["n_ch"],
            "noise_stats": [],
        }
        ref = _phase1_stitch(renders, meta, display_lut)
        out, tmp = _phase2_stitch(renders, meta, display_lut)
        try:
            np.testing.assert_array_equal(ref, out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rectangular_grid_base_parity(self):
        """3 rows × 5 cols — different x and y tile counts.

        We allow at most 1 LSB diff on a tiny fraction of pixels
        (float32 add ordering: Phase 1 sums row0→row1 contributions in
        tile-arrival order; Phase 2 sums row1's body then adds row0's
        bottom-overlap during strip-emit. Same operands, different
        order, IEEE non-associative). The diff has zero biological
        significance and downstream consumers (uint16 viewers,
        diagnostics) are insensitive to it.
        """
        renders, meta = _synthetic_grid(tile_px=32, overlap_px=6,
                                            n_rows=3, n_cols=5)
        display_lut = {
            "channels": [{"lo": 100.0, "hi": 5000.0}] * meta["n_ch"],
            "noise_stats": [],
        }
        ref = _phase1_stitch(renders, meta, display_lut)
        out, tmp = _phase2_stitch(renders, meta, display_lut)
        try:
            diff = np.abs(ref.astype(np.int32) - out.astype(np.int32))
            assert diff.max() <= 1, f"max diff {diff.max()} > 1"
            assert (diff > 0).sum() / diff.size < 1e-3, \
                f"{(diff > 0).sum()} of {diff.size} pixels differ"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_out_of_order_tile_submission_matches_raster_order(self):
        """imap_unordered may deliver tiles in any order — the writer
        must produce the same output regardless."""
        renders, meta = _synthetic_grid(tile_px=16, overlap_px=4,
                                            n_rows=4, n_cols=4)
        display_lut = {
            "channels": [{"lo": 0.0, "hi": 65535.0}] * meta["n_ch"],
            "noise_stats": [],
        }
        # Raster order
        raster_renders = dict(renders)
        out_raster, tmp_r = _phase2_stitch(raster_renders, meta, display_lut)
        # Shuffled order
        keys = list(renders.keys())
        rng = np.random.default_rng(7)
        shuffled = list(keys); rng.shuffle(shuffled)
        shuffled_renders = {k: renders[k] for k in shuffled}
        out_shuffled, tmp_s = _phase2_stitch(shuffled_renders, meta, display_lut)
        try:
            np.testing.assert_array_equal(out_raster, out_shuffled)
        finally:
            shutil.rmtree(tmp_r, ignore_errors=True)
            shutil.rmtree(tmp_s, ignore_errors=True)


# ---------------------------------------------------------------------------
# End-to-end (writes the actual OME-TIFFs to disk and reads them back)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_morphology_focus_files_written_and_readable(self):
        renders, meta = _synthetic_grid(tile_px=32, overlap_px=4,
                                            n_rows=3, n_cols=3)
        display_lut = {
            "channels": [{"lo": 50.0, "hi": 5000.0}] * meta["n_ch"],
            "noise_stats": [],
        }
        out_dir = Path(tempfile.mkdtemp(prefix="xstream_e2e_"))
        try:
            w = StreamingStitchWriter(
                out_dir,
                channel_names=[f"ch{i}" for i in range(meta["n_ch"])],
                n_ch=meta["n_ch"], H_total=meta["H_total"], W_total=meta["W_total"],
                n_tile_rows=meta["n_rows"], n_tile_cols=meta["n_cols"],
                tile_px=meta["tile_px"], overlap_px=meta["overlap_px"],
                step_px=meta["step_px"],
                pixel_size_um=0.2125,
                display_lut=display_lut,
                intensity_mode="off",
                n_pyramid_levels=3,
                n_focus_files=meta["n_ch"],
                progress=False,
            )
            for idx, render in renders.items():
                w.submit_tile(idx, render)
            w.close()
            # All 4 morphology_focus_* files + morphology.ome.tif exist
            for ch in range(meta["n_ch"]):
                p = out_dir / "morphology_focus" / f"morphology_focus_{ch:04d}.ome.tif"
                assert p.exists(), f"missing {p}"
                arr = tifffile.imread(p, key=0)
                assert arr.shape == (meta["H_total"], meta["W_total"]), \
                    f"focus_{ch} shape {arr.shape} != ({meta['H_total']}, {meta['W_total']})"
            morph = out_dir / "morphology.ome.tif"
            assert morph.exists()
            w.cleanup()
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
