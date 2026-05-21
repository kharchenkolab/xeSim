"""Streaming tile-to-TIFF writer (Phase 2).

Drop-in replacement for Phase 1's full-plane memmap stitch buffer +
end-of-run bundle write. Tiles arrive from the renderer Pool, get
feather-stitched into a small row-ring, and finalized rows are written
to per-channel OME-TIFFs strip-by-strip as they become available. After
all tiles have been processed, pyramid levels are built sequentially
via memmap downsampling and written as sub-IFDs.

Memory profile: O(few tile rows + one memmap level under construction)
regardless of bundle size. RAM cost ≈ 500 MB - 2 GB at peak. Disk cost
≈ base-level uint16 size + pyramid levels (~30% extra).

The streaming writer is opt-in (auto-threshold on bundle area) and
preserves pixel-data parity with the Phase 1 memmap path: same renderer
output, same feather math, same per-channel display_lut calibration.
Noise injection uses a per-strip deterministic RNG; the resulting byte
sequence differs from Phase 1's whole-image-at-once noise but the
statistical properties match.

See ``misc/phase2_streaming_writer.md`` for the design rationale.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np

from .intensity import calibrate_to_uint16

# Heuristic auto-threshold: bundles with base-level pixel count above
# this go through the streaming writer; smaller bundles stay on the
# Phase 1 memmap path. 1e9 pixels ≈ 4 GB / channel uint16 ≈ where the
# Phase 1 RAM cost starts to bite.
AUTOSTREAM_PIXELS = 1_000_000_000


def should_stream(n_ch: int, h: int, w: int) -> bool:
    """Pick streaming-writer vs Phase 1 memmap based on bundle pixel area.

    Override with env var ``XESIM_FORCE_STREAMING_WRITER=1|0``.
    """
    import os
    forced = os.environ.get("XESIM_FORCE_STREAMING_WRITER", "").strip()
    if forced == "1":
        return True
    if forced == "0":
        return False
    return (n_ch * h * w) >= AUTOSTREAM_PIXELS


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _feather_mask(tile_px: int, overlap_px: int) -> np.ndarray:
    """Linear-ramp feather mask — must stay byte-identical to
    ``scene_pipeline._feather_mask`` (parity contract).

    Note: ``ramp[0]`` is ``1/(overlap+1)``, NOT zero. The edge pixels
    receive a small but non-zero weight; after the feather-divide they
    inherit that contribution from the lone bordering tile.
    """
    if overlap_px <= 0:
        return np.ones((tile_px, tile_px), dtype=np.float32)
    ramp = np.ones(tile_px, dtype=np.float32)
    for k in range(overlap_px):
        w = (k + 1) / (overlap_px + 1)
        ramp[k] = w
        ramp[tile_px - 1 - k] = w
    return ramp[:, None] * ramp[None, :]


def _grid_dims(n_tiles: int, n_cols: int) -> tuple[int, int]:
    """Recover (n_rows, n_cols) from the linear tile-grid count."""
    n_rows = int(math.ceil(n_tiles / n_cols))
    return n_rows, n_cols


# ---------------------------------------------------------------------------
# StreamingStitchWriter
# ---------------------------------------------------------------------------


class StreamingStitchWriter:
    """Tile-streaming OME-TIFF writer.

    Construction is from `build_scene`, after the tile grid is known
    but before any tile has been rendered. ``submit_tile`` is called
    once per tile (in any order — `imap_unordered` is supported). After
    the last tile, call ``close`` to flush the row-ring, build the
    pyramid, and finalize all five OME-TIFFs.

    Memory: a small row-ring (≤ 2 input tile rows) for accumulation,
    plus per-channel memmap-backed pyramid intermediates. Disk: ~30%
    extra over the final bundle size during construction; cleaned up
    on close.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        channel_names: list[str],
        n_ch: int,
        H_total: int,
        W_total: int,
        n_tile_rows: int,
        n_tile_cols: int,
        tile_px: int,
        overlap_px: int,
        step_px: int,
        pixel_size_um: float,
        display_lut: dict | None = None,
        intensity_mode: str = "scale",
        target_intensity_stats: dict | None = None,
        target_intensity_quantiles: dict | None = None,
        n_pyramid_levels: int = 8,
        n_focus_files: int = 4,
        progress: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        (self.out_dir / "morphology_focus").mkdir(parents=True, exist_ok=True)
        self.channel_names = list(channel_names)
        self.n_ch = int(n_ch)
        self.H = int(H_total)
        self.W = int(W_total)
        self.n_rows = int(n_tile_rows)
        self.n_cols = int(n_tile_cols)
        self.tile_px = int(tile_px)
        self.overlap_px = int(overlap_px)
        self.step_px = int(step_px)
        self.pixel_size_um = float(pixel_size_um)
        self.display_lut = display_lut
        self.intensity_mode = intensity_mode
        self.target_intensity_stats = target_intensity_stats
        self.target_intensity_quantiles = target_intensity_quantiles
        self.n_pyramid_levels = int(n_pyramid_levels)
        self.n_focus_files = int(n_focus_files)
        self.progress = progress
        self.feather = _feather_mask(self.tile_px, self.overlap_px) \
                       if self.overlap_px > 0 else None
        # OME-TIFF compression. Real Xenium morphology TIFFs are
        # compressed (JPEG-2000); for the streaming path (large bundles)
        # we default to ZSTD — lossless, fast, and crucial for the
        # morphology.ome.tif z-stack which broadcasts the single focal
        # plane to 12 identical z-slices (12x redundancy → compresses to
        # ~1x). Disable with XESIM_TIFF_COMPRESSION=none.
        import os as _os
        _comp = _os.environ.get("XESIM_TIFF_COMPRESSION", "zstd").strip().lower()
        self.compression = None if _comp in ("none", "", "raw") else _comp

        # Disk workspace
        self._tmpdir = Path(tempfile.mkdtemp(prefix="xesim_stream_"))

        # Row-ring: 2 consecutive input rows live at any time. Each
        # entry is a (value, weight) tuple of in-RAM float32 arrays of
        # shape (n_ch, tile_px, W_total) and (tile_px, W_total). Size
        # is bounded by ``2 * n_ch * tile_px * W * 4 bytes`` — ~1 GB
        # on breast 5K (W=51720, tile=602, 4 ch). Memmap-backing
        # showed ~50% per-tile slowdown vs RAM in benches; the ring
        # fits in RAM on any bundle we care about.
        self._ring_band_h = self.tile_px
        self._ring_value: dict[int, np.ndarray] = {}
        self._ring_weight: dict[int, np.ndarray] = {}

        # Per-row tile-arrival count
        self._arrived = np.zeros(self.n_rows, dtype=np.int32)
        self._row_done = np.zeros(self.n_rows, dtype=bool)
        self._strip_emitted = np.zeros(self.n_rows + 1, dtype=bool)

        # Per-channel uint16 memmap holding the base level — used for
        # pyramid construction after streaming. We could skip this if
        # tifffile let us re-read partial tiles cheaply, but going via a
        # memmap keeps the code simple and the cost is a single base-
        # level disk write extra.
        self._base_paths: list[Path] = []
        self._base_arrays: list[np.memmap] = []
        for c in range(self.n_ch):
            p = self._tmpdir / f"base_ch{c}.u16"
            self._base_paths.append(p)
            self._base_arrays.append(
                np.memmap(p, dtype=np.uint16, mode="w+",
                          shape=(self.H, self.W)))

        # Strip y-band lookup: strip s covers y ∈ [s*step_px, (s+1)*step_px),
        # except strip n_rows which is the trailing overlap_px tail.
        # Total strips written to base = n_rows + (1 if remainder>0 else 0).
        self._tail_rows = self.H - self.n_rows * self.step_px
        self._n_strips = self.n_rows + (1 if self._tail_rows > 0 else 0)

        if self.progress:
            mb_ring = (2 * self.n_ch * self.tile_px * self.W * 4
                        + 2 * self.tile_px * self.W * 4) / (1 << 20)
            gb_base = sum(int(np.prod(a.shape)) * a.dtype.itemsize
                            for a in self._base_arrays) / (1 << 30)
            print(f"[stream] tmpdir: {self._tmpdir}", flush=True)
            print(f"[stream] row-ring ~{mb_ring:.0f} MB; base memmap "
                  f"{gb_base:.1f} GB on disk; "
                  f"{self._n_strips} strips × {self.step_px} rows", flush=True)

    # -----------------------------------------------------------------
    # Tile submission
    # -----------------------------------------------------------------

    def submit_tile(self, idx: int, render: np.ndarray) -> None:
        """Stage a single rendered tile into the row-ring.

        ``render`` is the float32 (C, tile_px, tile_px) tile output by
        the renderer (worker side). ``idx`` is the linear tile index in
        row-major order over the tile grid.
        """
        row = idx // self.n_cols
        col = idx % self.n_cols
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise IndexError(
                f"tile idx={idx} out of grid ({self.n_rows}×{self.n_cols})")

        # Lazily allocate the row's ring entry (memmap-backed).
        if row not in self._ring_value:
            self._alloc_ring_row(row)

        # Accumulate the (feather-weighted) tile into the row's band.
        x0 = col * self.step_px
        x1 = x0 + self.tile_px
        # Clamp x1 to W in the rare case the tile extends past the image
        # (right-edge tiles).
        x1_clamped = min(x1, self.W)
        w_use = x1_clamped - x0
        if w_use <= 0:
            return
        render_ch = render.astype(np.float32, copy=False)[:, :, :w_use]
        if self.feather is not None:
            feather = self.feather[:, :w_use]
            value_band = self._ring_value[row]
            weight_band = self._ring_weight[row]
            value_band[:, :, x0:x0 + w_use] += render_ch * feather[None, :, :]
            weight_band[:, x0:x0 + w_use] += feather
        else:
            self._ring_value[row][:, :, x0:x0 + w_use] = render_ch
            self._ring_weight[row][:, x0:x0 + w_use] = 1.0

        self._arrived[row] += 1
        if self._arrived[row] >= self.n_cols:
            self._row_done[row] = True
            self._try_finalize_around(row)

    def _alloc_ring_row(self, row: int) -> None:
        """Allocate one row of the ring in RAM."""
        self._ring_value[row] = np.zeros(
            (self.n_ch, self.tile_px, self.W), dtype=np.float32)
        self._ring_weight[row] = np.zeros(
            (self.tile_px, self.W), dtype=np.float32)

    def _drop_ring_row(self, row: int) -> None:
        """Release a ring entry once its strips have been emitted."""
        if row not in self._ring_value:
            return
        del self._ring_value[row]
        del self._ring_weight[row]

    # -----------------------------------------------------------------
    # Strip finalization
    # -----------------------------------------------------------------

    def _try_finalize_around(self, row: int) -> None:
        """When row R completes, check strips that might now be finalizable.

        Strip s covers y ∈ [s*step, s*step + step). Its dependencies are
        input rows s-1 (top overlap) and s (body + bottom-overlap-with-
        s+1 OR strip-tail). Edges:
          - s=0: only row 0 needed.
          - s=n_rows-1: only rows n_rows-2 and n_rows-1.
          - s=n_rows (the tail strip, present only if H_total %
            step_px != 0): only row n_rows-1 needed.
        """
        candidates = (row - 1, row, row + 1)
        for s in candidates:
            if s < 0 or s >= self._n_strips:
                continue
            if self._strip_emitted[s]:
                continue
            if self._strip_ready(s):
                self._emit_strip(s)

    def _strip_ready(self, s: int) -> bool:
        if s == 0:
            return bool(self._row_done[0])
        if s == self.n_rows:
            # Tail strip below the last input row (height < step_px).
            return bool(self._row_done[self.n_rows - 1])
        # Middle strip: needs input row s-1 (top overlap) and row s.
        return bool(self._row_done[s - 1] and self._row_done[s])

    def _emit_strip(self, s: int) -> None:
        """Compute pixels for strip ``s`` and write to base memmap."""
        if s == self.n_rows:
            # Tail strip: only one contributing row (n_rows-1), and we
            # need its bottom `tail_rows` pixels.
            row = self.n_rows - 1
            y_global0 = self.n_rows * self.step_px
            y_global1 = self.H
            y_local0 = self.step_px       # bottom-tail in the ring
            y_local1 = y_local0 + (y_global1 - y_global0)
            value = self._ring_value[row][:, y_local0:y_local1, :].copy()
            weight = self._ring_weight[row][y_local0:y_local1, :].copy()
        else:
            # Strip s spans y ∈ [s*step, s*step + step). Its pixels come
            # from rows s (whole step) plus the top overlap of row s
            # (first `overlap_px` rows of strip) which is also covered
            # by row s-1's tile (its bottom `overlap_px` rows).
            value_local = self._ring_value[s][:, :self.step_px, :].copy()
            weight_local = self._ring_weight[s][:self.step_px, :].copy()
            if s > 0:
                # Top overlap_px rows of strip s receive contributions
                # from row s-1's tile-rows [step_px:step_px + overlap_px].
                prev_v = self._ring_value[s - 1]
                prev_w = self._ring_weight[s - 1]
                top = self.overlap_px
                value_local[:, :top, :] += \
                    prev_v[:, self.step_px:self.step_px + top, :]
                weight_local[:top, :] += \
                    prev_w[self.step_px:self.step_px + top, :]
            value = value_local
            weight = weight_local

        # Feather-divide → float32 row band
        np.maximum(weight, 1e-6, out=weight)
        for c in range(self.n_ch):
            np.divide(value[c], weight, out=value[c])

        # Calibrate to uint16 per channel using the supplied LUT mode.
        # We hand `calibrate_to_uint16` the full per-channel band; it
        # handles per-channel scaling + noise injection.
        u16_band = calibrate_to_uint16(
            value,
            channel_names=self.channel_names,
            target_stats=self.target_intensity_stats,
            target_quantiles=self.target_intensity_quantiles,
            mode=self.intensity_mode,
            display_lut=self.display_lut,
        )

        # Write to base memmap at the right y offset.
        if s == self.n_rows:
            y0 = self.n_rows * self.step_px
        else:
            y0 = s * self.step_px
        y1 = y0 + u16_band.shape[1]
        for c in range(self.n_ch):
            self._base_arrays[c][y0:y1, :] = u16_band[c, :, :u16_band.shape[2]]

        self._strip_emitted[s] = True
        if self.progress and (s % 20 == 0 or s == self._n_strips - 1):
            print(f"[stream] strip {s + 1}/{self._n_strips} emitted "
                  f"(y={y0}..{y1})", flush=True)

        # Drop any ring rows that no future strip will need.
        # Row R is needed by strip R (always) and strip R+1 (top
        # overlap). So we can drop row R once strips R and R+1 are
        # both done. Strip n_rows-1's dependency on row n_rows-2 + row
        # n_rows-1 means we drop row R when self._strip_emitted[R]
        # AND self._strip_emitted[R+1] (the next strip — be careful of
        # the tail-strip index n_rows).
        for r in range(self.n_rows):
            if r not in self._ring_value:
                continue
            next_strip = r + 1
            if next_strip > self._n_strips - 1:
                # No more strips can need this row.
                if self._strip_emitted[r] if r < len(self._strip_emitted) else True:
                    self._drop_ring_row(r)
            elif (self._strip_emitted[r] if r < len(self._strip_emitted) else True) \
                 and self._strip_emitted[next_strip]:
                self._drop_ring_row(r)

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------

    def close(self) -> dict:
        """Flush remaining strips, build the pyramid, write OME-TIFFs.

        Returns a manifest with the written file paths + metadata.
        """
        # Sanity: all rows must have been submitted.
        if not bool(self._row_done.all()):
            missing = np.where(~self._row_done)[0].tolist()
            raise RuntimeError(
                f"close() called before all rows arrived; missing rows: "
                f"{missing[:5]} (total {len(missing)}); arrived counts: "
                f"{self._arrived[missing[:5]].tolist()}")
        # Force any unemitted strips through (shouldn't happen if the
        # finalize-on-row-done logic is correct, but be defensive).
        for s in range(self._n_strips):
            if not self._strip_emitted[s] and self._strip_ready(s):
                self._emit_strip(s)
        if not bool(self._strip_emitted.all()):
            missing = np.where(~self._strip_emitted)[0].tolist()
            raise RuntimeError(
                f"close() missed strips: {missing}")

        # Flush per-channel base memmaps so the writer reads see them.
        for a in self._base_arrays:
            a.flush()

        # Build pyramid + write OME-TIFFs. Each per-channel
        # morphology_focus_NNNN gets its base + 7 sub-IFD pyramid levels.
        if self.progress:
            print(f"[stream] building pyramid + writing OME-TIFFs…", flush=True)
        written = {"focus_files": []}
        t0 = time.time()
        for fi in range(self.n_focus_files):
            ch_idx = min(fi, self.n_ch - 1)
            ch_name = (self.channel_names[ch_idx]
                       if ch_idx < len(self.channel_names) else f"channel_{ch_idx}")
            path = self.out_dir / "morphology_focus" / f"morphology_focus_{fi:04d}.ome.tif"
            self._write_focus_file(path, ch_idx, ch_name)
            written["focus_files"].append(str(path))
            if self.progress:
                print(f"  morphology_focus_{fi:04d}.ome.tif done "
                      f"({time.time()-t0:.1f}s)", flush=True)

        # morphology.ome.tif: a 1-z DAPI z-stack pyramid. Match
        # bundle_writer._write_morphology_z's contract.
        dapi_idx = 0   # convention: DAPI is channel 0
        path = self.out_dir / "morphology.ome.tif"
        self._write_morphology_z_file(path, dapi_idx)
        written["z_stack"] = str(path)
        if self.progress:
            print(f"  morphology.ome.tif done "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # Rewrite OME-XML of morphology_focus_NNNN files to cross-
        # reference siblings (matches real Xenium convention).
        from .bundle_writer import _rewrite_morphology_focus_ome_xml
        _rewrite_morphology_focus_ome_xml(
            self.out_dir / "morphology_focus",
            self.channel_names[:self.n_focus_files],
            self.pixel_size_um)

        return written

    def cleanup(self) -> None:
        """Release temp files. Call after the bundle directory is done
        being written. Safe to call multiple times."""
        try:
            for a in self._base_arrays:
                try: a.flush()
                except Exception: pass
            self._base_arrays.clear()
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Per-file pyramid build + write
    # -----------------------------------------------------------------

    def _write_focus_file(self, out_path: Path, ch_idx: int, ch_name: str) -> None:
        """Write one morphology_focus_NNNN.ome.tif (single channel + 8-level
        pyramid)."""
        import tifffile

        base = self._base_arrays[ch_idx]
        levels = self._build_pyramid_levels_for_channel(ch_idx)

        metadata = {
            "axes": "YX",
            "PhysicalSizeX": self.pixel_size_um, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": self.pixel_size_um, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": [ch_name]},
        }
        bytes_total = sum(int(np.prod(lvl.shape)) * lvl.dtype.itemsize
                          for lvl in levels)
        use_bigtiff = bytes_total > 3 * (1 << 30)
        n_sub = max(0, len(levels) - 1)

        if out_path.exists():
            out_path.unlink()
        with tifffile.TiffWriter(out_path, ome=True, bigtiff=use_bigtiff) as tw:
            tw.write(np.asarray(levels[0]), photometric="minisblack",
                     metadata=metadata, subifds=n_sub,
                     compression=self.compression)
            for sub in levels[1:]:
                tw.write(np.asarray(sub), photometric="minisblack",
                         subfiletype=1, compression=self.compression)

    def _write_morphology_z_file(self, out_path: Path, dapi_idx: int) -> None:
        """Write morphology.ome.tif (n_z-broadcast DAPI z-stack pyramid).

        Real Xenium morphology.ome.tif has 12 z-slices; the 2D writer
        broadcasts the single focal plane to all n_z slices so Xenium
        Explorer's z-slider works. We write each z-plane via a generator
        so the full ``(n_z, H, W)`` array is NEVER materialised — on full
        breast that array would be ~93 GB and OOM. Per-plane streaming
        keeps RAM at one (H, W) plane (~7.7 GB on breast).
        """
        import tifffile
        n_z = 12

        levels = self._build_pyramid_levels_for_channel(dapi_idx)
        metadata = {
            "axes": "ZYX",
            "PhysicalSizeX": self.pixel_size_um, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": self.pixel_size_um, "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 0.75, "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": ["DAPI"]},
        }
        # bytes_total counts ONE copy per level × n_z (the on-disk size,
        # pre-compression) to decide BigTIFF.
        bytes_total = n_z * sum(int(np.prod(lvl.shape)) * lvl.dtype.itemsize
                                for lvl in levels)
        use_bigtiff = bytes_total > 3 * (1 << 30)
        n_sub = max(0, len(levels) - 1)

        def _z_planes(level: np.ndarray):
            """Yield the same (H, W) plane n_z times — lazy z-broadcast."""
            plane = np.asarray(level)
            for _ in range(n_z):
                yield plane

        if out_path.exists():
            out_path.unlink()
        with tifffile.TiffWriter(out_path, ome=True, bigtiff=use_bigtiff) as tw:
            tw.write(_z_planes(levels[0]),
                     shape=(n_z, *levels[0].shape), dtype=levels[0].dtype,
                     photometric="minisblack", metadata=metadata,
                     subifds=n_sub, compression=self.compression)
            for sub in levels[1:]:
                tw.write(_z_planes(sub),
                         shape=(n_z, *sub.shape), dtype=sub.dtype,
                         photometric="minisblack", subfiletype=1,
                         compression=self.compression)

    def _build_pyramid_levels_for_channel(
        self, ch_idx: int,
    ) -> list[np.ndarray]:
        """Return [base, level1, …] for one channel.

        Sequential 2× block-average downsample, materialised one level
        at a time as a uint16 array. The base is the per-channel memmap;
        subsequent levels are computed in memory (each ¼ of the prior).
        """
        base = self._base_arrays[ch_idx]
        levels: list[np.ndarray] = [base]
        cur = base
        for _ in range(1, self.n_pyramid_levels):
            cur = _downsample_2x_2d(cur)
            if cur is None:
                break
            levels.append(cur)
        return levels


def _downsample_2x_2d(arr: np.ndarray) -> np.ndarray | None:
    """2×2 block-average downsample for a 2D uint16 array. Returns None
    if the input would shrink below 16 px on any axis (mirrors
    bundle_writer._downsample_2x's stopping rule)."""
    h, w = arr.shape
    if h < 16 or w < 16:
        return None
    h2, w2 = h // 2, w // 2
    return arr[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).astype(
        np.float32).mean(axis=(1, 3)).astype(arr.dtype)
