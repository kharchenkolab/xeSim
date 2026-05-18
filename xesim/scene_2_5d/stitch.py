"""Tile-and-stitch for 2.5D scenes.

`build_scene_25d` tiles a region into overlapping chunks, runs
`compose_region_scene_25d` per tile, and stitches multi-z DAPI +
transcripts + cells_3d into a single bundle. Memory bounded by the
per-tile cost (~1 GB at 500 µm tiles) regardless of total region size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# Module-level globals for multiprocessing workers (set by _worker_init).
# In scene-first mode, the workers receive a SharedScene25D once and reuse it.
_WORKER_MODEL = None
_WORKER_SHARED = None


def _worker_init(model_path: str, device: str, shared,
                  compile_renderer: bool = False):
    """Initializer for each worker process. Loads model once + sets shared.

    If `compile_renderer=True`, wrap the renderer module with
    `torch.compile(mode='reduce-overhead')` after load. First few forward
    passes pay a JIT cost; later ones are faster on Conv2d-heavy graphs.
    """
    global _WORKER_MODEL, _WORKER_SHARED
    from ..model import XesimModel
    _WORKER_MODEL = XesimModel.load(model_path, device=device)
    _WORKER_SHARED = shared
    if compile_renderer:
        try:
            import torch
            # Trigger renderer load
            _ = _WORKER_MODEL.v37b
            _WORKER_MODEL._v37b = torch.compile(
                _WORKER_MODEL._v37b, mode="reduce-overhead", dynamic=True
            )
        except Exception as e:
            print(f"[worker] torch.compile failed (continuing without): {e}", flush=True)


def _worker_render_tile(arg):
    """Worker: render one tile from shared state. Returns slim tuple."""
    k, bbox, seed = arg
    from .scene_first import render_tile_from_shared
    import numpy as _np
    global _WORKER_MODEL, _WORKER_SHARED
    res = render_tile_from_shared(
        _WORKER_MODEL, _WORKER_SHARED, tile_bounds_um=bbox,
        rng=_np.random.default_rng(int(seed)),
        progress=False, rescale_dapi=False,
    )
    return (k, bbox,
            res.dapi_zstack.astype(_np.float32, copy=False),
            res.focal_2d_render.astype(_np.float32, copy=False),
            res.molecules, res.cells_3d)


@dataclass
class StitchResult:
    """Output of build_scene_25d."""
    dapi_zstack_path: str | None        # path to mmap'd DAPI z-stack
    focal_2d_render: np.ndarray         # (C, H_full, W_full) float32, non-DAPI channels stitched
    molecules: pd.DataFrame             # combined across tiles, deduped
    cells_3d: pd.DataFrame              # combined across tiles, deduped
    region_bounds_um: tuple[float, float, float, float]
    pixel_size_um: float
    z_slices_um: np.ndarray
    H_full: int
    W_full: int

    def to_compose_result(self):
        """Load the mmap DAPI z-stack into a `Compose25DResult` so the
        existing `write_bundle_25d` writer accepts it as-is.

        For whole-bundle scale (~500 MB at 7.2×2.9 mm × 12 z), this
        fits comfortably in RAM on the build host.
        """
        from .compose import Compose25DResult
        n_z = len(self.z_slices_um)
        dapi_mmap = np.memmap(self.dapi_zstack_path, dtype=np.uint16, mode='r',
                                shape=(n_z, self.H_full, self.W_full))
        # Materialize as float32 in [0, 1] so the writer's [0, 4095] uint16
        # scaler works the same as for in-memory compose results.
        dapi_zstack = (np.asarray(dapi_mmap, dtype=np.float32) / 4095.0)
        # cell_label_3d is not propagated through stitch (per-tile only);
        # the writer doesn't actually consume it.
        return Compose25DResult(
            cell_label_3d=np.zeros((1, 1, 1), dtype=np.int32),
            z_slices_um=self.z_slices_um,
            dapi_zstack=dapi_zstack,
            focal_2d_render=self.focal_2d_render,
            molecules=self.molecules,
            cells_3d=self.cells_3d,
            region_bounds_um=self.region_bounds_um,
            pixel_size_um=self.pixel_size_um,
        )


def build_scene_25d(
    model,
    bundle_path: str | Path,
    *,
    scene_bounds_um: tuple[float, float, float, float],
    tile_um: float = 500.0,
    overlap_um: float = 50.0,
    z_step_um: float = 3.0,
    imaged_depth_um: float = 33.0,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    num_workers: int = 1,
    tmp_dir: str | Path | None = None,
    model_path: str | None = None,   # required when num_workers > 1
    device: str = "cuda",
    compile_renderer: bool = False,
    model_dir: str | Path | None = None,  # for nucleus-priors auto-discover
) -> StitchResult:
    """Build a 2.5D scene over `scene_bounds_um` by tiling + stitching.

    Uses scene-first 2.5D: one bundle-wide compute pass (cell placement,
    tilt MRF, encoder anchors) then per-tile rendering. Workers
    receive the shared state once via Pool initargs.
    """
    from .scene_first import precompute_scene_25d, render_tile_from_shared

    rng = rng or np.random.default_rng(0)
    from ..xenium import resolve_bundle
    bundle = resolve_bundle(str(bundle_path))
    psz = float(bundle.pixel_size)
    xmin, ymin, xmax, ymax = scene_bounds_um
    W_um = xmax - xmin; H_um = ymax - ymin
    W_full = int(round(W_um / psz))
    H_full = int(round(H_um / psz))
    z_slices = np.arange(0.0, imaged_depth_um + 1e-3, z_step_um)
    n_z = len(z_slices)
    n_ch = int(model.manifest.get("n_channels", 3))

    # Build tile grid (with overlap)
    step = tile_um - overlap_um
    xs = list(np.arange(xmin, xmax, step))
    ys = list(np.arange(ymin, ymax, step))
    grid = [(x, y) for y in ys for x in xs]
    if progress:
        print(f"[build_scene_25d] bounds {xmin:.0f}-{xmax:.0f}, {ymin:.0f}-{ymax:.0f} µm "
              f"-> {len(grid)} tiles ({tile_um}µm, overlap {overlap_um}µm)")
        print(f"  full output: ({n_z}, {H_full}, {W_full})")

    # Memory-mapped DAPI accumulator (float32 for feather-blend accumulation;
    # converted to uint16 after global p99 rescale).
    tmp_dir = Path(tmp_dir) if tmp_dir else Path("/tmp/xesim_25d_stitch")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dapi_accum_path = tmp_dir / "dapi_accum.dat"
    dapi_weight_path = tmp_dir / "dapi_weight.dat"
    dapi_path = tmp_dir / "dapi_stitch.dat"      # final uint16 output
    dapi_accum = np.memmap(str(dapi_accum_path), dtype=np.float32, mode='w+',
                              shape=(n_z, H_full, W_full))
    dapi_weight = np.memmap(str(dapi_weight_path), dtype=np.float32, mode='w+',
                                shape=(H_full, W_full))
    # Focal 2D render stitched on RAM (smaller — 4 ch × full image float32)
    focal_render = np.zeros((n_ch, H_full, W_full), dtype=np.float32)
    focal_weight = np.zeros((H_full, W_full), dtype=np.float32)
    tile_px = int(round(tile_um / psz))
    overlap_px = int(round(overlap_um / psz))

    def _feather_for_tile(tx0, ty0, tx1, ty1, h_eff, w_eff):
        """Build feather weight for this tile, with tapering DISABLED on
        sides that touch the region boundary (no neighbor to blend
        with → don't ramp down)."""
        w_y = np.ones(h_eff, dtype=np.float32)
        w_x = np.ones(w_eff, dtype=np.float32)
        if overlap_px > 0:
            # Top edge tapers down only if there IS a tile above
            if ty0 > ymin + 0.01:
                ramp = np.linspace(0.05, 1, min(overlap_px, h_eff), dtype=np.float32)
                w_y[:len(ramp)] = ramp
            # Bottom edge
            if ty1 < ymax - 0.01:
                ramp = np.linspace(1, 0.05, min(overlap_px, h_eff), dtype=np.float32)
                w_y[-len(ramp):] = ramp
            # Left edge
            if tx0 > xmin + 0.01:
                ramp = np.linspace(0.05, 1, min(overlap_px, w_eff), dtype=np.float32)
                w_x[:len(ramp)] = ramp
            # Right edge
            if tx1 < xmax - 0.01:
                ramp = np.linspace(1, 0.05, min(overlap_px, w_eff), dtype=np.float32)
                w_x[-len(ramp):] = ramp
        return np.outer(w_y, w_x)

    all_molecules: list[pd.DataFrame] = []
    all_cells_3d: list[pd.DataFrame] = []
    seen_cell_ids: set[str] = set()

    def _run_tile(k_xy):
        k, (tx0, ty0) = k_xy
        tx1 = min(tx0 + tile_um, xmax)
        ty1 = min(ty0 + tile_um, ymax)
        if tx1 - tx0 < 5 or ty1 - ty0 < 5:
            return None
        res = compose_region_scene_25d(
            model, bundle_path, region_bounds_um=(tx0, ty0, tx1, ty1),
            z_step_um=z_step_um, imaged_depth_um=imaged_depth_um,
            rng=np.random.default_rng(int(rng.integers(0, 2**31-1))),
            progress=False,
            rescale_dapi=False,    # disable per-tile rescale; we'll do global after stitch
            model_dir=model_dir,
        )
        return (k, (tx0, ty0, tx1, ty1),
                res.dapi_zstack.astype(np.float32, copy=False),
                res.focal_2d_render.astype(np.float32, copy=False),
                res.molecules, res.cells_3d)

    def _consume(k, bbox, dapi_zstack, focal_tile_full, mol, c3d):
        nonlocal all_molecules, all_cells_3d, seen_cell_ids
        tx0, ty0, tx1, ty1 = bbox
        x0_px = int(round((tx0 - xmin) / psz))
        y0_px = int(round((ty0 - ymin) / psz))
        h_tile, w_tile = dapi_zstack.shape[1:]
        x1_px = min(W_full, x0_px + w_tile)
        y1_px = min(H_full, y0_px + h_tile)
        h_eff = y1_px - y0_px; w_eff = x1_px - x0_px
        if h_eff <= 0 or w_eff <= 0:
            return

        wt = _feather_for_tile(tx0, ty0, tx1, ty1, h_eff, w_eff)

        dapi_tile = dapi_zstack[:, :h_eff, :w_eff]
        cur_accum = np.asarray(dapi_accum[:, y0_px:y1_px, x0_px:x1_px])
        cur_accum += dapi_tile * wt[None, :, :]
        dapi_accum[:, y0_px:y1_px, x0_px:x1_px] = cur_accum
        cur_w = np.asarray(dapi_weight[y0_px:y1_px, x0_px:x1_px])
        cur_w += wt
        dapi_weight[y0_px:y1_px, x0_px:x1_px] = cur_w

        focal_tile = focal_tile_full[:, :h_eff, :w_eff]
        focal_render[:, y0_px:y1_px, x0_px:x1_px] += focal_tile * wt[None, :, :]
        focal_weight[y0_px:y1_px, x0_px:x1_px] += wt

        exc_x0 = tx0 + (overlap_um / 2.0 if tx0 > xmin else 0.0)
        exc_y0 = ty0 + (overlap_um / 2.0 if ty0 > ymin else 0.0)
        exc_x1 = tx1 - (overlap_um / 2.0 if tx1 < xmax else 0.0)
        exc_y1 = ty1 - (overlap_um / 2.0 if ty1 < ymax else 0.0)
        if len(mol) > 0 and 'x_true' in mol.columns:
            in_excl = ((mol['x_true'] >= exc_x0) & (mol['x_true'] < exc_x1) &
                        (mol['y_true'] >= exc_y0) & (mol['y_true'] < exc_y1))
            all_molecules.append(mol[in_excl].copy())

        if len(c3d) > 0 and 'centroid_x' in c3d.columns:
            in_excl_c = ((c3d['centroid_x'] >= exc_x0) & (c3d['centroid_x'] < exc_x1) &
                          (c3d['centroid_y'] >= exc_y0) & (c3d['centroid_y'] < exc_y1))
            c_owned = c3d[in_excl_c].copy()
            c_owned = c_owned[~c_owned['cell_id'].isin(seen_cell_ids)]
            seen_cell_ids.update(c_owned['cell_id'].astype(str).tolist())
            all_cells_3d.append(c_owned)

    # Build the tile-arg list (each tile's bbox is fully resolved up front).
    tile_args = []
    for k, (tx0, ty0) in enumerate(grid):
        tx1 = min(tx0 + tile_um, xmax)
        ty1 = min(ty0 + tile_um, ymax)
        if tx1 - tx0 < 5 or ty1 - ty0 < 5:
            continue
        seed = int(rng.integers(0, 2**31 - 1))
        tile_args.append((k, (tx0, ty0, tx1, ty1), seed))

    # Bundle-wide compose: cells, tilts, MRF, nucleus polygons — runs ONCE.
    print(f"[build_scene_25d] scene-first precompute over the whole region", flush=True)
    shared = precompute_scene_25d(
        model, bundle_path,
        scene_bounds_um=scene_bounds_um,
        z_step_um=z_step_um, imaged_depth_um=imaged_depth_um,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        progress=progress,
        model_dir=model_dir,
    )

    if num_workers <= 1:
        for idx, ta in enumerate(tile_args):
            k_, bbox, seed = ta
            res = render_tile_from_shared(
                model, shared, tile_bounds_um=bbox,
                rng=np.random.default_rng(int(seed)),
                progress=False, rescale_dapi=False,
            )
            _consume(k_, bbox,
                       res.dapi_zstack.astype(np.float32, copy=False),
                       res.focal_2d_render.astype(np.float32, copy=False),
                       res.molecules, res.cells_3d)
            del res
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if progress:
                print(f"  tile {idx+1}/{len(tile_args)}", flush=True)
    else:
        # Multiprocessing.Pool with N workers, each holding its own model
        # on GPU. 4 CUDA contexts time-share the SMs and parallelize at the
        # CUDA-context level. (Single-process multi-stream was tried —
        # measured slightly slower at small scale and underutilized the GPU
        # because each kernel is too small to saturate the SMs even with
        # 4 concurrent streams. CUDA-context multiplexing wins for our
        # lightweight UNet renderer.)
        if model_path is None:
            raise ValueError("num_workers>1 requires model_path so each worker "
                              "can load the model on GPU")
        print(f"[build_scene_25d] tile parallel: {num_workers} workers", flush=True)
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        done = 0
        with ctx.Pool(processes=num_workers,
                        initializer=_worker_init,
                        initargs=(str(model_path), device, shared,
                                   bool(compile_renderer))) as pool:
            for result in pool.imap_unordered(_worker_render_tile, tile_args):
                k_, bbox, dapi_zstack, focal_tile, mol, c3d = result
                _consume(k_, bbox, dapi_zstack, focal_tile, mol, c3d)
                done += 1
                if progress:
                    print(f"  tile {done}/{len(tile_args)}", flush=True)

    # Normalize focal_render by weight
    focal_render = focal_render / np.maximum(focal_weight[None, :, :], 1e-6)

    # Normalize stitched DAPI: divide by feather weight, global p99 rescale,
    # cast to uint16 mmap output.
    if progress: print("[build_scene_25d] finalizing DAPI: weight-normalize, global p99 rescale, uint16 cast")
    dapi_mmap = np.memmap(str(dapi_path), dtype=np.uint16, mode='w+',
                            shape=(n_z, H_full, W_full))
    # Find global p99 of nucleus-region pixels for rescaling
    accum_flat = np.asarray(dapi_accum)
    weight_flat = np.asarray(dapi_weight)
    nz_pixels = []
    for zi in range(n_z):
        v = accum_flat[zi] / np.maximum(weight_flat, 1e-6)
        nz_p = v[v > 0]
        if nz_p.size > 0:
            nz_pixels.append(nz_p)
    target_max_u16 = 4095
    if nz_pixels:
        all_nz = np.concatenate(nz_pixels)
        global_p99 = float(np.percentile(all_nz, 99))
    else:
        global_p99 = 1.0
    if progress:
        print(f"  global p99 = {global_p99:.4f} (rescale factor: {target_max_u16 / max(global_p99, 1e-6):.1f})")
    scale = target_max_u16 / max(global_p99, 1e-6)
    for zi in range(n_z):
        v = accum_flat[zi] / np.maximum(weight_flat, 1e-6)
        dapi_mmap[zi] = np.clip(v * scale, 0, 65535).astype(np.uint16)

    # Combine molecules / cells
    molecules = pd.concat(all_molecules, ignore_index=True) if all_molecules else pd.DataFrame()
    cells_3d = pd.concat(all_cells_3d, ignore_index=True) if all_cells_3d else pd.DataFrame()

    if progress:
        print(f"[build_scene_25d] done. {len(cells_3d)} cells, {len(molecules)} molecules")

    return StitchResult(
        dapi_zstack_path=str(dapi_path),
        focal_2d_render=focal_render,
        molecules=molecules, cells_3d=cells_3d,
        region_bounds_um=tuple(scene_bounds_um), pixel_size_um=psz,
        z_slices_um=z_slices, H_full=H_full, W_full=W_full,
    )


__all__ = ["StitchResult", "build_scene_25d"]
