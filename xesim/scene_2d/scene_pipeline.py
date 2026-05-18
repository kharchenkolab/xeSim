"""Multi-tile scene orchestrator: tile grid → render → stitch → ownership filter.

Phase 2.A workflow:

    scene_bounds_um (xmin, ymin, xmax, ymax)
        │
        ├─ tile_grid(...) → list of tile_center_um
        │
    for each tile:
        ├─ build_tile(...)  → Scene2D (contains all cells in tile + buffer)
        ├─ render_tile(...) → (C, tile_px, tile_px) uint16 patch
        ├─ filter_to_owned(...) → Scene2D (only cells whose centroid ∈ tile)
        ├─ place patch into stitched_image buffer
        │
    returns (filtered_scenes, stitched_image, ghost_id_offset)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..mechanistic_scene import MechanisticScene
from . import Scene2D
from .intensity import calibrate_to_uint16
from .render_tile import render_tile           # noqa: F401 (for scripts that import via this module)
from .explain_region import explain_region


# ---------------------------------------------------------------------------
# Multiprocessing workers — spawn context, each holds its own model on GPU.
# Used by build_scene's parallel branch.
# ---------------------------------------------------------------------------
_WORKER_MODEL_2D = None
_WORKER_BUNDLE_PATH_2D: str | None = None
_WORKER_KWARGS_2D: dict = {}


def _worker_init_2d(model_path: str, device: str, bundle_path: str, kwargs: dict):
    """Initializer for each build_scene worker process. Loads model once;
    subsequent tile calls reuse it. Sets module globals consumed by
    `_worker_render_tile_2d`."""
    global _WORKER_MODEL_2D, _WORKER_BUNDLE_PATH_2D, _WORKER_KWARGS_2D
    from ..model import XesimModel
    _WORKER_MODEL_2D = XesimModel.load(model_path, device=device)
    _WORKER_BUNDLE_PATH_2D = bundle_path
    _WORKER_KWARGS_2D = kwargs


def _worker_render_tile_2d(arg):
    """Worker: render one tile via explain_region using the worker's model.

    `arg = (k, tile_bounds_xyxy, seed)`. Returns `(k, ExplainRegionResult)`.
    """
    k, tile_bounds_xyxy, seed = arg
    global _WORKER_MODEL_2D, _WORKER_BUNDLE_PATH_2D, _WORKER_KWARGS_2D
    res = explain_region(
        _WORKER_MODEL_2D, _WORKER_BUNDLE_PATH_2D,
        region_bounds_um=tile_bounds_xyxy,
        rng=np.random.default_rng(int(seed)),
        sample_molecules=True,
        background_mask_sigma=3.0,
        **_WORKER_KWARGS_2D,
    )
    return k, res


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------


@dataclass
class TileGridCell:
    """One cell of the non-overlapping tile grid."""

    grid_i: int                     # column index (x direction)
    grid_j: int                     # row index (y direction)
    center_x_um: float
    center_y_um: float
    tile_bounds_um: tuple[float, float, float, float]   # (xmin, xmax, ymin, ymax)


def tile_grid(
    scene_bounds_um: tuple[float, float, float, float],
    tile_size_um: float,
    overlap_um: float = 0.0,
) -> list[TileGridCell]:
    """Cover ``scene_bounds_um = (xmin, ymin, xmax, ymax)`` with a tile grid.

    Each tile spans ``tile_size_um`` and is spaced by
    ``tile_size_um - overlap_um``. With overlap > 0, adjacent tiles share an
    overlap zone used for feathering during stitch.

    Tiles are anchored to ``(xmin, ymin)``; the bottom and right edges may
    extend slightly past ``(xmax, ymax)`` if not divisible.
    """
    xmin, ymin, xmax, ymax = scene_bounds_um
    spacing = tile_size_um - overlap_um
    if spacing <= 0:
        raise ValueError("overlap_um must be < tile_size_um")
    # nx tiles cover (nx-1)*spacing + tile_size_um >= xmax-xmin
    nx = max(1, int(np.ceil(((xmax - xmin) - tile_size_um) / spacing)) + 1)
    ny = max(1, int(np.ceil(((ymax - ymin) - tile_size_um) / spacing)) + 1)
    cells: list[TileGridCell] = []
    for j in range(ny):
        for i in range(nx):
            tx_min = xmin + i * spacing
            ty_min = ymin + j * spacing
            tx_max = tx_min + tile_size_um
            ty_max = ty_min + tile_size_um
            cx = 0.5 * (tx_min + tx_max)
            cy = 0.5 * (ty_min + ty_max)
            cells.append(TileGridCell(
                grid_i=i, grid_j=j,
                center_x_um=cx, center_y_um=cy,
                tile_bounds_um=(tx_min, tx_max, ty_min, ty_max),
            ))
    return cells


# ---------------------------------------------------------------------------
# Cell ownership filtering
# ---------------------------------------------------------------------------


def _cell_centroid_um(
    cell, label_mask: np.ndarray, tile_bounds_um: tuple[float, float, float, float],
    pixel_size_um: float,
) -> tuple[float, float] | None:
    """Centroid of a MechanisticCell in scene-global µm (from its label mask)."""
    mask = (label_mask == cell.label)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    xmin, _, ymin, _ = tile_bounds_um
    cx = xmin + float(xs.mean()) * pixel_size_um
    cy = ymin + float(ys.mean()) * pixel_size_um
    return (cx, cy)


def filter_to_owned(
    scene: Scene2D,
    *,
    tile_bounds_um: tuple[float, float, float, float],
    keep_ghosts: bool = True,
    seen_anchor_ids: set[str] | None = None,
) -> Scene2D:
    """Return a new Scene2D containing only cells owned by this tile.

    Ownership rule for ANCHOR cells:
      - centroid (computed from the rasterized mask in this tile) falls
        inside ``tile_bounds_um``, AND
      - the cell_id has not already been claimed by a previous tile
        (``seen_anchor_ids`` is updated in-place).

    Ownership rule for GHOST cells: always owned by their home tile (they
    are sampled inside the tile by construction).

    The rasterized ``cell_label`` and ``nucleus_label`` masks are kept
    unchanged so the rendered image is unaffected — only ``mech_scene.cells``
    and ``molecules`` are filtered.
    """
    xmin, xmax, ymin, ymax = tile_bounds_um
    mech = scene.mech_scene
    seen = seen_anchor_ids if seen_anchor_ids is not None else set()

    kept_cells = []
    owned_ids: set[str] = set()
    for cell in mech.cells:
        is_ghost = bool(cell.provenance.get("is_ghost", False))
        if is_ghost and keep_ghosts:
            kept_cells.append(cell)
            owned_ids.add(cell.cell_id)
            continue
        if cell.cell_id in seen:
            continue
        c = _cell_centroid_um(cell, mech.cell_label,
                                tile_bounds_um=scene.tile_bounds_um,
                                pixel_size_um=scene.pixel_size)
        if c is None:
            continue
        if (xmin <= c[0] < xmax) and (ymin <= c[1] < ymax):
            kept_cells.append(cell)
            owned_ids.add(cell.cell_id)
            seen.add(cell.cell_id)

    new_mech = MechanisticScene(
        image_shape=mech.image_shape, pixel_size=mech.pixel_size,
        cell_label=mech.cell_label, nucleus_label=mech.nucleus_label,
        cells=tuple(kept_cells), scene_id=mech.scene_id,
        provenance=dict(mech.provenance or {}),
    )

    mols = scene.molecules
    if len(mols) > 0:
        mols = mols[mols["true_cell_id"].isin(owned_ids)].reset_index(drop=True)

    prov = dict(scene.provenance or {})
    prov["owning_tile_bounds_um"] = list(tile_bounds_um)
    prov["n_owned_cells"] = len(kept_cells)

    return Scene2D(
        mech_scene=new_mech, molecules=mols,
        tile_bounds_um=scene.tile_bounds_um,
        pixel_size=scene.pixel_size,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Ghost id remapping
# ---------------------------------------------------------------------------


def _remap_ghost_ids(scene: Scene2D, id_offset: int) -> tuple[Scene2D, int]:
    """Shift this tile's ghost cell ids to a globally-unique range.

    Returns (new_scene, next_offset). Local ghost IDs ``ghost_NNNNNN`` are
    renamed to ``ghost_GGGGGGG`` where G = id_offset + N + 1.
    """
    mech = scene.mech_scene
    n_remapped = 0
    rename_map: dict[str, str] = {}

    new_cells = []
    for cell in mech.cells:
        is_ghost = bool(cell.provenance.get("is_ghost", False))
        if not is_ghost:
            new_cells.append(cell)
            continue
        new_id = f"ghost_{id_offset + n_remapped:07d}"
        rename_map[cell.cell_id] = new_id
        n_remapped += 1
        # MechanisticCell is frozen; rebuild
        from dataclasses import replace
        new_cells.append(replace(cell, cell_id=new_id))

    new_mech = MechanisticScene(
        image_shape=mech.image_shape, pixel_size=mech.pixel_size,
        cell_label=mech.cell_label, nucleus_label=mech.nucleus_label,
        cells=tuple(new_cells), scene_id=mech.scene_id,
        provenance=dict(mech.provenance or {}),
    )

    mols = scene.molecules
    if len(mols) > 0 and rename_map:
        mols = mols.copy()
        mols["true_cell_id"] = mols["true_cell_id"].map(
            lambda c: rename_map.get(c, c))

    new_scene = Scene2D(
        mech_scene=new_mech, molecules=mols,
        tile_bounds_um=scene.tile_bounds_um,
        pixel_size=scene.pixel_size,
        provenance=dict(scene.provenance or {}),
    )
    return new_scene, id_offset + n_remapped


# ---------------------------------------------------------------------------
# Image stitching helper
# ---------------------------------------------------------------------------


def _drop_ghosts_to_target(
    scenes: list[Scene2D], target_ghost_mols: float,
    *, rng: np.random.Generator,
) -> tuple[list[Scene2D], int]:
    """Drop random ghost cells across all scenes until total ghost mols ≤ target.

    Returns (updated_scenes, n_ghosts_dropped). Each scene is rebuilt with
    a filtered mech_scene.cells and molecules.
    """
    # Build a (scene_idx, ghost_cell_id, mol_count) list across all scenes
    candidates: list[tuple[int, str, int]] = []
    for si, sc in enumerate(scenes):
        if len(sc.molecules) == 0:
            continue
        gm = sc.molecules[sc.molecules["is_ghost"]]
        for cid, group in gm.groupby("true_cell_id"):
            candidates.append((si, str(cid), int(len(group))))

    if not candidates:
        return scenes, 0

    # Shuffle and drop until we're at or below target
    rng.shuffle(candidates)
    total_ghost_mols = sum(c[2] for c in candidates)
    dropped_per_scene: dict[int, set[str]] = {}
    dropped = 0
    while candidates and total_ghost_mols > target_ghost_mols:
        si, cid, n = candidates.pop()
        dropped_per_scene.setdefault(si, set()).add(cid)
        total_ghost_mols -= n
        dropped += 1

    # Rebuild scenes with dropped ghosts removed
    out: list[Scene2D] = []
    for si, sc in enumerate(scenes):
        drop_ids = dropped_per_scene.get(si, set())
        if not drop_ids:
            out.append(sc); continue
        new_cells = tuple(c for c in sc.mech_scene.cells
                           if c.cell_id not in drop_ids)
        new_mech = MechanisticScene(
            image_shape=sc.mech_scene.image_shape,
            pixel_size=sc.mech_scene.pixel_size,
            cell_label=sc.mech_scene.cell_label,
            nucleus_label=sc.mech_scene.nucleus_label,
            cells=new_cells, scene_id=sc.mech_scene.scene_id,
            provenance=dict(sc.mech_scene.provenance or {}),
        )
        new_mols = sc.molecules[~sc.molecules["true_cell_id"].isin(drop_ids)
                                  ].reset_index(drop=True)
        out.append(Scene2D(
            mech_scene=new_mech, molecules=new_mols,
            tile_bounds_um=sc.tile_bounds_um, pixel_size=sc.pixel_size,
            provenance=dict(sc.provenance or {}),
        ))
    return out, dropped


def _feather_mask(tile_px: int, overlap_px: int) -> np.ndarray:
    """Linear-ramp feather mask: 1 inside, ramping to 0 at each edge.

    Used during multi-tile image stitching to smoothly blend overlapping
    tile renders so the seam between tiles is invisible.
    """
    ramp = np.ones(tile_px, dtype=np.float32)
    for k in range(overlap_px):
        w = (k + 1) / (overlap_px + 1)
        ramp[k] = w
        ramp[tile_px - 1 - k] = w
    return ramp[:, None] * ramp[None, :]


def _global_scale_to_uint16(
    stitched_float: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    target_stats: dict[str, tuple[float, float]] | None = None,
    target_quantiles: dict[str, np.ndarray] | None = None,
    mode: str = "scale",
    p99: float = 99.5,
) -> np.ndarray:
    """Per-channel uint16 conversion over the entire stitched image.

    Percentiles are computed over the full multi-tile buffer so every tile
    shares the same scale factor → no per-tile brightness seam. Delegates to
    :func:`xesim.scene_2d.intensity.calibrate_to_uint16` for the actual
    mapping; ``mode`` controls which calibration scheme is used.
    """
    return calibrate_to_uint16(
        stitched_float, channel_names=channel_names,
        target_stats=target_stats, target_quantiles=target_quantiles,
        mode=mode, p99=p99,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_scene(
    model,
    bundle_path: str,
    *,
    scene_bounds_um: tuple[float, float, float, float],
    add_ghosts: bool = True,
    add_transcript_proposed: bool = False,
    noise_fraction: float = 0.23,
    ghost_count_scale: float = 1.0,
    annotation_path: str | None = None,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    align_to_tile_origin: bool = True,
    overlap_um: float = 6.0,
    target_intensity_stats: dict[str, tuple[float, float]] | None = None,
    target_intensity_quantiles: dict[str, np.ndarray] | None = None,
    intensity_mode: str = "scale",
    inference_tile_px: int | None = None,
    num_workers: int = 1,
    model_path: str | None = None,
    device: str = "cuda",
    emission_backend: str = "legacy",
    stpuppeteer_config: str | None = None,
) -> dict[str, Any]:
    """Build + render a multi-tile scene over ``scene_bounds_um``.

    Each tile is built with full anchor context (cells within ~25 µm of the
    tile bounds), rendered through the model, then FILTERED to cells whose
    centroid falls inside the tile's exclusive bounds. The result is a list
    of Scene2D objects with no cell duplicates across tiles, plus a single
    stitched ``(C, H, W)`` uint16 morphology image.

    Returns
    -------
    dict with:
        scenes : list[Scene2D]
        stitched_image : np.ndarray (C, H, W) float32 — raw renderer output,
            feather-stitched. Calibration to uint16 happens at the bundle-writer
            boundary, not here, so this matches the single-tile path's contract.
        scene_bounds_um : tuple
        pixel_size_um : float
        tile_size_um : float
        n_tiles : int
        n_anchor_cells : int  (across all owned tiles, deduped)
        n_ghost_cells : int
        n_transcripts : int
    """
    rng = rng or np.random.default_rng(0)
    pixel_size = float(model.pixel_size)
    # Default to 2× the training tile size (or 512, whichever is bigger).
    # The renderer is fully-conv with no norm layers, so larger inference
    # tiles produce visually-identical output (verified per-channel p50/p99
    # within 1% on a 600µm A/B vs the training-tile default) but amortize
    # per-tile Python/IPC overhead — measured +23% wall-time win on a
    # 600µm region. Task 9.AX (2026-05-17).
    if inference_tile_px is None:
        tile_px = max(int(model.tile_px) * 2, 512)
    else:
        tile_px = int(inference_tile_px)
    tile_size_um = float(tile_px) * pixel_size
    n_ch = int(model.manifest.get("n_channels", 3))
    if progress and tile_px != int(model.tile_px):
        src = "override" if inference_tile_px is not None else "auto-2x"
        print(f"[build_scene] inference_tile_px {src}: {tile_px} "
              f"(model trained at {int(model.tile_px)})")

    # Optionally align scene origin to a tile-grid multiple to simplify image math
    if align_to_tile_origin:
        xmin_raw, ymin_raw, xmax_raw, ymax_raw = scene_bounds_um
        sb = (xmin_raw, ymin_raw, xmax_raw, ymax_raw)
    else:
        sb = tuple(scene_bounds_um)
    # Snap tile spacing to an exact multiple of pixel_size so that the
    # tile-grid µm coords align with the stitched-buffer pixel grid. Without
    # this snap, the grid uses spacing_um = tile_size_um - overlap_um but the
    # stitcher pastes at i * round(spacing_um / pixel_size) pixels, drifting
    # tile_i by (step_px*psz - spacing_um) µm — at default 6 µm overlap on
    # 0.2125 µm/px, that's +0.05 µm/tile, ~3.5 µm across a 35k-px bundle —
    # visible as cell-polygon misalignment in saved morphology.
    spacing_um_raw = tile_size_um - overlap_um
    step_px = int(round(spacing_um_raw / pixel_size))
    spacing_um = step_px * pixel_size                # snap to pixel grid
    overlap_um_snapped = tile_size_um - spacing_um
    grid = tile_grid(sb, tile_size_um, overlap_um=overlap_um_snapped)

    nx = max(c.grid_i for c in grid) + 1
    ny = max(c.grid_j for c in grid) + 1
    overlap_px = tile_px - step_px
    H_total = (ny - 1) * step_px + tile_px
    W_total = (nx - 1) * step_px + tile_px

    if progress:
        print(f"[build_scene] grid: {nx} × {ny} = {len(grid)} tiles "
                f"(overlap={overlap_um:.1f} µm = {overlap_px} px), "
                f"image {W_total} × {H_total} px = "
                f"{(W_total*pixel_size):.1f} × {(H_total*pixel_size):.1f} µm")

    # Feather-weighted accumulation: value buffer + weight buffer.
    # At the end, stitched = accum_value / accum_weight.
    feather = _feather_mask(tile_px, overlap_px) if overlap_px > 0 else None
    accum_value = np.zeros((n_ch, H_total, W_total), dtype=np.float32)
    accum_weight = np.zeros((H_total, W_total), dtype=np.float32)
    scenes_out: list[Scene2D] = []
    seen_anchor_ids: set[str] = set()    # global anchor-ownership dedupe
    ghost_id_offset = 0
    total_anchors = 0
    total_ghosts = 0
    total_mols = 0
    total_tx_proposed = 0

    # Pre-derive tile seeds so the parallel and serial paths produce the
    # same RNG sequence per tile (only the master rng touches integers()).
    tile_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in grid]

    def _render_one_tile(idx: int):
        gc = grid[idx]
        tile_rng = np.random.default_rng(tile_seeds[idx])
        tile_bounds_xyxy = (gc.tile_bounds_um[0], gc.tile_bounds_um[2],
                              gc.tile_bounds_um[1], gc.tile_bounds_um[3])
        res = explain_region(
            model, bundle_path,
            region_bounds_um=tile_bounds_xyxy,
            annotation_path=annotation_path,
            add_ghosts=add_ghosts,
            add_transcript_proposed=add_transcript_proposed,
            noise_fraction=noise_fraction,
            ghost_count_scale=ghost_count_scale,
            sample_molecules=True,
            rng=tile_rng,
            background_mask_sigma=3.0,
            emission_backend=emission_backend,
            stpuppeteer_config=stpuppeteer_config,
        )
        return idx, res

    def _consume(idx: int, res) -> None:
        nonlocal ghost_id_offset, total_anchors, total_ghosts
        nonlocal total_tx_proposed, total_mols
        gc = grid[idx]
        sc = res.scene
        render = res.image.astype(np.float32)[:, :tile_px, :tile_px]
        y0 = gc.grid_j * step_px
        x0 = gc.grid_i * step_px
        if feather is not None:
            accum_value[:, y0:y0 + tile_px, x0:x0 + tile_px] += (
                render * feather[None, :, :])
            accum_weight[y0:y0 + tile_px, x0:x0 + tile_px] += feather
        else:
            accum_value[:, y0:y0 + tile_px, x0:x0 + tile_px] = render
            accum_weight[y0:y0 + tile_px, x0:x0 + tile_px] = 1.0
        owned = filter_to_owned(
            sc, tile_bounds_um=gc.tile_bounds_um,
            seen_anchor_ids=seen_anchor_ids,
        )
        owned, ghost_id_offset = _remap_ghost_ids(owned, ghost_id_offset)
        n_anchor_t = sum(1 for c in owned.mech_scene.cells
                            if c.source == "observed_anchor")
        n_ghost_t = sum(1 for c in owned.mech_scene.cells
                           if c.provenance.get("is_ghost", False))
        n_tx_t = sum(1 for c in owned.mech_scene.cells
                        if c.source == "tx_inferred")
        total_anchors += n_anchor_t
        total_ghosts += n_ghost_t
        total_tx_proposed += n_tx_t
        total_mols += int(len(owned.molecules))
        scenes_out.append(owned)

    import time as _time
    _t_start = _time.time()
    _milestone = max(1, len(grid) // 20)    # print every 5% (was 10%)

    def _print_progress(done: int) -> None:
        if not progress: return
        if done % _milestone != 0 and done != len(grid):
            return
        elapsed = _time.time() - _t_start
        rate = done / max(elapsed, 1e-6)
        remaining = max(0, len(grid) - done)
        eta = remaining / max(rate, 1e-6)
        pct = 100 * done / len(grid)
        def _fmt(s: float) -> str:
            if s < 60:   return f"{s:.0f}s"
            if s < 3600: return f"{int(s//60)}m{int(s%60):02d}s"
            return f"{int(s//3600)}h{int((s%3600)//60):02d}m"
        print(f"  tile {done}/{len(grid)} ({pct:5.1f}%)  "
              f"elapsed {_fmt(elapsed)}  ETA {_fmt(eta)}  "
              f"({rate:.1f} tiles/s)", flush=True)

    if num_workers <= 1:
        # Serial path (default)
        for k in range(len(grid)):
            idx, res = _render_one_tile(k)
            _consume(idx, res)
            _print_progress(k + 1)
    else:
        # Parallel path: multiprocessing.Pool with spawn — each worker has
        # its own model + CUDA context, parallel GPU usage. The previous
        # ThreadPoolExecutor approach was GIL-bound and stuck at ~2× even
        # with 4 workers (numpy/scipy hot paths in CPU prep). Multiprocess
        # gives true 3-4× speedup on render.
        # We consume results serially as they arrive so the shared stitch
        # buffer + seen_anchor_ids set don't need locking.
        if model_path is None:
            raise ValueError("num_workers>1 requires model_path so each "
                              "worker can load its own model on GPU")
        if progress:
            print(f"[build_scene] parallel multiprocess: num_workers={num_workers}", flush=True)
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        # Pre-build per-tile arg list (tile_bounds + seed + flags)
        tile_args = []
        for k, gc in enumerate(grid):
            tile_bounds_xyxy = (gc.tile_bounds_um[0], gc.tile_bounds_um[2],
                                 gc.tile_bounds_um[1], gc.tile_bounds_um[3])
            tile_args.append((k, tile_bounds_xyxy, tile_seeds[k]))
        done = 0
        worker_kwargs = {
            "annotation_path": annotation_path,
            "add_ghosts": add_ghosts,
            "add_transcript_proposed": add_transcript_proposed,
            "noise_fraction": noise_fraction,
            "ghost_count_scale": ghost_count_scale,
            "emission_backend": emission_backend,
            "stpuppeteer_config": stpuppeteer_config,
        }
        with ctx.Pool(processes=num_workers,
                        initializer=_worker_init_2d,
                        initargs=(str(model_path), device, bundle_path,
                                   worker_kwargs)) as pool:
            for result in pool.imap_unordered(_worker_render_tile_2d, tile_args):
                idx, res = result
                _consume(idx, res)
                done += 1
                _print_progress(done)

    if progress:
        tx_part = f" + {total_tx_proposed} tx-proposed" if total_tx_proposed > 0 else ""
        print(f"[build_scene] done. {total_anchors} anchor cells{tx_part} + "
                f"{total_ghosts} ghosts, {total_mols} transcripts")
        if total_mols > 0:
            ghost_mols = sum(int(s.molecules["is_ghost"].sum())
                                for s in scenes_out
                                if len(s.molecules) > 0)
            print(f"  empirical noise_fraction = {ghost_mols / total_mols:.4f}")

    # Post-calibration: drop random ghost cells until empirical noise_fraction
    # matches the target. The per-tile calibration over-estimates anchor mol
    # output (uses NB mean rather than empirical), so without this step the
    # noise fraction routinely lands 1.5× above target.
    if total_mols > 0 and noise_fraction > 0:
        ghost_mols = sum(int(s.molecules["is_ghost"].sum())
                            for s in scenes_out if len(s.molecules) > 0)
        anchor_mols = total_mols - ghost_mols
        emp_nf = ghost_mols / total_mols
        if emp_nf > noise_fraction + 0.01 and anchor_mols > 0:
            target_ghost_mols = noise_fraction * anchor_mols / (1 - noise_fraction)
            if progress:
                print(f"[build_scene] post-calibration: emp_nf={emp_nf:.3f} "
                        f"> target={noise_fraction:.3f}; dropping ghosts to hit "
                        f"target ghost_mols={int(target_ghost_mols)} "
                        f"(was {ghost_mols})")
            scenes_out, ghosts_dropped = _drop_ghosts_to_target(
                scenes_out, target_ghost_mols, rng=rng)
            total_ghosts -= ghosts_dropped
            total_mols = sum(int(len(s.molecules)) for s in scenes_out)

    # Feather-weighted normalization. We return the float32 stitched image
    # directly — calibration to uint16 (if requested) happens at the
    # bundle-writer boundary, the same as the single-tile `--tile` path.
    # This unifies all explain paths on a float-output contract and keeps
    # intermediate buffers free of the per-channel "off"-mode rescale
    # that destroyed inter-channel ratios.
    if progress:
        print(f"[build_scene] feather-normalize (float32 output)…")
    stitched = (accum_value / np.maximum(accum_weight[None, :, :], 1e-6)).astype(np.float32)

    full_bounds_um = (
        sb[0], sb[1],
        sb[0] + W_total * pixel_size,
        sb[1] + H_total * pixel_size,
    )

    return {
        "scenes": scenes_out,
        "stitched_image": stitched,
        "scene_bounds_um": full_bounds_um,
        "pixel_size_um": pixel_size,
        "tile_size_um": tile_size_um,
        "overlap_um": overlap_um,
        "grid_nx": nx, "grid_ny": ny,
        "n_tiles": len(grid),
        "n_anchor_cells": total_anchors,
        "n_ghost_cells": total_ghosts,
        "n_transcript_proposed": total_tx_proposed,
        "n_transcripts": total_mols,
    }


__all__ = ["build_scene", "tile_grid", "filter_to_owned", "TileGridCell"]
