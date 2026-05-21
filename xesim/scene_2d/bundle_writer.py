"""Write a Scene2D (or list of Scene2D) to a Xenium-compatible bundle directory.

Output layout (Phase 2.B):

    output_dir/
      experiment.xenium                          # JSON metadata
      morphology.ome.tif                         # DAPI z-stack (single-Z replicated)
      morphology_focus/
        morphology_focus_0000.ome.tif            # CYX, all channels
      transcripts.parquet                        # all molecules, true cell_id
      transcripts.csv.gz                         # same content
      cell_boundaries.parquet                    # anchors only
      cell_boundaries.csv.gz
      nucleus_boundaries.parquet                 # anchors only
      nucleus_boundaries.csv.gz
      gene_panel.json                            # copied from real bundle
      ground_truth/                              # ground-truth extras
        molecule_provenance.parquet              # transcript_id, true_cell_id, is_ghost, ...
        cells_synth.parquet                      # anchor + ghost metadata
        ghost_cell_boundaries.parquet            # ghost polygons (kept separate)
        ghost_nucleus_boundaries.parquet
        config.json                              # generation settings + auto-tuned values

Conventions:
- ``cell_id`` in ``transcripts.*`` is the TRUE source cell id (anchor's real id
  or ghost's synthetic id like "ghost_000123"). Ghost cells do NOT appear in
  ``cell_boundaries.*`` so a naive segmenter sees them as unattributable.
- Downstream segmenter evaluations MUST re-segment from scratch (using only
  the morphology image) and compare to the truth in ``transcripts.parquet``
  or ``ground_truth/molecule_provenance.parquet`` — do NOT trust the existing
  ``cell_id`` column when measuring segmenter accuracy.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import Scene2D
from .intensity import auto_tune_intensity_stats, calibrate_to_uint16


UNASSIGNED_LABEL = "UNASSIGNED"


# ---------------------------------------------------------------------------
# noise_fraction auto-tune
# ---------------------------------------------------------------------------


def auto_tune_noise_fraction(bundle_path: str | Path) -> float:
    """Compute the UNASSIGNED rate from a real Xenium bundle.

    Reads ``transcripts.parquet`` and returns the fraction of transcripts
    whose ``cell_id`` is the sentinel value "UNASSIGNED". This is the
    natural target for the synthetic ghost-cell molecule budget — total
    ghost molecules / total molecules should match this rate.
    """
    from ..xenium import resolve_bundle

    bundle = resolve_bundle(str(bundle_path))
    tp = Path(bundle.transcripts_path) if bundle.transcripts_path else None
    if tp is None or not tp.exists():
        # Try a few standard locations
        root = Path(bundle_path)
        for cand in [root / "transcripts.parquet",
                       root / "data" / "transcripts.parquet"]:
            if cand.exists():
                tp = cand; break
    if tp is None or not tp.exists():
        raise FileNotFoundError(
            f"transcripts.parquet not found in bundle {bundle_path}")

    df = pd.read_parquet(tp, columns=["cell_id"])
    n = len(df)
    if n == 0:
        return 0.0
    return float((df["cell_id"] == UNASSIGNED_LABEL).sum()) / n


# ---------------------------------------------------------------------------
# Transcripts assembly
# ---------------------------------------------------------------------------


def _build_transcripts_df(
    scenes: list[Scene2D],
    *,
    rng: np.random.Generator,
    nucleus_label_arrays: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a unified Xenium-style transcripts DataFrame.

    Columns mirror the real bundle:
        transcript_id (uint64), cell_id (str), overlaps_nucleus (uint8),
        feature_name (str), x_location, y_location, z_location (float32),
        qv (float32), fov_name (str), nucleus_distance (float32),
        codeword_index (int32)

    ``cell_id`` carries the TRUE source assignment (anchor or ghost id).
    """
    parts: list[pd.DataFrame] = []
    for ti, sc in enumerate(scenes):
        mols = sc.molecules
        if len(mols) == 0:
            continue

        # overlaps_nucleus: rasterize the molecule's pixel into the nucleus mask.
        # Note: mols["x"], mols["y"] are TILE-LOCAL µm (from sample_scene_transcripts).
        # For the Xenium-compatible output, x_location/y_location must be in
        # GLOBAL bundle µm coords — add the tile origin.
        xmin, _, ymin, _ = sc.tile_bounds_um
        x_local = mols["x"].to_numpy(dtype=np.float64)
        y_local = mols["y"].to_numpy(dtype=np.float64)
        x_global = (x_local + xmin).astype(np.float32)
        y_global = (y_local + ymin).astype(np.float32)
        # overlaps_nucleus: precomputed per-tile while the mask was alive
        # (mask-free streaming path), else computed here from the resident
        # nucleus mask (legacy path).
        if "overlaps_nucleus" in mols.columns:
            overlaps_nuc = mols["overlaps_nucleus"].to_numpy(dtype=np.uint8)
        else:
            px = x_local / sc.pixel_size
            py = y_local / sc.pixel_size
            nuc_lbl = sc.mech_scene.nucleus_label
            h, w = nuc_lbl.shape
            px_i = np.clip(np.round(px).astype(np.int64), 0, w - 1)
            py_i = np.clip(np.round(py).astype(np.int64), 0, h - 1)
            overlaps_nuc = (nuc_lbl[py_i, px_i] > 0).astype(np.uint8)

        n_mol = len(mols)
        z_loc = (mols["z"].to_numpy(dtype=np.float32)
                   if "z" in mols.columns else np.zeros(n_mol, dtype=np.float32))
        qv = (mols["qv"].to_numpy(dtype=np.float32)
                if "qv" in mols.columns else np.full(n_mol, 30.0, dtype=np.float32))

        # Ghost-derived molecules represent uncertain / out-of-plane signal
        # — they don't correspond to a confirmed cell in the morphology
        # image, so they shouldn't claim a synthesized cell_id either.
        # Tag them as UNASSIGNED to match real-Xenium orphan semantics.
        # The full ghost provenance (which synthesized ghost emitted which
        # molecule) is preserved in ground_truth/molecule_provenance.parquet.
        cell_ids = mols["true_cell_id"].astype(str).to_numpy().copy()
        is_ghost_mol = mols.get("is_ghost",
                                  pd.Series([False] * len(mols))).to_numpy(dtype=bool)
        cell_ids[is_ghost_mol] = "UNASSIGNED"

        df_out = pd.DataFrame({
            "cell_id": cell_ids,
            "overlaps_nucleus": overlaps_nuc,
            "feature_name": mols["gene"].astype(str).to_numpy(),
            "x_location": x_global,
            "y_location": y_global,
            "z_location": z_loc,
            "qv": qv,
            "fov_name": np.full(n_mol, f"T{ti:03d}", dtype=object),
            "nucleus_distance": np.where(overlaps_nuc > 0, 0.0, 5.0).astype(np.float32),
            "codeword_index": np.zeros(n_mol, dtype=np.int32),
        })
        parts.append(df_out)

    if not parts:
        cols = ["transcript_id", "cell_id", "overlaps_nucleus", "feature_name",
                "x_location", "y_location", "z_location", "qv", "fov_name",
                "nucleus_distance", "codeword_index"]
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    df = pd.concat(parts, ignore_index=True)
    # Synthetic 64-bit transcript IDs (do not collide with real bundle IDs)
    tids = rng.integers(low=10**12, high=10**14, size=len(df))
    df.insert(0, "transcript_id", tids.astype(np.uint64))

    col_order = ["transcript_id", "cell_id", "overlaps_nucleus", "feature_name",
                 "x_location", "y_location", "z_location", "qv", "fov_name",
                 "nucleus_distance", "codeword_index"]
    return df[col_order]


# ---------------------------------------------------------------------------
# Polygon DataFrame assembly
# ---------------------------------------------------------------------------


def _polygons_to_long_df(
    polygons: list[tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Flatten ``(cell_id, x_arr, y_arr)`` polygons to Xenium long-form
    DataFrame: cell_id, vertex_x, vertex_y, label_id.
    """
    if not polygons:
        return pd.DataFrame({
            "cell_id": pd.Series(dtype=object),
            "vertex_x": pd.Series(dtype=np.float32),
            "vertex_y": pd.Series(dtype=np.float32),
            "label_id": pd.Series(dtype=np.int64),
        })

    # Vectorized assembly: build per-polygon arrays then concatenate
    # once at the end. Avoids the per-vertex Python `float(x) for x in
    # xs` loop that was ~4M iterations on a whole-bundle write.
    cid_to_label: dict[str, int] = {}
    next_label = 1
    xs_arrs: list[np.ndarray] = []
    ys_arrs: list[np.ndarray] = []
    cid_arrs: list[np.ndarray] = []
    lbl_arrs: list[np.ndarray] = []
    for cid, xs, ys in polygons:
        if cid not in cid_to_label:
            cid_to_label[cid] = next_label
            next_label += 1
        lbl = cid_to_label[cid]
        n = len(xs)
        if n == 0:
            continue
        xs_arrs.append(np.asarray(xs, dtype=np.float32))
        ys_arrs.append(np.asarray(ys, dtype=np.float32))
        cid_arrs.append(np.full(n, cid, dtype=object))
        lbl_arrs.append(np.full(n, lbl, dtype=np.int64))
    xs_out = np.concatenate(xs_arrs).tolist() if xs_arrs else []
    ys_out = np.concatenate(ys_arrs).tolist() if ys_arrs else []
    cids_out = np.concatenate(cid_arrs).tolist() if cid_arrs else []
    labels_out = np.concatenate(lbl_arrs).tolist() if lbl_arrs else []
    df = pd.DataFrame({
        "cell_id": cids_out,
        "vertex_x": np.asarray(xs_out, dtype=np.float32),
        "vertex_y": np.asarray(ys_out, dtype=np.float32),
        "label_id": np.asarray(labels_out, dtype=np.int64),
    })
    return df


def extract_scene_geometry(scene: Scene2D) -> dict[str, dict]:
    """Extract per-cell geometry + contour polygons from one tile scene's
    rasterized label masks, keyed by ``cell_id``.

    This is the per-tile counterpart of the mask reads that
    ``_collect_polygons`` / ``_build_real_cells_df`` / ``_build_cells_df``
    do in bulk at write time. Running it as each tile finishes lets the
    pipeline free the (large) label masks immediately instead of holding
    every tile's masks in RAM until the bundle write — the dominant term
    in parent memory on big bundles. The math is copied verbatim from
    those functions so the resulting tables are byte-identical (verified
    by tests in test_metadata_streaming.py).

    Returns ``{cell_id: {"cx","cy","area","nuc_area","cell_poly","nuc_poly"}}``
    where ``*_poly`` is ``(xs_um, ys_um)`` float32 arrays (or ``None`` if no
    contour could be extracted). Covers anchors AND ghosts; the caller
    decides per-cell (via the cell's own ``is_ghost``) where each lands.
    """
    from skimage.measure import find_contours
    from scipy.ndimage import find_objects as _find_objects

    mech = scene.mech_scene
    cell_lbl = mech.cell_label
    nuc_lbl = mech.nucleus_label
    xmin, _, ymin, _ = scene.tile_bounds_um
    psz = scene.pixel_size

    out: dict[str, dict] = {}
    if cell_lbl is None or cell_lbl.size == 0:
        return out

    max_lbl = int(cell_lbl.max())
    cell_slices = _find_objects(cell_lbl) if max_lbl > 0 else []
    if nuc_lbl is not None and nuc_lbl.size and int(nuc_lbl.max()) > 0:
        nuc_slices = _find_objects(nuc_lbl)
    else:
        nuc_slices = []

    def _contour(label_arr, slices, lbl):
        sl = slices[lbl - 1] if 1 <= lbl <= len(slices) else None
        if sl is None:
            return None
        sub = label_arr[sl] == lbl
        if not sub.any():
            return None
        padded = np.pad(sub.astype(np.uint8), 1, mode="constant")
        contours = find_contours(padded, level=0.5)
        if not contours:
            return None
        contour = max(contours, key=len)
        y_off, x_off = sl[0].start, sl[1].start
        ys_px = contour[:, 0] - 1 + y_off
        xs_px = contour[:, 1] - 1 + x_off
        return (np.asarray(xmin + xs_px * psz, dtype=np.float32),
                np.asarray(ymin + ys_px * psz, dtype=np.float32))

    for cell in mech.cells:
        lbl = int(cell.label)
        rec: dict = {"cx": None, "cy": None, "area": 0.0, "nuc_area": 0.0,
                     "cell_poly": None, "nuc_poly": None}
        # centroid + area (from cell mask)
        sl = cell_slices[lbl - 1] if 1 <= lbl <= len(cell_slices) else None
        if sl is not None:
            sub = cell_lbl[sl] == lbl
            if sub.any():
                y_off, x_off = sl[0].start, sl[1].start
                cys, cxs = np.where(sub)
                rec["cx"] = float(xmin + (cxs.mean() + x_off) * psz)
                rec["cy"] = float(ymin + (cys.mean() + y_off) * psz)
                rec["area"] = float(int(sub.sum()) * psz * psz)
        # nucleus area
        if nuc_slices and 1 <= lbl <= len(nuc_slices):
            nsl = nuc_slices[lbl - 1]
            if nsl is not None:
                nsub = nuc_lbl[nsl] == lbl
                if nsub.any():
                    rec["nuc_area"] = float(int(nsub.sum()) * psz * psz)
        # contour polygons
        rec["cell_poly"] = _contour(cell_lbl, cell_slices, lbl)
        rec["nuc_poly"] = _contour(nuc_lbl, nuc_slices, lbl)
        out[cell.cell_id] = rec
    return out


def _collect_polygons(
    scenes: list[Scene2D],
    *,
    kind: str,         # "cell" or "nucleus"
    is_ghost: bool,    # whether to collect ghost or anchor cells
    real_polys_lookup: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    geom_stash: dict[str, dict] | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Pull polygon (cell_id, xs, ys) tuples across all scenes.

    For anchor cells, polygons can be sourced from a real-bundle lookup
    (``real_polys_lookup: cell_id → (xs_um, ys_um)``) to avoid the
    tile-boundary clipping that affects tile-local contour extraction.
    Cells not in the lookup (and ghosts / tx-proposed cells) fall back to
    contour extraction.

    Contour polygons come from one of two sources, in order:
      * ``geom_stash`` (cell_id → {"cell_poly","nuc_poly", ...}), the
        per-tile extraction produced by :func:`extract_scene_geometry`
        before the masks were freed. Preferred when present.
      * direct ``find_contours`` on the scene's still-resident label mask
        (legacy path, used by tests / single-scene callers).
    """
    from skimage.measure import find_contours
    from scipy.ndimage import find_objects as _find_objects

    poly_key = "cell_poly" if kind == "cell" else "nuc_poly"
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sc in scenes:
        label_arr = (sc.mech_scene.cell_label
                       if kind == "cell" else sc.mech_scene.nucleus_label)
        xmin, _, ymin, _ = sc.tile_bounds_um
        psz = sc.pixel_size
        have_mask = label_arr is not None and label_arr.size > 0
        if have_mask:
            max_lbl = int(label_arr.max())
            slices = _find_objects(label_arr) if max_lbl > 0 else []
        else:
            slices = []
        # Map label → cell_id from the mech_scene
        for cell in sc.mech_scene.cells:
            if bool(cell.provenance.get("is_ghost", False)) != is_ghost:
                continue
            if (real_polys_lookup is not None and not is_ghost
                    and kind == "cell" and cell.cell_id in real_polys_lookup):
                xs_um, ys_um = real_polys_lookup[cell.cell_id]
                out.append((cell.cell_id,
                              xs_um.astype(np.float32),
                              ys_um.astype(np.float32)))
                continue
            if geom_stash is not None:
                rec = geom_stash.get(cell.cell_id)
                poly = rec.get(poly_key) if rec else None
                if poly is not None:
                    out.append((cell.cell_id, poly[0], poly[1]))
                continue
            if not have_mask:
                continue
            lbl = int(cell.label)
            sl = slices[lbl - 1] if 1 <= lbl <= len(slices) else None
            if sl is None:
                continue
            sub = label_arr[sl] == lbl
            if not sub.any():
                continue
            # Pad sub by 1 so find_contours can close at borders, then
            # offset back to tile-local pixel coords.
            padded = np.pad(sub.astype(np.uint8), 1, mode="constant")
            contours = find_contours(padded, level=0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            y_off, x_off = sl[0].start, sl[1].start
            ys_px = contour[:, 0] - 1 + y_off
            xs_px = contour[:, 1] - 1 + x_off
            xs_um = xmin + xs_px * psz
            ys_um = ymin + ys_px * psz
            out.append((cell.cell_id, xs_um.astype(np.float32),
                          ys_um.astype(np.float32)))
    return out


def _load_real_polys_lookup(real_bundle_path: str | Path,
                                kind: str = "cell",
                                cell_ids: set[str] | None = None,
                                ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a real bundle's cell or nucleus polygon table into a per-cell
    lookup. Filtered to ``cell_ids`` if given (significantly speeds up the
    read for large bundles)."""
    fname = ("cell_boundaries.parquet" if kind == "cell"
              else "nucleus_boundaries.parquet")
    p = Path(real_bundle_path) / fname
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["cell_id", "vertex_x", "vertex_y"])
    if cell_ids is not None:
        df = df[df["cell_id"].isin(cell_ids)]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid, g in df.groupby("cell_id", sort=False):
        out[str(cid)] = (g["vertex_x"].to_numpy(dtype=np.float32),
                          g["vertex_y"].to_numpy(dtype=np.float32))
    return out


# ---------------------------------------------------------------------------
# Cells table (ground truth)
# ---------------------------------------------------------------------------


def _build_real_cells_df(
    scenes: list[Scene2D],
    transcripts_df: pd.DataFrame,
    geom_stash: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Build the public ``cells.parquet`` table — anchors only, matching the
    real Xenium schema column-for-column.

    Columns: cell_id, x_centroid, y_centroid, transcript_counts,
    control_probe_counts, control_codeword_counts, unassigned_codeword_counts,
    deprecated_codeword_counts, total_counts, cell_area, nucleus_area.

    Ghosts are excluded (they live in ``ground_truth/cells_synth.parquet``).

    Centroid/area come from ``geom_stash`` (per-tile extraction, masks
    already freed) when provided, else from the still-resident label masks.
    """
    counts: dict[str, int] = {}
    if len(transcripts_df) > 0:
        vc = transcripts_df.groupby("cell_id", sort=False).size()
        counts = {str(k): int(v) for k, v in vc.items()}

    # Per-cell bbox lookup via scipy.ndimage.find_objects avoids the
    # full-tile `(cell_lbl == cell.label)` scan that dominated bundle-
    # write wall-time (367× faster at 500 cells / 2048² tile).
    from scipy.ndimage import find_objects as _find_objects

    rows = []
    for sc in scenes:
        xmin, _, ymin, _ = sc.tile_bounds_um
        psz = sc.pixel_size
        cell_lbl = sc.mech_scene.cell_label
        nuc_lbl = sc.mech_scene.nucleus_label
        use_stash = geom_stash is not None
        if not use_stash:
            max_lbl = int(cell_lbl.max()) if cell_lbl.size else 0
            cell_slices = _find_objects(cell_lbl) if max_lbl > 0 else []
            if nuc_lbl is not None and nuc_lbl.size and int(nuc_lbl.max()) > 0:
                nuc_slices = _find_objects(nuc_lbl)
            else:
                nuc_slices = []
        for cell in sc.mech_scene.cells:
            if bool(cell.provenance.get("is_ghost", False)):
                continue
            if use_stash:
                rec = geom_stash.get(cell.cell_id)
                if rec is None or rec["cx"] is None:
                    continue
                cx_um, cy_um = rec["cx"], rec["cy"]
                cell_area_um2 = rec["area"]
                nuc_area_um2 = rec["nuc_area"]
            else:
                lbl = int(cell.label)
                sl = cell_slices[lbl - 1] if 1 <= lbl <= len(cell_slices) else None
                if sl is None:
                    continue
                sub = cell_lbl[sl] == lbl
                if not sub.any():
                    continue
                y_off, x_off = sl[0].start, sl[1].start
                cys, cxs = np.where(sub)
                cx_um = xmin + (cxs.mean() + x_off) * psz
                cy_um = ymin + (cys.mean() + y_off) * psz
                cell_area_um2 = float(sub.sum()) * psz * psz
                nuc_area_um2 = 0.0
                if nuc_lbl is not None and 1 <= lbl <= len(nuc_slices):
                    nsl = nuc_slices[lbl - 1]
                    if nsl is not None:
                        nsub = nuc_lbl[nsl] == lbl
                        if nsub.any():
                            nuc_area_um2 = float(nsub.sum()) * psz * psz
            n_mol = counts.get(cell.cell_id, 0)
            rows.append({
                "cell_id": cell.cell_id,
                "x_centroid": float(cx_um),
                "y_centroid": float(cy_um),
                "transcript_counts": int(n_mol),
                "control_probe_counts": 0,
                "control_codeword_counts": 0,
                "unassigned_codeword_counts": 0,
                "deprecated_codeword_counts": 0,
                "total_counts": int(n_mol),
                "cell_area": float(cell_area_um2),
                "nucleus_area": float(nuc_area_um2),
            })
    if not rows:
        return pd.DataFrame({
            "cell_id": pd.Series(dtype=object),
            "x_centroid": pd.Series(dtype=np.float64),
            "y_centroid": pd.Series(dtype=np.float64),
            "transcript_counts": pd.Series(dtype=np.int64),
            "control_probe_counts": pd.Series(dtype=np.int64),
            "control_codeword_counts": pd.Series(dtype=np.int64),
            "unassigned_codeword_counts": pd.Series(dtype=np.int64),
            "deprecated_codeword_counts": pd.Series(dtype=np.int64),
            "total_counts": pd.Series(dtype=np.int64),
            "cell_area": pd.Series(dtype=np.float64),
            "nucleus_area": pd.Series(dtype=np.float64),
        })
    df = pd.DataFrame(rows)
    # Cast to real-bundle dtypes
    for c in ("transcript_counts", "control_probe_counts",
              "control_codeword_counts", "unassigned_codeword_counts",
              "deprecated_codeword_counts", "total_counts"):
        df[c] = df[c].astype(np.int64)
    for c in ("x_centroid", "y_centroid", "cell_area", "nucleus_area"):
        df[c] = df[c].astype(np.float64)
    return df


def _build_cells_df(scenes: list[Scene2D],
                    geom_stash: dict[str, dict] | None = None) -> pd.DataFrame:
    """Build a per-cell metadata table for ground_truth/cells_synth.parquet.

    EXCLUDES ghost cells — ghosts represent uncertain noise sources, not
    confirmed cells. Their molecule attribution is tracked separately in
    molecule_provenance.parquet (where the ghost-id is preserved so the
    full noise-source trail is auditable), and they're tagged UNASSIGNED
    in the public transcripts.parquet.

    Type-resolution columns (Phase 3 of the cell_type_resolver refactor):
      cell_type_source       — which tier produced the type call
                                (annotation|transcripts|training|stain_knn).
      cell_type_confidence   — float in [0, 1], tier-comparable.
      cell_type_evidence     — JSON string with tier-specific evidence.

    These columns live in ground_truth/cells_synth.parquet (xeSim-specific)
    and are NOT added to the public 10x-format cells.parquet (kept
    schema-clean for Xenium Explorer / downstream consumers).

    Centroid/area come from ``geom_stash`` (per-tile extraction, masks
    already freed) when provided, else from the still-resident label masks.
    """
    from scipy.ndimage import find_objects as _find_objects
    from ..cell_type_resolver import evidence_to_json

    rows = []
    for sc in scenes:
        xmin, _, ymin, _ = sc.tile_bounds_um
        psz = sc.pixel_size
        cell_lbl = sc.mech_scene.cell_label
        use_stash = geom_stash is not None
        if not use_stash:
            max_lbl = int(cell_lbl.max()) if cell_lbl.size else 0
            cell_slices = _find_objects(cell_lbl) if max_lbl > 0 else []
        for cell in sc.mech_scene.cells:
            # Skip ghosts — they're noise-source emitters, not confirmed cells.
            if bool(cell.provenance.get("is_ghost", False)):
                continue
            if use_stash:
                rec = geom_stash.get(cell.cell_id)
                if rec is None or rec["cx"] is None:
                    continue
                cx_um, cy_um, area_um2 = rec["cx"], rec["cy"], rec["area"]
            else:
                lbl = int(cell.label)
                sl = cell_slices[lbl - 1] if 1 <= lbl <= len(cell_slices) else None
                if sl is None:
                    continue
                sub = cell_lbl[sl] == lbl
                if not sub.any():
                    continue
                y_off, x_off = sl[0].start, sl[1].start
                ys, xs = np.where(sub)
                cx_um = xmin + (xs.mean() + x_off) * psz
                cy_um = ymin + (ys.mean() + y_off) * psz
                area_um2 = float(int(sub.sum()) * psz * psz)
            # Pull resolver-stamped provenance, if present. Cells from a
            # pre-refactor run (or that fell through all four tiers) get
            # `None` / 0.0 / "{}" — Phase 4 of the refactor makes the
            # stain bank mandatory so the None case disappears.
            res = cell.provenance.get("type_resolution")
            rows.append({
                "cell_id": cell.cell_id,
                "cell_type": cell.cell_type or "unknown",
                "is_ghost": bool(cell.provenance.get("is_ghost", False)),
                "centroid_x": float(cx_um),
                "centroid_y": float(cy_um),
                "area_um2": float(area_um2),
                "source": cell.source,
                "cell_type_source": (res["source"] if res else None),
                "cell_type_confidence": (float(res["confidence"])
                                            if res else 0.0),
                "cell_type_evidence": (evidence_to_json(res["evidence"])
                                          if res else "{}"),
            })
    df = pd.DataFrame(rows)
    # Invariant: every anchor cell must end with a real type. The
    # cell_type_resolver guarantees this when the stain-classifier bank
    # exists (mandatory at fit time). If this fires, the model was fit
    # before the Phase 4 refactor (missing bank) or the resolver had a
    # bug. Loud failure beats silent dim-rendered "unknown" cells.
    if len(df) > 0:
        bad_mask = (df["cell_type"].isin(["unknown", "Unknown", "UNKNOWN", ""])
                    | df["cell_type"].isna())
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            sample_ids = df.loc[bad_mask, "cell_id"].head(5).tolist()
            raise RuntimeError(
                f"cells_synth invariant violated: {n_bad} of {len(df)} "
                f"cells have cell_type ∈ {{'', None, 'unknown'}}. The "
                f"cell-type resolver must produce a real type for every "
                f"anchor; this typically means the model's stain-classifier "
                f"bank (cell_latent_bank.npz) is missing or the resolver "
                f"hit a bug. First offending cell_ids: {sample_ids}.")
    return df


# ---------------------------------------------------------------------------
# Morphology image
# ---------------------------------------------------------------------------


def _scale_to_uint16(
    arr: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    target_stats: dict[str, tuple[float, float]] | None = None,
    target_quantiles: dict[str, np.ndarray] | None = None,
    mode: str = "scale",
    display_lut: dict | None = None,
) -> np.ndarray:
    """Thin wrapper around :func:`calibrate_to_uint16` kept for back-compat.

    Already-uint16 inputs (e.g. pre-stitched multi-tile images from
    ``scene_pipeline.build_scene``) are returned unchanged.
    """
    return calibrate_to_uint16(
        arr, channel_names=channel_names, target_stats=target_stats,
        target_quantiles=target_quantiles, mode=mode, display_lut=display_lut)


# ---------------------------------------------------------------------------
# OME-TIFF pyramid helper
# ---------------------------------------------------------------------------


def _downsample_2x(cur: np.ndarray, channel_axis: int | None) -> np.ndarray | None:
    """One 2x downsample step. Returns None if too small to halve."""
    if channel_axis == 0 and cur.ndim == 3:
        _, h, w = cur.shape
        if h < 16 or w < 16: return None
        h2, w2 = h // 2, w // 2
        return cur[:, :h2*2, :w2*2].reshape(
            cur.shape[0], h2, 2, w2, 2).astype(np.float32).mean(
            axis=(2, 4)).astype(cur.dtype)
    if cur.ndim == 3:    # ZYX
        z, h, w = cur.shape
        if h < 16 or w < 16: return None
        h2, w2 = h // 2, w // 2
        return cur[:, :h2*2, :w2*2].reshape(
            z, h2, 2, w2, 2).astype(np.float32).mean(
            axis=(2, 4)).astype(cur.dtype)
    if cur.ndim == 2:
        h, w = cur.shape
        if h < 16 or w < 16: return None
        h2, w2 = h // 2, w // 2
        return cur[:h2*2, :w2*2].reshape(
            h2, 2, w2, 2).astype(np.float32).mean(
            axis=(1, 3)).astype(cur.dtype)
    return None


def _build_pyramid_levels(
    arr: np.ndarray, n_levels: int = 8, channel_axis: int | None = 0,
    parallel: bool = False,    # parallel-from-base OOMs on whole-bundle scale
) -> list[np.ndarray]:
    """Return a list of ``n_levels`` 2× downsampled arrays starting from ``arr``.

    For a (C, H, W) array, downsampling preserves the channel axis. Levels stop
    early if the spatial extent would shrink below 8 px. Uses simple 2× block
    averaging (then cast back to the array's dtype) — close enough to
    Xenium's pyramid for viewer/segmenter use.
    """
    levels = [arr]

    # Serial-by-default path: each level is computed from the previous
    # level (already 4× smaller) instead of from the base. This avoids
    # OOM on huge images — the parallel-from-base approach materialized
    # N float32 promotions of the base in memory, which for a 34k×14k×4
    # uint16 base is ~7.5 GB × N concurrent threads. Serial keeps one
    # intermediate at a time.
    cur = arr
    for _ in range(1, n_levels):
        cur = _downsample_2x(cur, channel_axis)
        if cur is None: break
        levels.append(cur)
    return levels


def _write_pyramid_ome_tiff(
    out_path: Path,
    base: np.ndarray,
    *,
    axes: str,                       # "CYX" or "ZYX"
    pixel_size_um: float,
    channel_names: list[str],
    physical_size_z: float | None = None,
    n_levels: int = 8,
) -> None:
    """Write an OME-TIFF with an 8-level (2× per step) pyramid.

    Matches the real Xenium morphology layout (8 pyramid levels) using
    tifffile's ``subifds`` mechanism: the base IFD declares N sub-resolutions,
    then each downsampled level is written as a sub-IFD.
    """
    import tifffile
    levels = _build_pyramid_levels(
        base, n_levels=n_levels,
        channel_axis=0 if axes == "CYX" else None)
    metadata: dict = {
        "axes": axes,
        "PhysicalSizeX": pixel_size_um, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size_um, "PhysicalSizeYUnit": "µm",
    }
    if physical_size_z is not None:
        metadata.update({"PhysicalSizeZ": physical_size_z,
                         "PhysicalSizeZUnit": "µm"})
    if channel_names:
        metadata["Channel"] = {"Name": list(channel_names)}

    # Use BigTIFF when total pixel volume exceeds ~3 GB (gives headroom
    # below the 4 GB classic-TIFF cap). All sub-resolutions count.
    bytes_total = sum(int(np.prod(lvl.shape)) * int(lvl.dtype.itemsize)
                       for lvl in levels)
    use_bigtiff = bytes_total > 3 * (1 << 30)
    n_sub = max(0, len(levels) - 1)
    import time as _time
    _t0 = _time.time()
    tag = out_path.name
    print(f"  [pyramid] {tag}: writing {len(levels)} levels "
          f"({levels[0].shape} → {levels[-1].shape}, "
          f"{bytes_total/(1<<30):.2f} GiB "
          f"{'bigtiff' if use_bigtiff else 'classic-tiff'})", flush=True)
    with tifffile.TiffWriter(out_path, ome=True, bigtiff=use_bigtiff) as tw:
        # Base resolution: declares the sub-IFD slots, gets OME metadata
        tw.write(levels[0], photometric="minisblack",
                 metadata=metadata, subifds=n_sub)
        print(f"  [pyramid] {tag}: level 0/{len(levels)-1} done "
              f"(elapsed {_time.time()-_t0:.1f}s)", flush=True)
        # Pyramid levels: each as a subfiletype=1 reduced-resolution sub-IFD
        for li, sub in enumerate(levels[1:], start=1):
            tw.write(sub, photometric="minisblack", subfiletype=1)
            print(f"  [pyramid] {tag}: level {li}/{len(levels)-1} done "
                  f"(elapsed {_time.time()-_t0:.1f}s)", flush=True)


def _make_xenium_morphology_ome_xml(
    file_index: int,
    n_channels: int,
    channel_names: list[str],
    h: int,
    w: int,
    pixel_size_um: float,
    file_uuids: list[str],
) -> str:
    """Build a Xenium-format OME-XML for morphology_focus_NNNN.ome.tif.

    Each file declares ``SizeC=n_channels`` and lists ``TiffData`` entries
    with ``UUID`` cross-references pointing to all sibling files. This
    matches the layout produced by real 10x Genomics Xenium runs so that
    Xenium Explorer and other OME-aware tools recognise the multi-channel
    morphology stack assembled across the four single-channel TIFFs.
    """
    # Match real Xenium's StructuredAnnotation purposes for the standard
    # 4-channel panel (DAPI / boundary / RNA / interior). For non-standard
    # panels we default to "Other".
    purposes_default = ["Nuclear", "Boundary", "RNA", "Interior"]
    channels_xml = "\n".join(
        f'      <Channel ID="Channel:{i}" Name="{channel_names[i]}" SamplesPerPixel="1">\n'
        f'        <AnnotationRef ID="Annotation:{i}"/>\n'
        f'      </Channel>'
        for i in range(n_channels))
    tiffdata_xml = "\n".join(
        f'      <TiffData FirstZ="0" FirstT="0" FirstC="{i}" PlaneCount="1">\n'
        f'        <UUID FileName="morphology_focus_{i:04d}.ome.tif">urn:uuid:{file_uuids[i]}</UUID>\n'
        f'      </TiffData>'
        for i in range(n_channels))
    annot_refs = "\n".join(
        f'    <AnnotationRef ID="Annotation:{i}"/>'
        for i in range(n_channels))
    annotations_xml = "\n".join(
        f'    <MapAnnotation ID="Annotation:{i}">\n'
        f'      <Value>\n'
        f'        <M K="Purpose">{purposes_default[i] if i < len(purposes_default) else "Other"}</M>\n'
        f'      </Value>\n'
        f'    </MapAnnotation>'
        for i in range(n_channels))
    return (
        f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
        f'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
        f'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" '
        f'UUID="urn:uuid:{file_uuids[file_index]}">\n'
        f'  <Plate ID="Plate:0" WellOriginX="0.0" WellOriginXUnit="&#xB5;m" '
        f'WellOriginY="0.0" WellOriginYUnit="&#xB5;m"/>\n'
        f'  <Instrument ID="Instrument:0">\n'
        f'    <Microscope Manufacturer="10x Genomics" Model="Xenium"/>\n'
        f'  </Instrument>\n'
        f'  <Image ID="Image:0">\n'
        f'    <InstrumentRef ID="Instrument:0"/>\n'
        f'    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
        f'SizeX="{w}" SizeY="{h}" SizeZ="1" SizeC="{n_channels}" SizeT="1" '
        f'PhysicalSizeX="{pixel_size_um}" PhysicalSizeXUnit="&#xB5;m" '
        f'PhysicalSizeY="{pixel_size_um}" PhysicalSizeYUnit="&#xB5;m">\n'
        f'{channels_xml}\n'
        f'{tiffdata_xml}\n'
        f'    </Pixels>\n'
        f'{annot_refs}\n'
        f'  </Image>\n'
        f'  <StructuredAnnotations>\n'
        f'{annotations_xml}\n'
        f'  </StructuredAnnotations>\n'
        f'</OME>\n'
    )


def _rewrite_morphology_focus_ome_xml(
    focus_dir: Path, channel_names: list[str], pixel_size_um: float,
) -> None:
    """Overwrite each ``morphology_focus_NNNN.ome.tif``'s OME-XML in-place
    with Xenium-format metadata (SizeC=N, cross-referenced TiffData/UUIDs).
    Image pixels and pyramid IFDs are left untouched."""
    import tifffile
    import uuid
    paths = sorted(focus_dir.glob("morphology_focus_*.ome.tif"))
    n_ch = len(paths)
    if n_ch == 0:
        return
    # Deterministic UUIDs derived from absolute path + index — stable
    # across re-runs of the same bundle so external indexers see the same
    # UUIDs after a regen.
    seed_ns = uuid.uuid5(uuid.NAMESPACE_URL, str(focus_dir.resolve()))
    file_uuids = [
        str(uuid.uuid5(seed_ns, f"morphology_focus_{i:04d}"))
        for i in range(n_ch)]
    # Pull canvas size from file 0.
    with tifffile.TiffFile(paths[0]) as tf:
        h, w = tf.series[0].shape[-2:]
    for i, p in enumerate(paths):
        xml = _make_xenium_morphology_ome_xml(
            i, n_ch, channel_names, h, w, pixel_size_um, file_uuids)
        tifffile.tiffcomment(p, xml)


def _write_morphology_focus(
    image: np.ndarray,             # (C, H, W) float32 OR uint16
    channel_names: list[str],
    pixel_size_um: float,
    out_dir: Path,
    *,
    target_stats: dict[str, tuple[float, float]] | None = None,
    target_quantiles: dict[str, np.ndarray] | None = None,
    intensity_mode: str = "scale",
    n_replica_files: int = 4,
    n_pyramid_levels: int = 8,
    display_lut: dict | None = None,
) -> None:
    """Write ``morphology_focus/morphology_focus_NNNN.ome.tif`` (one per
    channel) following Xenium's convention: each file is a SINGLE-channel
    pyramid OME-TIFF whose OME-XML still advertises ``SizeC=N`` across
    the bundle. The downstream reader assembles the multi-channel stack
    by reading channel 0 of each file.

    The number of files (``n_replica_files``) is fixed by Xenium
    convention; we use ``len(channel_names)`` capped at
    ``n_replica_files``. If a model has fewer than 4 channels we still
    write 4 files (real bundles always ship 4) — extra files repeat the
    last channel so OME-XML and on-disk file count stay consistent."""
    focus_dir = out_dir / "morphology_focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    arr_u16 = _scale_to_uint16(image, channel_names=channel_names,
                                target_stats=target_stats,
                                target_quantiles=target_quantiles,
                                mode=intensity_mode,
                                display_lut=display_lut)
    n_ch = arr_u16.shape[0]
    n_files = max(n_replica_files, n_ch)

    def _write_one(i: int) -> None:
        ch_idx = min(i, n_ch - 1)
        per_ch = arr_u16[ch_idx]
        dst = focus_dir / f"morphology_focus_{i:04d}.ome.tif"
        if dst.exists(): dst.unlink()
        # Pass just THIS file's channel name so the per-file OME-XML
        # labels the single plane correctly. The bundle-wide channel
        # roster is reconstructed by reading channel_names from each
        # file in order (the downstream reader does this).
        per_ch_name = (channel_names[ch_idx]
                          if ch_idx < len(channel_names) else f"channel_{ch_idx}")
        _write_pyramid_ome_tiff(
            dst, per_ch, axes="YX",
            pixel_size_um=pixel_size_um,
            channel_names=[per_ch_name],
            n_levels=n_pyramid_levels,
        )

    # Parallelize the per-channel writes via threads. Each thread shares
    # arr_u16 read-only and writes to its own file. Pyramid build holds
    # ~600 MB per channel at peak; 4 channels in parallel ~2.4 GB —
    # within budget. NOT multiprocess (commit 113226e shows that OOM'd
    # at whole-bundle scale due to pickle-copying the base image).
    from concurrent.futures import ThreadPoolExecutor
    max_workers = min(n_files, 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_write_one, range(n_files)))
    # After all per-channel files are written, overwrite each one's
    # OME-XML so the bundle's `morphology_focus/` advertises the multi-
    # channel structure Xenium Explorer (and other OME-aware readers)
    # expect — SizeC=N with TiffData/UUID cross-references between the
    # sibling files. Pixel data is untouched.
    _rewrite_morphology_focus_ome_xml(
        focus_dir, channel_names[:n_files], pixel_size_um)


def _write_morphology_z(
    image: np.ndarray,             # (C, H, W); we extract the DAPI channel
    channel_names: list[str],
    pixel_size_um: float,
    out_path: Path,
    n_z: int = 12,
    *,
    target_stats: dict[str, tuple[float, float]] | None = None,
    target_quantiles: dict[str, np.ndarray] | None = None,
    intensity_mode: str = "scale",
    n_pyramid_levels: int = 8,
    display_lut: dict | None = None,
) -> None:
    """Write ``morphology.ome.tif`` (ZYX uint16, DAPI-only z-stack) with an
    8-level pyramid.

    Real Xenium's morphology.ome.tif is a multi-z DAPI image. We replicate
    the same 2D DAPI across ``n_z`` slices; tools that use this file for
    3D nucleus segmentation will see a flat z-stack — accepted simplification.
    """
    # Find DAPI channel (or fall back to channel 0)
    dapi_idx = 0
    for i, n in enumerate(channel_names):
        if n.upper() == "DAPI":
            dapi_idx = i; break
    # Apply intensity calibration on the full image, then take DAPI slice
    full_u16 = _scale_to_uint16(image, channel_names=channel_names,
                                  target_stats=target_stats,
                                  target_quantiles=target_quantiles,
                                  mode=intensity_mode,
                                  display_lut=display_lut)
    dapi_2d = full_u16[dapi_idx]                           # (H, W)
    stack = np.broadcast_to(dapi_2d, (n_z, *dapi_2d.shape)).copy()
    _write_pyramid_ome_tiff(
        out_path, stack, axes="ZYX",
        pixel_size_um=pixel_size_um, channel_names=["DAPI"],
        physical_size_z=0.75, n_levels=n_pyramid_levels,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_bundle(
    *,
    output_dir: str | Path,
    scenes: Scene2D | Sequence[Scene2D],
    render_images: np.ndarray | Sequence[np.ndarray] | None,
    channel_names: list[str],
    pixel_size_um: float,
    gene_panel_source: str | Path | None = None,
    config: dict | None = None,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
    target_intensity_stats: dict[str, tuple[float, float]] | None = None,
    target_intensity_quantiles: dict[str, np.ndarray] | None = None,
    intensity_mode: str = "scale",
    n_morphology_focus_files: int = 4,
    n_pyramid_levels: int = 8,
    real_bundle_path: str | Path | None = None,
    display_lut: dict | None = None,
    geom_stash: dict[str, dict] | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a synthetic Xenium-compatible bundle directory.

    ``geom_stash`` (cell_id → per-cell geometry + contour polygons, from
    :func:`extract_scene_geometry` during tile build) lets the writer build
    boundaries/cells tables without re-reading the per-tile label masks —
    which the pipeline frees as soon as each tile is extracted to keep
    parent RAM bounded on large bundles. When ``None``, geometry is read
    from the scenes' still-resident masks (legacy / small-run path).

    Parameters
    ----------
    output_dir
        Destination directory. Created if missing.
    scenes
        Single Scene2D or sequence of Scene2D (Phase 2.A multi-tile).
    render_images
        Single (C,H,W) float array or sequence of them. May be None to skip
        morphology image writes. Sequence length must match ``scenes``.
        Multi-tile stitching is currently NOT implemented — pass a single
        image.
    channel_names
        Per-channel name list (e.g. ["DAPI", "ATP1A1/CD45/E-Cadherin", ...]).
    pixel_size_um
        Image pixel size (µm).
    gene_panel_source
        Path to a ``gene_panel.json`` in a real bundle. Copied verbatim
        if provided.
    config
        Generation config to be saved to ``ground_truth/config.json``
        (noise_fraction target, auto-tuned value, seed, etc.).
    rng
        Random generator for synthetic transcript IDs.
    overwrite
        If True, replace an existing directory.

    Returns
    -------
    dict with paths written.
    """
    if isinstance(scenes, Scene2D):
        scenes_list = [scenes]
    else:
        scenes_list = list(scenes)

    if render_images is None:
        render_list: list[np.ndarray] = []
    elif isinstance(render_images, np.ndarray):
        render_list = [render_images]
    else:
        render_list = list(render_images)

    # Allowed input combinations:
    #   single scene + single image  → tile mode (Phase 2.B)
    #   multiple scenes + single (pre-stitched) image  → scene mode (Phase 2.A)
    #   multiple scenes + no image  → caller built bundle without rendering
    if len(render_list) > 1 and len(render_list) != len(scenes_list):
        raise ValueError(
            "Pass either a single (pre-stitched) image or one image per scene")
    if len(scenes_list) > 1 and len(render_list) > 1:
        raise NotImplementedError(
            "Per-tile image inputs not supported; pre-stitch the image yourself")

    out_dir = Path(output_dir)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    rng = rng or np.random.default_rng(0)

    written: dict[str, Any] = {"output_dir": str(out_dir)}

    # CSV.gz mirror writes are expensive (~25% of bundle-write wall time
    # on whole-pancreas profile via py-spy). They duplicate .parquet
    # data 1:1 for tools that don't speak Arrow. Set XESIM_SKIP_CSV_GZ=1
    # to skip them (parquet covers all data; synth bundles consumed
    # directly by xeSim diagnostics / Xenium Explorer don't need csv).
    import os as _os
    _skip_csv = _os.environ.get("XESIM_SKIP_CSV_GZ", "").strip() == "1"

    # 1. Transcripts (with TRUE cell_id; ghost-cell molecules carry ghost IDs)
    transcripts_df = _build_transcripts_df(scenes_list, rng=rng)
    transcripts_df.to_parquet(out_dir / "transcripts.parquet", index=False)
    if not _skip_csv:
        transcripts_df.to_csv(out_dir / "transcripts.csv.gz",
                                index=False, compression="gzip")
    written["transcripts"] = {
        "n_rows": int(len(transcripts_df)),
        "parquet": str(out_dir / "transcripts.parquet"),
        **({"csv_gz": str(out_dir / "transcripts.csv.gz")} if not _skip_csv else {}),
    }

    # 2. Cell boundaries — anchors only (ground-truth polygons).
    # For anchor cells, prefer the real bundle's polygons (they are
    # the genuine ground truth; tile-local contour extraction truncates
    # cells at tile boundaries). Ghosts/tx-proposed fall back to contour
    # extraction since they have no real polygon.
    real_cell_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    real_nuc_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if real_bundle_path is not None:
        anchor_ids: set[str] = set()
        for sc in scenes_list:
            for cell in sc.mech_scene.cells:
                if not bool(cell.provenance.get("is_ghost", False)):
                    anchor_ids.add(cell.cell_id)
        try:
            real_cell_lookup = _load_real_polys_lookup(
                real_bundle_path, kind="cell", cell_ids=anchor_ids)
            real_nuc_lookup = _load_real_polys_lookup(
                real_bundle_path, kind="nucleus", cell_ids=anchor_ids)
        except Exception as e:
            print(f"[bundle_writer] real_bundle polygon lookup failed: {e}; "
                  f"falling back to tile-local contour extraction.")
    anchor_cell_polys = _collect_polygons(
        scenes_list, kind="cell", is_ghost=False,
        real_polys_lookup=real_cell_lookup or None, geom_stash=geom_stash)
    anchor_nuc_polys = _collect_polygons(
        scenes_list, kind="nucleus", is_ghost=False,
        real_polys_lookup=real_nuc_lookup or None, geom_stash=geom_stash)
    cells_df = _polygons_to_long_df(anchor_cell_polys)
    nucs_df = _polygons_to_long_df(anchor_nuc_polys)
    cells_df.to_parquet(out_dir / "cell_boundaries.parquet", index=False)
    nucs_df.to_parquet(out_dir / "nucleus_boundaries.parquet", index=False)
    if not _skip_csv:
        cells_df.to_csv(out_dir / "cell_boundaries.csv.gz",
                          index=False, compression="gzip")
        nucs_df.to_csv(out_dir / "nucleus_boundaries.csv.gz",
                         index=False, compression="gzip")
    written["cell_boundaries"] = {"n_anchor_cells": int(cells_df["cell_id"].nunique())}

    # 2b. cells.parquet + cells.csv.gz — public per-cell metadata (anchors only)
    real_cells_df = _build_real_cells_df(scenes_list, transcripts_df,
                                         geom_stash=geom_stash)
    real_cells_df.to_parquet(out_dir / "cells.parquet", index=False)
    if not _skip_csv:
        real_cells_df.to_csv(out_dir / "cells.csv.gz",
                               index=False, compression="gzip")
    written["cells"] = {
        "n_cells": int(len(real_cells_df)),
        "parquet": str(out_dir / "cells.parquet"),
        **({"csv_gz": str(out_dir / "cells.csv.gz")} if not _skip_csv else {}),
    }

    # 3. Morphology image(s) — calibrated uint16 + 8-level pyramid + 4 focus files
    if render_list:
        if len(render_list) == 1:
            img = render_list[0]
            import time as _t
            _s = _t.time()
            print(f"[bundle_writer] morphology_focus: 4 channel files "
                  f"(shape {img.shape})...", flush=True)
            _write_morphology_focus(
                img, channel_names, pixel_size_um, out_dir,
                target_stats=target_intensity_stats,
                target_quantiles=target_intensity_quantiles,
                intensity_mode=intensity_mode,
                n_replica_files=n_morphology_focus_files,
                n_pyramid_levels=n_pyramid_levels,
                display_lut=display_lut,
            )
            print(f"[bundle_writer] morphology_focus done ({_t.time()-_s:.1f}s); "
                  f"now morphology.ome.tif (Z-stack pyramid)...", flush=True)
            _s = _t.time()
            _write_morphology_z(
                img, channel_names, pixel_size_um,
                out_dir / "morphology.ome.tif",
                target_stats=target_intensity_stats,
                target_quantiles=target_intensity_quantiles,
                intensity_mode=intensity_mode,
                n_pyramid_levels=n_pyramid_levels,
                display_lut=display_lut,
            )
            print(f"[bundle_writer] morphology.ome.tif done ({_t.time()-_s:.1f}s)",
                  flush=True)
            written["morphology"] = {
                "focus": str(out_dir / "morphology_focus" /
                              "morphology_focus_0000.ome.tif"),
                "n_focus_files": n_morphology_focus_files,
                "z_stack": str(out_dir / "morphology.ome.tif"),
                "shape": list(img.shape),
                "pyramid_levels": n_pyramid_levels,
                "intensity_mode": intensity_mode,
                "intensity_calibrated": (
                    target_intensity_stats is not None
                    or target_intensity_quantiles is not None),
            }

    # 4. Gene panel (copy verbatim if source provided)
    if gene_panel_source is not None:
        gp_src = Path(gene_panel_source)
        if gp_src.exists():
            shutil.copy2(gp_src, out_dir / "gene_panel.json")
            written["gene_panel"] = str(out_dir / "gene_panel.json")

    # 5. experiment.xenium metadata
    n_anchor_cells = int(cells_df["cell_id"].nunique()) if len(cells_df) else 0
    ghost_cells_total = int(sum(sc.n_ghosts for sc in scenes_list))
    region_name = "synth_bundle"
    if scenes_list:
        xmin = min(sc.tile_bounds_um[0] for sc in scenes_list)
        xmax = max(sc.tile_bounds_um[1] for sc in scenes_list)
        ymin = min(sc.tile_bounds_um[2] for sc in scenes_list)
        ymax = max(sc.tile_bounds_um[3] for sc in scenes_list)
    else:
        xmin = xmax = ymin = ymax = 0.0
    experiment = {
        "analysis_sw_name": "xeSim",
        "analysis_sw_version": "plan3.2D",
        "region_name": region_name,
        "panel_design_id": "synthetic",
        "panel_name": "synth_panel",
        "num_cells": n_anchor_cells,
        "num_unassigned_transcripts": int(0),   # synthetic bundle has no UNASSIGNED
        "num_transcripts": int(len(transcripts_df)),
        "pixel_size": float(pixel_size_um),
        "tile_bounds_um": [xmin, ymin, xmax, ymax],
        "num_channels": len(channel_names),
        "channel_names": list(channel_names),
        "synth_metadata": {
            "n_ghost_cells": ghost_cells_total,
            "ground_truth_in_cell_id_column": True,
            # Render provenance — so a bundle self-documents how it was made
            # (a missing model_dir is exactly what made an earlier bundle's
            # render impossible to reconstruct / mismatched in diagnostics).
            "model_dir": str(model_dir) if model_dir is not None else None,
            "source_bundle": str(real_bundle_path) if real_bundle_path is not None else None,
            "intensity_calibration": (
                {k: list(v) for k, v in target_intensity_stats.items()}
                if target_intensity_stats else None
            ),
            "intensity_mode": intensity_mode,
            "noise_calibrated": bool(display_lut and display_lut.get("noise_stats")),
            "n_morphology_focus_files": n_morphology_focus_files,
            "n_pyramid_levels": n_pyramid_levels,
        },
    }
    (out_dir / "experiment.xenium").write_text(json.dumps(experiment, indent=2))
    written["experiment_xenium"] = str(out_dir / "experiment.xenium")

    # 6. Ground truth: extras
    # 6a. Molecule provenance (richer than transcripts.parquet — has is_ghost).
    # x_um/y_um are GLOBAL bundle µm coordinates (mols["x"] is tile-local;
    # add the tile origin). Vectorized per tile + concat — pandas
    # iterrows() over the 6M-row molecule table took ~4 min; vectorized
    # is ~0.1s (~2000x speedup, see commit msg).
    prov_dfs = []
    for ti, sc in enumerate(scenes_list):
        if len(sc.molecules) == 0:
            continue
        xmin, _, ymin, _ = sc.tile_bounds_um
        m = sc.molecules
        df_tile = pd.DataFrame({
            "tile_idx": ti,
            "true_cell_id": m["true_cell_id"].astype(str).to_numpy(),
            "is_ghost": (m["is_ghost"].astype(bool).to_numpy()
                          if "is_ghost" in m.columns
                          else np.zeros(len(m), dtype=bool)),
            "gene": m["gene"].astype(str).to_numpy(),
            "true_factor": (m["true_factor"].astype(int).to_numpy()
                             if "true_factor" in m.columns
                             else np.full(len(m), -1, dtype=int)),
            "x_um": (m["x"].astype(float).to_numpy() + float(xmin)),
            "y_um": (m["y"].astype(float).to_numpy() + float(ymin)),
        })
        prov_dfs.append(df_tile)
    if prov_dfs:
        prov_df = pd.concat(prov_dfs, ignore_index=True)
    else:
        prov_df = pd.DataFrame(columns=[
            "tile_idx", "true_cell_id", "is_ghost", "gene",
            "true_factor", "x_um", "y_um"])
    prov_df.to_parquet(gt_dir / "molecule_provenance.parquet", index=False)

    # 6b. Cells table (anchor + ghost with is_ghost flag)
    cells_meta = _build_cells_df(scenes_list, geom_stash=geom_stash)
    cells_meta.to_parquet(gt_dir / "cells_synth.parquet", index=False)

    # 6c. Ghost cell boundaries (kept separate so the public bundle is "clean")
    ghost_cell_polys = _collect_polygons(scenes_list, kind="cell", is_ghost=True,
                                         geom_stash=geom_stash)
    ghost_nuc_polys = _collect_polygons(scenes_list, kind="nucleus", is_ghost=True,
                                        geom_stash=geom_stash)
    _polygons_to_long_df(ghost_cell_polys).to_parquet(
        gt_dir / "ghost_cell_boundaries.parquet", index=False)
    _polygons_to_long_df(ghost_nuc_polys).to_parquet(
        gt_dir / "ghost_nucleus_boundaries.parquet", index=False)

    # 6d. Generation config
    cfg = dict(config or {})
    cfg.update({
        "n_scenes": len(scenes_list),
        "n_anchor_cells": n_anchor_cells,
        "n_ghost_cells": ghost_cells_total,
        "n_transcripts": int(len(transcripts_df)),
        "channel_names": list(channel_names),
        "pixel_size_um": float(pixel_size_um),
    })
    (gt_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    written["ground_truth_dir"] = str(gt_dir)
    written["config"] = cfg

    return written


__all__ = ["write_bundle", "auto_tune_noise_fraction",
           "auto_tune_intensity_stats", "UNASSIGNED_LABEL"]
