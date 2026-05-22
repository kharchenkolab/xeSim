"""Stage E-prime: decode the trained UNet proposer into Stage B candidates.

Wraps a trained ``TinyProposerUNet`` checkpoint with a function whose signature
matches ``propose_latent_cells`` so it can drop into the existing Stage B
pipeline. Decoding strategy:

1. Forward the proposer on (real_dapi, real_membrane, ablated_cell_mask,
   ablated_nucleus_mask) → (present_prob, offsets, type_logits).
2. Find local maxima of ``present_prob`` above ``threshold`` with NMS window
   ``nms_size``; refine each peak's centroid using the predicted offsets.
3. Build a footprint per centroid: a small disk of radius ``cell_radius_px``
   for the cell, and a smaller disk for the nucleus. (Footprint quality is
   modest but adequate as a starting point — can be replaced by predicted
   masks later.)
4. Look up the predicted cell type at the centroid (argmax of
   ``type_logits``) and store on the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from .proposer_train import TinyProposerUNet
from .scene_completion import CandidateCell


def load_proposer(checkpoint_path: Path, device: torch.device | str = "cuda") -> dict[str, Any]:
    """Load a trained proposer checkpoint and instantiate the model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model"]
    model = TinyProposerUNet(
        in_channels=int(cfg["in_channels"]),
        hidden=int(cfg["hidden"]),
        n_types_plus_bg=int(cfg["n_types_plus_bg"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "type_names": list(ckpt.get("type_names", [])),
        "device": device,
    }


def propose_latent_cells_learned(
    real_dapi: np.ndarray,
    real_membrane: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    proposer: dict[str, Any],
    threshold: float = 0.7,
    nms_size: int = 21,
    cell_radius_px: int = 7,
    nucleus_radius_px: int = 4,
    shape_head: Any = None,
    real_polya: np.ndarray | None = None,
) -> list[CandidateCell]:
    """Stage E-prime drop-in replacement for ``propose_latent_cells``.

    Returns CandidateCell objects whose centroids come from the learned
    proposer's local-maxima decoding. Footprint is a disk approximation;
    swap for predicted instance segmentation when available.
    """

    model = proposer["model"]
    device = proposer["device"]
    type_names = proposer["type_names"]
    H, W = real_dapi.shape

    cell_mask = (cell_label > 0).astype(np.float32)
    nuc_mask = (nucleus_label > 0).astype(np.float32)
    inputs = np.stack([
        real_dapi.astype(np.float32),
        real_membrane.astype(np.float32),
        cell_mask,
        nuc_mask,
    ], axis=0)
    with torch.no_grad():
        x = torch.from_numpy(inputs).unsqueeze(0).to(device)
        out = model(x)
        present_prob = torch.sigmoid(out["present_logits"])[0, 0].cpu().numpy()
        offsets = out["offsets"][0].cpu().numpy()
        type_logits = out["type_logits"][0].cpu().numpy()

    peaks = (present_prob == maximum_filter(present_prob, size=nms_size)) & (present_prob > threshold)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []
    # Watershed-shape footprints from real-image evidence:
    #   markers = predicted centroid pixels (one label per peak)
    #   barrier = real membrane signal (the ridge), so cells stop growing
    #             at membrane boundaries
    #   restrict expansion to extracellular pixels (don't overwrite existing cells)
    extracellular = (cell_label == 0)
    seeds = np.zeros((H, W), dtype=np.int32)
    centroids: list[tuple[float, float, float, int, int]] = []  # (cy, cx, score, peak_y, peak_x)
    for k, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()), start=1):
        cy = float(y) + float(offsets[0, y, x])
        cx = float(x) + float(offsets[1, y, x])
        cyi = int(round(cy)); cxi = int(round(cx))
        if not (0 <= cyi < H and 0 <= cxi < W):
            continue
        if not extracellular[cyi, cxi]:
            continue  # predicted centroid landed inside an existing cell — drop
        seeds[cyi, cxi] = k
        centroids.append((cy, cx, float(present_prob[y, x]), int(y), int(x)))
    if not centroids:
        return []
    # Watershed on real_membrane (high values = barrier). Constrain growth to
    # extracellular pixels so we never overwrite real 10x cells. This produces
    # one labeled region per centroid, shaped by membrane ridge evidence.
    try:
        from skimage.segmentation import watershed as _watershed
        from skimage.morphology import dilation as _dilation, disk as _disk
        # Dilate seeds slightly so the watershed has a non-trivial source region.
        seeds_dil = _dilation(seeds, _disk(1))
        labels = _watershed(real_membrane, markers=seeds_dil, mask=extracellular).astype(np.int32)
    except Exception:  # noqa: BLE001 - fallback to disks if skimage unavailable
        labels = np.zeros_like(seeds)
        yy, xx = np.ogrid[:H, :W]
        for k, (cy, cx, _, _, _) in enumerate(centroids, start=1):
            cyi = int(round(cy)); cxi = int(round(cx))
            disk = (yy - cyi)**2 + (xx - cxi)**2 <= cell_radius_px**2
            disk &= extracellular & (labels == 0)
            labels[disk] = k

    # Note: Hybrid offset-vote extension was tested (extends watershed with
    # extracellular pixels voting for the cand centroid). Raw IoU win (some
    # tiles 0.03 -> 0.89) was lost at multi-stage level because merge already
    # consolidates similar regions. Neutral effect on bench. Skipped.
    # Shape head: if provided, replace watershed comp with learned mask
    # prediction. Trained on (real, ablated_masks, present, offsets) ->
    # cell footprint. Val IoU 0.817.
    if shape_head is not None:
        # Build offsets and polya tensors once
        polya_for_head = real_polya if real_polya is not None else real_dapi
        real3 = np.stack([
            real_dapi.astype(np.float32),
            real_membrane.astype(np.float32),
            np.asarray(polya_for_head, dtype=np.float32),
        ], axis=0)

    candidates: list[CandidateCell] = []
    for k, (cy, cx, score, peak_y, peak_x) in enumerate(centroids, start=1):
        # Gate shape head: it was trained with scaffold context. When local
        # scaffold density is very low (≈100% drop), fall back to watershed —
        # the shape head over-predicts area in that OOD regime.
        local_scaffold = 0.0
        if shape_head is not None:
            H_, W_ = cell_label.shape
            y0g, y1g = max(0, peak_y-32), min(H_, peak_y+32)
            x0g, x1g = max(0, peak_x-32), min(W_, peak_x+32)
            local_scaffold = float((cell_label[y0g:y1g, x0g:x1g] > 0).mean())
        if shape_head is not None and local_scaffold > 0.05:
            shape_pred = shape_head.predict_mask(
                real=real3,
                ablated_cell_mask=(cell_label > 0).astype(np.float32),
                ablated_nuc_mask=(nucleus_label > 0).astype(np.float32),
                present_map=present_prob,
                offset_y_map=offsets[0],
                offset_x_map=offsets[1],
                peak_y=peak_y, peak_x=peak_x,
                threshold=0.7,
            )
            comp = shape_pred & extracellular
        else:
            comp = labels == k
        if not np.any(comp):
            continue
        # Bound the maximum cell size — guard against runaway watershed when
        # ridge evidence is weak. ~600 px ≈ 27 µm² which is huge for a single cell.
        if int(comp.sum()) > 1200:
            cyi = int(round(cy)); cxi = int(round(cx))
            yy, xx = np.ogrid[:H, :W]
            disk = (yy - cyi)**2 + (xx - cxi)**2 <= (2 * cell_radius_px)**2
            comp = comp & disk
            if not np.any(comp):
                continue
        ys_c, xs_c = np.where(comp)
        cell_pixels = np.stack([ys_c, xs_c], axis=1)
        # Nucleus footprint: derive from real DAPI evidence. Three changes
        # from the previous version (which used a 0.5*peak threshold INSIDE
        # the watershed footprint):
        #   1. Threshold lowered to 0.35 * tile_DAPI_peak (or 0.5 * cell
        #      peak, whichever is lower) so soft nucleus edges are kept.
        #   2. Search NOT confined to cell footprint — use a wider disk
        #      around the proposer's predicted centroid (offset-corrected),
        #      so cells with a misaligned watershed still find their nucleus.
        #   3. After getting a DAPI-bright connected component near the
        #      predicted centroid, clip to be NOT inside other 10x cells
        #      (so we don't steal nuclei from existing cells).
        cyi = int(round(cy)); cxi = int(round(cx))
        cyi = int(np.clip(cyi, 0, H - 1)); cxi = int(np.clip(cxi, 0, W - 1))
        nucleus_pixels: np.ndarray | None = None
        # Search region: disk of radius ~10 px around predicted centroid.
        # That covers a typical Xenium nucleus diameter.
        SEARCH_RADIUS = 12
        yy, xx = np.ogrid[:H, :W]
        search_disk = ((yy - cyi)**2 + (xx - cxi)**2 <= SEARCH_RADIUS**2)
        # Forbid pixels currently labeled as OTHER cells (not this one).
        # Since `cell_label` is the ablated input, those pixels are 10x
        # cells we shouldn't grab nucleus from.
        allowed = (cell_label == 0) | comp
        search_region = search_disk & allowed
        if search_region.any():
            from scipy.ndimage import label as _cc, binary_dilation as _dil
            dapi_in_search = real_dapi[search_region]
            # Adaptive threshold: lower of (0.5 * peak in search) and (0.35 * tile peak)
            peak_local = float(dapi_in_search.max())
            peak_tile = float(real_dapi.max())
            nuc_thr = min(0.5 * peak_local, 0.35 * peak_tile)
            nuc_thr = max(nuc_thr, 0.10)
            nuc_candidate = (real_dapi >= nuc_thr) & search_region
            if nuc_candidate.any():
                cc_arr, _ = _cc(nuc_candidate)
                center_cc = int(cc_arr[cyi, cxi])
                if center_cc > 0:
                    nuc_mask_final = (cc_arr == center_cc)
                else:
                    # Centroid landed off-nucleus — pick the closest cc
                    # to the centroid (not the largest)
                    nz_idx = np.unique(cc_arr); nz_idx = nz_idx[nz_idx > 0]
                    best_d = 1e9; best_cc_id = 0
                    for cid in nz_idx:
                        ys_, xs_ = np.where(cc_arr == cid)
                        d = (float(ys_.mean()) - cyi)**2 + (float(xs_.mean()) - cxi)**2
                        if d < best_d:
                            best_d = d; best_cc_id = int(cid)
                    nuc_mask_final = (cc_arr == best_cc_id) if best_cc_id > 0 else None
                if nuc_mask_final is not None and nuc_mask_final.any():
                    # Dilate by 1 px to soften edges
                    nuc_mask_final = _dil(nuc_mask_final, iterations=1) & allowed
                    ys_nuc, xs_nuc = np.where(nuc_mask_final)
                    nucleus_pixels = np.stack([ys_nuc, xs_nuc], axis=1)
        # Fallback: small disk if DAPI didn't give a usable nucleus
        if nucleus_pixels is None:
            nuc_disk = ((yy - cyi)**2 + (xx - cxi)**2 <= nucleus_radius_px**2) & allowed
            if np.any(nuc_disk):
                ys_nuc, xs_nuc = np.where(nuc_disk)
                nucleus_pixels = np.stack([ys_nuc, xs_nuc], axis=1)
        candidates.append(CandidateCell(
            cy=cy, cx=cx,
            radius_px=float(np.sqrt(comp.sum() / np.pi)),
            cell_pixels=cell_pixels,
            nucleus_pixels=nucleus_pixels,
            dapi_peak_score=score,
            ridge_enclosure_score=0.0,
            provenance_methods=("E-prime-learned",),
        ))
    # Annotate with predicted type via setattr (CandidateCell is a frozen-style dataclass)
    for cand, lbl in zip(
        candidates,
        [int(np.argmax(type_logits[:, int(round(c.cy)), int(round(c.cx))])) for c in candidates],
    ):
        # type 0 = "no cell" / unknown — convert to type name when available
        cand_type_name = type_names[lbl] if 0 <= lbl < len(type_names) else "unknown"
        # Stash on provenance_methods or as a side-channel attribute
        cand.provenance_methods = tuple(list(cand.provenance_methods) + [f"type:{cand_type_name}"])
    return candidates


def propose_latent_cells_iterative(
    real_dapi: np.ndarray,
    real_membrane: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    proposer: dict[str, Any],
    *,
    thresholds: tuple[float, ...] = (0.7, 0.5, 0.3),
    nms_size: int = 21,
    dedup_distance_px: float = 6.0,
    **kw: Any,
) -> list[CandidateCell]:
    """Iterative propose -> condition -> propose decode (reseg.md sec 5, C4a).

    Wraps the single-shot :func:`propose_latent_cells_learned` in a most-certain-
    first cascade: at each (descending) threshold, decode candidates, then
    **rasterize the accepted cells back into the scaffold** (``cell_label`` /
    ``nucleus_label``) so the next pass sees them as context and the
    extracellular guard stops it re-proposing them. This makes recovery joint —
    cells tile space and don't double-count — and progressively rebuilds the
    scaffold inside a cleared cluster, the regime where a single pass plateaus.

    Returns the union of accepted candidates (deduplicated by centroid).
    """
    scaffold_cl = cell_label.astype(np.int32, copy=True)
    scaffold_nl = nucleus_label.astype(np.int32, copy=True)
    next_lab = int(scaffold_cl.max()) + 1
    accepted: list[CandidateCell] = []
    centers = np.zeros((0, 2), dtype=np.float32)

    for th in thresholds:
        cands = propose_latent_cells_learned(
            real_dapi, real_membrane, scaffold_cl, scaffold_nl, proposer,
            threshold=th, nms_size=nms_size, **kw)
        for cand in cands:
            if centers.shape[0] > 0:
                d = np.sqrt(((centers[:, 0] - cand.cy) ** 2 +
                             (centers[:, 1] - cand.cx) ** 2))
                if float(d.min()) < dedup_distance_px:
                    continue  # already have a cell here
            # accept + write into the scaffold so later passes condition on it
            px = cand.cell_pixels
            if px is None or len(px) == 0:
                continue
            ys = px[:, 0].astype(np.int64); xs = px[:, 1].astype(np.int64)
            scaffold_cl[ys, xs] = next_lab
            if cand.nucleus_pixels is not None and len(cand.nucleus_pixels):
                nys = cand.nucleus_pixels[:, 0].astype(np.int64)
                nxs = cand.nucleus_pixels[:, 1].astype(np.int64)
                scaffold_nl[nys, nxs] = next_lab
            next_lab += 1
            accepted.append(cand)
            centers = np.vstack([centers, [[cand.cy, cand.cx]]])
    return accepted


def propose_latent_cells_union(
    real_dapi: np.ndarray,
    real_membrane: np.ndarray,
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    proposer: dict[str, Any],
    *,
    dedup_distance_px: float = 8.0,
    threshold: float = 0.7,
    nms_size: int = 21,
) -> list[CandidateCell]:
    """Union of heuristic A1+A2+A3 and learned proposers, deduplicated by centroid distance.

    The heuristic produces well-shaped footprints when ridge evidence exists.
    The learned proposer covers cells the heuristic misses (no DAPI peak,
    no enclosed ridge). When both fire near the same location, prefer the
    heuristic candidate (better footprint quality from real-image evidence).
    """

    from .scene_completion import propose_latent_cells

    heur = propose_latent_cells(real_dapi, real_membrane, cell_label, nucleus_label)
    learned = propose_latent_cells_learned(
        real_dapi, real_membrane, cell_label, nucleus_label,
        proposer, threshold=threshold, nms_size=nms_size,
    )
    if not heur:
        return learned
    if not learned:
        return heur
    h_centers = np.array([(c.cy, c.cx) for c in heur], dtype=np.float32)
    out = list(heur)
    for cand in learned:
        d = np.sqrt((h_centers[:, 0] - cand.cy)**2 + (h_centers[:, 1] - cand.cx)**2)
        if d.min() < dedup_distance_px:
            continue  # heuristic already covers this
        out.append(cand)
    return out
