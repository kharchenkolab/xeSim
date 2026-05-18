"""Phase 4 — transcript-based latent cell completion.

Given a `MechanisticScene` and the crop's bundle bbox, find molecules that
are *not* explained by any segmented cell, cluster them by biology factor +
spatial density, and propose new cells with inferred type. Each proposal is
emitted as a `CandidateCell` so it can flow into the existing scene-
completion / rendering machinery.

This is the "headline win" of the transcripts integration: cells that 10x
segmentation drops out can be recovered from their orphan transcripts.

Within-bundle scope: we read molecules from cellAdmix's per-fit
`molecules.parquet` (via `XesimModel.molecule_factor_assignments`), which
already has each molecule's factor label and host-cell assignment from the
canonical fit. No re-projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .scene_completion import CandidateCell


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Tiny DBSCAN — returns cluster labels (-1 for noise). O(n log n) with KD-tree."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, r=eps)
    core_mask = np.array([len(nb) >= min_samples for nb in neighbors])
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not core_mask[i]:
            continue
        # BFS over core points reachable from i
        stack = [i]
        while stack:
            j = stack.pop()
            if labels[j] != -1:
                continue
            labels[j] = cluster_id
            if core_mask[j]:
                for k in neighbors[j]:
                    if labels[k] == -1:
                        stack.append(k)
        cluster_id += 1
    return labels

if TYPE_CHECKING:
    from .mechanistic_scene import MechanisticScene
    from .model import XesimModel


@dataclass(frozen=True)
class CropGeometry:
    """Crop's mapping between bundle-µm coordinates and scene pixel coordinates."""
    xmin_um: float
    ymin_um: float
    pixel_size_um: float
    image_size: int

    @classmethod
    def from_manifest_crop(cls, crop: dict, image_size: int = 256) -> "CropGeometry":
        bx = crop["crop_box"]
        # Each canonical crop covers a fixed crop_size_um square; image_size pixels span it
        xspan = float(bx["xmax"]) - float(bx["xmin"])
        return cls(
            xmin_um=float(bx["xmin"]),
            ymin_um=float(bx["ymin"]),
            pixel_size_um=xspan / float(image_size),
            image_size=int(image_size),
        )

    @classmethod
    def from_region(cls, region_bounds_um: tuple[float, float, float, float],
                     pixel_size_um: float, image_size: int) -> "CropGeometry":
        """Construct from explain_region-style bounds (xmin, ymin, xmax, ymax)."""
        xmin, ymin, xmax, ymax = region_bounds_um
        return cls(xmin_um=float(xmin), ymin_um=float(ymin),
                    pixel_size_um=float(pixel_size_um),
                    image_size=int(image_size))

    def molecules_in_crop(self, mol_df) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Return (px, py, mask_in_crop) for molecules whose (x,y) µm coords fall
        within the crop bounds. Pixel coords are float — caller can floor for indexing."""
        xs = mol_df["x"].to_numpy(dtype=np.float64)
        ys = mol_df["y"].to_numpy(dtype=np.float64)
        xmax = self.xmin_um + self.pixel_size_um * self.image_size
        ymax = self.ymin_um + self.pixel_size_um * self.image_size
        in_crop = (xs >= self.xmin_um) & (xs < xmax) & (ys >= self.ymin_um) & (ys < ymax)
        px = (xs - self.xmin_um) / self.pixel_size_um
        py = (ys - self.ymin_um) / self.pixel_size_um
        return px, py, in_crop


# ---------------------------------------------------------------------------


def _circle_mask(cy: float, cx: float, radius_px: float, h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius_px ** 2


def propose_transcript_cells(
    model: "XesimModel",
    scene: "MechanisticScene",
    crop: dict[str, Any] | None = None,
    *,
    geom: "CropGeometry | None" = None,
    min_cluster_molecules: int = 3,
    explained_dilation_um: float = 1.0,
    cluster_eps_um: float = 3.5,
    rng: np.random.Generator | None = None,
    use_shape_exemplars: bool = True,
) -> tuple[list[CandidateCell], dict[str, Any]]:
    """Propose new cells from orphan transcripts in a canonical crop.

    Strategy:
      1. Pull all biology-factor-assigned molecules in this crop.
      2. Drop molecules whose pixel falls inside (or within
         `explained_dilation_um`) of an existing segmented cell — those
         are "explained". Keep the rest as orphans.
      3. DBSCAN orphans in µm-space (eps=`cluster_eps_um`,
         min_samples=`min_cluster_molecules`). Don't cluster per-factor:
         per-factor counts are too sparse on a 256-px Xenium tile, and
         the within-cell factor mix is itself meaningful (admixture).
      4. For each cluster: assign type from the modal biology factor's
         source label, sample a shape exemplar, place at centroid with
         a radius derived from cluster spread.

    Parameters
    ----------
    min_cluster_molecules
        DBSCAN min_samples and the minimum cluster size for a proposal.
    explained_dilation_um
        Margin to dilate existing cells before deciding which molecules
        are orphans.
    cluster_eps_um
        DBSCAN eps in µm.
    use_shape_exemplars
        If True and the model has exemplars, sample a real shape mask
        of the inferred type. Otherwise use a circular mask.
    """
    priors = model._require_transcripts()
    if geom is None:
        if crop is None:
            raise ValueError("Must pass either `crop` (manifest dict) or `geom` (CropGeometry)")
        geom = CropGeometry.from_manifest_crop(crop, image_size=scene.image_shape[0])
    rng = rng or np.random.default_rng(0)

    mol = model.molecule_factor_assignments(
        columns=["x", "y", "factor", "factor_margin", "cell_idx"],
        with_gene_names=False,
        include_unassigned=True,
    )
    px, py, in_crop = geom.molecules_in_crop(mol)
    mol_in = mol[in_crop].copy()
    mol_in["_px"] = px[in_crop]
    mol_in["_py"] = py[in_crop]
    mol_in["_x_um"] = mol_in["x"].astype(float)
    mol_in["_y_um"] = mol_in["y"].astype(float)

    biology = set(int(f) for f in priors["factor_signatures"]["biology_factors"])
    mol_in = mol_in[mol_in["factor"].isin(biology)]

    cell_label = scene.cell_label
    explained_mask = cell_label > 0
    dilation_iters = max(0, int(round(explained_dilation_um / geom.pixel_size_um)))
    if dilation_iters > 0:
        explained_mask = binary_dilation(explained_mask, iterations=dilation_iters)

    px_floor = np.clip(mol_in["_px"].astype(np.int64), 0, geom.image_size - 1)
    py_floor = np.clip(mol_in["_py"].astype(np.int64), 0, geom.image_size - 1)
    is_in_cell = explained_mask[py_floor, px_floor]
    orphan = mol_in[~is_in_cell].copy().reset_index(drop=True)

    factor_to_type: dict[int, str] = {}
    for f_str, src in priors.get("factor_source_labels", {}).items():
        t = src.get("type")
        if t:
            factor_to_type[int(f_str)] = t

    sampler = None
    if use_shape_exemplars:
        try:
            from .cell_shape_exemplar import CellShapeExemplarSampler
            if model.paths.exemplars.exists():
                sampler = CellShapeExemplarSampler(model.paths.exemplars)
        except Exception:
            sampler = None

    h, w = geom.image_size, geom.image_size
    candidates: list[CandidateCell] = []
    diagnostics: dict[str, Any] = {
        "n_molecules_in_crop": int(len(mol_in)),
        "n_orphan_molecules": int(len(orphan)),
        "explained_dilation_px": int(dilation_iters),
        "cluster_eps_um": float(cluster_eps_um),
        "min_cluster_molecules": int(min_cluster_molecules),
    }
    proposed_coverage = np.zeros_like(cell_label, dtype=bool)

    if len(orphan) < min_cluster_molecules:
        diagnostics["n_clusters"] = 0
        diagnostics["n_candidates"] = 0
        return candidates, diagnostics

    coords = orphan[["_x_um", "_y_um"]].to_numpy()
    cluster_labels = _dbscan(coords, eps=cluster_eps_um,
                                min_samples=min_cluster_molecules)
    n_clusters = int((np.unique(cluster_labels) >= 0).sum())
    diagnostics["n_clusters"] = n_clusters

    factor_arr = orphan["factor"].to_numpy()
    px_arr = orphan["_px"].to_numpy()
    py_arr = orphan["_py"].to_numpy()

    for cid in range(n_clusters):
        sel = cluster_labels == cid
        if sel.sum() < min_cluster_molecules:
            continue
        px_c = px_arr[sel]; py_c = py_arr[sel]
        cy = float(np.mean(py_c)); cx = float(np.mean(px_c))
        sd = max(float(np.std(py_c)), float(np.std(px_c)), 1.0)
        radius_px = float(max(3.0, min(2.0 * sd, 0.8 * h / 8)))

        # Modal factor → cell type
        f_counts = np.bincount(factor_arr[sel].astype(np.int64))
        modal_f = int(np.argmax(f_counts))
        cell_type = factor_to_type.get(modal_f, "unknown")

        cell_mask_full = None
        if sampler is not None:
            try:
                cell_mask, _nuc, _info = sampler.sample(
                    rng, type_name=cell_type, target_area_um2=None,
                )
                cell_mask_full = _place_mask(
                    cell_mask, cy=cy, cx=cx, h=h, w=w,
                    target_radius_px=radius_px,
                )
            except Exception:
                cell_mask_full = None
        if cell_mask_full is None:
            cell_mask_full = _circle_mask(cy, cx, radius_px, h, w)

        cell_mask_full &= ~explained_mask & ~proposed_coverage
        if cell_mask_full.sum() < max(8, min_cluster_molecules):
            continue

        cell_pixels_yx = np.argwhere(cell_mask_full)
        candidates.append(CandidateCell(
            cy=cy, cx=cx, radius_px=radius_px,
            cell_pixels=cell_pixels_yx,
            nucleus_pixels=None,
            dapi_peak_score=0.0,
            ridge_enclosure_score=float(sel.sum()),
            provenance_methods=("transcripts", f"factor_{modal_f}", cell_type),
        ))
        proposed_coverage |= cell_mask_full

    diagnostics["n_candidates"] = len(candidates)
    return candidates, diagnostics


def propose_hybrid_cells(
    model: "XesimModel",
    scene: "MechanisticScene",
    crop: dict[str, Any],
    real_dapi: np.ndarray,
    real_membrane: np.ndarray,
    *,
    match_radius_px: int = 4,
    rng: np.random.Generator | None = None,
    use_shape_exemplars: bool = True,
    image_proposer_kwargs: dict[str, Any] | None = None,
    **tx_kwargs,
) -> tuple[list[CandidateCell], dict[str, Any]]:
    """Hybrid image+transcript proposer (Phase 4 v2).

    Strategy: take the **deduped union** of image-based and
    transcript-based candidates. Image candidates have better shapes
    (ridge components + DAPI peaks) and find some missed cells the
    transcripts cluster cannot resolve; transcript candidates find some
    cells the image cannot resolve (low-DAPI cytoplasm-only cells, or
    cells whose membrane is masked by neighbors). The union captures
    both.

    Dedup rule: if a transcript candidate's centroid is within
    `match_radius_px` of an image candidate's centroid, the image
    candidate wins (it has the better shape).

    Returns (candidates, diagnostics).
    """
    from .scene_completion import propose_latent_cells

    rng = rng or np.random.default_rng(0)
    image_kwargs = image_proposer_kwargs or {}
    image_cands = propose_latent_cells(
        real_dapi.astype(np.float32), real_membrane.astype(np.float32),
        scene.cell_label, scene.nucleus_label,
        **image_kwargs,
    )
    tx_cands, tx_diag = propose_transcript_cells(
        model, scene, crop, rng=rng, use_shape_exemplars=use_shape_exemplars,
        **tx_kwargs,
    )

    cell_label = scene.cell_label
    h, w = cell_label.shape
    existing_mask = cell_label > 0
    proposed_coverage = np.zeros_like(cell_label, dtype=bool)
    out_cands: list[CandidateCell] = []

    def _take(cand: CandidateCell, *, extra_prov: tuple = ()) -> bool:
        nonlocal proposed_coverage
        cm = np.zeros((h, w), dtype=bool)
        cm[cand.cell_pixels[:, 0], cand.cell_pixels[:, 1]] = True
        cm &= ~existing_mask & ~proposed_coverage
        if cm.sum() < 8:
            return False
        new_cand = CandidateCell(
            cy=cand.cy, cx=cand.cx, radius_px=cand.radius_px,
            cell_pixels=np.argwhere(cm),
            nucleus_pixels=cand.nucleus_pixels,
            dapi_peak_score=cand.dapi_peak_score,
            ridge_enclosure_score=cand.ridge_enclosure_score,
            provenance_methods=cand.provenance_methods + extra_prov,
        )
        out_cands.append(new_cand)
        proposed_coverage |= cm
        return True

    # 1. Take all image candidates first (they have refined shapes).
    image_centroids: list[tuple[float, float]] = []
    for cand in image_cands:
        if _take(cand):
            image_centroids.append((cand.cy, cand.cx))

    # 2. Add transcript candidates whose centroid is NOT near any image
    #    candidate (the unique-to-tx slice).
    n_tx_added = 0; n_tx_skipped_duplicate = 0
    for tx_cand in tx_cands:
        is_dup = False
        for icy, icx in image_centroids:
            if (tx_cand.cy - icy) ** 2 + (tx_cand.cx - icx) ** 2 \
                    <= match_radius_px ** 2:
                is_dup = True; break
        if is_dup:
            n_tx_skipped_duplicate += 1
            continue
        if _take(tx_cand, extra_prov=("hybrid_union",)):
            n_tx_added += 1

    diagnostics = {
        **tx_diag,
        "n_image_proposed": len(image_cands),
        "n_tx_proposed": len(tx_cands),
        "n_tx_added_after_dedup": n_tx_added,
        "n_tx_skipped_duplicate": n_tx_skipped_duplicate,
        "n_hybrid_candidates": len(out_cands),
    }
    return out_cands, diagnostics


def _place_mask(
    mask_patch: np.ndarray,
    *,
    cy: float, cx: float, h: int, w: int,
    target_radius_px: float,
) -> np.ndarray:
    """Center the binary patch in a (h, w) canvas, scaled so its effective
    radius matches `target_radius_px`. Simple nearest-neighbor rescale."""
    if mask_patch.sum() == 0:
        return _circle_mask(cy, cx, target_radius_px, h, w)
    # Compute the patch's effective radius from its area
    patch_size = mask_patch.shape[0]
    patch_radius = max(1.0, float(np.sqrt(mask_patch.sum() / np.pi)))
    scale = max(0.5, min(2.5, target_radius_px / patch_radius))
    new_size = max(3, int(round(patch_size * scale)))
    # Nearest-neighbor resize
    from scipy.ndimage import zoom
    zoom_factor = new_size / patch_size
    resized = zoom(mask_patch.astype(np.float32), zoom_factor, order=0) > 0.5
    ny, nx = resized.shape
    canvas = np.zeros((h, w), dtype=bool)
    top = int(round(cy - ny / 2))
    left = int(round(cx - nx / 2))
    # Clip into canvas
    y0 = max(0, top); y1 = min(h, top + ny)
    x0 = max(0, left); x1 = min(w, left + nx)
    if y1 <= y0 or x1 <= x0:
        return canvas
    canvas[y0:y1, x0:x1] = resized[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas
