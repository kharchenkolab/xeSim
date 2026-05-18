"""Segmentation comparison metrics.

Given two label images on the same canvas, score how well they agree.
Used by Phase 3 to compare:

  - Cellpose-on-real vs 10x's published cell_boundaries (real baseline)
  - Cellpose-on-synth vs the ground-truth labels we baked in

Matching strategy:
  1. Compute centroids of every nonzero label in each image.
  2. Hungarian-match centroids with a max-distance cutoff (≈5 µm).
  3. For each matched pair, compute IoU between the two cell masks.
  4. Summary stats: match rate, spurious rate, median IoU, count delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchedPair:
    label_a: int
    label_b: int
    centroid_dist_um: float
    iou: float
    area_a_px: int
    area_b_px: int


@dataclass
class MatchSummary:
    n_cells_a: int
    n_cells_b: int
    n_matched: int
    match_rate_a: float        # matched / n_cells_a
    match_rate_b: float        # matched / n_cells_b
    spurious_rate_b: float     # (n_cells_b - matched) / n_cells_b
    median_iou: float
    mean_iou: float
    iou_at_50: float           # fraction of pairs with iou >= 0.5
    iou_at_75: float           # fraction of pairs with iou >= 0.75
    count_delta: int           # n_cells_b - n_cells_a

    def to_dict(self) -> dict:
        return {k: (None if v is None else (float(v) if isinstance(v, float)
                                                    else int(v)))
                for k, v in self.__dict__.items()}


def _label_centroids(label_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, centroids_yx_px, areas_px) for nonzero labels."""
    labs, inv, counts = np.unique(label_image.ravel(), return_inverse=True,
                                       return_counts=True)
    # filter background
    keep = labs > 0
    labs = labs[keep]
    counts = counts[keep]
    if len(labs) == 0:
        return labs, np.zeros((0, 2)), counts
    # sum-y, sum-x per label
    H, W = label_image.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # use bincount keyed by index back into labs
    # rebuild the label→index for nonzero
    label_to_idx = {int(l): i for i, l in enumerate(labs)}
    flat = label_image.ravel()
    mask = flat > 0
    idx = np.array([label_to_idx[int(v)] for v in flat[mask]], dtype=np.int64)
    sum_y = np.bincount(idx, weights=yy.ravel()[mask], minlength=len(labs))
    sum_x = np.bincount(idx, weights=xx.ravel()[mask], minlength=len(labs))
    centroids = np.stack([sum_y / counts, sum_x / counts], axis=1)
    return labs, centroids, counts


def match_segmentations(label_a: np.ndarray, label_b: np.ndarray, *,
                          pixel_size_um: float,
                          max_centroid_um: float = 5.0,
                          ) -> tuple[list[MatchedPair], MatchSummary]:
    """Compare two label images defined on the same canvas.

    Hungarian-matches per-cell centroids with a max-distance cutoff.
    Returns (matched_pairs, summary).

    `label_a` is treated as ground truth in the match_rate computation
    (match_rate_a = how many of A's cells got matched). `label_b` is
    the test set.
    """
    if label_a.shape != label_b.shape:
        raise ValueError(f"label_a {label_a.shape} != label_b {label_b.shape}")

    labs_a, ca, areas_a = _label_centroids(label_a)
    labs_b, cb, areas_b = _label_centroids(label_b)

    pairs: list[MatchedPair] = []
    if len(labs_a) > 0 and len(labs_b) > 0:
        # cost = centroid distance in px; infeasible pairs set to very large
        diff = ca[:, None, :] - cb[None, :, :]
        dist_px = np.sqrt((diff ** 2).sum(axis=-1))
        max_dist_px = max_centroid_um / pixel_size_um
        cost = np.where(dist_px <= max_dist_px, dist_px, 1e6)
        ai, bj = linear_sum_assignment(cost)
        for a_i, b_j in zip(ai, bj):
            if cost[a_i, b_j] >= 1e6 - 1:
                continue
            la = int(labs_a[a_i])
            lb = int(labs_b[b_j])
            mask_a = label_a == la
            mask_b = label_b == lb
            inter = int((mask_a & mask_b).sum())
            uni = int((mask_a | mask_b).sum())
            iou = inter / uni if uni > 0 else 0.0
            pairs.append(MatchedPair(
                label_a=la, label_b=lb,
                centroid_dist_um=float(dist_px[a_i, b_j] * pixel_size_um),
                iou=float(iou),
                area_a_px=int(areas_a[a_i]),
                area_b_px=int(areas_b[b_j]),
            ))

    n_a = int(len(labs_a))
    n_b = int(len(labs_b))
    n_m = int(len(pairs))
    ious = np.array([p.iou for p in pairs], dtype=np.float32)
    summary = MatchSummary(
        n_cells_a=n_a, n_cells_b=n_b, n_matched=n_m,
        match_rate_a=(n_m / n_a) if n_a > 0 else 0.0,
        match_rate_b=(n_m / n_b) if n_b > 0 else 0.0,
        spurious_rate_b=((n_b - n_m) / n_b) if n_b > 0 else 0.0,
        median_iou=float(np.median(ious)) if ious.size else 0.0,
        mean_iou=float(ious.mean()) if ious.size else 0.0,
        iou_at_50=float((ious >= 0.5).mean()) if ious.size else 0.0,
        iou_at_75=float((ious >= 0.75).mean()) if ious.size else 0.0,
        count_delta=int(n_b - n_a),
    )
    return pairs, summary


def rasterize_polygons(polys_df: pd.DataFrame, *,
                          bounds_um: tuple[float, float, float, float],
                          pixel_size_um: float,
                          image_shape: tuple[int, int],
                          poly_id_col: str = "cell_id",
                          ) -> np.ndarray:
    """Rasterize a polygon table (e.g. 10x's cell_boundaries.parquet) into
    an int32 label image over the given window.

    `polys_df` columns expected: ``cell_id`` (or ``poly_id_col``),
    ``vertex_x``, ``vertex_y`` in µm, with one row per vertex
    (Xenium's per-vertex layout).
    """
    from skimage.draw import polygon as sk_polygon
    xmin, ymin, _, _ = bounds_um
    H, W = image_shape
    label_image = np.zeros((H, W), dtype=np.int32)
    grouped = polys_df.groupby(poly_id_col, sort=False)
    label_counter = 0
    for poly_id, g in grouped:
        ys = (g["vertex_y"].to_numpy() - ymin) / pixel_size_um
        xs = (g["vertex_x"].to_numpy() - xmin) / pixel_size_um
        if len(ys) < 3:
            continue
        rr, cc = sk_polygon(ys, xs, shape=(H, W))
        if len(rr) == 0:
            continue
        label_counter += 1
        # If overlap exists, give priority to the first label (10x's
        # cells should not overlap in practice)
        empty = label_image[rr, cc] == 0
        label_image[rr[empty], cc[empty]] = label_counter
    return label_image


def crop_to_window(polys_df: pd.DataFrame,
                     bounds_um: tuple[float, float, float, float],
                     *,
                     vx_col: str = "vertex_x",
                     vy_col: str = "vertex_y",
                     id_col: str = "cell_id") -> pd.DataFrame:
    """Filter a vertex-level polygon DF to those whose CENTROID falls in
    `bounds_um`. We keep all vertices of any included polygon so its
    rasterization is correct near borders.
    """
    xmin, ymin, xmax, ymax = bounds_um
    cents = (polys_df.groupby(id_col)
                .agg(cx=(vx_col, "mean"), cy=(vy_col, "mean"))
                .reset_index())
    keep = cents[(cents["cx"] >= xmin) & (cents["cx"] < xmax) &
                 (cents["cy"] >= ymin) & (cents["cy"] < ymax)][id_col]
    return polys_df[polys_df[id_col].isin(keep)]
