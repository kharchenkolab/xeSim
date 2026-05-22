"""Cell-set ablation for drop-and-recover segmentation benchmarks.

Selects a set of cell labels to remove from a 10x segmentation so a recovery
method (heuristic / learned proposer / renderer-in-the-loop) can be scored
against the held-out truth. Single-cell ablation tests gap-filling; cluster and
region ablation test the joint multi-cell regime (the misc/reseg.md motivation)
where cells must be recovered together with no local scaffold to lean on.

The nucleus footprint of an ablated cell is removed wherever it lies inside the
cell's footprint (nucleus is a subset of the cell), matching the convention in
``proposer_corpus.build_proposer_corpus``.
"""
from __future__ import annotations

import numpy as np

from .metrics import _label_centroids


def select_ablation_set(
    cell_label: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "cluster",
    size: int = 4,
) -> set[int]:
    """Pick a set of cell labels to ablate.

    mode:
      - ``single``  : ``size`` independent random cells (gap-filling regime).
      - ``cluster`` : a contiguous clump — a random seed cell plus its
        ``size``-1 nearest neighbours by centroid.
      - ``region``  : every cell whose centroid falls inside the bounding box
        of such a clump, i.e. a fully-cleared contiguous patch (the hardest
        regime: no interior scaffold survives).
    """
    labs, cents, _areas = _label_centroids(cell_label)
    n = len(labs)
    if n == 0:
        return set()
    size = int(max(1, min(size, n)))

    if mode == "single" or size == 1:
        idx = rng.choice(n, size=size, replace=False)
        return {int(labs[i]) for i in idx}

    seed = int(rng.integers(n))
    d = np.sqrt(((cents - cents[seed]) ** 2).sum(axis=1))
    order = np.argsort(d)[:size]

    if mode == "cluster":
        return {int(labs[i]) for i in order}

    if mode == "region":
        sel = cents[order]
        y0, x0 = sel.min(axis=0)
        y1, x1 = sel.max(axis=0)
        inbox = ((cents[:, 0] >= y0) & (cents[:, 0] <= y1) &
                 (cents[:, 1] >= x0) & (cents[:, 1] <= x1))
        return {int(l) for l in labs[inbox]}

    raise ValueError(f"unknown ablation mode: {mode!r}")


def ablate(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    labels_to_remove: set[int] | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell_label, nucleus_label) copies with the given cells removed.

    Nucleus pixels are zeroed inside the ablated cells' footprints (nucleus is
    a subset of the cell), so the recovery method sees the same evidence a
    real missed cell would leave: extracellular morphology, no scaffold."""
    labels = list(labels_to_remove)
    mask = np.isin(cell_label, labels)
    cl = cell_label.copy()
    nl = nucleus_label.copy()
    cl[mask] = 0
    nl[mask] = 0
    return cl, nl


def truth_label(
    cell_label: np.ndarray,
    labels_to_remove: set[int] | list[int],
) -> np.ndarray:
    """Label image holding ONLY the ablated cells, relabelled 1..K.

    This is the ground truth for the recovery match (label_a in
    ``match_segmentations``): the recovery is judged only on whether it puts
    the right cells back where the ablated ones were."""
    labels = sorted(int(l) for l in labels_to_remove)
    out = np.zeros_like(cell_label, dtype=np.int32)
    for new, l in enumerate(labels, start=1):
        out[cell_label == l] = new
    return out, labels  # also return ordered labels so callers can map relabel->original


def local_density_per_100um2(
    cell_label: np.ndarray,
    labels_to_remove: set[int] | list[int],
    pixel_size_um: float,
) -> float:
    """Cells per 100 µm² in the bbox around the ablated set — a context measure
    for stratifying recovery difficulty (dense clusters are harder)."""
    labs, cents, _ = _label_centroids(cell_label)
    if len(labs) == 0:
        return 0.0
    sel_mask = np.isin(labs, list(labels_to_remove))
    if not sel_mask.any():
        return 0.0
    sel = cents[sel_mask]
    pad = 20.0 / pixel_size_um  # ~20µm halo around the set
    y0, x0 = sel.min(axis=0) - pad
    y1, x1 = sel.max(axis=0) + pad
    in_box = ((cents[:, 0] >= y0) & (cents[:, 0] <= y1) &
              (cents[:, 1] >= x0) & (cents[:, 1] <= x1))
    n = int(in_box.sum())
    area_um2 = max((y1 - y0) * (x1 - x0) * pixel_size_um ** 2, 1e-6)
    return 100.0 * n / area_um2


__all__ = ["select_ablation_set", "ablate", "truth_label",
           "local_density_per_100um2"]
