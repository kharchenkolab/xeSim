"""Planted-error operators for the correction benchmark (misc/reseg.md, S2).

Deliberately corrupt a 10x label map so we can test whether a method REPAIRS
the error (correction) vs reproduces it — isolated from circularity because we
planted it and know the right answer. Operators:

- ``merge_cells``  : two adjacent cells -> one label (simulates under-segmentation)
- ``split_cell``   : one cell -> two labels split by a line through its centroid
                     (simulates over-segmentation)
- ``jitter_boundary``: erode/dilate one cell's footprint (wrong contour)
"""
from __future__ import annotations

import numpy as np

from .metrics import _label_centroids


def nearest_pair(cl: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    """A random close (adjacent) pair of cell labels, for a merge target."""
    labs, cents, _ = _label_centroids(cl)
    if len(labs) < 2:
        return None
    seed = int(rng.integers(len(labs)))
    d = np.sqrt(((cents - cents[seed]) ** 2).sum(1)); d[seed] = 1e9
    return int(labs[seed]), int(labs[int(d.argmin())])


def merge_cells(cl: np.ndarray, nl: np.ndarray, a: int, b: int):
    """Relabel cell ``b`` into ``a`` (one cell where 10x had two)."""
    clm = cl.copy(); nlm = nl.copy()
    clm[clm == b] = a; nlm[nlm == b] = a
    return clm, nlm


def split_cell(cl: np.ndarray, nl: np.ndarray, a: int,
               rng: np.random.Generator, new_label: int):
    """Split cell ``a`` into two labels by a random line through its centroid
    (two cells where 10x had one). Returns (cl', nl', new_label)."""
    ys, xs = np.where(cl == a)
    if len(ys) < 8:
        return cl.copy(), nl.copy(), None
    cy, cx = ys.mean(), xs.mean()
    theta = float(rng.uniform(0, np.pi))
    nx, ny = np.cos(theta), np.sin(theta)        # normal of the split line
    side = ((xs - cx) * nx + (ys - cy) * ny) > 0
    clm = cl.copy(); nlm = nl.copy()
    clm[ys[side], xs[side]] = new_label
    # split the nucleus the same way
    nys, nxs = np.where(nl == a)
    if len(nys):
        nside = ((nxs - cx) * nx + (nys - cy) * ny) > 0
        nlm[nys[nside], nxs[nside]] = new_label
    return clm, nlm, new_label


def jitter_boundary(cl: np.ndarray, nl: np.ndarray, a: int,
                    rng: np.random.Generator, px: int = 2):
    """Erode or dilate cell ``a``'s footprint by ``px`` (wrong contour), without
    overwriting other cells (dilation only grows into background)."""
    from scipy.ndimage import binary_erosion, binary_dilation
    mask = cl == a
    clm = cl.copy()
    if rng.random() < 0.5:
        new = binary_erosion(mask, iterations=px)
        clm[mask & ~new] = 0
    else:
        grown = binary_dilation(mask, iterations=px) & (cl == 0)
        clm[grown] = a
    return clm, nl.copy()


__all__ = ["nearest_pair", "merge_cells", "split_cell", "jitter_boundary"]
