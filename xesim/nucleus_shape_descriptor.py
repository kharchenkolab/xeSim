"""Per-cell nucleus shape descriptor — small deterministic per-cell features
that capture the structural information the renderer learned from 10x
nucleus annotations.

After we switched to DAPI-anchored nucleus re-derivation (which optimizes
pixel-level alignment with real DAPI), the per-cell variation in nucleus
*shape* — irregularity, eccentricity, relative size — got smoothed away.
The renderer's expressive range collapsed because all nucleus inputs
became more uniform.

This module recovers three scalars per cell from the segmentation
itself (no real-image features needed):

    area_ratio   = nuc_area / cell_area   ∈ [0, 1]
    circularity  = 4π · nuc_area / nuc_perimeter²   ∈ (0, 1]   (1 = disk)
    eccentricity = from second-central-moments   ∈ [0, 1)      (0 = disk)

For cells with no nucleus (sectioning state where 10x decided the nucleus
is absent / out-of-plane), the descriptor is all zeros — that itself is
information ("no nucleus here").

The 3-vector is meant to be concatenated to the appearance latent and fed
through the per-cell LUT injection into the v37b U-Net. Training: v37b
sees an expanded 15-dim per-cell conditioning. At inference, the
descriptor is recomputed from whichever nucleus_label the pipeline
produced (10x for survivors, DAPI-rederived for added cells).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def cell_shape_descriptors(cell_label: np.ndarray,
                            nucleus_label: np.ndarray) -> dict[int, np.ndarray]:
    """Compute 3-vector per cell. Returns {cell_label: np.array([3], float32)}.

    Cells with no nucleus get zeros. The function is deterministic and
    does not depend on real-image content.
    """
    out: dict[int, np.ndarray] = {}
    cell_labels = np.unique(cell_label)
    cell_labels = cell_labels[cell_labels > 0]

    for L in cell_labels:
        cm = cell_label == int(L)
        cell_area = float(cm.sum())
        if cell_area < 1.0:
            continue
        nm = (nucleus_label == int(L)) & cm
        nuc_area = float(nm.sum())
        if nuc_area < 4.0:
            # No / negligible nucleus — emit zeros (encodes "no nucleus")
            out[int(L)] = np.zeros(3, dtype=np.float32)
            continue

        area_ratio = nuc_area / cell_area

        # Perimeter via boundary pixels of the nucleus mask
        eroded = binary_erosion(nm)
        boundary = nm & ~eroded
        perimeter = float(boundary.sum())
        circularity = (4.0 * np.pi * nuc_area) / max(perimeter * perimeter, 1.0)
        circularity = float(np.clip(circularity, 0.0, 1.0))

        # Eccentricity from second central moments of the nucleus mask
        ys, xs = np.where(nm)
        cy = ys.mean(); cx = xs.mean()
        dy = ys - cy; dx = xs - cx
        m20 = float((dy * dy).mean())
        m02 = float((dx * dx).mean())
        m11 = float((dy * dx).mean())
        # eigenvalues of [[m20, m11], [m11, m02]]
        tr = m20 + m02
        det = m20 * m02 - m11 * m11
        disc = max(tr * tr / 4.0 - det, 0.0)
        lam1 = tr / 2.0 + np.sqrt(disc)
        lam2 = tr / 2.0 - np.sqrt(disc)
        if lam1 > 1e-6:
            eccentricity = float(np.sqrt(max(1.0 - lam2 / lam1, 0.0)))
        else:
            eccentricity = 0.0

        out[int(L)] = np.array([area_ratio, circularity, eccentricity], dtype=np.float32)

    return out


def descriptor_array(cell_labels_present: np.ndarray,
                      descriptors: dict[int, np.ndarray]) -> np.ndarray:
    """Convert a {label: 3-vec} dict to an ordered (N, 3) array following
    the same ordering as `cell_labels_present` (the order returned by
    `np.unique(cell_label)[1:]`)."""
    out = np.zeros((len(cell_labels_present), 3), dtype=np.float32)
    for i, L in enumerate(cell_labels_present):
        v = descriptors.get(int(L))
        if v is not None:
            out[i] = v
    return out
