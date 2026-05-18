"""Segmenter abstraction. Cellpose ships first, but Baysor/Stardist/etc.
need the same harness — so define the surface here.

Convention: a ``Segmenter`` consumes a `(C, H, W)` float32 image plus a
parallel list of channel role names (``"nuclear"``, ``"membrane"``,
``"cyto"`` …) and returns a `SegmentationResult` with an integer label
image at the same resolution as the input. 0 = background; labels start
at 1.

The harness (`xesim.segeval.bundle_io`) picks channels by NAME from the
bundle's morphology image and the model manifest's `channel_names`, so
upstream changes in stain order don't break this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


# --------------------------------------------------------------------- types

#: Channel roles understood by segmenters. Add freely; segmenters
#: declare which they use via `Segmenter.required_channels`.
CHANNEL_ROLE_NUCLEAR = "nuclear"
CHANNEL_ROLE_MEMBRANE = "membrane"
CHANNEL_ROLE_CYTO = "cyto"


@dataclass(frozen=True)
class ChannelSpec:
    """Maps a channel-role name to a column index into a `(C, H, W)` stack
    plus the source channel's display name from the bundle/model manifest.

    Built by `bundle_io.build_channel_spec` from a manifest's
    ``channel_names`` table and a role→name mapping.
    """
    role: str          # e.g. "nuclear"
    name: str          # e.g. "DAPI" — name in bundle/manifest
    index: int         # row index into the (C, H, W) stack


@dataclass(frozen=True)
class SegmentationResult:
    """Output of a `Segmenter.segment` call.

    ``label_image`` is an `(H, W)` int32 array. 0 is background; cell
    labels are arbitrary positive integers (not necessarily contiguous,
    but the segmenter is responsible for keeping them unique).

    ``cell_table`` is an optional per-cell pandas-friendly summary
    (centroid_y_px, centroid_x_px, area_px, ...). Some segmenters compute
    it cheaply; others leave it None and the harness derives it from
    `label_image` if needed.

    ``config`` carries the segmenter's runtime configuration (model
    name, channel mapping, threshold params, etc.) so a result is
    self-describing.
    """
    label_image: np.ndarray
    segmenter_name: str
    channels: tuple[ChannelSpec, ...]
    pixel_size_um: float
    cell_table: Any = None    # optional pd.DataFrame; type-loose to keep base.py pandas-free
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        labs = np.unique(self.label_image)
        return int((labs > 0).sum())


@runtime_checkable
class Segmenter(Protocol):
    """Minimal interface a segmenter must implement.

    Implementations may extend with model-specific kwargs in `__init__`
    but the runtime call signature here is the only thing the harness
    depends on.
    """
    name: str

    @property
    def required_channels(self) -> tuple[str, ...]:
        """Channel roles this segmenter needs, in order. e.g.
        ``("nuclear", "membrane")`` for Cellpose."""
        ...

    def segment(self, image: np.ndarray, *,
                  channels: Sequence[ChannelSpec],
                  pixel_size_um: float) -> SegmentationResult:
        """Run the segmenter on `image` (shape ``(C, H, W)``, float32).
        `channels` MUST list at least the segmenter's required roles
        (extras are ignored). Returns a `SegmentationResult`.
        """
        ...


# --------------------------------------------------------------------- helpers


def label_image_to_centroids(label_image: np.ndarray) -> np.ndarray:
    """Compute per-label centroids from an int label image.

    Returns an array of shape (n_cells, 3): ``[label, cy_px, cx_px]``.
    Background label 0 is excluded. Empty labels (no pixels) are omitted.
    """
    labels = np.unique(label_image)
    labels = labels[labels > 0]
    out = np.empty((len(labels), 3), dtype=np.float64)
    for i, lab in enumerate(labels):
        ys, xs = np.where(label_image == lab)
        if len(ys) == 0:
            out[i] = (lab, np.nan, np.nan)
            continue
        out[i] = (lab, ys.mean(), xs.mean())
    return out


def select_channels(
    image: np.ndarray,
    channels: Sequence[ChannelSpec],
    required: Sequence[str],
) -> dict[str, np.ndarray]:
    """Pull the required channel roles out of `image` using the
    `ChannelSpec` mapping. Returns ``{role: (H, W) array}``."""
    by_role = {c.role: c for c in channels}
    missing = [r for r in required if r not in by_role]
    if missing:
        raise ValueError(
            f"Segmenter requires channels {required!r}; "
            f"missing: {missing!r}. Provided: {[c.role for c in channels]!r}"
        )
    out = {}
    for r in required:
        spec = by_role[r]
        out[r] = np.asarray(image[spec.index], dtype=np.float32)
    return out
