"""xeSim segmenter-evaluation harness.

Phase 3 round-trip validation: feed a bundle (real or synth) to an
off-the-shelf segmenter (Cellpose first; Baysor/Stardist later), then
compare the recovered cells to a ground-truth label image.

Public surface kept thin so that swapping segmenters is one import
change.
"""
from .base import (
    CHANNEL_ROLE_CYTO,
    CHANNEL_ROLE_MEMBRANE,
    CHANNEL_ROLE_NUCLEAR,
    ChannelSpec,
    SegmentationResult,
    Segmenter,
    label_image_to_centroids,
    select_channels,
)
from .bundle_io import (
    build_channel_spec,
    bundle_channel_names,
    read_window,
    synth_bundle_channel_stack,
)
from .cellpose_impl import CellposeSegmenter
from .metrics import (
    MatchedPair,
    MatchSummary,
    crop_to_window,
    match_segmentations,
    rasterize_polygons,
)

__all__ = [
    "CHANNEL_ROLE_CYTO",
    "CHANNEL_ROLE_MEMBRANE",
    "CHANNEL_ROLE_NUCLEAR",
    "CellposeSegmenter",
    "ChannelSpec",
    "MatchedPair",
    "MatchSummary",
    "SegmentationResult",
    "Segmenter",
    "build_channel_spec",
    "bundle_channel_names",
    "crop_to_window",
    "label_image_to_centroids",
    "match_segmentations",
    "rasterize_polygons",
    "read_window",
    "select_channels",
    "synth_bundle_channel_stack",
]
