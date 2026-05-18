"""Cellpose SAM (Cellpose 4.x) implementation of the `Segmenter`
protocol.

CellposeSAM is a single multi-purpose model loaded lazily on first
``segment`` call. Default ``diameter`` is left to Cellpose's auto-sizer;
callers can override via ``CellposeSegmenter(diameter=15.0)``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import (
    CHANNEL_ROLE_MEMBRANE,
    CHANNEL_ROLE_NUCLEAR,
    ChannelSpec,
    SegmentationResult,
    select_channels,
)


class CellposeSegmenter:
    """CellposeSAM-backed segmenter.

    ``required_channels = (nuclear, membrane)``. The wrapper packs the
    two channels into Cellpose's expected ``[chan_cyto, chan_nuc]``
    interleaving (channel axis = 0) and asks Cellpose to use both. Auto
    diameter unless ``diameter`` is set explicitly.
    """
    name = "cellpose-sam"

    def __init__(self, *, gpu: bool = True, diameter: float | None = None,
                  flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                  min_size: int = 15, model_path: str | None = None) -> None:
        self.gpu = gpu
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.model_path = model_path
        self._model = None   # lazy

    @property
    def required_channels(self) -> tuple[str, ...]:
        return (CHANNEL_ROLE_NUCLEAR, CHANNEL_ROLE_MEMBRANE)

    def _load(self):
        if self._model is None:
            from cellpose.models import CellposeModel
            kwargs = {"gpu": self.gpu}
            if self.model_path is not None:
                kwargs["pretrained_model"] = self.model_path
            self._model = CellposeModel(**kwargs)
        return self._model

    def segment(self, image: np.ndarray, *,
                  channels: Sequence[ChannelSpec],
                  pixel_size_um: float) -> SegmentationResult:
        if image.ndim != 3:
            raise ValueError(f"image must be (C, H, W); got shape {image.shape}")
        ch_data = select_channels(image, channels, self.required_channels)
        # Cellpose 4.x expects (C, H, W) with channel_axis specified. The
        # CellposeSAM model has a `nchan` knob, but at runtime simply
        # feeding [nuclear, membrane] as a 2-channel stack with
        # channel_axis=0 works.
        stack = np.stack([ch_data[CHANNEL_ROLE_NUCLEAR],
                            ch_data[CHANNEL_ROLE_MEMBRANE]], axis=0)

        # Normalize each channel into the [0,1] dynamic range Cellpose
        # expects. Cellpose 4 has internal normalization=True, but for
        # heterogeneous bundles it's more robust to pre-clip extreme
        # outliers ourselves.
        def _prenorm(c: np.ndarray) -> np.ndarray:
            lo = float(np.percentile(c, 1.0))
            hi = float(np.percentile(c, 99.5))
            if hi <= lo + 1e-6:
                return np.zeros_like(c, dtype=np.float32)
            return np.clip((c - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        stack = np.stack([_prenorm(stack[0]), _prenorm(stack[1])], axis=0)

        model = self._load()
        # CellposeSAM expects diameter; if user didn't set one, give it
        # a Xenium-ish default (~12-15 µm cell, ~25-50 px @ 0.2125 µm/px).
        diam = self.diameter
        if diam is None:
            diam = 30.0  # px; Cellpose internally rescales relative to this
        result = model.eval(
            stack,
            channel_axis=0,
            normalize=True,
            diameter=diam,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
            compute_masks=True,
        )
        # Cellpose API returns (masks, flows, styles) or similar. Be
        # liberal about shape.
        if isinstance(result, tuple):
            masks = result[0]
        else:
            masks = result
        masks = np.asarray(masks, dtype=np.int32)
        if masks.ndim == 3:
            # CellposeSAM occasionally returns a (1, H, W) shape.
            masks = masks[0] if masks.shape[0] == 1 else masks.max(axis=0)

        return SegmentationResult(
            label_image=masks,
            segmenter_name=self.name,
            channels=tuple(channels),
            pixel_size_um=float(pixel_size_um),
            cell_table=None,
            config={
                "model": "cpsam",
                "gpu": self.gpu,
                "diameter_px": float(diam),
                "flow_threshold": float(self.flow_threshold),
                "cellprob_threshold": float(self.cellprob_threshold),
                "min_size": int(self.min_size),
                "model_path": self.model_path,
            },
        )
