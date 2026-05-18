"""Read a Xenium bundle's morphology window + assemble a channel stack
for a segmenter.

The harness deliberately resolves channel roles by NAME (DAPI, ATP1A1,
…) against the bundle's morphology channel names, not by position. So
models trained with different channel orders work without harness
changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..xenium import resolve_bundle
from .base import ChannelSpec


# Default role → channel-name candidates. First match wins, case-insensitive,
# substring match.
DEFAULT_ROLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "nuclear": ("dapi", "hoechst", "nucleus"),
    "membrane": ("atp1a1", "e-cadherin", "ecad", "cd45", "membrane"),
    "cyto": ("18s", "ribosom", "cyto"),
}


def detect_channel_name(role: str, candidates: Sequence[str],
                          extra: Mapping[str, Sequence[str]] | None = None) -> str | None:
    """Return the first channel-name in `candidates` whose lowercase form
    contains any candidate substring for `role`. Substrings come from
    `DEFAULT_ROLE_CANDIDATES` plus `extra` overlay."""
    table = dict(DEFAULT_ROLE_CANDIDATES)
    if extra:
        table.update({k: tuple(v) for k, v in extra.items()})
    needles = table.get(role, ())
    for ch in candidates:
        lo = ch.lower()
        for n in needles:
            if n in lo:
                return ch
    return None


def build_channel_spec(channel_names: Sequence[str],
                         roles: Sequence[str],
                         *,
                         overrides: Mapping[str, str] | None = None,
                         extra_candidates: Mapping[str, Sequence[str]] | None = None,
                         ) -> tuple[ChannelSpec, ...]:
    """Map each role in `roles` to (name, index) using `channel_names`.

    `overrides` lets the caller force a specific role→name pairing
    ("nuclear": "DAPI"); otherwise we fall back to substring detection.
    Raises ``ValueError`` if any role cannot be resolved.
    """
    names = list(channel_names)
    out: list[ChannelSpec] = []
    overrides = dict(overrides or {})
    for role in roles:
        if role in overrides:
            chosen = overrides[role]
            if chosen not in names:
                raise ValueError(
                    f"Override for role {role!r} = {chosen!r} not in channel_names "
                    f"{names!r}")
        else:
            chosen = detect_channel_name(role, names, extra=extra_candidates)
            if chosen is None:
                raise ValueError(
                    f"Could not auto-detect a channel for role {role!r} "
                    f"among {names!r}. Pass overrides={{'{role}': '<name>'}}.")
        out.append(ChannelSpec(role=role, name=chosen, index=names.index(chosen)))
    return tuple(out)


def bundle_channel_names(bundle_path: str | Path) -> tuple[list[str], float]:
    """Return (channel_names, pixel_size_um) for the bundle's morphology
    image. Channel order is the order in `morphology.ome.tif` /
    `morphology_focus/`."""
    from ..images import channel_names as _channel_names
    bundle = resolve_bundle(bundle_path)
    if not bundle.morphology_focus_paths:
        raise ValueError(f"bundle {bundle_path} has no morphology image")
    names = list(_channel_names(bundle.morphology_focus_paths))
    return names, float(bundle.pixel_size)


def read_window(bundle_path: str | Path,
                  bounds_um: tuple[float, float, float, float],
                  *,
                  pixel_size_um: float | None = None,
                  ) -> tuple[np.ndarray, list[str], float]:
    """Read a window of the bundle's morphology image.

    Returns (image, channel_names, pixel_size_um) where ``image`` is
    ``(C, H, W)`` float32 at native bundle resolution (or
    ``pixel_size_um`` if given — currently only native is supported).

    `bounds_um = (xmin, ymin, xmax, ymax)`.
    """
    from ..images import ImageStackReader
    from ..models import CropBox

    bundle = resolve_bundle(bundle_path)
    if not bundle.morphology_focus_paths:
        raise ValueError(f"bundle {bundle_path} has no morphology image")
    psz = pixel_size_um or float(bundle.pixel_size)
    if abs(psz - float(bundle.pixel_size)) > 1e-3:
        raise NotImplementedError(
            "Resampling to a different pixel_size_um is not yet supported")
    reader = ImageStackReader(bundle.morphology_focus_paths, psz)
    xmin, ymin, xmax, ymax = bounds_um
    crop = CropBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, crop_id="segeval_window")
    img = reader.read(crop).astype(np.float32)
    from ..images import channel_names as _channel_names
    names = list(_channel_names(bundle.morphology_focus_paths))
    return img, names, psz


def synth_bundle_channel_stack(bundle_path: str | Path,
                                  ) -> tuple[np.ndarray, list[str], float]:
    """For a synth bundle (output of `xesim explain --format bundle`),
    read the full morphology image. Returns (image, channel_names,
    pixel_size_um). Used when the entire synth bundle is small (a few
    tiles); for large synth bundles use ``read_window``."""
    return read_window(bundle_path,
                         bounds_um=_bundle_full_bounds_um(bundle_path))


def _bundle_full_bounds_um(bundle_path: str | Path
                                ) -> tuple[float, float, float, float]:
    from ..images import ome_image_shape
    bundle = resolve_bundle(bundle_path)
    h, w = ome_image_shape(bundle.morphology_focus_paths[0])
    psz = float(bundle.pixel_size)
    return (0.0, 0.0, w * psz, h * psz)
