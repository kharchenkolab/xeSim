"""2D synth bundle generator (Phase 1: tile-level MVP).

Two cell populations + tissue-context-aware ghost cells absorb all
non-anchor signal. See ``misc/2D.md`` for the full plan.

Public API:
- :class:`Scene2D` — composed scene (anchor + ghost cells + molecules + GT)
- :func:`compose_tile_scene` — build a Scene2D for one tile from a bundle
- :func:`render_tile` — render a Scene2D through the v16 renderer
- :func:`write_tile` — write a Scene2D to a flat output directory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..mechanistic_scene import MechanisticScene


@dataclass
class Scene2D:
    """A composed 2D scene ready for rendering + emission.

    A small wrapper around ``MechanisticScene`` that adds the molecule list
    (with per-molecule ground truth tags) and tile metadata.

    Fields
    ------
    mech_scene
        The underlying ``MechanisticScene`` with the union of anchor and
        ghost cells. Anchor cells have ``provenance["is_ghost"] = False``;
        ghost cells have ``True``. ``cell_label`` and ``nucleus_label``
        rasterize the union.
    molecules
        Per-molecule DataFrame with columns:
          - ``x, y`` (µm in tile-local coords)
          - ``gene`` (string gene name)
          - ``true_cell_id`` (str — anchor or ghost cell id)
          - ``is_ghost`` (bool — True if the source was a ghost cell)
          - ``true_factor`` (int — factor index)
          - ``qv`` (synthetic quality value)
    tile_bounds_um
        ``(xmin, xmax, ymin, ymax)`` in scene-coordinate µm.
    pixel_size
        µm per pixel for the rasterized masks.
    """

    mech_scene: MechanisticScene
    molecules: pd.DataFrame
    tile_bounds_um: tuple[float, float, float, float]
    pixel_size: float
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---- Convenience properties -----------------------------------------

    @property
    def anchor_cells(self) -> list:
        return [c for c in self.mech_scene.cells if not c.provenance.get("is_ghost", False)]

    @property
    def ghost_cells(self) -> list:
        return [c for c in self.mech_scene.cells if c.provenance.get("is_ghost", False)]

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_cells)

    @property
    def n_ghosts(self) -> int:
        return len(self.ghost_cells)

    @property
    def noise_fraction(self) -> float:
        """Empirical fraction of molecules attributed to ghost cells."""
        if len(self.molecules) == 0:
            return 0.0
        return float(self.molecules["is_ghost"].mean())

    def summary(self) -> dict[str, Any]:
        return {
            "n_anchor_cells": self.n_anchors,
            "n_ghost_cells": self.n_ghosts,
            "n_molecules": int(len(self.molecules)),
            "noise_fraction": self.noise_fraction,
            "tile_bounds_um": list(self.tile_bounds_um),
            "pixel_size": float(self.pixel_size),
            "provenance": self.provenance,
        }


__all__ = ["Scene2D"]


def __getattr__(name):
    if name in {"write_bundle", "auto_tune_noise_fraction",
                "auto_tune_intensity_stats", "UNASSIGNED_LABEL"}:
        from . import bundle_writer
        return getattr(bundle_writer, name)
    if name in {"build_tile"}:
        from . import tile_pipeline
        return getattr(tile_pipeline, name)
    if name in {"build_scene", "tile_grid", "filter_to_owned"}:
        from . import scene_pipeline
        return getattr(scene_pipeline, name)
    if name in {"calibrate_to_uint16"}:
        from . import intensity
        return getattr(intensity, name)
    raise AttributeError(name)
