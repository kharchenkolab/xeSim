"""Plan3 v37: scene editing primitives.

Modifications operate on a copy of MechanisticScene and return a new
Scene that can be passed to RefinerPipeline.render_scene().

All edits respect cell-source principle — every cell stays a discrete
object with its own label, type, latent. No free fields.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .mechanistic_scene import MechanisticCell, MechanisticScene


def _next_label(scene: MechanisticScene) -> int:
    """Return a fresh integer label not in use."""
    used = set(int(c.label) for c in scene.cells)
    used.update(int(x) for x in np.unique(scene.cell_label) if x > 0)
    return (max(used) if used else 0) + 1


class SceneEditor:
    """Builder/editor wrapping a MechanisticScene with mutation methods.

    All methods return self for chaining. Call .finalize() to get the
    edited immutable MechanisticScene.
    """

    def __init__(self, scene: MechanisticScene):
        self.image_shape = scene.image_shape
        self.pixel_size = float(scene.pixel_size)
        self.cell_label = scene.cell_label.copy()
        self.nucleus_label = scene.nucleus_label.copy()
        self._cells = {int(c.label): c for c in scene.cells}
        self.scene_id = scene.scene_id
        self.provenance = dict(scene.provenance) if scene.provenance else {}

    # ---------- queries ----------

    @property
    def cells(self) -> list[MechanisticCell]:
        return list(self._cells.values())

    def cell_count(self) -> int:
        return len(self._cells)

    # ---------- mutations ----------

    def remove_cell(self, label: int) -> "SceneEditor":
        lbl = int(label)
        if lbl not in self._cells:
            return self
        self.cell_label[self.cell_label == lbl] = 0
        self.nucleus_label[self.nucleus_label == lbl] = 0
        del self._cells[lbl]
        return self

    def remove_cells_of_type(self, type_name: str) -> "SceneEditor":
        to_remove = [int(lbl) for lbl, c in self._cells.items() if c.cell_type == type_name]
        for lbl in to_remove:
            self.remove_cell(lbl)
        return self

    def retype_cell(self, label: int, new_type: str) -> "SceneEditor":
        lbl = int(label)
        if lbl in self._cells:
            self._cells[lbl] = dataclasses.replace(self._cells[lbl], cell_type=new_type)
        return self

    def set_latent(self, label: int, latent: np.ndarray | tuple[float, ...] | None) -> "SceneEditor":
        lbl = int(label)
        if lbl not in self._cells:
            return self
        lat = None if latent is None else tuple(float(v) for v in latent)
        self._cells[lbl] = dataclasses.replace(self._cells[lbl], latent_vector=lat)
        return self

    def resample_latent(self, label: int, rng: np.random.Generator | None = None,
                        latent_dim: int = 4) -> "SceneEditor":
        rng = rng or np.random.default_rng()
        return self.set_latent(label, rng.standard_normal(latent_dim).astype(np.float32))

    def resample_all_latents(self, rng: np.random.Generator | None = None,
                             latent_dim: int = 4) -> "SceneEditor":
        rng = rng or np.random.default_rng()
        for lbl in list(self._cells.keys()):
            self.resample_latent(int(lbl), rng=rng, latent_dim=latent_dim)
        return self

    def move_cell(self, label: int, dy: int, dx: int) -> "SceneEditor":
        """Translate a cell by (dy, dx) pixels. Drops the cell if it moves out of bounds."""
        lbl = int(label)
        if lbl not in self._cells:
            return self
        h, w = self.image_shape
        cell_pix = self.cell_label == lbl
        nuc_pix = self.nucleus_label == lbl
        ys, xs = np.where(cell_pix)
        if len(ys) == 0:
            return self
        new_ys = ys + dy; new_xs = xs + dx
        if (new_ys < 0).any() or (new_ys >= h).any() or (new_xs < 0).any() or (new_xs >= w).any():
            # Cell would leave the frame; remove instead
            return self.remove_cell(lbl)
        # Move pixels
        self.cell_label[cell_pix] = 0
        self.nucleus_label[nuc_pix] = 0
        # Check for collision: target pixels already occupied
        target_occupied = self.cell_label[new_ys, new_xs] > 0
        keep = ~target_occupied
        self.cell_label[new_ys[keep], new_xs[keep]] = lbl
        if nuc_pix.any():
            n_ys, n_xs = np.where(nuc_pix)
            nn_ys = n_ys + dy; nn_xs = n_xs + dx
            in_bounds = (nn_ys >= 0) & (nn_ys < h) & (nn_xs >= 0) & (nn_xs < w)
            self.nucleus_label[nn_ys[in_bounds], nn_xs[in_bounds]] = lbl
        return self

    def add_cell_from_template(self, *, source_label: int, target_yx: tuple[int, int],
                                new_type: str | None = None,
                                latent: np.ndarray | None = None) -> int:
        """Add a copy of an existing cell at a new location.

        Returns the new cell's label. Copies the source cell's mask
        (translated), nucleus mask, type, and most other params. New
        cell gets a fresh label and optionally a new type / latent.
        Returns -1 if placement fails (collision / out-of-bounds).
        """
        src_lbl = int(source_label)
        if src_lbl not in self._cells:
            return -1
        src_cell = self._cells[src_lbl]
        h, w = self.image_shape
        cell_pix = self.cell_label == src_lbl
        nuc_pix = self.nucleus_label == src_lbl
        ys, xs = np.where(cell_pix)
        if len(ys) == 0: return -1
        cy = int(round(ys.mean())); cx = int(round(xs.mean()))
        ty, tx = target_yx
        dy = int(ty - cy); dx = int(tx - cx)
        new_ys = ys + dy; new_xs = xs + dx
        in_bounds = (new_ys >= 0) & (new_ys < h) & (new_xs >= 0) & (new_xs < w)
        if not in_bounds.all():
            return -1
        # Check collision
        if (self.cell_label[new_ys[in_bounds], new_xs[in_bounds]] > 0).any():
            return -1
        new_lbl = _next_label(self)
        self.cell_label[new_ys, new_xs] = new_lbl
        if nuc_pix.any():
            n_ys, n_xs = np.where(nuc_pix)
            nn_ys = n_ys + dy; nn_xs = n_xs + dx
            ib = (nn_ys >= 0) & (nn_ys < h) & (nn_xs >= 0) & (nn_xs < w)
            self.nucleus_label[nn_ys[ib], nn_xs[ib]] = new_lbl
        new_cell = dataclasses.replace(
            src_cell,
            cell_id=f"added_{new_lbl}",
            label=new_lbl,
            source="synthetic",
            cell_type=(new_type if new_type is not None else src_cell.cell_type),
            nucleus_label=new_lbl if nuc_pix.any() else None,
            latent_vector=(None if latent is None else tuple(float(v) for v in latent)),
        )
        self._cells[new_lbl] = new_cell
        return new_lbl

    # ---------- finalize ----------

    def finalize(self, scene_id: str | None = None) -> MechanisticScene:
        # Re-anchor cells dict in label order
        cell_list = tuple(self._cells[k] for k in sorted(self._cells.keys()))
        return MechanisticScene(
            image_shape=self.image_shape,
            pixel_size=self.pixel_size,
            cell_label=self.cell_label.astype(np.int32, copy=True),
            nucleus_label=self.nucleus_label.astype(np.int32, copy=True),
            cells=cell_list,
            scene_id=scene_id or self.scene_id,
            provenance={**self.provenance, "edits": "via SceneEditor"},
        )
