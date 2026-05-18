from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mechanistic_schema import (
    MECHANISTIC_CELL_SOURCES,
    MECHANISTIC_PARAMS_TYPE,
    MECHANISTIC_SCENE_TYPE,
    MECHANISTIC_SECTIONING_STATES,
)
from .schema import MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class MechanisticCell:
    cell_id: str
    label: int
    source: str = "synthetic"
    sectioning_state: str = "full"
    nucleus_label: int | None = None
    cell_type: str | None = None
    segmentation_reliability: float = 1.0
    visibility_fraction: float = 1.0
    membrane_efficiency: float = 1.0
    dapi_efficiency: float = 1.0
    polya_efficiency: float = 1.0
    # Optional per-cell physical parameters; ``None`` means use the scene-global
    # MechanisticParams default. Set per-cell when the scene builder has type-
    # aware information.
    membrane_width_px: float | None = None
    membrane_dropout_keep_prob: float | None = None
    membrane_interior_fraction: float | None = None
    dapi_texture_scale: float | None = None
    polya_texture_scale: float | None = None
    # Plan3 v34: per-cell 3D anatomy state. ``z_position_um`` is the offset
    # of the cell's centroid from the section midplane (positive =
    # cell-center above the slab midplane). ``z_extent_um`` overrides the
    # per-type default. ``apical_axis`` is an in-plane unit vector (dy, dx)
    # that points from the basal pole toward the apical pole *projected
    # onto the slab*; ``None`` means no in-plane apical asymmetry (used
    # for stromal/immune cells with no clear polarity).
    z_position_um: float = 0.0
    z_extent_um: float | None = None
    apical_axis: tuple[float, float] | None = None
    # Plan3 v37: per-cell 4-dim latent style vector. Stochastic "look"
    # signal — same-type cells with different latents render with
    # different fine-grained texture (chromatin pattern, polyA puncta
    # placement, membrane intensity variation). Set to None for synth
    # scenes (renderer will draw random latents); set explicitly when a
    # cell was inferred from a real tile via the latent encoder so the
    # re-render reproduces that specific cell's appearance.
    latent_vector: tuple[float, ...] | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "label": int(self.label),
            "source": self.source,
            "sectioning_state": self.sectioning_state,
            "nucleus_label": None if self.nucleus_label is None else int(self.nucleus_label),
            "cell_type": self.cell_type,
            "segmentation_reliability": float(self.segmentation_reliability),
            "visibility_fraction": float(self.visibility_fraction),
            "membrane_efficiency": float(self.membrane_efficiency),
            "dapi_efficiency": float(self.dapi_efficiency),
            "polya_efficiency": float(self.polya_efficiency),
            "membrane_width_px": None if self.membrane_width_px is None else float(self.membrane_width_px),
            "membrane_dropout_keep_prob": None if self.membrane_dropout_keep_prob is None else float(self.membrane_dropout_keep_prob),
            "membrane_interior_fraction": None if self.membrane_interior_fraction is None else float(self.membrane_interior_fraction),
            "dapi_texture_scale": None if self.dapi_texture_scale is None else float(self.dapi_texture_scale),
            "polya_texture_scale": None if self.polya_texture_scale is None else float(self.polya_texture_scale),
            "z_position_um": float(self.z_position_um),
            "z_extent_um": None if self.z_extent_um is None else float(self.z_extent_um),
            "apical_axis": None if self.apical_axis is None else [float(self.apical_axis[0]), float(self.apical_axis[1])],
            "latent_vector": None if self.latent_vector is None else [float(v) for v in self.latent_vector],
            "quality_flags": list(self.quality_flags),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MechanisticCell":
        def _opt(key: str) -> float | None:
            v = value.get(key)
            return None if v is None else float(v)

        return cls(
            cell_id=str(value["cell_id"]),
            label=int(value["label"]),
            source=str(value.get("source", "synthetic")),
            sectioning_state=str(value.get("sectioning_state", "full")),
            nucleus_label=None if value.get("nucleus_label") is None else int(value["nucleus_label"]),
            cell_type=None if value.get("cell_type") is None else str(value["cell_type"]),
            segmentation_reliability=float(value.get("segmentation_reliability", 1.0)),
            visibility_fraction=float(value.get("visibility_fraction", 1.0)),
            membrane_efficiency=float(value.get("membrane_efficiency", 1.0)),
            dapi_efficiency=float(value.get("dapi_efficiency", 1.0)),
            polya_efficiency=float(value.get("polya_efficiency", 1.0)),
            membrane_width_px=_opt("membrane_width_px"),
            membrane_dropout_keep_prob=_opt("membrane_dropout_keep_prob"),
            membrane_interior_fraction=_opt("membrane_interior_fraction"),
            dapi_texture_scale=_opt("dapi_texture_scale"),
            polya_texture_scale=_opt("polya_texture_scale"),
            z_position_um=float(value.get("z_position_um", 0.0)),
            z_extent_um=_opt("z_extent_um"),
            apical_axis=(
                None if value.get("apical_axis") is None
                else (float(value["apical_axis"][0]), float(value["apical_axis"][1]))
            ),
            latent_vector=(
                None if value.get("latent_vector") is None
                else tuple(float(v) for v in value["latent_vector"])
            ),
            quality_flags=tuple(str(flag) for flag in value.get("quality_flags", [])),
            provenance=dict(value.get("provenance", {})),
        )


@dataclass(frozen=True)
class MechanisticScene:
    image_shape: tuple[int, int]
    pixel_size: float
    cell_label: np.ndarray
    nucleus_label: np.ndarray
    cells: tuple[MechanisticCell, ...]
    scene_id: str = "scene"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = True) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "type": MECHANISTIC_SCENE_TYPE,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            "pixel_size": float(self.pixel_size),
            "cells": [cell.to_dict() for cell in self.cells],
            "provenance": dict(self.provenance),
        }
        if include_arrays:
            obj["cell_label"] = self.cell_label.astype(np.int32, copy=False).tolist()
            obj["nucleus_label"] = self.nucleus_label.astype(np.int32, copy=False).tolist()
        return obj

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MechanisticScene":
        if value.get("type") != MECHANISTIC_SCENE_TYPE:
            raise ValueError(f"not a mechanistic scene: {value.get('type')!r}")
        if "cell_label" not in value or "nucleus_label" not in value:
            raise ValueError("mechanistic scene dictionary must include label arrays")
        image_shape = tuple(int(x) for x in value["image_shape"])
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain two integers")
        scene = cls(
            image_shape=(image_shape[0], image_shape[1]),
            pixel_size=float(value["pixel_size"]),
            cell_label=np.asarray(value["cell_label"], dtype=np.int32),
            nucleus_label=np.asarray(value["nucleus_label"], dtype=np.int32),
            cells=tuple(MechanisticCell.from_dict(cell) for cell in value.get("cells", [])),
            scene_id=str(value.get("scene_id", "scene")),
            provenance=dict(value.get("provenance", {})),
        )
        validate_mechanistic_scene(scene)
        return scene


@dataclass(frozen=True)
class MechanisticParams:
    dapi_base: float = 0.85
    dapi_texture_scale: float = 0.12
    dapi_blur_sigma_px: float = 1.1
    membrane_base: float = 0.75
    membrane_width_px: float = 1.8
    membrane_dropout_keep_prob: float = 0.78
    membrane_dropout_scale_px: float = 10.0
    membrane_interior_fraction: float = 0.10
    membrane_out_of_focus_fraction: float = 0.35
    membrane_out_of_focus_sigma_px: float = 4.0
    haze_level: float = 0.035
    haze_scale_px: float = 18.0
    # Explicit extracellular tissue component fitted from real background
    # pixels. Distinct from ``haze_level`` (which is bundle-level optical glow):
    # this represents continuous low-amplitude membrane signal in tissue
    # regions outside cell masks, reflecting unsegmented cell membrane
    # contributions and out-of-plane membrane projection. Renders even in
    # synthetic mode so the simulator does not produce black backgrounds.
    extracellular_tissue_level: float = 0.10
    extracellular_tissue_scale_px: float = 24.0
    extracellular_tissue_texture: float = 0.35
    # PolyA stain (channel 2): cytoplasmic mRNA signal. Concentrated in
    # cytoplasm of metabolically active cells (epithelial, endocrine), low
    # in fibroblast / immune. ``polya_base`` is the global cytoplasm-mean
    # polyA intensity; per-type efficiency multiplies this in
    # ``MechanisticCell``.
    polya_base: float = 0.04
    polya_nucleus_fraction: float = 0.30
    # Plan3 B10: polyA within-cell texture amplitude. 0.4 ≈ real per-cell
    # CV (intra-cell std / mean ≈ 0.35-0.40 across pancreas types). Blur
    # tightened to 0.3 px to keep speckle visible. Cuts B7 std_polya gap.
    polya_texture_scale: float = 0.4
    polya_blur_sigma_px: float = 0.3
    # Plan3 B15: Poisson shot-noise scale on polyA (per-pixel
    # Poisson(amp * polya_shot_lambda) / polya_shot_lambda). Default 0
    # disables (B15 sweep showed regression at shot_lambda=50, 200, 500;
    # additive Poisson on top of the existing smooth-noise overshoots
    # the per-cell std distribution). Code path retained for future use
    # when smooth_noise is replaced rather than augmented.
    polya_shot_lambda: float = 0.0
    # Plan3 v35: per-cell texture sampling for polyA and DAPI. v14 style —
    # each cell gets its own independent smooth-noise field. v34's perf
    # optimization replaced these with a single shared tile-level field
    # which made all cells show identical texture patterns (only differing
    # in brightness). Set False for backward-compat / speed.
    polya_per_cell_texture: bool = True
    dapi_per_cell_texture: bool = True
    # Plan3 v35 P2: per-cell stochastic polyA puncta (texture model — not
    # molecule modeling). When >0, sample N=Poisson(area*amp*density)
    # points uniformly inside each cell's cytoplasm and add puncta_amp
    # at each point (then PSF-blurred with polya_blur_sigma_px). Yields
    # heavy-tailed pixel distribution. Defaults to 0 (disabled) for
    # backward-compat; v35 turn-on requires both > 0.
    polya_puncta_density_per_amp: float = 0.0
    polya_puncta_amp: float = 0.0
    # Plan3 v35 P6: nucleolus dark spot dimming factor. 0=no spot,
    # 0.5=spot center is 50% dim. Real nuclei show 1-2 dark spots
    # (nucleoli) within DAPI signal — without these, synth nuclei look
    # over-uniform.
    dapi_nucleolus_dim_factor: float = 0.0
    # Plan3 v35 P7: nuclear envelope brightness boost. 0=no boost,
    # 0.4=envelope is 40% brighter than nuclear interior. Real DAPI shows
    # heterochromatin adhered to nuclear membrane.
    dapi_envelope_boost: float = 0.0
    # Plan3 v35 P3: per-cell extracellular membrane bleed. Each cell's
    # boundary extends a halo into immediate extracellular space
    # (basement membrane proteins, secreted matrix). Cell-source
    # compliant — replaces v14's cell-free `green_tissue_haze`.
    membrane_extracellular_bleed_amp: float = 0.0
    membrane_extracellular_bleed_decay_px: float = 3.0
    # Plan3 v35 P10: per-pixel multiplicative jitter along membrane
    # boundary. 0=uniform line, 0.5=±50% intensity variation along line.
    # Real membranes show this from junction density variation.
    membrane_line_texture_amp: float = 0.0
    # Plan3 v35 P9: junction boost at cell-cell contact pixels. 0=no
    # boost; 0.5=intercellular ridge 50% brighter than cell-extracellular.
    membrane_junction_boost: float = 0.0
    # Plan3 v35 P10b: intercellular ridge sigma. Default v34 0.9 (tight,
    # creates lattice look). v14-era ~1.5-3 with diffuse interior +
    # out-of-focus added on top.
    intercell_ridge_sigma_px: float = 0.9
    # Plan3 v35 P8: per-cell cytoplasmic DAPI bleed fraction. 0=no bleed,
    # 0.15=cytoplasm at 15% of nuclear DAPI amplitude. Real DAPI shows
    # dim signal in cytoplasm (mtDNA, chromatin extension, optical bleed
    # from cells above/below the section).
    dapi_cytoplasm_fraction: float = 0.0
    # Plan3 T-rasterize-polya: optional additive rasterized-transcript
    # contribution to polyA cytoplasm. Default 0 (disabled): Xenium
    # polyA is a CONTINUOUS polyA-tagged stain, not single-molecule
    # rasterization, and replacing the smooth model with rasterized
    # points regressed B4 0.546 -> 0.29 and B7 std_polya 17 -> 127.
    # Code path retained as additive contribution for future use; set
    # to a small value (e.g., 0.05) if a real-data study confirms a
    # sparse-spot contribution at top of the polyA stain.
    polya_per_transcript: float = 0.0
    polya_extracellular_level: float = 0.005
    illumination_scale: float = 0.05
    noise_sigma: float = 0.015
    seed: int = 1
    # Plan3 v32 (HELD, superseded by v34 S3): per-type extracellular membrane
    # ridge. Empty dicts disable; will be removed once S3 lands.
    per_type_extracellular_membrane_amp: dict[str, float] = field(default_factory=dict)
    per_type_extracellular_membrane_width_px: dict[str, float] = field(default_factory=dict)
    # Plan3 v34: per-type 3D anatomy. Each entry is a dict of
    # ``z_extent_um``, ``z_position_nucleus_relative``, ``layers`` (z-layered
    # per-channel emission factors). See ``xesim/mechanistic_anatomy.py``
    # for the schema and ``project_anatomy_through_slab()`` for the
    # integration semantics. Empty dict disables anatomy projection and the
    # renderer falls back to the v33 boundary-distance Gaussian model
    # (default for backward compatibility).
    per_type_anatomy: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Section slab thickness in µm; used by the section-projection renderer.
    section_slab_thickness_um: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": MECHANISTIC_PARAMS_TYPE,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MechanisticParams":
        kwargs = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        return cls(**kwargs)


def scene_from_label_masks(
    cell_label: np.ndarray,
    nucleus_label: np.ndarray,
    pixel_size: float,
    scene_id: str = "scene",
    source: str = "observed_anchor",
    cell_ids: list[str] | None = None,
    cell_id_to_type: dict[str, int] | None = None,
    type_names: list[str] | None = None,
    per_type_efficiencies: dict[str, dict[str, float]] | None = None,
) -> MechanisticScene:
    """Build a simple mechanistic scene from aligned cell/nucleus label masks.

    When ``cell_ids``, ``cell_id_to_type``, ``type_names`` and
    ``per_type_efficiencies`` are all provided, each cell receives type-aware
    DAPI and membrane efficiencies that scale its contribution to the
    mechanistic render.
    """

    cell_arr = np.asarray(cell_label, dtype=np.int32)
    nucleus_arr = np.asarray(nucleus_label, dtype=np.int32)
    labels = [int(x) for x in np.unique(cell_arr) if int(x) > 0]
    sorted_cell_ids: list[str] = list(cell_ids) if cell_ids is not None else []
    have_types = (
        cell_ids is not None
        and cell_id_to_type is not None
        and type_names is not None
        and per_type_efficiencies is not None
    )
    cells: list[MechanisticCell] = []
    for idx, label in enumerate(labels):
        nucleus_labels = np.unique(nucleus_arr[cell_arr == label])
        nucleus_ids = [int(x) for x in nucleus_labels if int(x) > 0]
        cell_type_name: str | None = None
        dapi_eff = 1.0
        memb_eff = 1.0
        polya_eff = 1.0
        memb_width: float | None = None
        memb_dropout: float | None = None
        memb_interior: float | None = None
        dapi_texture: float | None = None
        polya_texture: float | None = None
        if have_types and idx < len(sorted_cell_ids):
            cell_id = sorted_cell_ids[idx]
            type_idx = int(cell_id_to_type.get(cell_id, 0))
            if 0 <= type_idx < len(type_names):
                cell_type_name = type_names[type_idx]
                eff = per_type_efficiencies.get(cell_type_name, {})
                dapi_eff = float(eff.get("dapi_efficiency", 1.0))
                memb_eff = float(eff.get("membrane_efficiency", 1.0))
                polya_eff = float(eff.get("polya_efficiency", 1.0))
                # Per-type physical parameters (optional in the type table).
                memb_width = _opt_param(eff, "membrane_width_px")
                memb_dropout = _opt_param(eff, "membrane_dropout_keep_prob")
                memb_interior = _opt_param(eff, "membrane_interior_fraction")
                dapi_texture = _opt_param(eff, "dapi_texture_scale")
                polya_texture = _opt_param(eff, "polya_texture_scale")
        cell_id_str = sorted_cell_ids[idx] if idx < len(sorted_cell_ids) else f"cell_{label}"
        cells.append(
            MechanisticCell(
                cell_id=cell_id_str,
                label=label,
                source=source,
                nucleus_label=nucleus_ids[0] if nucleus_ids else None,
                sectioning_state="full" if nucleus_ids else "nucleus_poor",
                cell_type=cell_type_name,
                dapi_efficiency=dapi_eff,
                membrane_efficiency=memb_eff,
                polya_efficiency=polya_eff,
                membrane_width_px=memb_width,
                membrane_dropout_keep_prob=memb_dropout,
                membrane_interior_fraction=memb_interior,
                dapi_texture_scale=dapi_texture,
                polya_texture_scale=polya_texture,
            )
        )
    scene = MechanisticScene(
        image_shape=tuple(int(x) for x in cell_arr.shape),
        pixel_size=float(pixel_size),
        cell_label=cell_arr,
        nucleus_label=nucleus_arr,
        cells=tuple(cells),
        scene_id=scene_id,
    )
    validate_mechanistic_scene(scene)
    return scene


def validate_mechanistic_scene(scene: MechanisticScene) -> None:
    h, w = scene.image_shape
    if h <= 0 or w <= 0:
        raise ValueError("mechanistic scene image_shape must be positive")
    if scene.cell_label.shape != (h, w):
        raise ValueError("cell_label shape does not match image_shape")
    if scene.nucleus_label.shape != (h, w):
        raise ValueError("nucleus_label shape does not match image_shape")
    if scene.pixel_size <= 0:
        raise ValueError("pixel_size must be positive")

    ids: set[str] = set()
    labels: set[int] = set()
    present_cell_labels = {int(x) for x in np.unique(scene.cell_label) if int(x) > 0}
    present_nucleus_labels = {int(x) for x in np.unique(scene.nucleus_label) if int(x) > 0}
    for cell in scene.cells:
        if not cell.cell_id:
            raise ValueError("cell_id must be non-empty")
        if cell.cell_id in ids:
            raise ValueError(f"duplicated cell_id: {cell.cell_id}")
        ids.add(cell.cell_id)
        if cell.label <= 0:
            raise ValueError(f"cell {cell.cell_id} has non-positive label")
        if cell.label in labels:
            raise ValueError(f"duplicated cell label: {cell.label}")
        labels.add(cell.label)
        if cell.label not in present_cell_labels:
            raise ValueError(f"cell {cell.cell_id} references missing label {cell.label}")
        if cell.source not in MECHANISTIC_CELL_SOURCES:
            raise ValueError(f"invalid cell source: {cell.source}")
        if cell.sectioning_state not in MECHANISTIC_SECTIONING_STATES:
            raise ValueError(f"invalid sectioning state: {cell.sectioning_state}")
        _validate_fraction(cell.segmentation_reliability, "segmentation_reliability")
        _validate_fraction(cell.visibility_fraction, "visibility_fraction")
        if cell.membrane_efficiency < 0:
            raise ValueError("membrane_efficiency must be non-negative")
        if cell.dapi_efficiency < 0:
            raise ValueError("dapi_efficiency must be non-negative")
        if cell.nucleus_label is not None and cell.nucleus_label not in present_nucleus_labels:
            raise ValueError(f"cell {cell.cell_id} references missing nucleus label {cell.nucleus_label}")


def make_tiny_mechanistic_scene() -> MechanisticScene:
    """Return a deterministic 64x64 two-cell scene for tests and smoke docs."""

    h, w = 64, 64
    yy, xx = np.mgrid[:h, :w]
    cell_label = np.zeros((h, w), dtype=np.int32)
    nucleus_label = np.zeros((h, w), dtype=np.int32)

    cell1 = ((xx - 23) / 15.0) ** 2 + ((yy - 31) / 20.0) ** 2 <= 1.0
    cell2 = ((xx - 42) / 14.0) ** 2 + ((yy - 31) / 18.0) ** 2 <= 1.0
    cell_label[cell1] = 1
    cell_label[cell2] = 2
    nucleus_label[((xx - 21) / 5.0) ** 2 + ((yy - 30) / 7.0) ** 2 <= 1.0] = 1
    nucleus_label[((xx - 43) / 6.0) ** 2 + ((yy - 32) / 6.0) ** 2 <= 1.0] = 2
    return scene_from_label_masks(cell_label, nucleus_label, pixel_size=0.5, scene_id="tiny_plan3", source="synthetic")


def _validate_fraction(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _opt_param(table: dict[str, Any], key: str) -> float | None:
    v = table.get(key)
    return None if v is None else float(v)

