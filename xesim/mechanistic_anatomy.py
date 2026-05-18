"""Plan3 v34 S1: per-type 3D anatomy.

Each cell type has a parametric anatomy that the renderer integrates over
the section slab. Anatomy fields:

- ``z_extent_um``                       : cell height in µm.
- ``z_position_nucleus_relative``      : nucleus center as a fraction of
                                         z_extent measured from the basal
                                         end (0 = at basal pole, 1 = at
                                         apical pole). Epithelials have
                                         basally-located nuclei (~0.2-0.3),
                                         stromal/immune typically central
                                         (~0.5).
- ``z_position_apical_relative``       : apical pole as a fraction of
                                         z_extent (typically 1.0).
- ``layers``                           : per-layer z-fraction range and
                                         per-channel emission factor. The
                                         emission factor is the contribution
                                         of that layer to the pixel value
                                         when integrated. Channels: dapi,
                                         membrane, polya.

The renderer projects each cell's anatomy into the slab as follows: for a
cell with z_position p (offset of cell center from slab midplane in µm)
and z_extent e, the cell occupies z ∈ [p - e/2, p + e/2] in absolute
section coordinates. Slab is [-half_slab, +half_slab] centered at z=0.
The in-plane integral of layer L for channel c is

    ∫_{slab ∩ cell ∩ L} emission_factor[L, c] dz  (in µm units)

normalized by the slab thickness. A fully-in-plane cell with z_extent ≈
slab_thickness contributes roughly the sum of all layer emissions; an
"apex_above" cell with p > 0 large contributes only the basal layers.

This module focuses on (a) defining the anatomy schema, (b) supplying
biological priors, and (c) validating priors against real whole-cell
intensity statistics. Renderer changes that *use* the anatomy are
implemented in S3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Default Xenium-like section thickness. Real Xenium is typically ~5 µm
# physical section but optical depth can extend further; 8 µm is a
# reasonable default. Per-bundle override is allowed via params.
DEFAULT_SLAB_THICKNESS_UM = 8.0


# Default / fallback anatomy for cells without a specific type assignment
# (cell_type == "unknown" or None, which includes latent cells proposed at
# Stage B and any 10x cells the type classifier left unclassified). The
# default is a generic 'mid-size cell' anatomy that gives moderate
# emission across all channels — biologically plausible as a baseline and
# avoids dim/missing renders for the ~80% of cells without specific type
# assignment in the canonical128 corpus.
DEFAULT_ANATOMY: dict[str, Any] = {
    "z_extent_um": 9.0,
    "z_position_nucleus_relative": 0.50,
    "z_position_apical_relative": 1.0,
    "layers": {
        "membrane_shell":     {"z_lo": 0.00, "z_hi": 0.06, "dapi": 0.0, "membrane": 0.30, "polya": 0.0},
        "cytoplasm":          {"z_lo": 0.06, "z_hi": 0.94, "dapi": 0.10, "membrane": 0.05, "polya": 0.5},
        "nucleus":            {"z_lo": 0.30, "z_hi": 0.70, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.15},
        "membrane_shell_top": {"z_lo": 0.94, "z_hi": 1.00, "dapi": 0.0,  "membrane": 0.30, "polya": 0.0},
    },
}


# Biological priors for per-type 3D anatomy. Keys match the cell-type
# names used elsewhere in the pipeline (cell_types.json).
PER_TYPE_ANATOMY_PRIORS: dict[str, dict[str, Any]] = {
    "Ductal/tumor epithelial": {
        "z_extent_um": 15.0,
        "z_position_nucleus_relative": 0.25,
        "z_position_apical_relative": 1.0,
        "layers": {
            "basal_membrane":     {"z_lo": 0.00, "z_hi": 0.05, "dapi": 0.0, "membrane": 0.55, "polya": 0.0},
            "basal_cytoplasm":    {"z_lo": 0.05, "z_hi": 0.20, "dapi": 0.05, "membrane": 0.05, "polya": 0.6},
            "nucleus":            {"z_lo": 0.15, "z_hi": 0.40, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.2},
            "supranuclear_cyto":  {"z_lo": 0.40, "z_hi": 0.90, "dapi": 0.05, "membrane": 0.05, "polya": 1.0},
            "apical_mucin":       {"z_lo": 0.85, "z_hi": 0.98, "dapi": 0.0,  "membrane": 0.30, "polya": 1.4},
            "apical_membrane":    {"z_lo": 0.98, "z_hi": 1.00, "dapi": 0.0,  "membrane": 0.80, "polya": 0.0},
        },
    },
    "Exocrine epithelial": {
        "z_extent_um": 14.0,
        "z_position_nucleus_relative": 0.20,
        "z_position_apical_relative": 1.0,
        "layers": {
            "basal_membrane":     {"z_lo": 0.00, "z_hi": 0.05, "dapi": 0.0, "membrane": 0.50, "polya": 0.0},
            "basal_cytoplasm":    {"z_lo": 0.05, "z_hi": 0.20, "dapi": 0.05, "membrane": 0.05, "polya": 1.2},
            "nucleus":            {"z_lo": 0.15, "z_hi": 0.35, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.3},
            "supranuclear_cyto":  {"z_lo": 0.35, "z_hi": 0.85, "dapi": 0.05, "membrane": 0.05, "polya": 1.5},
            "apical_zymogen":     {"z_lo": 0.85, "z_hi": 0.98, "dapi": 0.0,  "membrane": 0.20, "polya": 1.8},
            "apical_membrane":    {"z_lo": 0.98, "z_hi": 1.00, "dapi": 0.0,  "membrane": 0.70, "polya": 0.0},
        },
    },
    "Endocrine": {
        "z_extent_um": 10.0,
        "z_position_nucleus_relative": 0.45,
        "z_position_apical_relative": 1.0,
        "layers": {
            "basal_membrane":     {"z_lo": 0.00, "z_hi": 0.04, "dapi": 0.0, "membrane": 0.30, "polya": 0.0},
            "basal_cytoplasm":    {"z_lo": 0.04, "z_hi": 0.40, "dapi": 0.05, "membrane": 0.04, "polya": 0.4},
            "nucleus":            {"z_lo": 0.30, "z_hi": 0.60, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.15},
            "apical_cytoplasm":   {"z_lo": 0.60, "z_hi": 0.96, "dapi": 0.05, "membrane": 0.04, "polya": 0.4},
            "apical_membrane":    {"z_lo": 0.96, "z_hi": 1.00, "dapi": 0.0,  "membrane": 0.30, "polya": 0.0},
        },
    },
    "Endothelial": {
        "z_extent_um": 7.0,
        "z_position_nucleus_relative": 0.50,
        "z_position_apical_relative": 1.0,
        "layers": {
            "basal_membrane":     {"z_lo": 0.00, "z_hi": 0.06, "dapi": 0.0, "membrane": 0.40, "polya": 0.0},
            "cytoplasm":          {"z_lo": 0.06, "z_hi": 0.94, "dapi": 0.10, "membrane": 0.05, "polya": 0.6},
            "nucleus":            {"z_lo": 0.30, "z_hi": 0.70, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.15},
            "apical_membrane":    {"z_lo": 0.94, "z_hi": 1.00, "dapi": 0.0,  "membrane": 0.40, "polya": 0.0},
        },
    },
    "Fibroblast / CAF": {
        "z_extent_um": 8.0,
        "z_position_nucleus_relative": 0.50,
        "z_position_apical_relative": 1.0,
        "layers": {
            "membrane_shell":     {"z_lo": 0.00, "z_hi": 0.06, "dapi": 0.0, "membrane": 0.20, "polya": 0.0},
            "cytoplasm":          {"z_lo": 0.06, "z_hi": 0.94, "dapi": 0.10, "membrane": 0.05, "polya": 0.5},
            "nucleus":            {"z_lo": 0.30, "z_hi": 0.70, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.15},
            "membrane_shell_top": {"z_lo": 0.94, "z_hi": 1.00, "dapi": 0.0, "membrane": 0.20, "polya": 0.0},
        },
    },
    "Immune": {
        "z_extent_um": 7.0,
        "z_position_nucleus_relative": 0.50,
        "z_position_apical_relative": 1.0,
        "layers": {
            "membrane_shell":     {"z_lo": 0.00, "z_hi": 0.06, "dapi": 0.0, "membrane": 0.18, "polya": 0.0},
            "cytoplasm":          {"z_lo": 0.06, "z_hi": 0.94, "dapi": 0.10, "membrane": 0.04, "polya": 0.4},
            "nucleus":            {"z_lo": 0.30, "z_hi": 0.70, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.10},
            "membrane_shell_top": {"z_lo": 0.94, "z_hi": 1.00, "dapi": 0.0, "membrane": 0.18, "polya": 0.0},
        },
    },
    "Mural / pericyte": {
        "z_extent_um": 8.0,
        "z_position_nucleus_relative": 0.50,
        "z_position_apical_relative": 1.0,
        "layers": {
            "membrane_shell":     {"z_lo": 0.00, "z_hi": 0.06, "dapi": 0.0, "membrane": 0.18, "polya": 0.0},
            "cytoplasm":          {"z_lo": 0.06, "z_hi": 0.94, "dapi": 0.12, "membrane": 0.04, "polya": 0.3},
            "nucleus":            {"z_lo": 0.30, "z_hi": 0.70, "dapi": 1.0,  "membrane": 0.0,  "polya": 0.05},
            "membrane_shell_top": {"z_lo": 0.94, "z_hi": 1.00, "dapi": 0.0, "membrane": 0.18, "polya": 0.0},
        },
    },
}


def project_anatomy_through_slab(
    anatomy: dict[str, Any],
    z_position_um: float = 0.0,
    slab_thickness_um: float = DEFAULT_SLAB_THICKNESS_UM,
    channels: tuple[str, ...] = ("dapi", "membrane", "polya"),
) -> dict[str, float]:
    """Integrate per-layer anatomy through the section slab.

    Returns a dict of per-channel projected emission magnitudes (averaged
    over the cell's in-plane footprint, in same units as the layer
    ``emission_factor``). Layer emissions are summed, weighted by the
    fraction of the layer that intersects the slab.

    Args
    ----
    anatomy : dict
        Per-type anatomy (one entry from PER_TYPE_ANATOMY_PRIORS).
    z_position_um : float
        Cell-center offset from slab midplane (positive = above plane).
    slab_thickness_um : float
        Thickness of the section slab (default 8 µm).
    channels : tuple
        Which channels to compute.

    Returns
    -------
    dict[channel_name -> float]
        Per-channel projected emission, normalized by slab thickness.
    """
    z_extent = float(anatomy["z_extent_um"])
    half_extent = z_extent * 0.5
    half_slab = slab_thickness_um * 0.5
    # Cell occupies z in absolute coords [z_position - half_extent, z_position + half_extent].
    # Slab is [-half_slab, +half_slab].
    cell_lo = z_position_um - half_extent
    cell_hi = z_position_um + half_extent
    slab_lo = -half_slab
    slab_hi = +half_slab
    intersect_lo = max(cell_lo, slab_lo)
    intersect_hi = min(cell_hi, slab_hi)
    intersect_len = max(0.0, intersect_hi - intersect_lo)
    out = {ch: 0.0 for ch in channels}
    if intersect_len <= 0:
        return out
    layers = anatomy.get("layers", {})
    for layer_name, layer in layers.items():
        # Layer is at z_relative ∈ [z_lo, z_hi] within the cell. Convert to
        # absolute z: layer_lo_abs = cell_lo + z_lo * z_extent, etc.
        layer_lo_abs = cell_lo + float(layer["z_lo"]) * z_extent
        layer_hi_abs = cell_lo + float(layer["z_hi"]) * z_extent
        layer_in_slab_lo = max(layer_lo_abs, slab_lo)
        layer_in_slab_hi = min(layer_hi_abs, slab_hi)
        layer_in_slab_len = max(0.0, layer_in_slab_hi - layer_in_slab_lo)
        if layer_in_slab_len <= 0:
            continue
        # Contribution = layer emission factor * (length of layer-in-slab / slab_thickness).
        # This makes the integral dimensionally consistent and
        # automatically handles partial slab intersections.
        weight = layer_in_slab_len / slab_thickness_um
        for ch in channels:
            out[ch] += float(layer.get(ch, 0.0)) * weight
    return out


def fit_per_type_anatomy(
    crop_manifest_path: Path,
    params_path: Path,
    output_path: Path,
    *,
    cell_types_path: Path | None = None,
    slab_thickness_um: float = DEFAULT_SLAB_THICKNESS_UM,
    splits: tuple[str, ...] = ("train",),
    max_crops: int = 128,
) -> dict[str, Any]:
    """S1: configure per-type anatomy and validate against real per-type
    intensity distributions.

    For now this writes the biological priors (PER_TYPE_ANATOMY_PRIORS)
    into ``output_path`` and produces a validation report comparing the
    in-plane projection at z_position=0 to real per-type whole-cell
    intensity statistics.

    Future iterations of this fitter can tune the layer emission factors
    by minimising the bias between projected anatomy and real per-type
    means. For now we ship the priors and rely on S3 to validate.
    """
    from .validation import validate_crop_manifest
    from .cell_types import load_cell_type_assignment

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(params_path).open() as fh:
        params = json.load(fh)
    params["per_type_anatomy"] = {**PER_TYPE_ANATOMY_PRIORS}
    # Default anatomy under "unknown" key handles cells without a specific
    # type classification (~80% of canonical128 cells).
    params["per_type_anatomy"]["unknown"] = {**DEFAULT_ANATOMY}
    params["section_slab_thickness_um"] = float(slab_thickness_um)

    # Validation: compare projected per-type intensity (at z=0) to real
    # whole-cell-mask means from canonical128 train tiles.
    real_means: dict[str, dict[str, list[float]]] = {
        t: {"dapi": [], "membrane": [], "polya": []} for t in PER_TYPE_ANATOMY_PRIORS
    }
    if cell_types_path is not None:
        manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
        artifact = load_cell_type_assignment(cell_types_path)
        type_names = list(artifact["type_names"])
        cid_to_type = {}
        for crop in artifact.get("crops", []):
            cid_to_type.update({str(k): int(v) for k, v in crop.get("cell_id_to_type_index", {}).items()})
        root = Path(crop_manifest_path).parent
        n_used = 0
        for record in manifest.get("crops", []):
            if record.get("split") not in splits:
                continue
            if n_used >= max_crops:
                break
            with np.load(root / record["npz_path"], allow_pickle=True) as data:
                images = np.asarray(data["images"], dtype=np.float32)[:3]
                cl = np.asarray(data["cell_label"], dtype=np.int32)
                cell_ids = [str(v) for v in data["cell_ids"]]
            for j, lv in enumerate(np.unique(cl)[1:]):
                cid = cell_ids[j] if j < len(cell_ids) else None
                tidx = cid_to_type.get(str(cid), 0)
                if 0 < tidx < len(type_names):
                    tname = type_names[tidx]
                    if tname not in real_means:
                        continue
                    cell_pix = cl == int(lv)
                    if cell_pix.sum() < 16:
                        continue
                    for ci, ch in enumerate(["dapi", "membrane", "polya"]):
                        real_means[tname][ch].append(float(images[ci][cell_pix].mean()))
            n_used += 1

    # Compute predicted per-channel mean from the anatomy projected
    # through the slab at z_position=0 (fully-in-plane). This is the
    # "fully-in-plane" reference; in a slab where many cells are out
    # of plane, the population mean would be a mix.
    validation_report: dict[str, Any] = {}
    for tname, anat in PER_TYPE_ANATOMY_PRIORS.items():
        proj = project_anatomy_through_slab(anat, z_position_um=0.0, slab_thickness_um=slab_thickness_um)
        rec = {"projected_at_z0": proj}
        for ch in ("dapi", "membrane", "polya"):
            samples = real_means.get(tname, {}).get(ch, [])
            if samples:
                rec[f"real_mean_{ch}"] = float(np.median(samples))
                rec[f"n_cells_{ch}"] = len(samples)
        validation_report[tname] = rec

    params["per_type_anatomy_validation"] = validation_report

    with output_path.open("w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    return validation_report
