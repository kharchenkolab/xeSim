"""Plan3 A4 — fitted polyA spatial autocorrelation.

Fits per-type radial power spectrum of real polyA cytoplasm patches and
produces spectrum-matched noise fields at render time. Replaces the
isotropic smooth_noise polya_texture in mechanistic_render so the
refiner has structure to refine.

Method: for each cell, extract a square cytoplasm patch (cell mask minus
nucleus mask), compute its zero-mean FFT, take radial-binned mean of
|FFT|^2. Average across cells of each type. At render time, generate
spectrum-matched noise by:

  noise_fft = sqrt(target_radial_power) * exp(2*pi*i * uniform_phase)
  noise = iFFT(noise_fft).real

This produces a stationary Gaussian field whose radial power matches
real polyA of that type. The synthesis is per-image (one noise field per
render), then per-cell modulation (existing per-cell amplitude LUT)
applies on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cell_types import load_cell_type_assignment
from .schema import MANIFEST_SCHEMA_VERSION
from .validation import validate_crop_manifest


POLYA_SPECTRUM_TYPE = "xesim.polya_spectrum.v0"

PATCH_SIZE = 32  # cytoplasm patch in pixels


def fit_polya_spectrum(
    crop_manifest_path: Path,
    cell_types_path: Path,
    output_path: Path,
    patch_size: int = PATCH_SIZE,
    min_cytoplasm_px: int = 100,
    splits: tuple[str, ...] = ("train", "val", "test"),
    n_radial_bins: int = 12,
) -> Path:
    """Fit per-type radial power spectrum of real polyA cytoplasm fields."""

    manifest = validate_crop_manifest(crop_manifest_path, check_files=True)
    ct_artifact = load_cell_type_assignment(cell_types_path)
    type_names = list(ct_artifact["type_names"])
    n_types = len(type_names)
    cell_id_to_type: dict[str, int] = {}
    for crop in ct_artifact.get("crops", []):
        for cid, idx in crop.get("cell_id_to_type_index", {}).items():
            cell_id_to_type[str(cid)] = int(idx)

    accept = set(splits)
    root = crop_manifest_path.parent
    spectra_per_type: list[list[np.ndarray]] = [[] for _ in range(n_types)]
    fy = np.fft.fftfreq(patch_size).astype(np.float32)
    fx = np.fft.fftfreq(patch_size).astype(np.float32)
    fy2, fx2 = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(fy2 * fy2 + fx2 * fx2)
    r_max = float(r.max())
    bin_idx = ((r / max(r_max, 1e-6)) * n_radial_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_radial_bins - 1)
    bin_counts = np.bincount(bin_idx.ravel(), minlength=n_radial_bins).astype(np.float32)

    for crop in manifest.get("crops", []):
        if str(crop.get("split", "train")) not in accept:
            continue
        npz_path = root / crop["npz_path"]
        with np.load(npz_path, allow_pickle=True) as data:
            images = np.asarray(data["images"], dtype=np.float32)
            if images.shape[0] < 3:
                continue
            polya = images[2]
            cell_label = np.asarray(data["cell_label"], dtype=np.int32)
            nucleus_label = np.asarray(data["nucleus_label"], dtype=np.int32)
            cell_ids = [str(v) for v in data["cell_ids"].tolist()]
        h, w = polya.shape
        unique = np.unique(cell_label)
        nonzero = unique[unique > 0].tolist()
        if len(nonzero) != len(cell_ids):
            n = min(len(nonzero), len(cell_ids))
            nonzero = nonzero[:n]
            cell_ids = cell_ids[:n]
        for label_value, cell_id in zip(nonzero, cell_ids):
            type_idx = cell_id_to_type.get(cell_id, 0)
            if type_idx == 0:
                continue
            cell_mask = cell_label == int(label_value)
            cytoplasm = cell_mask & (nucleus_label == 0)
            if int(np.sum(cytoplasm)) < int(min_cytoplasm_px):
                continue
            ys, xs = np.nonzero(cytoplasm)
            if ys.size == 0:
                continue
            cy = int(round(float(np.mean(ys))))
            cx = int(round(float(np.mean(xs))))
            half = int(patch_size) // 2
            iy0 = cy - half
            ix0 = cx - half
            iy1 = iy0 + int(patch_size)
            ix1 = ix0 + int(patch_size)
            if iy0 < 0 or ix0 < 0 or iy1 > h or ix1 > w:
                continue
            patch = polya[iy0:iy1, ix0:ix1].astype(np.float32) - float(np.mean(polya[iy0:iy1, ix0:ix1]))
            ft = np.fft.fft2(patch)
            power = (ft.real * ft.real + ft.imag * ft.imag).astype(np.float32)
            radial = np.bincount(bin_idx.ravel(), weights=power.ravel(), minlength=n_radial_bins)
            radial = radial / np.maximum(bin_counts, 1.0)
            spectra_per_type[type_idx].append(radial.astype(np.float32))

    per_type_radial: list[list[float]] = []
    per_type_n: list[int] = []
    for ti in range(n_types):
        if not spectra_per_type[ti]:
            per_type_radial.append([1.0] * n_radial_bins)
            per_type_n.append(0)
            continue
        stack = np.stack(spectra_per_type[ti], axis=0)
        per_type_radial.append(stack.mean(axis=0).tolist())
        per_type_n.append(len(spectra_per_type[ti]))

    payload = {
        "type": POLYA_SPECTRUM_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(crop_manifest_path),
        "cell_types_path": str(cell_types_path),
        "type_names": type_names,
        "patch_size": int(patch_size),
        "n_radial_bins": int(n_radial_bins),
        "splits": list(splits),
        "per_type_radial_power": per_type_radial,
        "per_type_n_cells": per_type_n,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def load_polya_spectrum(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        obj = json.load(handle)
    if obj.get("type") != POLYA_SPECTRUM_TYPE:
        raise ValueError(f"{path}: not a polya_spectrum artifact")
    return obj


def synthesize_polya_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    radial_power: np.ndarray,
    n_radial_bins: int,
) -> np.ndarray:
    """Synthesize a stationary noise field with the target radial power
    spectrum.

    Builds a 2D amplitude map by interpolating the per-bin radial power,
    multiplies by a uniform random phase, takes the inverse FFT, and
    returns the real part. Output is approximately mean-zero, unit-std
    when ``radial_power`` is normalized.
    """

    h, w = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    fy2, fx2 = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(fy2 * fy2 + fx2 * fx2)
    r_max = float(r.max())
    # Map to bin indices as floats for linear interpolation.
    bin_pos = (r / max(r_max, 1e-6)) * n_radial_bins
    bin_lo = np.clip(np.floor(bin_pos).astype(np.int64), 0, n_radial_bins - 1)
    bin_hi = np.clip(bin_lo + 1, 0, n_radial_bins - 1)
    frac = (bin_pos - bin_lo).astype(np.float32)
    rp = np.asarray(radial_power, dtype=np.float32).clip(min=0.0)
    amplitude_2d = (1.0 - frac) * rp[bin_lo] + frac * rp[bin_hi]
    amplitude_2d = np.sqrt(np.clip(amplitude_2d, 1e-12, None))
    # Uniform random phase, with hermitian symmetry implicitly enforced
    # by taking the real part of the inverse FFT (real-valued output).
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(h, w)).astype(np.float32)
    fft = amplitude_2d * (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)
    field = np.fft.ifft2(fft).real.astype(np.float32)
    # Normalize to unit std.
    std = float(np.std(field))
    if std > 1e-6:
        field = field / std
    return field
