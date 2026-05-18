"""Plan3 step-18 evaluation harness for the mechanistic refiner.

This module computes population-level distributional metrics that go beyond
the paired MAE numbers in `xesim.mechanistic_refiner_samples.v0`. It is
explicitly designed around the redux from the prior `adv_xeSim` sprint:

- Aggregate scalar metrics frequently disagree with visual realism, so every
  distributional metric is reported alongside a real-vs-real baseline computed
  on a random split of the held-out crops. That baseline is the ceiling of
  "indistinguishable from real" given finite samples.
- DINOv2 perceptual features were the single load-bearing tool that broke
  through MSE blur in `adv_xeSim`; we reuse them here as a perceptual
  distance, applied inside foreground-support masks so the score cannot
  reward leakage.
- Real Xenium membrane fibers concentrate at very fine wavelengths
  (~3 px); aggregate radial spectrum bins must be foreground-only and
  per-channel, otherwise the structure averages out.
- Anisotropy from segmentation-axis features was a dead end; we measure
  cardinal-axis preference at the *image* level instead, as a distributional
  property to match.

The CLI command `xesim eval-mechanistic-refiner` produces a manifest of type
`xesim.mechanistic_refiner_eval.v0` with all of the above plus a visual
sheet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .qc import assemble_compare_rows, mask_tile, residual_tile_from_arrays, stain_tile, write_png
from .schema import MANIFEST_SCHEMA_VERSION, MECHANISTIC_REFINER_EVAL_TYPE
from .torch_utils import get_device
from .validation import (
    ManifestValidationError,
    validate_mechanistic_refiner_corpus,
    validate_mechanistic_refiner_samples,
)

DEFAULT_PATCH_SIZE = 192
DEFAULT_PATCH_STRIDE = 96
DEFAULT_DINO_INPUT_SIZE = 224
DEFAULT_SPECTRUM_BINS = 24
DEFAULT_POLAR_RADIAL_BINS = 8
DEFAULT_POLAR_ORIENT_BINS = 8

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# DINOv2 perceptual features
# ---------------------------------------------------------------------------


class DINOv2Embedder:
    """Lazy DINOv2 feature extractor for 2-channel Xenium patches.

    Loads `facebookresearch/dinov2:dinov2_vits14` on first call. The first
    load downloads weights via torch.hub; in offline contexts the constructor
    may fail. Callers should treat this as best-effort and skip DINO metrics
    when ``available`` is False.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: torch.device | str | None = None):
        self.model_name = str(model_name)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._model: torch.nn.Module | None = None
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)
        self._available: bool | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                self._ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - we intentionally swallow
                self._available = False
                self._load_error = str(exc)
        return bool(self._available)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model = torch.hub.load("facebookresearch/dinov2", self.model_name, verbose=False)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(self.device)
        self._model = model
        self._available = True

    @staticmethod
    def repack_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """Repack image tensor to (B, 3, H, W).

        2-channel input: DAPI->R, membrane->G, mean(DAPI, membrane)->B.
        3-channel input: DAPI->R, membrane->G, polyA->B (Plan3 polyA pipeline).
        """
        if x.ndim != 4:
            raise ValueError("DINOv2Embedder expects a (B, C, H, W) tensor")
        if x.shape[1] == 3:
            return x
        if x.shape[1] == 2:
            dapi = x[:, 0:1]
            memb = x[:, 1:2]
            avg = (dapi + memb) * 0.5
            return torch.cat([dapi, memb, avg], dim=1)
        raise ValueError(f"DINOv2Embedder supports 2 or 3 channels, got {x.shape[1]}")

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repack_to_rgb(x)
        target = DEFAULT_DINO_INPUT_SIZE
        if x.shape[-1] != target or x.shape[-2] != target:
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embedding per patch as ``(B, C)``. No-grad inference path."""

        self._ensure_loaded()
        assert self._model is not None
        prep = self._prep(x.to(self.device))
        with torch.no_grad():
            feats = self._model(prep)
        return feats.detach()

    def features_for_training(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embeddings keeping the graph open through ``x``.

        DINO weights stay frozen (set ``requires_grad=False`` in
        ``_ensure_loaded``) but the forward pass remains differentiable so a
        gradient can flow back into the refined image.
        """

        self._ensure_loaded()
        assert self._model is not None
        prep = self._prep(x.to(self.device))
        return self._model(prep)


# ---------------------------------------------------------------------------
# Spectrum helpers (ported from adv_xeSim/xesim/gen/cell_diffusion.py)
# ---------------------------------------------------------------------------


def _make_radial_bins(h: int, w: int, num_bins: int, device: torch.device):
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fy2 * fy2 + fx2 * fx2)
    r_max = float(r.max().item())
    bin_idx = ((r / (r_max + 1e-9)) * num_bins).long().clamp(max=num_bins - 1)
    counts = torch.bincount(bin_idx.flatten(), minlength=num_bins).float().clamp(min=1.0)
    return bin_idx, counts


def radial_log_power(patches: torch.Tensor, num_bins: int = DEFAULT_SPECTRUM_BINS) -> torch.Tensor:
    """Compute per-(patch, channel) radial-averaged log power-spectrum.

    Returns a tensor of shape ``(N, C, num_bins)``.
    """

    if patches.ndim != 4:
        raise ValueError("radial_log_power expects (N, C, H, W)")
    n, c, h, w = patches.shape
    bin_idx, counts = _make_radial_bins(h, w, num_bins, patches.device)
    fft = torch.fft.rfft2(patches, norm="ortho")
    power = fft.real * fft.real + fft.imag * fft.imag
    flat = power.reshape(n, c, -1)
    idx = bin_idx.flatten().reshape(1, 1, -1).expand(n, c, -1)
    out = torch.zeros(n, c, num_bins, device=patches.device, dtype=patches.dtype)
    out.scatter_add_(2, idx, flat)
    out = out / counts.reshape(1, 1, -1)
    return torch.log(out + 1e-8)


def _make_polar_bins(h: int, w: int, num_radial: int, num_orient: int, device: torch.device):
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fy2 * fy2 + fx2 * fx2)
    r_max = float(r.max().item())
    r_idx = ((r / (r_max + 1e-9)) * num_radial).long().clamp(max=num_radial - 1)
    angle = torch.atan2(fy2, fx2)
    angle = torch.remainder(angle, math.pi)
    o_idx = ((angle / math.pi) * num_orient).long().clamp(max=num_orient - 1)
    bin_idx = r_idx * num_orient + o_idx
    n_bins = num_radial * num_orient
    counts = torch.bincount(bin_idx.flatten(), minlength=n_bins).float().clamp(min=1.0)
    return bin_idx, counts, n_bins


def polar_log_power(patches: torch.Tensor, num_radial: int = DEFAULT_POLAR_RADIAL_BINS,
                    num_orient: int = DEFAULT_POLAR_ORIENT_BINS) -> torch.Tensor:
    """Compute per-(patch, channel) polar-binned log power. Shape ``(N, C, R*O)``."""

    if patches.ndim != 4:
        raise ValueError("polar_log_power expects (N, C, H, W)")
    n, c, h, w = patches.shape
    bin_idx, counts, total = _make_polar_bins(h, w, num_radial, num_orient, patches.device)
    fft = torch.fft.rfft2(patches, norm="ortho")
    power = fft.real * fft.real + fft.imag * fft.imag
    flat = power.reshape(n, c, -1)
    idx = bin_idx.flatten().reshape(1, 1, -1).expand(n, c, -1)
    out = torch.zeros(n, c, total, device=patches.device, dtype=patches.dtype)
    out.scatter_add_(2, idx, flat)
    out = out / counts.reshape(1, 1, -1)
    return torch.log(out + 1e-8)


def cardinal_anisotropy(patches: torch.Tensor, num_radial: int = DEFAULT_POLAR_RADIAL_BINS,
                       num_orient: int = DEFAULT_POLAR_ORIENT_BINS) -> torch.Tensor:
    """Per-(patch, channel) cardinal-vs-diagonal anisotropy ratio.

    Returns ``(N, C)`` of ``mean log power at orientation indices closest to 0
    or pi/2 minus the mean across the diagonal indices``. Higher = more
    cardinal-axis structure (the regime real Xenium membrane sits in).
    """

    log_power = polar_log_power(patches, num_radial=num_radial, num_orient=num_orient)
    n, c, _ = log_power.shape
    rad = log_power.reshape(n, c, num_radial, num_orient)
    # Use the mid-band radial bins (skip DC and very high freqs) to avoid noise.
    mid = rad[..., max(1, num_radial // 4): max(2, num_radial - 1), :]
    mid = mid.mean(dim=2)  # (N, C, num_orient)
    o = num_orient
    # closest-to-cardinal vs closest-to-diagonal indices
    cardinal_idx = [0, o // 2]
    diag_idx = [o // 4, (3 * o) // 4]
    cardinal = mid[..., cardinal_idx].mean(dim=-1)
    diagonal = mid[..., diag_idx].mean(dim=-1)
    return cardinal - diagonal


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def _foreground_mask_threshold(support: np.ndarray, valid: np.ndarray, min_fraction: float = 0.4) -> bool:
    """Whether a patch has enough valid foreground to be evaluated."""

    if support.size == 0:
        return False
    valid_pixels = float(np.mean(valid))
    if valid_pixels < 0.5:
        return False
    return float(np.mean(support * valid)) >= float(min_fraction)


def extract_patches(image: np.ndarray, support: np.ndarray, valid: np.ndarray,
                    patch_size: int = DEFAULT_PATCH_SIZE,
                    stride: int = DEFAULT_PATCH_STRIDE,
                    min_foreground_fraction: float = 0.4,
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sliding-window extraction of foreground-rich patches.

    Returns ``(patches, support_patches)`` where each patch is a ``(C, P, P)``
    array with image values masked outside ``valid`` zeroed and ``support``
    coverage applied. Patches with insufficient valid/foreground coverage are
    skipped so distributional metrics are not dominated by padding.
    """

    if image.ndim != 3:
        raise ValueError("extract_patches expects (C, H, W) image")
    c, h, w = image.shape
    if patch_size > h or patch_size > w:
        return [], []
    patches: list[np.ndarray] = []
    sup_patches: list[np.ndarray] = []
    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))
    if not y_positions or y_positions[-1] != h - patch_size:
        y_positions.append(h - patch_size)
    if not x_positions or x_positions[-1] != w - patch_size:
        x_positions.append(w - patch_size)
    seen: set[tuple[int, int]] = set()
    for y in y_positions:
        for x in x_positions:
            if (y, x) in seen:
                continue
            seen.add((y, x))
            sup = support[y:y + patch_size, x:x + patch_size]
            val = valid[y:y + patch_size, x:x + patch_size]
            if not _foreground_mask_threshold(sup, val, min_foreground_fraction):
                continue
            img = image[:, y:y + patch_size, x:x + patch_size].copy()
            img *= val[None, ...]
            patches.append(img.astype(np.float32, copy=False))
            sup_patches.append((sup * val).astype(np.float32, copy=False))
    return patches, sup_patches


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = np.sort(a.ravel())
    bb = np.sort(b.ravel())
    if aa.shape[0] != bb.shape[0]:
        # Resample by linear interpolation to common quantile grid.
        n = max(aa.shape[0], bb.shape[0])
        u = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
        aa = np.interp(u, np.linspace(0.0, 1.0, aa.shape[0], endpoint=False) + 0.5 / aa.shape[0], aa)
        bb = np.interp(u, np.linspace(0.0, 1.0, bb.shape[0], endpoint=False) + 0.5 / bb.shape[0], bb)
    return float(np.mean(np.abs(aa - bb)))


def two_sample_nn_accuracy(features_a: np.ndarray, features_b: np.ndarray, seed: int = 7) -> float:
    """Leave-one-out nearest-neighbor binary accuracy in feature space.

    0.5 = indistinguishable, 1.0 = perfectly separable. Useful as a paired
    check on distributional distances. ``features_a`` and ``features_b`` are
    ``(N, D)`` arrays, not necessarily the same N.
    """

    if features_a.size == 0 or features_b.size == 0:
        return 0.5
    fa = np.ascontiguousarray(features_a, dtype=np.float32)
    fb = np.ascontiguousarray(features_b, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if fa.shape[0] > fb.shape[0]:
        idx = rng.choice(fa.shape[0], size=fb.shape[0], replace=False)
        fa = fa[idx]
    elif fb.shape[0] > fa.shape[0]:
        idx = rng.choice(fb.shape[0], size=fa.shape[0], replace=False)
        fb = fb[idx]
    feats = np.concatenate([fa, fb], axis=0)
    labels = np.concatenate([np.zeros(fa.shape[0], dtype=np.int64), np.ones(fb.shape[0], dtype=np.int64)])
    n = feats.shape[0]
    if n < 4:
        return 0.5
    # Compute squared pairwise distances; mask diagonal.
    norms = np.sum(feats * feats, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (feats @ feats.T)
    np.fill_diagonal(dist2, np.inf)
    nn = np.argmin(dist2, axis=1)
    nn_labels = labels[nn]
    return float(np.mean(nn_labels == labels))


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_mechanistic_refiner(
    corpus_path: Path,
    samples_path: Path,
    output_dir: Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_stride: int = DEFAULT_PATCH_STRIDE,
    spectrum_bins: int = DEFAULT_SPECTRUM_BINS,
    polar_radial_bins: int = DEFAULT_POLAR_RADIAL_BINS,
    polar_orient_bins: int = DEFAULT_POLAR_ORIENT_BINS,
    min_foreground_fraction: float = 0.4,
    enable_dino: bool = True,
    dino_model: str = "dinov2_vits14",
    real_vs_real_seed: int = 7,
    device_name: str | None = None,
    tile_size: int = 128,
    num_visual_rows: int = 6,
    cell_types_path: Path | None = None,
    canonical_manifest_path: Path | None = None,
) -> Path:
    """Evaluate a sampled mechanistic refiner against held-out real crops.

    Reads the refiner corpus (real + rough + masks) and the refiner samples
    (refined + paired metrics), extracts foreground patches, computes
    distributional metrics including DINOv2 perceptual distance, radial
    log-power spectrum L1, and cardinal anisotropy, and reports each metric
    against a real-vs-real baseline derived from a random split of the
    held-out crops.
    """

    corpus = validate_mechanistic_refiner_corpus(corpus_path, check_files=True)
    samples = validate_mechanistic_refiner_samples(samples_path, check_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = _load_corpus_arrays(corpus_path, corpus)
    refined_arrays = _load_sample_arrays(samples_path, samples)
    selected_indices = refined_arrays["indices"]

    real = arrays["target"][selected_indices]
    rough = arrays["rough"][selected_indices]
    refined = refined_arrays["refined"]
    support = arrays["support_mask"][selected_indices]
    valid = arrays["valid_mask"][selected_indices]
    crop_ids = list(arrays["crop_ids"][selected_indices])
    splits = list(arrays["splits"][selected_indices])
    if real.shape != refined.shape or rough.shape != refined.shape:
        raise ManifestValidationError(
            f"shape mismatch: real {real.shape} rough {rough.shape} refined {refined.shape}"
        )

    real_patches, _ = _extract_role_patches(real, support, valid, patch_size, patch_stride, min_foreground_fraction)
    rough_patches, _ = _extract_role_patches(rough, support, valid, patch_size, patch_stride, min_foreground_fraction)
    refined_patches, _ = _extract_role_patches(refined, support, valid, patch_size, patch_stride, min_foreground_fraction)

    # Optional per-type stratification: produce a parallel patch-type label for
    # every extracted real patch, recording the dominant cell-type index.
    patch_type_labels: list[int] | None = None
    type_names: list[str] = []
    if cell_types_path is not None and canonical_manifest_path is not None:
        from .cell_types import build_per_pixel_type_label, load_cell_type_assignment

        ct_artifact = load_cell_type_assignment(cell_types_path)
        type_names = list(ct_artifact["type_names"])
        ct_lookup = {
            str(c["crop_id"]): {str(k): int(v) for k, v in c["cell_id_to_type_index"].items()}
            for c in ct_artifact.get("crops", [])
        }
        from .validation import validate_crop_manifest

        canon = validate_crop_manifest(canonical_manifest_path, check_files=True)
        canon_root = canonical_manifest_path.parent
        canon_records = {str(c["crop_id"]): c for c in canon.get("crops", [])}
        patch_type_labels = []
        # Recompute the dominant type per patch by re-walking the records and crops in order.
        for record_idx, crop_id in enumerate(crop_ids):
            mapping = ct_lookup.get(str(crop_id))
            if mapping is None:
                # No type info for this crop; assign 0 to all patches from it
                _real_patches, _ = extract_patches(real[record_idx], support[record_idx], valid[record_idx],
                                                     patch_size, patch_stride, min_foreground_fraction)
                patch_type_labels.extend([0] * len(_real_patches))
                continue
            canon_npz_path = canon_root / canon_records[str(crop_id)]["npz_path"]
            with np.load(canon_npz_path, allow_pickle=True) as canon_data:
                canon_cell_label = np.asarray(canon_data["cell_label"], dtype=np.int32)
                canon_cell_ids = [str(v) for v in canon_data["cell_ids"].tolist()]
            type_label = build_per_pixel_type_label(canon_cell_label, canon_cell_ids, mapping)
            # Pad to render shape if needed (canonical and corpus shapes may differ in edge crops).
            if type_label.shape != real[record_idx].shape[1:]:
                # Pad smaller, crop larger.
                target_h, target_w = real[record_idx].shape[1:]
                src_h, src_w = type_label.shape
                resized = np.zeros((target_h, target_w), dtype=np.int16)
                resized[: min(src_h, target_h), : min(src_w, target_w)] = type_label[: min(src_h, target_h), : min(src_w, target_w)]
                type_label = resized
            # Walk patches in the same order extract_patches does and label by mode.
            patches_this, _ = extract_patches(real[record_idx], support[record_idx], valid[record_idx],
                                              patch_size, patch_stride, min_foreground_fraction)
            # Re-extract with the same iteration but read the type label for each accepted patch.
            h, w = type_label.shape
            y_positions = list(range(0, h - patch_size + 1, patch_stride))
            x_positions = list(range(0, w - patch_size + 1, patch_stride))
            if not y_positions or y_positions[-1] != h - patch_size:
                y_positions.append(h - patch_size)
            if not x_positions or x_positions[-1] != w - patch_size:
                x_positions.append(w - patch_size)
            seen: set[tuple[int, int]] = set()
            collected = 0
            for y in y_positions:
                for x in x_positions:
                    if (y, x) in seen:
                        continue
                    seen.add((y, x))
                    sup_patch = support[record_idx, y:y + patch_size, x:x + patch_size]
                    val_patch = valid[record_idx, y:y + patch_size, x:x + patch_size]
                    if not _foreground_mask_threshold(sup_patch, val_patch, min_foreground_fraction):
                        continue
                    type_patch = type_label[y:y + patch_size, x:x + patch_size]
                    fg = (sup_patch * val_patch) > 0.5
                    if not np.any(fg):
                        patch_type_labels.append(0)
                    else:
                        # Mode of nonzero type values inside the foreground; fall back to 0.
                        vals = type_patch[fg]
                        nonzero = vals[vals > 0]
                        if nonzero.size:
                            counts = np.bincount(nonzero)
                            patch_type_labels.append(int(np.argmax(counts)))
                        else:
                            patch_type_labels.append(0)
                    collected += 1
            assert collected == len(patches_this), f"patch enumeration mismatch for {crop_id}"

    n_real = len(real_patches)
    n_rough = len(rough_patches)
    n_ref = len(refined_patches)
    if n_real == 0 or n_rough == 0 or n_ref == 0:
        raise ManifestValidationError(
            f"insufficient foreground patches: real={n_real} rough={n_rough} refined={n_ref}"
        )

    rng = np.random.default_rng(real_vs_real_seed)
    real_split = rng.permutation(n_real)
    half = n_real // 2
    real_a_idx = real_split[:half]
    real_b_idx = real_split[half:2 * half]

    real_arr = np.stack(real_patches, axis=0)
    rough_arr = np.stack(rough_patches, axis=0)
    refined_arr = np.stack(refined_patches, axis=0)

    device = get_device(device_name)
    real_t = torch.from_numpy(real_arr).to(device)
    rough_t = torch.from_numpy(rough_arr).to(device)
    refined_t = torch.from_numpy(refined_arr).to(device)

    # ------------------------------------------------------------------
    # Distributional metrics: radial log-power spectrum and anisotropy.
    # ------------------------------------------------------------------
    real_radial = radial_log_power(real_t, num_bins=spectrum_bins).detach().cpu().numpy()
    rough_radial = radial_log_power(rough_t, num_bins=spectrum_bins).detach().cpu().numpy()
    refined_radial = radial_log_power(refined_t, num_bins=spectrum_bins).detach().cpu().numpy()
    real_aniso = cardinal_anisotropy(real_t, num_radial=polar_radial_bins, num_orient=polar_orient_bins).detach().cpu().numpy()
    rough_aniso = cardinal_anisotropy(rough_t, num_radial=polar_radial_bins, num_orient=polar_orient_bins).detach().cpu().numpy()
    refined_aniso = cardinal_anisotropy(refined_t, num_radial=polar_radial_bins, num_orient=polar_orient_bins).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # DINOv2 perceptual features (best-effort).
    # ------------------------------------------------------------------
    dino_block: dict[str, Any] = {"available": False}
    if enable_dino:
        embedder = DINOv2Embedder(model_name=dino_model, device=device)
        if embedder.available:
            real_feats = _embed_in_batches(embedder, real_t).cpu().numpy()
            rough_feats = _embed_in_batches(embedder, rough_t).cpu().numpy()
            refined_feats = _embed_in_batches(embedder, refined_t).cpu().numpy()
            dino_block = _summarize_dino(real_feats, rough_feats, refined_feats, real_a_idx, real_b_idx, real_vs_real_seed)
            dino_block["available"] = True
            dino_block["model"] = embedder.model_name
        else:
            dino_block = {
                "available": False,
                "error": embedder._load_error or "DINOv2 unavailable",
                "model": embedder.model_name,
            }

    # ------------------------------------------------------------------
    # Real-vs-real baseline + role distances on every distributional metric.
    # ------------------------------------------------------------------
    channel_names = ["dapi", "membrane", "polya"][: real.shape[1]]
    spectrum_block = _summarize_distribution(
        "radial_log_power_spectrum",
        real_radial,
        rough_radial,
        refined_radial,
        real_a_idx,
        real_b_idx,
        real_vs_real_seed,
        per_channel_names=channel_names,
    )
    aniso_block = _summarize_distribution(
        "cardinal_anisotropy",
        real_aniso[..., None],  # treat as one-feature distribution per channel
        rough_aniso[..., None],
        refined_aniso[..., None],
        real_a_idx,
        real_b_idx,
        real_vs_real_seed,
        per_channel_names=channel_names,
    )

    # ------------------------------------------------------------------
    # Pull paired-metric summaries directly from the samples report.
    # ------------------------------------------------------------------
    paired_block = {
        "num_samples": int(samples["num_samples"]),
        "selected_step": int(samples.get("selected_step", 0)),
        "summary": dict(samples.get("summary", {})),
    }

    # ------------------------------------------------------------------
    # Stratified-by-type evaluation.
    # ------------------------------------------------------------------
    stratified_block: dict[str, Any] = {"available": False}
    if patch_type_labels is not None and type_names:
        labels_arr = np.asarray(patch_type_labels, dtype=np.int64)
        if labels_arr.shape[0] == n_real:
            per_type: dict[str, Any] = {}
            for type_idx, type_name in enumerate(type_names):
                indices = np.where(labels_arr == type_idx)[0]
                if indices.size < 4:
                    continue
                real_subset = real_radial[indices]
                rough_subset = rough_radial[indices]
                refined_subset = refined_radial[indices]
                aniso_real_sub = real_aniso[indices][..., None]
                aniso_rough_sub = rough_aniso[indices][..., None]
                aniso_ref_sub = refined_aniso[indices][..., None]
                # Within-type real-vs-real split
                rng_t = np.random.default_rng(real_vs_real_seed + type_idx)
                perm = rng_t.permutation(indices.size)
                half = indices.size // 2
                a_idx = perm[:half]
                b_idx = perm[half:2 * half]
                spectrum_summary = _summarize_distribution(
                    f"radial_log_power_spectrum_{type_name}",
                    real_subset, rough_subset, refined_subset,
                    a_idx, b_idx, real_vs_real_seed + type_idx,
                    per_channel_names=channel_names,
                )
                aniso_summary = _summarize_distribution(
                    f"cardinal_anisotropy_{type_name}",
                    aniso_real_sub, aniso_rough_sub, aniso_ref_sub,
                    a_idx, b_idx, real_vs_real_seed + type_idx,
                    per_channel_names=channel_names,
                )
                per_type[type_name] = {
                    "num_patches": int(indices.size),
                    "spectrum": spectrum_summary,
                    "anisotropy": aniso_summary,
                }
                if dino_block.get("available") and 'real_feats' in locals():
                    dino_real_sub = real_feats[indices]
                    dino_rough_sub = rough_feats[indices]
                    dino_refined_sub = refined_feats[indices]
                    per_type[type_name]["dino"] = {
                        "real_vs_real_feature_l2": _feature_l2(dino_real_sub[a_idx], dino_real_sub[b_idx]),
                        "rough_vs_real_feature_l2": _feature_l2(dino_rough_sub, dino_real_sub),
                        "refined_vs_real_feature_l2": _feature_l2(dino_refined_sub, dino_real_sub),
                    }
            stratified_block = {
                "available": True,
                "type_names": type_names,
                "per_type": per_type,
            }

    # ------------------------------------------------------------------
    # Visual sheet — pick records with highest absolute residual delta so the
    # sheet exposes both wins and failures.
    # ------------------------------------------------------------------
    sample_records = samples.get("records", [])
    sample_records_sorted = sorted(
        sample_records,
        key=lambda rec: abs(float(rec.get("mae_delta", 0.0))),
        reverse=True,
    )
    n_rows = min(int(num_visual_rows), len(sample_records_sorted))
    rows: list[list[np.ndarray]] = []
    display_lut = corpus.get("display_lut")
    for record in sample_records_sorted[:n_rows]:
        idx = int(record["index"])
        target_np = arrays["target"][idx]
        rough_np = arrays["rough"][idx]
        sup = arrays["support_mask"][idx]
        leak = arrays["leakage_mask"][idx]
        # Find the matching refined sample (since sample order may differ).
        ref_position = int(np.where(refined_arrays["indices"] == idx)[0][0])
        refined_np = refined_arrays["refined"][ref_position]
        residual_np = refined_arrays["residual"][ref_position]
        leakage_residual = np.abs(residual_np).mean(axis=0) * leak
        rows.append([
            stain_tile(target_np, tile_size, display_lut=display_lut),
            stain_tile(rough_np, tile_size, display_lut=display_lut),
            stain_tile(refined_np, tile_size, display_lut=display_lut),
            residual_tile_from_arrays(target_np, rough_np, tile_size, display_lut=display_lut),
            residual_tile_from_arrays(target_np, refined_np, tile_size, display_lut=display_lut),
            _mono_tile(leakage_residual, tile_size),
        ])
    sheet_path = output_dir / "mechanistic_refiner_eval_sheet.png"
    if rows:
        write_png(sheet_path, assemble_compare_rows(rows, padding=4))

    # ------------------------------------------------------------------
    # RGB sheet (DAPI=red, membrane=green) — primary visual gate.
    # ------------------------------------------------------------------
    rgb_rows: list[list[np.ndarray]] = []
    for record in sample_records_sorted[:n_rows]:
        idx = int(record["index"])
        target_np = arrays["target"][idx]
        rough_np = arrays["rough"][idx]
        ref_position = int(np.where(refined_arrays["indices"] == idx)[0][0])
        refined_np = refined_arrays["refined"][ref_position]
        # Calibrate every panel to the real-image percentile range so the
        # comparison stays fair across panels.
        ref_target = target_np
        rgb_real = to_dapi_red_membrane_green_rgb(target_np, ref_target)
        rgb_rough = to_dapi_red_membrane_green_rgb(rough_np, ref_target)
        rgb_refined = to_dapi_red_membrane_green_rgb(refined_np, ref_target)
        rgb_rows.append([
            _rgb_tile(rgb_real, tile_size),
            _rgb_tile(rgb_rough, tile_size),
            _rgb_tile(rgb_refined, tile_size),
        ])
    rgb_sheet_path = output_dir / "mechanistic_refiner_eval_rgb_sheet.png"
    if rgb_rows:
        write_png(rgb_sheet_path, assemble_compare_rows(rgb_rows, padding=4))

    # ------------------------------------------------------------------
    # Manifest assembly.
    # ------------------------------------------------------------------
    manifest = {
        "type": MECHANISTIC_REFINER_EVAL_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_corpus": str(corpus_path),
        "source_samples": str(samples_path),
        "sheet_path": sheet_path.name if rows else None,
        "rgb_sheet_path": rgb_sheet_path.name if rgb_rows else None,
        "patch_size": int(patch_size),
        "patch_stride": int(patch_stride),
        "min_foreground_fraction": float(min_foreground_fraction),
        "num_real_patches": int(n_real),
        "num_rough_patches": int(n_rough),
        "num_refined_patches": int(n_ref),
        "num_records": int(len(crop_ids)),
        "num_visual_rows": int(n_rows),
        "splits": list(set(str(s) for s in splits)),
        "device": str(device),
        "paired_metrics": paired_block,
        "spectrum": spectrum_block,
        "anisotropy": aniso_block,
        "dino": dino_block,
        "stratified": stratified_block,
    }
    manifest_path = output_dir / "mechanistic_refiner_eval.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def _embed_in_batches(embedder: DINOv2Embedder, patches: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in range(0, patches.shape[0], batch_size):
        chunk = patches[start: start + batch_size]
        feats.append(embedder.features(chunk))
    return torch.cat(feats, dim=0)


def _summarize_dino(real: np.ndarray, rough: np.ndarray, refined: np.ndarray,
                    real_a_idx: np.ndarray, real_b_idx: np.ndarray, seed: int) -> dict[str, Any]:
    real_a = real[real_a_idx]
    real_b = real[real_b_idx]
    out = {
        "real_vs_real_feature_l2": _feature_l2(real_a, real_b),
        "rough_vs_real_feature_l2": _feature_l2(rough, real),
        "refined_vs_real_feature_l2": _feature_l2(refined, real),
        "real_vs_real_nn_accuracy": two_sample_nn_accuracy(real_a, real_b, seed=seed),
        "rough_vs_real_nn_accuracy": two_sample_nn_accuracy(rough, real, seed=seed),
        "refined_vs_real_nn_accuracy": two_sample_nn_accuracy(refined, real, seed=seed),
        "feature_dim": int(real.shape[1]),
    }
    return out


def _feature_l2(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    return float(np.sqrt(np.sum((mean_a - mean_b) ** 2)))


def _summarize_distribution(name: str, real: np.ndarray, rough: np.ndarray, refined: np.ndarray,
                             real_a_idx: np.ndarray, real_b_idx: np.ndarray, seed: int,
                             per_channel_names: list[str]) -> dict[str, Any]:
    """For each channel, report Wasserstein-1 distance per feature and mean."""

    if real.ndim != 3 or rough.ndim != 3 or refined.ndim != 3:
        raise ValueError(f"{name}: arrays must be (N, C, F)")
    n_channels = real.shape[1]
    out: dict[str, Any] = {"name": name, "num_features": int(real.shape[2])}
    real_a = real[real_a_idx]
    real_b = real[real_b_idx]
    per_channel: dict[str, dict[str, float]] = {}
    for c in range(n_channels):
        ch_name = per_channel_names[c] if c < len(per_channel_names) else f"channel_{c}"
        real_a_ch = real_a[:, c, :].reshape(-1)
        real_b_ch = real_b[:, c, :].reshape(-1)
        rough_ch = rough[:, c, :].reshape(-1)
        refined_ch = refined[:, c, :].reshape(-1)
        real_ch_full = real[:, c, :].reshape(-1)
        per_channel[ch_name] = {
            "real_vs_real_w1": wasserstein_1d(real_a_ch, real_b_ch),
            "rough_vs_real_w1": wasserstein_1d(rough_ch, real_ch_full),
            "refined_vs_real_w1": wasserstein_1d(refined_ch, real_ch_full),
        }
    out["per_channel"] = per_channel
    out["aggregate_real_vs_real_w1"] = float(np.mean([v["real_vs_real_w1"] for v in per_channel.values()]))
    out["aggregate_rough_vs_real_w1"] = float(np.mean([v["rough_vs_real_w1"] for v in per_channel.values()]))
    out["aggregate_refined_vs_real_w1"] = float(np.mean([v["refined_vs_real_w1"] for v in per_channel.values()]))
    return out


def _extract_role_patches(images: np.ndarray, support: np.ndarray, valid: np.ndarray,
                          patch_size: int, stride: int, min_fg: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_patches: list[np.ndarray] = []
    all_support: list[np.ndarray] = []
    for i in range(images.shape[0]):
        patches, sups = extract_patches(images[i], support[i], valid[i], patch_size, stride, min_fg)
        all_patches.extend(patches)
        all_support.extend(sups)
    return all_patches, all_support


def _load_corpus_arrays(corpus_path: Path, corpus: dict[str, Any]) -> dict[str, np.ndarray]:
    npz_path = corpus_path.parent / str(corpus["npz_path"])
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "target": data["target"].astype(np.float32, copy=False),
            "rough": data["rough"].astype(np.float32, copy=False),
            "support_mask": data["foreground_support_mask"].astype(np.float32, copy=False),
            "leakage_mask": data["leakage_mask"].astype(np.float32, copy=False),
            "valid_mask": data["valid_mask"].astype(np.float32, copy=False),
            "crop_ids": np.asarray([str(value) for value in data["crop_ids"].tolist()], dtype=object),
            "splits": np.asarray([str(value) for value in data["splits"].tolist()], dtype=object),
        }


def _load_sample_arrays(samples_path: Path, samples: dict[str, Any]) -> dict[str, np.ndarray]:
    npz_path = samples_path.parent / str(samples["npz_path"])
    with np.load(npz_path, allow_pickle=True) as data:
        required = ["refined", "residual", "indices"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ManifestValidationError(f"{npz_path}: missing arrays: {', '.join(missing)}")
        return {
            "refined": data["refined"].astype(np.float32, copy=False),
            "residual": data["residual"].astype(np.float32, copy=False),
            "indices": data["indices"].astype(np.int64, copy=False),
        }


def _mono_tile(values: np.ndarray, size: int) -> np.ndarray:
    arr = values.astype(np.float32, copy=True)
    if np.max(arr) > 0:
        arr = arr / (np.percentile(arr, 99.0) + 1e-6)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = _nn_resize_2d(arr, size)
    return np.stack([arr, arr, arr], axis=-1)


def to_dapi_red_membrane_green_rgb(img: np.ndarray, ref: np.ndarray | None = None,
                                    p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Render a 2- or 3-channel image as RGB.

    2-channel (DAPI, membrane): DAPI -> R, membrane -> G, blue stays 0.
    3-channel (DAPI, membrane, polyA): DAPI -> R, membrane -> G, polyA -> B.

    Each channel is independently rescaled to its reference percentile range
    so panels stay comparable. If ``ref`` is None, ``img`` is used as its own
    reference.
    """

    if img.ndim != 3 or img.shape[0] < 2:
        raise ValueError("to_dapi_red_membrane_green_rgb expects (C, H, W) with C >= 2")
    if ref is None:
        ref = img
    h, w = img.shape[1], img.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    n = min(img.shape[0], 3)
    mapping = [(0, 0), (1, 1), (2, 2)][:n]  # DAPI->R, membrane->G, polyA->B
    for ch_idx, color_idx in mapping:
        ref_ch = ref[ch_idx]
        lo = float(np.percentile(ref_ch, p_low))
        hi = float(np.percentile(ref_ch, p_high))
        rng = max(hi - lo, 1e-6)
        v = np.clip((img[ch_idx] - lo) / rng, 0.0, 1.0)
        rgb[..., color_idx] = (v * 255.0).astype(np.uint8)
    return rgb


def _rgb_tile(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    out = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        out[..., c] = _nn_resize_2d(rgb[..., c], size)
    return out


def _nn_resize_2d(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    if h == size and w == size:
        return arr
    scale = size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    ys = np.linspace(0, h - 1, new_h).astype(np.int64)
    xs = np.linspace(0, w - 1, new_w).astype(np.int64)
    out = arr[ys[:, None], xs[None, :]]
    canvas = np.zeros((size, size), dtype=arr.dtype)
    canvas[: out.shape[0], : out.shape[1]] = out
    return canvas
