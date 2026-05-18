"""Stain-based cell-type classifier — encoder-latent kNN fallback.

When `explain_region` types cells, it tries (in order):
  1. annotation file
  2. cellAdmix transcript-based classifier
  3. model's training-time cid_to_type

This module provides the 4th fallback: encode the cell's stain
appearance via the model's per-cell encoder and look up the nearest
neighbour in a "latent bank" of typed training cells. Catches the
~0.4% of cells that fall through all earlier fallbacks (typically
small/transcript-poor cells with no annotation).

Two entry points:
- ``build_latent_bank(model)``: encode all typed canonical-crop cells,
  return ``(latents, types, cell_ids)``. Save once per model with
  ``np.savez(MODEL_DIR/cell_latent_bank.npz, ...)``.
- ``classify_cells_in_bundle(bank, model, bundle_path, cells_df)``:
  for each row in ``cells_df`` (with ``centroid_x, centroid_y``),
  encode a fixed-radius crop and assign the kNN-predicted type.

Known limitations: kNN CV accuracy ~72% overall, biased toward
majority types (Exocrine on pancreas). Rare types (Endocrine, Mural)
have weak recall. The classifier is a quick-fix to avoid leaving
cells as "unknown" — not a precision typing tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def build_latent_bank(model, canonical_dir: str | Path | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode all typed cells in the model's canonical crops.

    Returns ``(latents (N, D), types (N,), cell_ids (N,))``.
    """
    from .mechanistic_scene import MechanisticScene, MechanisticCell

    canon = Path(canonical_dir or (Path(model.manifest['bundle']).parent
                                    if False else
                                    Path('/'.join(model.manifest.get(
                                        'renderer', 'renderer.pt'
                                    ).split('/')[:-1]))))
    # Fall back to model_dir/canonical
    if not (canon / 'cell_types.json').exists():
        canon = Path(getattr(model, '_model_dir', '.')) / 'canonical'
    ct = json.loads((canon / 'cell_types.json').read_text())
    type_names = ct['type_names']
    n_ch = int(model.manifest.get('n_channels', 4))

    latents: list[np.ndarray] = []
    types: list[str] = []
    cell_ids: list[str] = []

    for crop_meta in ct['crops']:
        npz_path = canon / crop_meta['npz_path']
        d = np.load(npz_path, allow_pickle=True)
        images = d['images']
        cell_label = d['cell_label']
        ids = [str(x) for x in d['cell_ids']]
        cid_to_type = crop_meta.get('cell_id_to_type_index', {})
        nz = np.unique(cell_label); nz = nz[nz > 0]
        mech_cells = []
        for i, lbl in enumerate(nz):
            cid = ids[i] if i < len(ids) else str(int(lbl))
            ti = cid_to_type.get(cid, 0)
            t = type_names[ti] if 0 <= ti < len(type_names) else 'unknown'
            mech_cells.append(MechanisticCell(
                cell_id=cid, label=int(lbl), source='canonical',
                cell_type=t, nucleus_label=int(lbl),
                provenance={'is_ghost': False}))
        scene = MechanisticScene(
            image_shape=cell_label.shape, pixel_size=float(model.pixel_size),
            cell_label=cell_label.astype(np.int32),
            nucleus_label=cell_label.astype(np.int32),
            cells=tuple(mech_cells), scene_id=crop_meta.get('crop_id', '?'),
            provenance={})
        rgb = images[:n_ch].astype(np.float32)
        enc = model.encode_real(scene, rgb)
        for c in mech_cells:
            v = enc.get(c.label)
            if v is not None and c.cell_type and c.cell_type != 'unknown':
                latents.append(v); types.append(c.cell_type); cell_ids.append(c.cell_id)
    return np.stack(latents), np.array(types), np.array(cell_ids)


def save_bank(bank_path: str | Path, latents: np.ndarray,
              types: np.ndarray, cell_ids: np.ndarray) -> None:
    np.savez(bank_path, latents=latents, types=types, cell_ids=cell_ids)


def load_bank(bank_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(bank_path, allow_pickle=True)
    return d['latents'], d['types'], d['cell_ids']


def classify_cells_by_centroid(
    model,
    bundle_path: str | Path,
    cell_centroids_um: Sequence[tuple[str, float, float]],
    bank: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    bank_path: str | Path | None = None,
    *,
    crop_half_um: float = 16.0,
    cell_radius_um: float = 5.0,
    n_neighbors: int = 10,
) -> dict[str, str]:
    """Encode a crop around each cell centroid and look up nearest type.

    ``cell_centroids_um`` is a sequence of ``(cell_id, x_um, y_um)``.
    Returns dict ``cell_id -> predicted cell_type`` (or 'unknown' on
    failure to encode).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from .mechanistic_scene import MechanisticScene, MechanisticCell
    from .images import OmeCropReader
    from .models import CropBox
    from .xenium import resolve_bundle

    if bank is None:
        if bank_path is None:
            raise ValueError("either bank= or bank_path= required")
        bank = load_bank(bank_path)
    latents, types, _ = bank
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(latents, types)

    bundle_path = Path(bundle_path)
    # Use cached OmeCropReader per channel — avoids re-decompressing the
    # ~300MB-compressed / ~1.9GB-uncompressed morphology TIFF per call.
    # The plane cache is process-wide (xesim/images.py:_PLANE_CACHE).
    bundle = resolve_bundle(str(bundle_path))
    readers = [OmeCropReader(p, bundle.pixel_size)
               for p in bundle.morphology_focus_paths]
    # LUT from model's canonical manifest
    model_dir = Path(model.paths.root) if hasattr(model, 'paths') else Path('.')
    canon_mf = json.loads(
        (model_dir / 'canonical' / 'manifest.json').read_text())
    lut = canon_mf['image_normalization']['lut']['channels']
    psz = float(model.pixel_size)
    n_ch = int(model.manifest.get('n_channels', 4))

    out: dict[str, str] = {}
    for cid, cx_um, cy_um in cell_centroids_um:
        xmin = cx_um - crop_half_um; xmax = cx_um + crop_half_um
        ymin = cy_um - crop_half_um; ymax = cy_um + crop_half_um
        try:
            crop_box = CropBox(
                crop_id=cid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            per_ch = []
            for ri in range(n_ch):
                if ri >= len(readers): break
                per_ch.append(readers[ri].read(crop_box))
            if len(per_ch) < n_ch:
                out[cid] = 'unknown'; continue
            h = min(c.shape[0] for c in per_ch)
            w = min(c.shape[1] for c in per_ch)
            crop = np.stack([c[:h, :w] for c in per_ch], axis=0)
        except Exception:
            out[cid] = 'unknown'; continue
        yy, xx = np.indices((h, w), dtype=np.float32)
        r_px = cell_radius_um / psz
        mask = ((yy - h/2) ** 2 + (xx - w/2) ** 2) < r_px ** 2
        label = np.where(mask, 1, 0).astype(np.int32)
        crop_norm = np.stack([
            ((crop[c].astype(np.float32) - lut[c]['lo']) /
             max(lut[c]['hi'] - lut[c]['lo'], 1e-6)).clip(0, 1)
            for c in range(n_ch)])
        scene = MechanisticScene(
            image_shape=(h, w), pixel_size=psz,
            cell_label=label, nucleus_label=label,
            cells=(MechanisticCell(
                cell_id=cid, label=1, source='stain_kNN_classify',
                cell_type='unknown', nucleus_label=1,
                provenance={'is_ghost': False}),),
            scene_id='cls', provenance={})
        try:
            enc = model.encode_real(scene, crop_norm)
            v = enc.get(1)
            out[cid] = str(knn.predict([v])[0]) if v is not None else 'unknown'
        except Exception:
            out[cid] = 'unknown'
    return out


__all__ = ['build_latent_bank', 'save_bank', 'load_bank',
           'classify_cells_by_centroid']
