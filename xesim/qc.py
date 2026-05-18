from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schema import CROP_MANIFEST_TYPE, MANIFEST_SCHEMA_VERSION, QC_REPORT_TYPE, SYNTHETIC_MANIFEST_TYPE
from .validation import validate_crop_manifest, validate_synthetic_manifest


STAIN_COLORS = (
    np.array([0.84, 0.73, 0.96], dtype=np.float32),  # DAPI / nuclear: purple
    np.array([0.75, 0.93, 0.74], dtype=np.float32),  # membrane: green
    np.array([0.72, 0.90, 0.98], dtype=np.float32),  # additional stain: cyan
    np.array([0.98, 0.80, 0.56], dtype=np.float32),  # additional stain: amber
    np.array([0.98, 0.72, 0.78], dtype=np.float32),  # additional stain: rose
)

DECOMPOSED_COMPONENT_KEYS = (
    "decomposed_mask_linked",
    "decomposed_nucleus_linked",
    "decomposed_boundary_linked",
    "decomposed_cell_interior_linked",
    "decomposed_background_haze",
    "decomposed_background_global_haze",
    "decomposed_residual_texture",
)


def fit_display_lut(
    image_stacks: list[np.ndarray],
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> dict[str, Any]:
    """Fit fixed per-channel display limits from real image stacks only."""

    n_channels = max((stack.shape[0] for stack in image_stacks if stack.ndim == 3 and stack.size), default=0)
    channels: list[dict[str, float | int]] = []
    for idx in range(n_channels):
        values = [
            stack[idx].reshape(-1).astype(np.float32, copy=False)
            for stack in image_stacks
            if stack.ndim == 3 and stack.shape[0] > idx and stack.size
        ]
        if values:
            merged = np.concatenate(values)
            lo = float(np.nanpercentile(merged, low_percentile))
            hi = float(np.nanpercentile(merged, high_percentile))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        channels.append({"channel_index": idx, "lo": lo, "hi": hi})
    return {
        "type": "xesim.display_lut.v0",
        "source": "real_crops",
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "channels": channels,
    }


def normalize_images_with_lut(images: np.ndarray, display_lut: dict[str, Any]) -> np.ndarray:
    """Linearly normalize an image stack with fixed per-channel LUT limits."""

    if images.ndim != 3 or images.size == 0:
        return images.astype(np.float32, copy=False)
    out = np.empty_like(images, dtype=np.float32)
    for idx in range(images.shape[0]):
        channel_lut = _channel_lut(display_lut, idx)
        if channel_lut is None:
            out[idx] = images[idx].astype(np.float32, copy=False)
            continue
        lo = float(channel_lut["lo"])
        hi = float(channel_lut["hi"])
        if hi <= lo:
            hi = lo + 1.0
        out[idx] = np.clip((images[idx].astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0)
    return out


def stain_tile(images: np.ndarray, tile_size: int, display_lut: dict[str, Any] | None = None) -> np.ndarray:
    """Return a Baysor-style white-background pseudocolor stain tile."""

    return _resize_nearest(_image_rgb(images, display_lut=display_lut), tile_size, tile_size)


def mask_tile(cell_label: np.ndarray, nucleus_label: np.ndarray, tile_size: int) -> np.ndarray:
    """Return a white-background pseudocolor mask tile."""

    return _resize_nearest(_mask_rgb(cell_label, nucleus_label), tile_size, tile_size)


def residual_tile_from_arrays(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    tile_size: int,
    display_lut: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return a white-background residual tile from two image stacks."""

    if real_images.ndim == 3 and synthetic_images.ndim == 3 and real_images.size and synthetic_images.size:
        c = min(real_images.shape[0], synthetic_images.shape[0])
        h = min(real_images.shape[1], synthetic_images.shape[1])
        w = min(real_images.shape[2], synthetic_images.shape[2])
        residual = np.abs(real_images[:c, :h, :w] - synthetic_images[:c, :h, :w])
        return _resize_nearest(_image_rgb(residual, display_lut=_residual_lut(display_lut)), tile_size, tile_size)
    return np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)


def assemble_contact_sheet(tiles: list[np.ndarray], padding: int = 4) -> np.ndarray:
    return _assemble_sheet(tiles, padding=padding)


def assemble_compare_rows(rows: list[list[np.ndarray]], padding: int = 4) -> np.ndarray:
    return _assemble_rows(rows, padding=padding)


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, rgb)


def image_rgb_float(images: np.ndarray, display_lut: dict[str, Any] | None = None) -> np.ndarray:
    """Return white-background pseudocolor RGB values as floats in [0, 1]."""

    if images.ndim != 3 or images.shape[0] == 0:
        return np.zeros((*images.shape[-2:], 3), dtype=np.float32)
    h, w = images.shape[1:]
    rgb = np.ones((h, w, 3), dtype=np.float32)
    for idx in range(images.shape[0]):
        norm = _normalize_channel(images[idx], _channel_lut(display_lut, idx))
        if norm is None:
            continue
        color = STAIN_COLORS[idx % len(STAIN_COLORS)]
        rgb *= 1.0 - norm[..., None] * (1.0 - color)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def visual_background_summary(
    image_records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    display_lut: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize displayed RGB background whiteness and tint for image records."""

    rgb_means: list[np.ndarray] = []
    whiteness: list[float] = []
    tint_ranges: list[float] = []
    for images, cell_label, nucleus_label in image_records:
        if images.ndim != 3 or images.shape[0] == 0:
            continue
        background = ~((cell_label > 0) | (nucleus_label > 0))
        if background.shape != images.shape[1:] or not np.any(background):
            continue
        rgb = image_rgb_float(images, display_lut=display_lut)
        bg_rgb = rgb[background]
        if bg_rgb.size == 0:
            continue
        mean_rgb = np.mean(bg_rgb, axis=0)
        rgb_means.append(mean_rgb)
        whiteness.append(float(np.mean(np.abs(1.0 - bg_rgb))))
        tint_ranges.append(float(np.max(mean_rgb) - np.min(mean_rgb)))
    if not rgb_means:
        return {
            "n": 0,
            "background_rgb_mean": [],
            "background_whiteness_mae": 0.0,
            "background_tint_range": 0.0,
        }
    return {
        "n": len(rgb_means),
        "background_rgb_mean": [float(x) for x in np.mean(np.stack(rgb_means, axis=0), axis=0).tolist()],
        "background_whiteness_mae": float(np.mean(whiteness)),
        "background_tint_range": float(np.mean(tint_ranges)),
    }


def save_contact_sheet(
    manifest_path: Path,
    output_path: Path,
    max_records: int = 16,
    tile_size: int = 128,
) -> Path:
    manifest, records_key = _load_manifest_for_qc(manifest_path)
    root = manifest_path.parent
    records = manifest.get(records_key, [])[: max(0, max_records)]
    tiles = [_record_tile(root / record["npz_path"], tile_size=tile_size) for record in records]
    if not tiles:
        tiles = [np.zeros((tile_size, tile_size, 3), dtype=np.uint8)]
    sheet = _assemble_sheet(tiles, padding=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(output_path, sheet)
    return output_path


def save_qc_report(
    manifest_path: Path,
    sheet_path: Path,
    report_path: Path,
    max_records: int = 16,
    tile_size: int = 128,
) -> Path:
    manifest, records_key = _load_manifest_for_qc(manifest_path)
    records = manifest.get(records_key, [])[: max(0, max_records)]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(
            {
                "type": QC_REPORT_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_path": str(manifest_path),
                "sheet_path": str(sheet_path),
                "records_key": records_key,
                "num_records": len(records),
                "tile_size": tile_size,
                "records": [
                    {
                        "id": record.get("crop_id") or record.get("sample_id"),
                        "npz_path": record.get("npz_path"),
                        "image_channels": record.get("image_channels", []),
                    }
                    for record in records
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return report_path


def save_compare_sheet(
    real_manifest_path: Path,
    synthetic_manifest_path: Path,
    output_path: Path,
    max_records: int = 16,
    tile_size: int = 128,
) -> Path:
    real_manifest = validate_crop_manifest(real_manifest_path, check_files=True)
    synthetic_manifest = validate_synthetic_manifest(synthetic_manifest_path, check_files=True)
    real_root = real_manifest_path.parent
    synthetic_root = synthetic_manifest_path.parent
    real_by_id = {record["crop_id"]: record for record in real_manifest.get("crops", [])}
    pairs = []
    for sample in synthetic_manifest.get("samples", [])[: max(0, max_records)]:
        source_id = sample.get("latent", {}).get("source_crop_id")
        real = real_by_id.get(source_id)
        if real is None:
            continue
        real_npz = real_root / real["npz_path"]
        synthetic_npz = synthetic_root / sample["npz_path"]
        pairs.append((real_npz, synthetic_npz))
    display_lut = _fit_lut_from_npz([real_npz for real_npz, _ in pairs])
    rows = []
    for real_npz, synthetic_npz in pairs:
        rows.append(
            [
                _record_tile(real_npz, tile_size, display_lut=display_lut),
                _record_tile(synthetic_npz, tile_size, display_lut=display_lut),
                _residual_tile(real_npz, synthetic_npz, tile_size, display_lut=display_lut),
            ]
        )
    if not rows:
        rows = [[np.zeros((tile_size, tile_size, 3), dtype=np.uint8) for _ in range(3)]]
    sheet = _assemble_rows(rows, padding=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(output_path, sheet)
    return output_path


def save_compare_qc_report(
    real_manifest_path: Path,
    synthetic_manifest_path: Path,
    sheet_path: Path,
    report_path: Path,
    max_records: int = 16,
    tile_size: int = 128,
) -> Path:
    real_manifest = validate_crop_manifest(real_manifest_path, check_files=True)
    synthetic_manifest = validate_synthetic_manifest(synthetic_manifest_path, check_files=True)
    real_root = real_manifest_path.parent
    real_by_id = {record["crop_id"]: record for record in real_manifest.get("crops", [])}
    pairs = []
    real_npz_paths = []
    for sample in synthetic_manifest.get("samples", [])[: max(0, max_records)]:
        source_id = sample.get("latent", {}).get("source_crop_id")
        real = real_by_id.get(source_id)
        if real is not None:
            pairs.append({"source_crop_id": source_id, "sample_id": sample.get("sample_id")})
            real_npz_paths.append(real_root / real["npz_path"])
    display_lut = _fit_lut_from_npz(real_npz_paths)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(
            {
                "type": QC_REPORT_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_path": str(synthetic_manifest_path),
                "real_manifest_path": str(real_manifest_path),
                "sheet_path": str(sheet_path),
                "records_key": "pairs",
                "num_records": len(pairs),
                "tile_size": tile_size,
                "display_lut": display_lut,
                "records": pairs,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return report_path


def save_decomposed_component_sheet(
    synthetic_manifest_path: Path,
    output_path: Path,
    max_records: int = 8,
    tile_size: int = 128,
) -> Path:
    manifest = validate_synthetic_manifest(synthetic_manifest_path, check_files=True)
    root = synthetic_manifest_path.parent
    records = manifest.get("samples", [])[: max(0, max_records)]
    synthetic_paths = [root / record["npz_path"] for record in records]
    display_lut = _fit_lut_from_npz(synthetic_paths)
    component_lut = _fit_component_lut_from_npz(synthetic_paths)
    rows: list[list[np.ndarray]] = []
    for path in synthetic_paths:
        data = np.load(path, allow_pickle=True)
        rows.append(
            [stain_tile(data["images"].astype(np.float32), tile_size, display_lut=display_lut)]
            + [_component_tile(data, key, tile_size, component_lut) for key in DECOMPOSED_COMPONENT_KEYS]
        )
    if not rows:
        rows = [
            [
                np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                for _ in range(1 + len(DECOMPOSED_COMPONENT_KEYS))
            ]
        ]
    sheet = _assemble_rows(rows, padding=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(output_path, sheet)
    return output_path


def save_decomposed_component_qc_report(
    synthetic_manifest_path: Path,
    sheet_path: Path,
    report_path: Path,
    max_records: int = 8,
    tile_size: int = 128,
) -> Path:
    manifest = validate_synthetic_manifest(synthetic_manifest_path, check_files=True)
    root = synthetic_manifest_path.parent
    records = manifest.get("samples", [])[: max(0, max_records)]
    report_records: list[dict[str, Any]] = []
    for record in records:
        data = np.load(root / record["npz_path"], allow_pickle=True)
        components = {}
        for key in DECOMPOSED_COMPONENT_KEYS:
            if key in data.files:
                arr = data[key].astype(np.float32, copy=False)
                components[key] = {
                    "mean_abs": float(np.mean(np.abs(arr))),
                    "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                }
        report_records.append(
            {
                "sample_id": record.get("sample_id"),
                "npz_path": record.get("npz_path"),
                "source_crop_id": record.get("latent", {}).get("source_crop_id"),
                "components": components,
            }
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(
            {
                "type": QC_REPORT_TYPE,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_path": str(synthetic_manifest_path),
                "sheet_path": str(sheet_path),
                "records_key": "decomposed_components",
                "num_records": len(report_records),
                "tile_size": tile_size,
                "columns": ["synthetic"] + [f"abs_{key}" for key in DECOMPOSED_COMPONENT_KEYS],
                "records": report_records,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return report_path


def _load_manifest_for_qc(path: Path) -> tuple[dict[str, Any], str]:
    with path.open() as handle:
        raw = json.load(handle)
    artifact_type = raw.get("type")
    if artifact_type == CROP_MANIFEST_TYPE:
        return validate_crop_manifest(path, check_files=True), "crops"
    if artifact_type == SYNTHETIC_MANIFEST_TYPE:
        return validate_synthetic_manifest(path, check_files=True), "samples"
    raise ValueError(f"{path} is not a crop or synthetic manifest")


def _record_tile(path: Path, tile_size: int, display_lut: dict[str, Any] | None = None) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    images = data["images"].astype(np.float32)
    if images.ndim == 3 and images.shape[0] > 0:
        return stain_tile(images, tile_size, display_lut=display_lut)
    return mask_tile(data["cell_label"], data["nucleus_label"], tile_size)


def _residual_tile(
    real_path: Path,
    synthetic_path: Path,
    tile_size: int,
    display_lut: dict[str, Any] | None = None,
) -> np.ndarray:
    real = np.load(real_path, allow_pickle=True)
    synthetic = np.load(synthetic_path, allow_pickle=True)
    real_images = real["images"].astype(np.float32)
    synthetic_images = synthetic["images"].astype(np.float32)
    if real_images.ndim == 3 and synthetic_images.ndim == 3 and real_images.size and synthetic_images.size:
        c = min(real_images.shape[0], synthetic_images.shape[0], 3)
        h = min(real_images.shape[1], synthetic_images.shape[1])
        w = min(real_images.shape[2], synthetic_images.shape[2])
        residual = np.abs(real_images[:c, :h, :w] - synthetic_images[:c, :h, :w])
        rgb = _resize_nearest(_image_rgb(residual, display_lut=_residual_lut(display_lut)), tile_size, tile_size)
    else:
        real_mask = real["cell_label"] > 0
        synthetic_mask = synthetic["cell_label"] > 0
        h = min(real_mask.shape[0], synthetic_mask.shape[0])
        w = min(real_mask.shape[1], synthetic_mask.shape[1])
        diff = real_mask[:h, :w] ^ synthetic_mask[:h, :w]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[diff, 0] = 230
        rgb = _resize_nearest(rgb, tile_size, tile_size)
    return rgb


def _fit_lut_from_npz(paths: list[Path]) -> dict[str, Any] | None:
    stacks = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        images = data["images"].astype(np.float32)
        if images.ndim == 3 and images.shape[0] > 0:
            stacks.append(images)
    return fit_display_lut(stacks) if stacks else None


def _fit_component_lut_from_npz(paths: list[Path]) -> dict[str, Any] | None:
    stacks = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        components = [
            np.abs(data[key].astype(np.float32, copy=False))
            for key in DECOMPOSED_COMPONENT_KEYS
            if key in data.files
        ]
        stacks.extend(components)
    return fit_display_lut(stacks, low_percentile=0.0, high_percentile=99.5) if stacks else None


def _component_tile(
    data: np.lib.npyio.NpzFile,
    key: str,
    tile_size: int,
    display_lut: dict[str, Any] | None = None,
) -> np.ndarray:
    if key not in data.files:
        h, w = data["images"].shape[-2:] if "images" in data.files and data["images"].ndim == 3 else (tile_size, tile_size)
        return _resize_nearest(np.full((h, w, 3), 255, dtype=np.uint8), tile_size, tile_size)
    arr = np.abs(data[key].astype(np.float32, copy=False))
    return stain_tile(arr, tile_size, display_lut=display_lut)


def _image_rgb(images: np.ndarray, display_lut: dict[str, Any] | None = None) -> np.ndarray:
    return _float_rgb_to_uint8(image_rgb_float(images, display_lut=display_lut))


def _mask_rgb(cell_label: np.ndarray, nucleus_label: np.ndarray) -> np.ndarray:
    cell = cell_label > 0
    nucleus = nucleus_label > 0
    rgb = np.ones((*cell.shape, 3), dtype=np.float32)
    rgb[cell] *= STAIN_COLORS[1]
    rgb[nucleus] *= STAIN_COLORS[0]
    return _float_rgb_to_uint8(rgb)


def _channel_lut(display_lut: dict[str, Any] | None, channel_index: int) -> dict[str, float] | None:
    if not display_lut:
        return None
    channels = display_lut.get("channels", [])
    if channel_index >= len(channels):
        return None
    value = channels[channel_index]
    if not isinstance(value, dict):
        return None
    return {"lo": float(value.get("lo", 0.0)), "hi": float(value.get("hi", 1.0))}


def _residual_lut(display_lut: dict[str, Any] | None) -> dict[str, Any] | None:
    if not display_lut:
        return None
    channels = []
    for channel in display_lut.get("channels", []):
        if not isinstance(channel, dict):
            continue
        lo = float(channel.get("lo", 0.0))
        hi = float(channel.get("hi", 1.0))
        span = max(hi - lo, 1e-6)
        channels.append({"channel_index": int(channel.get("channel_index", len(channels))), "lo": 0.0, "hi": span})
    return {
        "type": "xesim.display_lut.v0",
        "source": "real_crops_residual_span",
        "low_percentile": 0.0,
        "high_percentile": None,
        "channels": channels,
    }


def _normalize_channel(arr: np.ndarray, channel_lut: dict[str, float] | None = None) -> np.ndarray | None:
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32, copy=False)
    if channel_lut is None:
        lo = float(np.nanpercentile(arr, 1.0))
        hi = float(np.nanpercentile(arr, 99.8))
    else:
        lo = float(channel_lut["lo"])
        hi = float(channel_lut["hi"])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.power(norm, 0.85).astype(np.float32, copy=False)


def _float_rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _resize_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb.astype(np.uint8, copy=False)
    ys = np.linspace(0, rgb.shape[0] - 1, height).round().astype(int)
    xs = np.linspace(0, rgb.shape[1] - 1, width).round().astype(int)
    return rgb[ys][:, xs].astype(np.uint8, copy=False)


def _assemble_sheet(tiles: list[np.ndarray], padding: int) -> np.ndarray:
    n = len(tiles)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    th, tw = tiles[0].shape[:2]
    h = rows * th + (rows + 1) * padding
    w = cols * tw + (cols + 1) * padding
    sheet = np.full((h, w, 3), 255, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y = padding + row * (th + padding)
        x = padding + col * (tw + padding)
        sheet[y : y + th, x : x + tw] = tile
    return sheet


def _assemble_rows(rows: list[list[np.ndarray]], padding: int) -> np.ndarray:
    row_images = [_assemble_horizontal(row, padding=padding) for row in rows]
    h = sum(row.shape[0] for row in row_images) + padding * (len(row_images) + 1)
    w = max(row.shape[1] for row in row_images) + 2 * padding
    sheet = np.full((h, w, 3), 255, dtype=np.uint8)
    y = padding
    for row in row_images:
        sheet[y : y + row.shape[0], padding : padding + row.shape[1]] = row
        y += row.shape[0] + padding
    return sheet


def _assemble_horizontal(tiles: list[np.ndarray], padding: int) -> np.ndarray:
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    th = max(tile.shape[0] for tile in tiles)
    widths = [tile.shape[1] for tile in tiles]
    h = th + 2 * padding
    w = sum(widths) + (len(tiles) + 1) * padding
    row = np.full((h, w, 3), 255, dtype=np.uint8)
    x = padding
    for tile in tiles:
        y = padding + (th - tile.shape[0]) // 2
        row[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + padding
    return row


def _write_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG writer expects HxWx3 RGB data")
    rgb = rgb.astype(np.uint8, copy=False)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))
    payload = b"".join(
        [
            _png_chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(data, crc)
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", crc & 0xFFFFFFFF)
