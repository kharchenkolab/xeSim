from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import CropBox

try:  # pragma: no cover - optional dependency
    import zarr  # type: ignore
    from zarr.storage import ZipStore  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    zarr = None
    ZipStore = None


@dataclass(frozen=True)
class ZarrMaskCrop:
    label: np.ndarray
    ids: np.ndarray
    source: str


class CellsZarrMaskReader:
    """Reusable reader for Xenium `cells.zarr.zip` mask crops.

    Opening a zipped Zarr store is relatively expensive. Canonicalization and
    visual QC both read many nearby crops, so keep the store, group, transform,
    and cell-id table open for the duration of a run.
    """

    def __init__(self, cells_zarr_path: Path, pixel_size: float):
        if zarr is None or ZipStore is None:
            raise RuntimeError("zarr is required to read cells.zarr.zip masks. Install xesim[io].")
        self.cells_zarr_path = cells_zarr_path
        self.pixel_size = pixel_size
        self.store = ZipStore(str(cells_zarr_path), mode="r")
        self.root = zarr.open_group(store=self.store, mode="r")
        self.masks = self.root["masks"]
        self.transform = _read_transform(self.masks, pixel_size)
        self.cell_id_array = np.asarray(self.root["cell_id"][:]) if "cell_id" in self.root else None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "CellsZarrMaskReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def read(self, crop: CropBox, kind: str, shape: tuple[int, int] | None = None) -> ZarrMaskCrop:
        if kind not in {"cell", "nucleus"}:
            raise ValueError("kind must be 'cell' or 'nucleus'")
        mask_key = "1" if kind == "cell" else "0"
        if mask_key not in self.masks:
            raise KeyError(f"Missing masks/{mask_key} in {self.cells_zarr_path}")
        arr = self.masks[mask_key]
        x0, x1, y0, y1 = _crop_bounds(crop, arr.shape, self.transform)
        label = np.asarray(arr[y0:y1, x0:x1]).astype(np.int32, copy=False)
        if shape is not None:
            label = _match_shape(label, shape)
        ids = zarr_label_ids(label, self.cell_id_array)
        return ZarrMaskCrop(label=label, ids=ids, source=f"zarr:cells.zarr.zip:masks/{mask_key}")


def read_cells_zarr_mask_crop(
    cells_zarr_path: Path,
    crop: CropBox,
    kind: str,
    pixel_size: float,
    shape: tuple[int, int] | None = None,
) -> ZarrMaskCrop:
    """Read one cell or nucleus mask crop from Xenium `cells.zarr.zip`.

    Xenium stores nucleus masks under `masks/0` and cell masks under `masks/1`
    in the tested v2 bundle layout. The homogeneous transform maps microns to
    mask/image pixels; if it is absent, `pixel_size` is used.
    """

    with CellsZarrMaskReader(cells_zarr_path, pixel_size) as reader:
        return reader.read(crop, kind, shape=shape)


def zarr_label_ids(label: np.ndarray, cell_id_array: np.ndarray | None) -> np.ndarray:
    ids: list[str] = []
    for value in np.unique(label):
        label_value = int(value)
        if label_value <= 0:
            continue
        decoded = decode_cell_id_from_zarr(label_value, cell_id_array)
        ids.append(decoded if decoded is not None else f"zarr_label_{label_value}")
    return np.asarray(ids, dtype=object)


def decode_cell_id_from_zarr(label_value: int, cell_id_array: np.ndarray | None) -> str | None:
    """Decode a 1-based Zarr mask label to a Xenium cell ID string."""

    if cell_id_array is None:
        return None
    row = label_value - 1
    if row < 0 or row >= int(cell_id_array.shape[0]) or int(cell_id_array.shape[1]) < 2:
        return None
    encoded = int(cell_id_array[row, 0])
    suffix = int(cell_id_array[row, 1])
    return f"{decode_xenium_base16_letters(encoded)}-{suffix}"


def decode_xenium_base16_letters(value: int) -> str:
    """Decode Xenium's a-p nibble alphabet used in Zarr `cell_id` arrays."""

    alphabet = "abcdefghijklmnop"
    value = int(value) & 0xFFFFFFFF
    chars = []
    for shift in range(28, -1, -4):
        chars.append(alphabet[(value >> shift) & 0xF])
    return "".join(chars)


def _read_transform(masks_group: object, pixel_size: float) -> np.ndarray:
    try:
        transform = np.asarray(masks_group["homogeneous_transform"][:], dtype=np.float64)
        if transform.shape == (4, 4):
            return transform
    except Exception:
        pass
    transform = np.eye(4, dtype=np.float64)
    scale = 1.0 / pixel_size if pixel_size > 0 else 1.0
    transform[0, 0] = scale
    transform[1, 1] = scale
    return transform


def _crop_bounds(
    crop: CropBox,
    shape: tuple[int, int],
    transform: np.ndarray,
) -> tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    xs = np.asarray([crop.xmin, crop.xmax], dtype=np.float64)
    ys = np.asarray([crop.ymin, crop.ymax], dtype=np.float64)
    px = transform[0, 0] * xs + transform[0, 3]
    py = transform[1, 1] * ys + transform[1, 3]
    x0 = int(np.floor(np.min(px)))
    x1 = int(np.ceil(np.max(px)))
    y0 = int(np.floor(np.min(py)))
    y1 = int(np.ceil(np.max(py)))
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    return x0, x1, y0, y1


def _match_shape(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=label.dtype)
    hh = min(h, label.shape[0])
    ww = min(w, label.shape[1])
    if hh > 0 and ww > 0:
        out[:hh, :ww] = label[:hh, :ww]
    return out
