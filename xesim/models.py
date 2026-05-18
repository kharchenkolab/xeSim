from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CropBox:
    """Micron-space crop bounds with optional Z bounds."""

    crop_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float | None = None
    zmax: float | None = None

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        return 0.5 * (self.xmin + self.xmax)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.ymin + self.ymax)

    @property
    def has_z(self) -> bool:
        return self.zmin is not None and self.zmax is not None

    def contains_xy(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CropBox":
        return cls(**value)


@dataclass(frozen=True)
class XeniumBundle:
    """Resolved Xenium bundle paths and metadata."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    pixel_size: float
    z_step_size: float | None
    morphology_path: Path | None
    morphology_focus_paths: tuple[Path, ...]
    transcripts_path: Path | None
    cells_path: Path | None
    cells_zarr_path: Path | None
    cell_boundaries_path: Path | None
    nucleus_boundaries_path: Path | None

    @property
    def run_name(self) -> str:
        return str(self.manifest.get("run_name", self.root.name))

    def to_summary(self) -> dict[str, Any]:
        def p(path: Path | None) -> str | None:
            return None if path is None else str(path)

        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "run_name": self.run_name,
            "pixel_size": self.pixel_size,
            "z_step_size": self.z_step_size,
            "morphology_path": p(self.morphology_path),
            "morphology_focus_paths": [str(path) for path in self.morphology_focus_paths],
            "transcripts_path": p(self.transcripts_path),
            "cells_path": p(self.cells_path),
            "cells_zarr_path": p(self.cells_zarr_path),
            "cell_boundaries_path": p(self.cell_boundaries_path),
            "nucleus_boundaries_path": p(self.nucleus_boundaries_path),
        }


@dataclass(frozen=True)
class CropRecord:
    """Canonical crop metadata stored in a crop manifest."""

    crop_id: str
    crop_box: CropBox
    npz_path: Path
    split: str
    image_channels: tuple[str, ...]
    transcript_count: int = 0
    assigned_transcript_count: int = 0
    cell_count: int = 0
    nucleus_count: int = 0
    qc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        path = self.npz_path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return {
            "crop_id": self.crop_id,
            "crop_box": self.crop_box.to_dict(),
            "npz_path": str(path),
            "split": self.split,
            "image_channels": list(self.image_channels),
            "transcript_count": self.transcript_count,
            "assigned_transcript_count": self.assigned_transcript_count,
            "cell_count": self.cell_count,
            "nucleus_count": self.nucleus_count,
            "qc": self.qc,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], root: Path | None = None) -> "CropRecord":
        path = Path(value["npz_path"])
        if root is not None and not path.is_absolute():
            path = root / path
        return cls(
            crop_id=value["crop_id"],
            crop_box=CropBox.from_dict(value["crop_box"]),
            npz_path=path,
            split=value["split"],
            image_channels=tuple(value.get("image_channels", [])),
            transcript_count=int(value.get("transcript_count", 0)),
            assigned_transcript_count=int(value.get("assigned_transcript_count", 0)),
            cell_count=int(value.get("cell_count", 0)),
            nucleus_count=int(value.get("nucleus_count", 0)),
            qc=dict(value.get("qc", {})),
        )


@dataclass(frozen=True)
class LatentScene2p5D:
    """Saved latent state for one synthetic 2.5D crop."""

    source_crop_id: str
    seed: int
    perturbations: dict[str, Any]
    stain_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticCrop:
    """Synthetic crop artifact and exact provenance."""

    sample_id: str
    npz_path: Path
    latent: LatentScene2p5D
    image_channels: tuple[str, ...]

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        path = self.npz_path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return {
            "sample_id": self.sample_id,
            "npz_path": str(path),
            "latent": self.latent.to_dict(),
            "image_channels": list(self.image_channels),
        }
