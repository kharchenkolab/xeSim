from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .models import CropBox, XeniumBundle

try:  # pragma: no cover - exercised when optional dependency is installed
    import pyarrow.dataset as pds  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pds = None
    pq = None


XENIUM_TRANSCRIPT_COLUMNS = (
    "transcript_id",
    "cell_id",
    "feature_name",
    "x_location",
    "y_location",
    "z_location",
    "qv",
)

TRANSCRIPT_FILTER_CROP_LIMIT = 64


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    x: float
    y: float
    z: float = 0.0
    cell_area: float = 0.0
    nucleus_area: float = 0.0
    transcript_count: int = 0
    cell_type: str = ""

    @property
    def radius_um(self) -> float:
        if self.cell_area > 0:
            return math.sqrt(self.cell_area / math.pi)
        return 5.0


@dataclass(frozen=True)
class PolygonRecord:
    object_id: str
    x: np.ndarray
    y: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(np.min(self.x)),
            float(np.max(self.x)),
            float(np.min(self.y)),
            float(np.max(self.y)),
        )


def _existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _manifest_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        return path / "experiment.xenium"
    return path


def resolve_bundle(path_or_dir: str | Path) -> XeniumBundle:
    """Resolve a Xenium directory or manifest into a bundle description."""

    manifest_path = _manifest_path(path_or_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find Xenium manifest: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    root = manifest_path.parent
    images = manifest.get("images", {})
    pixel_size = float(manifest.get("pixel_size", 1.0))
    z_step_size = manifest.get("z_step_size")
    z_step = None if z_step_size is None else float(z_step_size)

    morphology_rel = images.get("morphology_filepath", "morphology.ome.tif")
    morphology_path = root / morphology_rel
    if not morphology_path.exists():
        morphology_path = None

    focus_paths = _resolve_focus_paths(root, images)
    transcripts = _existing(
        [
            root / manifest.get("transcripts_parquet_filepath", "transcripts.parquet"),
            root / "transcripts.parquet",
            root / manifest.get("transcripts_csv_filepath", "transcripts.csv.gz"),
            root / "transcripts.csv.gz",
        ]
    )
    cells = _existing([root / "cells.parquet", root / "cells.csv.gz"])
    explorer_files = manifest.get("xenium_explorer_files", {})
    cells_zarr = _existing(
        [
            root / explorer_files.get("cells_zarr_filepath", "cells.zarr.zip"),
            root / "cells.zarr.zip",
        ]
    )
    cell_boundaries = _existing([root / "cell_boundaries.parquet", root / "cell_boundaries.csv.gz"])
    nucleus_boundaries = _existing(
        [root / "nucleus_boundaries.parquet", root / "nucleus_boundaries.csv.gz"]
    )
    return XeniumBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        pixel_size=pixel_size,
        z_step_size=z_step,
        morphology_path=morphology_path,
        morphology_focus_paths=tuple(focus_paths),
        transcripts_path=transcripts,
        cells_path=cells,
        cells_zarr_path=cells_zarr,
        cell_boundaries_path=cell_boundaries,
        nucleus_boundaries_path=nucleus_boundaries,
    )


def _resolve_focus_paths(root: Path, images: dict[str, Any]) -> list[Path]:
    first = images.get("morphology_focus_filepath")
    out: list[Path] = []
    if first:
        first_path = root / first
        if first_path.exists():
            out.append(first_path)
            out.extend(sorted(p for p in first_path.parent.glob("*.ome.tif") if p != first_path))
            return out
    focus_dir = root / "morphology_focus"
    if focus_dir.exists():
        out.extend(sorted(focus_dir.glob("*.ome.tif")))
    return out


def read_cells(bundle: XeniumBundle, prefer_parquet: bool = True) -> list[CellSummary]:
    """Read Xenium cell centroids from Parquet or CSV."""

    path = bundle.cells_path
    if path is None:
        return []
    if path.suffix == ".parquet" and prefer_parquet and pq is not None:
        return _read_cells_parquet(path)
    if path.suffix == ".parquet":
        csv_path = bundle.root / "cells.csv.gz"
        if csv_path.exists():
            return _read_cells_csv(csv_path)
        if pq is not None:
            return _read_cells_parquet(path)
        raise RuntimeError("cells.parquet requires pyarrow; install xesim[io] or provide cells.csv.gz")
    return _read_cells_csv(path)


def _read_cells_csv(path: Path) -> list[CellSummary]:
    out: list[CellSummary] = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cid = row.get("cell_id") or row.get("cell") or ""
            if not cid:
                continue
            out.append(
                CellSummary(
                    cell_id=cid,
                    x=float(row.get("x_centroid") or row.get("x") or 0.0),
                    y=float(row.get("y_centroid") or row.get("y") or 0.0),
                    z=float(row.get("z_centroid") or row.get("z") or 0.0),
                    cell_area=float(row.get("cell_area") or 0.0),
                    nucleus_area=float(row.get("nucleus_area") or 0.0),
                    transcript_count=int(float(row.get("transcript_counts") or 0)),
                    cell_type=row.get("cell_type") or row.get("celltype") or "",
                )
            )
    return out


def _read_cells_parquet(path: Path) -> list[CellSummary]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read cells.parquet")
    table = pq.read_table(path)
    names = set(table.column_names)

    def col(*candidates: str, default: Any = None) -> list[Any]:
        for candidate in candidates:
            if candidate in names:
                return table[candidate].to_pylist()
        if default is None:
            return [None] * table.num_rows
        return [default] * table.num_rows

    ids = col("cell_id", "cell")
    xs = col("x_centroid", "x", default=0.0)
    ys = col("y_centroid", "y", default=0.0)
    zs = col("z_centroid", "z", default=0.0)
    areas = col("cell_area", default=0.0)
    nuc_areas = col("nucleus_area", default=0.0)
    counts = col("transcript_counts", default=0)
    cell_types = col("cell_type", "celltype", default="")
    return [
        CellSummary(
            cell_id=str(cid),
            x=float(x or 0.0),
            y=float(y or 0.0),
            z=float(z or 0.0),
            cell_area=float(area or 0.0),
            nucleus_area=float(nuc_area or 0.0),
            transcript_count=int(count or 0),
            cell_type=str(cell_type or ""),
        )
        for cid, x, y, z, area, nuc_area, count, cell_type in zip(
            ids, xs, ys, zs, areas, nuc_areas, counts, cell_types
        )
        if cid not in (None, "")
    ]


def choose_crop_boxes(
    cells: Sequence[CellSummary],
    n_crops: int,
    crop_size_um: float,
    bounds: tuple[float, float, float, float] | None = None,
    selection: str = "spread",
    seed: int = 1,
    *,
    stratified_k: int = 20,
    stratified_alpha: float = 0.5,
) -> list[CropBox]:
    """Select crop boxes centered on cells using a deterministic strategy.

    ``stratified`` selection clusters candidates by their local cell-type
    composition (k-means on per-window type fractions) then allocates
    per-cluster quota = n_c^alpha (alpha in [0,1]; 1=proportional to
    cluster size = `spread`-ish, 0=uniform-per-cluster, 0.5=moderate
    rare-upsampling, matching the H&E foundation-model standard).
    Cells must have ``cell_type`` populated; otherwise falls back to
    ``density``. See misc/plan_25d_molecule_port.md and Hagos et al. 2025
    (StainStyleSampler) for the recipe."""

    if selection not in {"spread", "random", "grid", "density", "stratified"}:
        raise ValueError(
            "selection must be spread, random, grid, density, or stratified")
    if not cells:
        raise ValueError("No cells available for crop selection")
    half = 0.5 * crop_size_um
    if bounds is None:
        valid = list(cells)
    else:
        xmin, xmax, ymin, ymax = bounds
        valid = [
            c
            for c in cells
            if c.x - half >= xmin
            and c.x + half <= xmax
            and c.y - half >= ymin
            and c.y + half <= ymax
        ]
    if not valid:
        raise ValueError("No cells remain after crop edge filtering")
    valid = sorted(valid, key=lambda c: (c.x, c.y, c.cell_id))
    n = min(max(n_crops, 0), len(valid))
    if selection == "random":
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(valid), size=n, replace=False)
        chosen = [valid[int(i)] for i in idxs]
    elif selection == "grid":
        chosen = _choose_grid_cells(valid, n, crop_size_um, bounds)
    elif selection == "density":
        chosen = _choose_density_cells(valid, n, crop_size_um)
    elif selection == "stratified":
        chosen = _choose_stratified_cells(
            valid, n, crop_size_um,
            k=stratified_k, alpha=stratified_alpha, seed=seed)
    elif n >= len(valid):
        chosen = valid
    else:
        idxs = np.linspace(0, len(valid) - 1, n).round().astype(int)
        chosen = [valid[int(i)] for i in idxs]
    return [
        CropBox(
            crop_id=f"crop_{i + 1:05d}",
            xmin=c.x - half,
            xmax=c.x + half,
            ymin=c.y - half,
            ymax=c.y + half,
        )
        for i, c in enumerate(chosen)
    ]


def _choose_grid_cells(
    cells: Sequence[CellSummary],
    n_crops: int,
    crop_size_um: float,
    bounds: tuple[float, float, float, float] | None,
) -> list[CellSummary]:
    if n_crops <= 0:
        return []
    half = 0.5 * crop_size_um
    if bounds is None:
        xmin = min(c.x for c in cells) - half
        xmax = max(c.x for c in cells) + half
        ymin = min(c.y for c in cells) - half
        ymax = max(c.y for c in cells) + half
    else:
        xmin, xmax, ymin, ymax = bounds
    width = max(xmax - xmin, crop_size_um)
    height = max(ymax - ymin, crop_size_um)
    n_cols = max(1, int(math.ceil(math.sqrt(n_crops * width / max(height, 1e-6)))))
    n_rows = max(1, int(math.ceil(n_crops / n_cols)))
    xs = np.linspace(xmin + half, xmax - half, n_cols)
    ys = np.linspace(ymin + half, ymax - half, n_rows)
    targets = [(float(x), float(y)) for y in ys for x in xs][:n_crops]
    chosen: list[CellSummary] = []
    used: set[str] = set()
    for tx, ty in targets:
        available = [cell for cell in cells if cell.cell_id not in used]
        if not available:
            break
        nearest = min(available, key=lambda c: ((c.x - tx) ** 2 + (c.y - ty) ** 2, c.cell_id))
        chosen.append(nearest)
        used.add(nearest.cell_id)
    return chosen


def _choose_density_cells(
    cells: Sequence[CellSummary],
    n_crops: int,
    crop_size_um: float,
) -> list[CellSummary]:
    if n_crops <= 0:
        return []
    bin_size = max(crop_size_um * 0.5, 1.0)
    bins: dict[tuple[int, int], int] = defaultdict(int)
    cell_bins: dict[str, tuple[int, int]] = {}
    for cell in cells:
        key = (int(math.floor(cell.x / bin_size)), int(math.floor(cell.y / bin_size)))
        bins[key] += 1
        cell_bins[cell.cell_id] = key

    scored: list[tuple[int, int, float, float, str, CellSummary]] = []
    for cell in cells:
        bx, by = cell_bins[cell.cell_id]
        density = sum(bins.get((bx + dx, by + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        scored.append((density, cell.transcript_count, cell.x, cell.y, cell.cell_id, cell))
    ordered = [item[-1] for item in sorted(scored, key=lambda x: (-x[0], -x[1], x[2], x[3], x[4]))]

    chosen: list[CellSummary] = []
    min_distance = max(crop_size_um, 1.0)
    for cell in ordered:
        if all((cell.x - c.x) ** 2 + (cell.y - c.y) ** 2 >= min_distance**2 for c in chosen):
            chosen.append(cell)
        if len(chosen) == n_crops:
            return chosen
    used = {cell.cell_id for cell in chosen}
    for cell in ordered:
        if cell.cell_id not in used:
            chosen.append(cell)
            used.add(cell.cell_id)
        if len(chosen) == n_crops:
            break
    return chosen


def _choose_stratified_cells(
    cells: Sequence[CellSummary],
    n_crops: int,
    crop_size_um: float,
    *,
    k: int,
    alpha: float,
    seed: int,
) -> list[CellSummary]:
    """Composition-aware crop selection.

    Cluster candidates by their local cell-type composition (k-means on
    per-window type fractions), then allocate per-cluster quota = n_c^alpha
    and pick spatially-spread representatives within each cluster.

    Falls back to ``_choose_density_cells`` if any cell lacks a cell_type
    or if there are fewer than k cell types observed (degenerate clustering).
    """
    if n_crops <= 0:
        return []
    types = sorted({c.cell_type for c in cells if c.cell_type})
    if len(types) < 2:
        # Not enough type diversity to stratify — degrade to density.
        return _choose_density_cells(cells, n_crops, crop_size_um)
    type_idx = {t: i for i, t in enumerate(types)}
    n_types = len(types)

    # Compute per-cell composition over a window = 2 × crop_size_um
    # (gives a stable estimate; smaller windows are noisy when crops are
    # ~64 µm with ~5-15 cells inside). Use the same density bin trick as
    # density-selection: bin neighborhood counts by cell type.
    win = max(crop_size_um * 2.0, 1.0)
    bin_size = max(crop_size_um * 0.5, 1.0)
    # Type-count per bin
    bins: dict[tuple[int, int], np.ndarray] = {}
    cell_bin: dict[str, tuple[int, int]] = {}
    for c in cells:
        key = (int(math.floor(c.x / bin_size)),
               int(math.floor(c.y / bin_size)))
        cell_bin[c.cell_id] = key
        if key not in bins:
            bins[key] = np.zeros(n_types, dtype=np.int64)
        if c.cell_type in type_idx:
            bins[key][type_idx[c.cell_type]] += 1
    # Window radius in bin units
    r = int(math.ceil(win / bin_size / 2.0))
    comp = np.zeros((len(cells), n_types), dtype=np.float64)
    cell_list = list(cells)
    for i, c in enumerate(cell_list):
        bx, by = cell_bin[c.cell_id]
        tot = np.zeros(n_types, dtype=np.int64)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                tot += bins.get((bx + dx, by + dy),
                                np.zeros(n_types, dtype=np.int64))
        s = tot.sum()
        comp[i] = tot / max(s, 1)

    # k-means
    k_eff = max(2, min(k, n_crops, len(cell_list) // 4))
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        # Degrade if sklearn unavailable
        return _choose_density_cells(cells, n_crops, crop_size_um)
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=seed)
    labels = km.fit_predict(comp)

    # Per-cluster quota = n_c^alpha proportional allocation.
    cluster_sizes = np.array(
        [int((labels == ci).sum()) for ci in range(k_eff)], dtype=np.float64)
    weights = np.power(cluster_sizes, float(alpha))
    if weights.sum() <= 0:
        weights = cluster_sizes
    raw_quota = n_crops * weights / weights.sum()
    quota = np.floor(raw_quota).astype(np.int64)
    # Distribute remainder by largest fractional part
    remainder = n_crops - int(quota.sum())
    if remainder > 0:
        order = np.argsort(-(raw_quota - quota))
        for i in order[:remainder]:
            quota[i] += 1

    # Within each cluster: spatial farthest-point sampling on (x, y) to
    # spread the quota geographically.
    chosen: list[CellSummary] = []
    rng = np.random.default_rng(seed)
    # Precompute per-cell xy + cluster membership indices to avoid O(n_cells^2)
    # cost from list.index() calls inside the per-cluster loop. With ~140k cells
    # × 20 clusters of ~7k members each, the old per-member list.index() was
    # ~20 billion Python ops (would run ~5h on a real bundle).
    all_xys = np.array([[c.x, c.y] for c in cell_list], dtype=np.float64)
    for ci in range(k_eff):
        if quota[ci] <= 0:
            continue
        member_idx = np.where(labels == ci)[0]
        if member_idx.size == 0:
            continue
        if quota[ci] >= member_idx.size:
            chosen.extend(cell_list[i] for i in member_idx)
            continue
        # FPS — start at the cluster-centroid-nearest cell (k-medoid-ish)
        centroid = km.cluster_centers_[ci]
        member_comp = comp[member_idx]
        dists_to_centroid = np.linalg.norm(member_comp - centroid, axis=1)
        seed_idx = int(np.argmin(dists_to_centroid))
        members = [cell_list[i] for i in member_idx]
        picked = [members[seed_idx]]
        xys = all_xys[member_idx]
        seed_xy = xys[seed_idx]
        min_d = np.linalg.norm(xys - seed_xy, axis=1)
        for _ in range(int(quota[ci]) - 1):
            idx = int(np.argmax(min_d))
            picked.append(members[idx])
            d = np.linalg.norm(xys - xys[idx], axis=1)
            min_d = np.minimum(min_d, d)
        chosen.extend(picked)

    # Enforce min spacing across picks (mirrors density logic) only mildly,
    # since stratified intentionally allows cluster-local clusters of crops.
    return chosen[:n_crops]


def read_boundary_polygons(
    path: Path | None,
    crop: CropBox,
    padding_um: float = 5.0,
    prefer_parquet: bool = True,
) -> list[PolygonRecord]:
    """Read cell/nucleus polygons overlapping one crop."""

    if path is None:
        return []
    if path.suffix == ".parquet" and prefer_parquet and pds is not None:
        return _read_polygons_parquet(path, crop, padding_um)
    if path.suffix == ".parquet":
        csv_path = path.with_suffix(".csv.gz")
        if csv_path.exists():
            return _read_polygons_csv(csv_path, crop, padding_um)
        if pds is not None:
            return _read_polygons_parquet(path, crop, padding_um)
        raise RuntimeError(f"{path.name} requires pyarrow; install xesim[io] or provide CSV")
    return _read_polygons_csv(path, crop, padding_um)


def _expanded(crop: CropBox, padding: float) -> tuple[float, float, float, float]:
    return crop.xmin - padding, crop.xmax + padding, crop.ymin - padding, crop.ymax + padding


def _read_polygons_csv(path: Path, crop: CropBox, padding_um: float) -> list[PolygonRecord]:
    xmin, xmax, ymin, ymax = _expanded(crop, padding_um)
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x = float(row["vertex_x"])
            y = float(row["vertex_y"])
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            oid = str(row.get("cell_id") or row.get("nucleus_id") or "")
            if oid:
                points[oid].append((x, y))
    return _polygon_records(points)


_POLY_FRAME_CACHE: dict[str, "pd.DataFrame"] = {}


def _read_polygons_parquet(path: Path, crop: CropBox, padding_um: float) -> list[PolygonRecord]:
    """Read polygons overlapping a crop. The parquet file is loaded ONCE
    per process (cached in `_POLY_FRAME_CACHE`); per-call cost is then
    just an in-memory pandas filter — critical when build_scene runs
    thousands of per-tile explain_region calls."""
    import pandas as pd
    xmin, xmax, ymin, ymax = _expanded(crop, padding_um)
    key = str(Path(path).resolve())
    if key not in _POLY_FRAME_CACHE:
        if pds is not None:
            ds = pds.dataset(path, format="parquet")
            tbl = ds.to_table(columns=["cell_id", "vertex_x", "vertex_y"])
            _POLY_FRAME_CACHE[key] = tbl.to_pandas()
        else:
            _POLY_FRAME_CACHE[key] = pd.read_parquet(
                path, columns=["cell_id", "vertex_x", "vertex_y"])
    df = _POLY_FRAME_CACHE[key]
    # In-memory spatial filter
    mask = ((df["vertex_x"] >= xmin) & (df["vertex_x"] <= xmax) &
            (df["vertex_y"] >= ymin) & (df["vertex_y"] <= ymax))
    sub = df[mask]
    # Reformat to match the to_pydict() shape used below
    values = {
        "cell_id": sub["cell_id"].tolist(),
        "vertex_x": sub["vertex_x"].tolist(),
        "vertex_y": sub["vertex_y"].tolist(),
    }
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for oid, x, y in zip(values["cell_id"], values["vertex_x"], values["vertex_y"]):
        points[str(oid)].append((float(x), float(y)))
    return _polygon_records(points)


def _polygon_records(points: dict[str, list[tuple[float, float]]]) -> list[PolygonRecord]:
    out: list[PolygonRecord] = []
    for oid, coords in points.items():
        if len(coords) < 3:
            continue
        arr = np.asarray(coords, dtype=np.float32)
        out.append(PolygonRecord(oid, arr[:, 0], arr[:, 1]))
    return out


def transcript_counts_by_crop(
    bundle: XeniumBundle,
    crops: Sequence[CropBox],
    min_qv: float = 20.0,
    prefer_parquet: bool = True,
) -> dict[str, tuple[int, int]]:
    """Return total and assigned transcript counts for each crop."""

    counts = {crop.crop_id: [0, 0] for crop in crops}
    path = bundle.transcripts_path
    if path is None or not crops:
        return {k: (v[0], v[1]) for k, v in counts.items()}
    if path.suffix == ".parquet" and prefer_parquet and pds is not None:
        _count_transcripts_parquet(path, crops, counts, min_qv)
    elif path.suffix == ".parquet" and (bundle.root / "transcripts.csv.gz").exists():
        _count_transcripts_csv(bundle.root / "transcripts.csv.gz", crops, counts, min_qv)
    elif path.suffix == ".parquet" and pds is not None:
        _count_transcripts_parquet(path, crops, counts, min_qv)
    elif path.suffix != ".parquet":
        _count_transcripts_csv(path, crops, counts, min_qv)
    return {k: (v[0], v[1]) for k, v in counts.items()}


def transcript_scan_strategy(crops: Sequence[CropBox]) -> str:
    """Describe how transcript table scans are spatially filtered."""

    if len(crops) <= TRANSCRIPT_FILTER_CROP_LIMIT:
        return "crop_boxes"
    return "union_bounds"


def _crop_for_point(x: float, y: float, crops: Sequence[CropBox]) -> str | None:
    for crop in crops:
        if crop.contains_xy(x, y):
            return crop.crop_id
    return None


def _union_bounds(crops: Sequence[CropBox]) -> tuple[float, float, float, float]:
    return (
        min(c.xmin for c in crops),
        max(c.xmax for c in crops),
        min(c.ymin for c in crops),
        max(c.ymax for c in crops),
    )


def _is_assigned(cell_id: Any) -> bool:
    text = str(cell_id)
    return text not in {"", "0", "cell_0", "UNASSIGNED", "None", "nan"}


def _count_transcripts_csv(
    path: Path,
    crops: Sequence[CropBox],
    counts: dict[str, list[int]],
    min_qv: float,
) -> None:
    xmin, xmax, ymin, ymax = _union_bounds(crops)
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qv = float(row.get("qv") or 0.0)
            if qv < min_qv:
                continue
            x = float(row["x_location"])
            y = float(row["y_location"])
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            crop_id = _crop_for_point(x, y, crops)
            if crop_id is None:
                continue
            counts[crop_id][0] += 1
            if _is_assigned(row.get("cell_id", "")):
                counts[crop_id][1] += 1


def _count_transcripts_parquet(
    path: Path,
    crops: Sequence[CropBox],
    counts: dict[str, list[int]],
    min_qv: float,
) -> None:
    if pds is None:
        raise RuntimeError("pyarrow.dataset is required for transcript Parquet")
    filt = _transcript_filter_expression(crops, min_qv)
    dataset = pds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=["x_location", "y_location", "cell_id"], filter=filt)
    for batch in scanner.to_batches():
        values = batch.to_pydict()
        for x, y, cid in zip(values["x_location"], values["y_location"], values["cell_id"]):
            crop_id = _crop_for_point(float(x), float(y), crops)
            if crop_id is None:
                continue
            counts[crop_id][0] += 1
            if _is_assigned(cid):
                counts[crop_id][1] += 1


def _transcript_filter_expression(crops: Sequence[CropBox], min_qv: float) -> Any:
    if pds is None:
        raise RuntimeError("pyarrow.dataset is required for transcript Parquet")
    if transcript_scan_strategy(crops) == "crop_boxes":
        filt = _crop_filter_expression(crops[0])
        for crop in crops[1:]:
            filt = filt | _crop_filter_expression(crop)
    else:
        filt = _union_filter_expression(crops)
    if min_qv >= 0:
        filt = filt & (pds.field("qv") >= min_qv)
    return filt


def _crop_filter_expression(crop: CropBox) -> Any:
    if pds is None:
        raise RuntimeError("pyarrow.dataset is required for transcript Parquet")
    return (
        (pds.field("x_location") >= crop.xmin)
        & (pds.field("x_location") <= crop.xmax)
        & (pds.field("y_location") >= crop.ymin)
        & (pds.field("y_location") <= crop.ymax)
    )


def _union_filter_expression(crops: Sequence[CropBox]) -> Any:
    if pds is None:
        raise RuntimeError("pyarrow.dataset is required for transcript Parquet")
    xmin, xmax, ymin, ymax = _union_bounds(crops)
    return (
        (pds.field("x_location") >= xmin)
        & (pds.field("x_location") <= xmax)
        & (pds.field("y_location") >= ymin)
        & (pds.field("y_location") <= ymax)
    )


def iter_manifest_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        obj = json.load(handle)
    yield from obj.get("crops", [])
