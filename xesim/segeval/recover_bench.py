"""Drop-and-recover segmentation benchmark (misc/reseg.md, Phase 0/1).

Ablate a set of cells from a 10x-segmented canonical crop, recover them from
image evidence with a pluggable recovery method, and score the recovery against
the held-out truth via :func:`segeval.metrics.match_segmentations`. Aggregate
across crops, stratified by ablation size and cell type, to produce the
baseline recovery curve every later improvement (joint proposer, topology
augmentation, renderer-in-the-loop) must beat.

Recovery methods (``recover_fn(real_dapi, real_membrane, cl_ab, nl_ab) -> list[CandidateCell]``):
  - :func:`recover_heuristic` — training-free DAPI-peak + ridge proposer.
  - :func:`make_learned_recover` — a trained TinyProposerUNet checkpoint.

Headline metric is **recall of the ablated cells** (``match_rate_a``): of the
cells we removed, how many got a recovered cell back near them. ``median_iou``
measures contour quality of the matches. ``spurious_rate_b`` is reported but
contaminated — a recovery method also proposes in genuine background / truly
missed cells, which look "spurious" against an ablated-only truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ablation import (select_ablation_set, ablate, truth_label,
                       local_density_per_100um2)
from .metrics import match_segmentations

RecoverFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list]


def candidates_to_label(candidates: list, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize recovered ``CandidateCell`` footprints into a label image."""
    out = np.zeros(shape, dtype=np.int32)
    H, W = shape
    for k, c in enumerate(candidates, start=1):
        px = getattr(c, "cell_pixels", None)
        if px is None or len(px) == 0:
            continue
        ys = px[:, 0].astype(np.int64)
        xs = px[:, 1].astype(np.int64)
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        out[ys[ok], xs[ok]] = k
    return out


def recover_heuristic(real_dapi, real_membrane, cl_ab, nl_ab) -> list:
    """Training-free A3 proposer (DAPI peaks + membrane-ridge components)."""
    from ..scene_completion import propose_latent_cells
    return propose_latent_cells(real_dapi, real_membrane, cl_ab, nl_ab)


def make_learned_recover(proposer: dict[str, Any], **kw) -> RecoverFn:
    """Build a recovery fn backed by a trained TinyProposerUNet checkpoint
    (load via ``proposer_decode.load_proposer``)."""
    from ..proposer_decode import propose_latent_cells_learned

    def _rec(real_dapi, real_membrane, cl_ab, nl_ab):
        return propose_latent_cells_learned(
            real_dapi, real_membrane, cl_ab, nl_ab, proposer, **kw)
    return _rec


def _crop_type_lookup(crop_id: str, cell_types_path: Path | None) -> dict[str, int]:
    if cell_types_path is None or not Path(cell_types_path).exists():
        return {}
    art = json.loads(Path(cell_types_path).read_text())
    for c in art.get("crops", []):
        if str(c.get("crop_id")) == crop_id:
            return {str(k): int(v) for k, v in c.get("cell_id_to_type_index", {}).items()}
    return {}


def bench_crop(
    npz_path: Path,
    rng: np.random.Generator,
    *,
    mode: str,
    size: int,
    recover_fn: RecoverFn,
    crop_id: str = "",
    type_lookup: dict[str, int] | None = None,
    type_names: list[str] | None = None,
    n_realizations: int = 2,
    max_centroid_um: float = 5.0,
    restrict_to_ablated_region: bool = True,
) -> list[dict[str, Any]]:
    """Run ``n_realizations`` ablate->recover->match trials on one crop.

    When ``restrict_to_ablated_region`` (default), recovered candidates are kept
    only if their centroid falls in the dilated footprint of the ablated cells.
    This de-contaminates precision: a recovered cell in true background or on a
    genuinely 10x-missed cell is out of scope for *this* benchmark (we only have
    truth for the cells we removed), so it should not count as spurious. Recall
    is unaffected (distant proposals never match an ablated truth cell anyway).
    """
    with np.load(npz_path, allow_pickle=True) as d:
        images = np.asarray(d["images"], dtype=np.float32)
        cl = np.asarray(d["cell_label"], dtype=np.int32)
        nl = np.asarray(d["nucleus_label"], dtype=np.int32)
        cell_ids = [str(v) for v in d["cell_ids"]]
        psz = float(d["pixel_size"])
    dapi, mem = images[0], images[1]
    # original cell label -> cell_id (cell_ids are ordered by sorted live labels)
    live = np.unique(cl)
    live = live[live > 0]
    lab_to_cid = {int(l): cell_ids[i] for i, l in enumerate(live)
                  if i < len(cell_ids)}
    type_lookup = type_lookup or {}

    results: list[dict[str, Any]] = []
    for _ in range(n_realizations):
        labels = select_ablation_set(cl, rng, mode=mode, size=size)
        if not labels:
            continue
        cl_ab, nl_ab = ablate(cl, nl, labels)
        truth, ordered = truth_label(cl, labels)
        density = local_density_per_100um2(cl, labels, psz)
        cands = recover_fn(dapi, mem, cl_ab, nl_ab)
        if restrict_to_ablated_region and cands:
            from scipy.ndimage import binary_dilation
            abl_mask = np.isin(cl, list(labels))
            rad = int(round(max_centroid_um / psz))
            region = binary_dilation(abl_mask, iterations=max(1, rad))
            H, W = cl.shape
            cands = [c for c in cands
                     if 0 <= int(round(c.cy)) < H and 0 <= int(round(c.cx)) < W
                     and region[int(round(c.cy)), int(round(c.cx))]]
        rec = candidates_to_label(cands, cl.shape)
        pairs, summ = match_segmentations(truth, rec, pixel_size_um=psz,
                                          max_centroid_um=max_centroid_um)
        # per-type recall: which truth (relabelled) cells got matched
        matched_truth = {p.label_a for p in pairs}
        per_type = {}
        for new_lab, orig in enumerate(ordered, start=1):
            cid = lab_to_cid.get(int(orig), "")
            t = type_lookup.get(cid, 0)
            tname = (type_names[t] if type_names and 0 <= t < len(type_names)
                     else str(t))
            slot = per_type.setdefault(tname, [0, 0])  # [matched, total]
            slot[1] += 1
            if new_lab in matched_truth:
                slot[0] += 1
        results.append({
            "crop_id": crop_id,
            "mode": mode,
            "size_requested": int(size),
            "n_ablated": int(len(labels)),
            "density_per_100um2": float(density),
            "summary": summ.to_dict(),
            "per_type": per_type,
        })
    return results


def run_benchmark(
    canonical_dir: str | Path,
    recover_fn: RecoverFn,
    *,
    split: str = "test",
    mode: str = "cluster",
    sizes: tuple[int, ...] = (1, 4, 12),
    n_realizations: int = 2,
    max_crops: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the drop-and-recover benchmark over a canonical-crop directory.

    Returns an aggregate report: per-size recall / median-IoU / count-error and
    a per-type recall breakdown, plus the raw per-trial records.
    """
    canonical_dir = Path(canonical_dir)
    manifest = json.loads((canonical_dir / "manifest.json").read_text())
    cell_types_path = canonical_dir / "cell_types.json"
    type_names = None
    if cell_types_path.exists():
        type_names = list(json.loads(cell_types_path.read_text()).get("type_names", []))

    crops = [c for c in manifest["crops"] if c.get("split") == split]
    if max_crops:
        crops = crops[:max_crops]
    rng = np.random.default_rng(seed)

    all_records: list[dict[str, Any]] = []
    for c in crops:
        crop_id = str(c.get("crop_id"))
        npz = canonical_dir / c["npz_path"]
        if not npz.exists():
            continue
        tl = _crop_type_lookup(crop_id, cell_types_path if cell_types_path.exists() else None)
        for size in sizes:
            all_records.extend(bench_crop(
                npz, rng, mode=mode, size=size, recover_fn=recover_fn,
                crop_id=crop_id, type_lookup=tl, type_names=type_names,
                n_realizations=n_realizations))

    # aggregate by ablation size
    by_size: dict[int, dict[str, Any]] = {}
    type_tally: dict[str, list[int]] = {}
    for r in all_records:
        s = r["size_requested"]
        agg = by_size.setdefault(s, {"recall": [], "median_iou": [], "iou50": [],
                                     "count_delta": [], "n_ablated": [],
                                     "density": [], "n_trials": 0})
        sm = r["summary"]
        agg["recall"].append(sm["match_rate_a"])
        agg["median_iou"].append(sm["median_iou"])
        agg["iou50"].append(sm["iou_at_50"])
        agg["count_delta"].append(sm["count_delta"])
        agg["n_ablated"].append(r["n_ablated"])
        agg["density"].append(r["density_per_100um2"])
        agg["n_trials"] += 1
        for tname, (m, tot) in r["per_type"].items():
            slot = type_tally.setdefault(tname, [0, 0])
            slot[0] += m
            slot[1] += tot

    size_summary = {}
    for s, agg in sorted(by_size.items()):
        size_summary[s] = {
            "n_trials": agg["n_trials"],
            "mean_n_ablated": float(np.mean(agg["n_ablated"])) if agg["n_ablated"] else 0.0,
            "recall_mean": float(np.mean(agg["recall"])) if agg["recall"] else 0.0,
            "median_iou_mean": float(np.mean(agg["median_iou"])) if agg["median_iou"] else 0.0,
            "iou50_mean": float(np.mean(agg["iou50"])) if agg["iou50"] else 0.0,
            "count_delta_mean": float(np.mean(agg["count_delta"])) if agg["count_delta"] else 0.0,
            "mean_density_per_100um2": float(np.mean(agg["density"])) if agg["density"] else 0.0,
        }
    type_recall = {t: {"recall": (m / tot if tot else 0.0), "n": tot}
                   for t, (m, tot) in sorted(type_tally.items(), key=lambda kv: -kv[1][1])}

    return {
        "config": {"split": split, "mode": mode, "sizes": list(sizes),
                   "n_realizations": n_realizations, "n_crops": len(crops),
                   "seed": seed},
        "by_size": size_summary,
        "by_type_recall": type_recall,
        "records": all_records,
    }


__all__ = ["candidates_to_label", "recover_heuristic", "make_learned_recover",
           "bench_crop", "run_benchmark"]
