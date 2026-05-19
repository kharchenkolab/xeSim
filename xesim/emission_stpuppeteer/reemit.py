"""``xesim re-emit-molecules``: re-sample transcripts on an existing bundle.

Copies an existing synth bundle to a new output directory, then
overwrites only ``transcripts.parquet`` and
``ground_truth/molecule_provenance.parquet`` with new transcripts
sampled against a (possibly different) STpuppeteer config. Morphology
images, cell geometries, and metadata are reused verbatim — that's the
whole point: skip the slow rendering step when only the emission model
changes.

By design the command **requires a new output directory** and errors out
if it already exists. No in-place writes, no .bak files; clean blast
radius.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Parquet/csv files re-emit fully rewrites via pandas to_parquet / to_csv.
# When the bundle was cloned with hardlinks we unlink these so writes
# don't mutate the source bundle's inodes.
_REWRITTEN_FILES = (
    "transcripts.parquet",
    "transcripts.csv.gz",
    "ground_truth/molecule_provenance.parquet",
)
# experiment.xenium is read by load_bundle THEN updated in place by
# _update_experiment_metadata; we can't unlink it pre-load. Real-copy it
# during the clone so the read finds it and the later in-place write
# breaks no link.
_REAL_COPY_FILES = ("experiment.xenium",)


def _clone_bundle(src: Path, out: Path, use_hard_links: bool) -> None:
    """Materialize a copy of ``src`` at ``out`` for re-emit to write into.

    With ``use_hard_links=True`` we run ``cp -al`` (Linux/macOS) so the
    20-GB morphology image and other static files are linked, not copied —
    the clone takes ~1s regardless of bundle size. Files re-emit fully
    rewrites are unlinked after the link clone; files re-emit updates
    in place are real-copied. Falls back to ``shutil.copytree`` if ``cp``
    is missing or the hardlink clone fails (e.g., cross-filesystem).
    """
    if not use_hard_links:
        shutil.copytree(src, out)
        return
    try:
        subprocess.run(["cp", "-al", str(src), str(out)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning(
            "cp -al failed (%s); falling back to full byte-copy", e)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src, out)
        return
    for rel in _REWRITTEN_FILES:
        p = out / rel
        if p.exists():
            os.unlink(p)
    for rel in _REAL_COPY_FILES:
        p_src = src / rel
        p_out = out / rel
        if p_src.exists():
            if p_out.exists():
                os.unlink(p_out)
            shutil.copy2(p_src, p_out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class ReemitResult:
    """Summary returned to the CLI for logging."""
    out_dir: Path
    scene_mode: str
    n_transcripts: int
    n_cells: int


def reemit_molecules(
    src_bundle: str | Path,
    out_bundle: str | Path,
    *,
    stpuppeteer_config: str | Path,
    seed: int = 0,
    use_hard_links: bool = False,
) -> ReemitResult:
    """Re-emit transcripts for an existing synth bundle.

    Parameters
    ----------
    src_bundle : path
        Existing xeSim explain output (2D or 2.5D).
    out_bundle : path
        New output directory; **must not exist**. Source is cloned here
        first, then the transcripts files are overwritten.
    stpuppeteer_config : path
        Path to STpuppeteer YAML config.
    seed : int
        RNG seed for the emission step. Same config + same seed →
        byte-identical output.
    use_hard_links : bool
        If True, clone the bundle with ``cp -al`` (hardlinks) instead of a
        full byte-copy. Skips the multi-GB copy of ``morphology.ome.tif``
        and other static files. Files we rewrite (transcripts.parquet,
        transcripts.csv.gz, experiment.xenium, ground_truth/
        molecule_provenance.parquet) are unlinked after the clone so
        writes break the link and never touch the source bundle. Falls
        back to full byte-copy if the hardlink clone fails (cross-fs,
        missing cp). Off by default — safe default for users who may move
        the source bundle later.
    """
    src = Path(src_bundle).resolve()
    out = Path(out_bundle).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source bundle not found: {src}")
    if src == out:
        raise ValueError(
            "out_bundle is the same as src_bundle — re-emit-molecules requires "
            "a new output directory (it never overwrites the source in place)."
        )
    if out.exists():
        raise FileExistsError(
            f"out_bundle already exists: {out}. Remove it or choose a new path; "
            "re-emit-molecules never overwrites an existing directory."
        )

    # Step 1: load the source bundle. load_bundle rasterises polygons
    # into cell_label / nucleus_label on first call and persists them to
    # SRC/ground_truth/{cell,nucleus}_label.npy so subsequent re-emits
    # mmap the cache (~1s) instead of re-rasterising (~5-7min). We load
    # from SRC, not OUT, so the cache benefits every future re-emit on
    # this bundle, not just runs that happen to write to the same OUT.
    from .bundle_reader import load_bundle
    lb = load_bundle(src)

    # Step 2: clone bundle. Default is shutil.copytree (full byte-copy);
    # --use-hard-links uses `cp -al` and unlinks the files we'll rewrite,
    # which avoids the multi-GB copy of morphology.ome.tif. Source bundle
    # is never modified either way (we only added the cache in step 1).
    if use_hard_links:
        logger.info("cloning (hardlinks) %s → %s", src, out)
    else:
        logger.info("copying %s → %s", src, out)
    _clone_bundle(src, out, use_hard_links=use_hard_links)

    # Step 3: emit. Dispatch on scene_mode. The emit_* helpers handle
    # config loading, count sampling, leakage, and placement; we just
    # need to thread the right inputs through.
    rng = np.random.default_rng(int(seed))
    if lb.meta.scene_mode == "2.5d":
        from .emit import emit_3d
        xmin, ymin, _, _ = lb.meta.tile_bounds_um
        mol_df = emit_3d(
            cells_records=lb.cells,
            cell_label_3d=lb.cell_label,
            nucleus_label_3d=lb.nucleus_label,
            z_slices_um=lb.z_slices_um.tolist(),
            tile_origin_um=(xmin, ymin),
            pixel_size_um=lb.meta.pixel_size_um,
            stpuppeteer_config=str(stpuppeteer_config),
            rng=rng,
        )
        _write_25d_transcripts(out, mol_df)
    elif lb.meta.scene_mode == "2d":
        from .emit import emit_2d
        # emit_2d expects a scene with .cells, .cell_label, .nucleus_label,
        # .pixel_size. Wrap LoadedBundle in a minimal SimpleNamespace.
        scene = SimpleNamespace(
            cells=lb.cells,
            cell_label=lb.cell_label,
            nucleus_label=lb.nucleus_label,
            pixel_size=lb.meta.pixel_size_um,
        )
        mol_df = emit_2d(
            scene=scene,
            stpuppeteer_config=str(stpuppeteer_config),
            rng=rng,
            pixel_size_um=lb.meta.pixel_size_um,
        )
        # emit_2d's output uses scene-format (cell_id, gene, x, y, ...);
        # the 2D bundle writer reshapes to true_cell_id + Xenium schema.
        # We replicate the relevant subset here for parity.
        _write_2d_transcripts(out, mol_df, lb)
    else:
        raise ValueError(f"unsupported scene_mode {lb.meta.scene_mode!r}")

    # Step 4: update experiment.xenium num_transcripts to match the new
    # output. Leaves the rest of the metadata (pixel_size, tile_bounds,
    # synth_metadata, etc.) untouched.
    _update_experiment_metadata(out, n_transcripts=int(len(mol_df)))

    n_cells = lb.meta.n_cells if lb.meta.n_cells > 0 else len(lb.cells)
    return ReemitResult(
        out_dir=out,
        scene_mode=lb.meta.scene_mode,
        n_transcripts=int(len(mol_df)),
        n_cells=n_cells,
    )


# ---------------------------------------------------------------------------
# Writers — mirror the 2D / 2.5D bundle writers' transcripts schema
# ---------------------------------------------------------------------------


def _write_25d_transcripts(out_dir: Path, mol: pd.DataFrame) -> None:
    """Mirror ``scene_2_5d/bundle_writer_25d.py``'s transcripts + provenance.

    Matches the schema that the 2.5D writer originally produced (lines
    78-89 and 290-312 of bundle_writer_25d.py) so downstream tooling
    sees a byte-compatible bundle.
    """
    n = len(mol)
    # Stable, ordered IDs — match the writer's "synth_NNNNNNNNN" convention.
    transcript_ids = np.asarray([f"synth_{i:09d}" for i in range(n)], dtype=object)
    feature_name = (mol["gene"].astype(str).to_numpy()
                      if "gene" in mol.columns
                      else np.full(n, "synth", dtype=object))
    qv_vals = (mol["qv"].astype(np.float32).to_numpy()
                 if "qv" in mol.columns
                 else np.full(n, 40.0, dtype=np.float32))

    # NOTE: the legacy 2.5D writer hard-codes overlaps_nucleus=True for
    # all transcripts (line 86 of bundle_writer_25d.py). We follow that
    # for output-parity even though emit_3d computes the real value per
    # transcript — that information is preserved in the provenance file
    # below for diagnostic use.
    transcripts_df = pd.DataFrame({
        "transcript_id": transcript_ids,
        "x_location": mol["x_true"].astype(np.float32).values,
        "y_location": mol["y_true"].astype(np.float32).values,
        "z_location": mol["z_true"].astype(np.float32).values,
        "qv": qv_vals,
        "feature_name": feature_name,
        "cell_id": mol["true_cell_id"].astype(str).values,
        "overlaps_nucleus": np.ones(n, dtype=bool),
        "nucleus_distance": np.zeros(n, dtype=np.float32),
    })
    # Overwrite the existing transcripts files. Both parquet AND csv.gz
    # (which the writer also produces) — keep them consistent.
    transcripts_df.to_parquet(out_dir / "transcripts.parquet", index=False)
    transcripts_df.to_csv(out_dir / "transcripts.csv.gz",
                            index=False, compression="gzip")

    # Provenance — same shape as the legacy writer.
    gt_dir = out_dir / "ground_truth"
    is_ghost = (mol["is_ghost"].astype(bool).to_numpy()
                  if "is_ghost" in mol.columns
                  else np.zeros(n, dtype=bool))
    true_factor = (mol["factor_label"].astype(np.int64).to_numpy()
                     if "factor_label" in mol.columns
                     else np.full(n, -1, dtype=np.int64))
    true_celltype = (mol["true_cell_type"].astype(str).to_numpy()
                       if "true_cell_type" in mol.columns
                       else np.full(n, "unknown", dtype=object))
    mol_prov = pd.DataFrame({
        "transcript_id": transcript_ids,
        "tile_idx": np.zeros(n, dtype=np.int64),
        "true_cell_id": mol["true_cell_id"].astype(str).to_numpy(),
        "is_ghost": is_ghost,
        "gene": feature_name,
        "true_factor": true_factor,
        "x_um": mol["x_true"].astype(np.float64).to_numpy(),
        "y_um": mol["y_true"].astype(np.float64).to_numpy(),
        "x_true": mol["x_true"].astype(np.float32).to_numpy(),
        "y_true": mol["y_true"].astype(np.float32).to_numpy(),
        "z_true": mol["z_true"].astype(np.float32).to_numpy(),
        "true_cell_type": true_celltype,
    })
    mol_prov.to_parquet(gt_dir / "molecule_provenance.parquet", index=False)


def _write_2d_transcripts(out_dir: Path, mol: pd.DataFrame, lb) -> None:
    """Mirror ``scene_2d/bundle_writer.py``'s transcripts + provenance.

    The 2D writer produces a slightly different schema than 2.5D (includes
    fov_name, codeword_index, uses uint64 transcript IDs, ghost-derived
    transcripts get cell_id="UNASSIGNED" while preserving true_cell_id in
    the provenance file).
    """
    n = len(mol)
    rng_ids = np.random.default_rng(0)
    # Synthetic uint64 IDs in the same range the legacy writer uses
    # (10**12..10**14); won't collide with real Xenium IDs.
    transcript_ids = rng_ids.integers(low=10**12, high=10**14, size=n).astype(np.uint64)

    # The 2D output of emit_2d uses scene-format columns (cell_id, gene,
    # x, y, ...). xeSim's wrapper in _emit_molecules renames cell_id →
    # true_cell_id and adds is_ghost; for re-emit we apply that mapping
    # here directly, then translate to the Xenium-spec columns.
    cell_id_to_is_ghost = {
        c.cell_id: bool(c.provenance.get("is_ghost", False))
        for c in lb.cells
    }
    true_cell_id = mol["cell_id"].astype(str).to_numpy()
    is_ghost = mol["cell_id"].map(cell_id_to_is_ghost).fillna(False).astype(bool).to_numpy()

    # Ghost-derived transcripts get cell_id="UNASSIGNED" in the public file
    # (line 145 of bundle_writer.py). True source is kept in the provenance.
    public_cell_id = true_cell_id.copy()
    public_cell_id[is_ghost] = "UNASSIGNED"

    overlaps_nuc = (mol["overlaps_nucleus"].astype(np.uint8).to_numpy()
                       if "overlaps_nucleus" in mol.columns
                       else np.zeros(n, dtype=np.uint8))
    transcripts_df = pd.DataFrame({
        "transcript_id": transcript_ids,
        "cell_id": public_cell_id,
        "overlaps_nucleus": overlaps_nuc,
        "feature_name": mol["gene"].astype(str).to_numpy(),
        "x_location": mol["x"].astype(np.float32).to_numpy(),
        "y_location": mol["y"].astype(np.float32).to_numpy(),
        # No z in 2D scene-format output; constant zero is the legacy default.
        "z_location": np.zeros(n, dtype=np.float32),
        "qv": (mol["qv"].astype(np.float32).to_numpy()
                  if "qv" in mol.columns
                  else np.full(n, 40.0, dtype=np.float32)),
        "fov_name": np.full(n, "T000", dtype=object),
        "nucleus_distance": np.where(overlaps_nuc > 0, 0.0, 5.0).astype(np.float32),
        "codeword_index": np.zeros(n, dtype=np.int32),
    })
    transcripts_df.to_parquet(out_dir / "transcripts.parquet", index=False)
    transcripts_df.to_csv(out_dir / "transcripts.csv.gz",
                            index=False, compression="gzip")

    # Provenance — 2D's schema (matches what's expected by downstream tools
    # that read both 2D and 2.5D bundles).
    gt_dir = out_dir / "ground_truth"
    mol_prov = pd.DataFrame({
        "transcript_id": transcript_ids,
        "tile_idx": np.zeros(n, dtype=np.int64),
        "true_cell_id": true_cell_id,
        "is_ghost": is_ghost,
        "gene": mol["gene"].astype(str).to_numpy(),
        "true_factor": (mol["factor_label"].astype(np.int64).to_numpy()
                          if "factor_label" in mol.columns
                          else np.full(n, -1, dtype=np.int64)),
        "x_um": mol["x"].astype(np.float64).to_numpy(),
        "y_um": mol["y"].astype(np.float64).to_numpy(),
    })
    mol_prov.to_parquet(gt_dir / "molecule_provenance.parquet", index=False)


def _update_experiment_metadata(out_dir: Path, *, n_transcripts: int) -> None:
    """Update ``experiment.xenium`` to reflect the new transcript count.

    Other metadata (pixel_size, tile_bounds, z_step, synth_metadata) is
    preserved as-is — re-emit doesn't change geometry or the rendering
    pipeline, only transcript content.
    """
    path = out_dir / "experiment.xenium"
    with open(path) as f:
        exp = json.load(f)
    exp["num_transcripts"] = int(n_transcripts)
    # Mark provenance so consumers know transcripts were re-emitted.
    synth = exp.setdefault("synth_metadata", {})
    synth["reemit"] = True
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
