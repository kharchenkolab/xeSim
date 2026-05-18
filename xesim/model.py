"""XesimModel — top-level API for the three plan3 tasks.

Wraps the full fitting + inference pipeline behind a clean class. Lives
on top of the existing per-module fit functions (`mechanistic_fit`,
`tissue_neighborhood`, etc.) and the v16 renderer trainer.

Tasks the class supports:

* ``XesimModel.fit(bundle, annotations, out_dir, ...)`` — one-shot pipeline
  that produces a self-contained model directory.
* ``XesimModel.load(model_dir)`` — load an already-fitted model.
* ``model.explain(bundle, crop_ids=..., complete=True)`` — render real
  tiles, optionally completing with the latent-cell proposer.
* ``model.generate(num_scenes=..., guide=..., seed=...)`` — sample new
  scenes; forward or slice-guided.
* ``model.render(scene)`` — deterministic render of a hand-built
  MechanisticScene.
* ``model.write(scene, render, out_dir, name=...)`` — standard
  serialization.

Model directory layout (see misc/cli_plan.md):

::

    MODEL_DIR/
    ├── manifest.json
    ├── canonical/manifest.json   (+ npz crops referenced by it)
    ├── cell_types.json
    ├── priors/
    │   ├── mechanistic_params.json
    │   ├── tissue_neighborhood.json
    │   ├── sectioning_states.json
    │   └── polya_spectrum.json
    ├── exemplars/exemplars.json
    └── renderer.pt
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image  # noqa: F401  (used in write())

from .canonicalize import canonicalize_bundle
from .cell_encoder import CROP_SIZE, CellEncoder, extract_cell_crops
from .cell_shape_exemplar import build_cell_shape_exemplars
from .cell_types import attach_cell_types
from .intensity_scatter import calibrate_render_means, fit_intensity_scatter
from .mechanistic_fit import fit_mechanistic_priors
from .mechanistic_scene import MechanisticScene
from .polya_spectrum import fit_polya_spectrum
from .scene_io import scene_from_canonical_crop
from .sectioning_states import fit_sectioning_states
from .slice_guide import extract_guide
from .slice_guided_sampler import sample_slice_guided_scene
from .structural_refiner import V37bUNet, build_structural_channels
from .tissue_neighborhood import fit_tissue_neighborhood, load_tissue_neighborhood
from .torch_utils import get_device
from .training.renderer import PatchDiscriminator, train_renderer
from .transcripts import (
    apply_admixture_rate,
    cell_h_dict_from_factor_fractions,
    classify_cells_by_centroids,
    fit_transcripts_nmf,
    load_cell_factor_fractions,
    load_molecule_factor_assignments,
    load_transcripts_priors,
    sample_scene_transcripts,
    tile_aware_per_type_alpha,
)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

MODEL_MANIFEST_VERSION = "xesim.model.v1"


@dataclass
class _Paths:
    """Resolves the canonical sub-paths inside a model directory."""
    root: Path

    @property
    def manifest(self) -> Path: return self.root / "manifest.json"
    @property
    def canonical(self) -> Path: return self.root / "canonical"
    @property
    def canonical_manifest(self) -> Path: return self.canonical / "manifest.json"
    @property
    def cell_types(self) -> Path: return self.canonical / "cell_types.json"
    @property
    def priors_dir(self) -> Path: return self.root / "priors"
    @property
    def mechanistic_params(self) -> Path: return self.priors_dir / "mechanistic_params.json"
    @property
    def tissue_neighborhood(self) -> Path: return self.priors_dir / "tissue_neighborhood.json"
    @property
    def sectioning_states(self) -> Path: return self.priors_dir / "sectioning_states.json"
    @property
    def polya_spectrum(self) -> Path: return self.priors_dir / "polya_spectrum.json"
    @property
    def exemplars(self) -> Path: return self.root / "exemplars" / "exemplars.json"
    @property
    def renderer(self) -> Path: return self.root / "renderer.pt"


# ---------------------------------------------------------------------------
# XesimModel
# ---------------------------------------------------------------------------


class XesimModel:
    """Fitted plan3 model: priors + renderer + per-type exemplars."""

    def __init__(self, model_dir: Path, *, device: str | None = None,
                 lazy_renderer: bool = True):
        self.paths = _Paths(Path(model_dir))
        self.device = get_device(device)
        self.manifest = json.loads(self.paths.manifest.read_text())
        ct = json.loads(self.paths.cell_types.read_text())
        self.type_names: list[str] = list(ct["type_names"])
        self.name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.type_names)}
        cid_to_type: dict[str, int] = {}
        for crop in ct.get("crops", []):
            for cid, idx in crop.get("cell_id_to_type_index", {}).items():
                cid_to_type[str(cid)] = int(idx)
        self.cid_to_type = cid_to_type
        self.n_type_one_hot = len(self.type_names) - 1

        self.tissue = load_tissue_neighborhood(self.paths.tissue_neighborhood)
        with self.paths.mechanistic_params.open() as f:
            self.mechanistic_params = json.load(f)
        self.transcripts_priors = load_transcripts_priors(self.paths.root)

        # Phase 2.G: when transcripts priors are present, the model's
        # transcript-aware methods (cell_factor_fractions,
        # classify_cells_by_transcripts, molecule_factor_assignments) need
        # to reach a cellAdmix run. Warn at load time if it's unreachable
        # so users see the issue before invoking an op that would fail.
        if self.transcripts_priors is not None and self.bundle_path:
            try:
                from .transcripts import _celladmix_run_dir
                run_dir = _celladmix_run_dir(self.paths.root,
                                              self.transcripts_priors,
                                              bundle_path=self.bundle_path)
                self._celladmix_run_dir = run_dir
            except FileNotFoundError as e:
                self._celladmix_run_dir = None
                print(f"[xesim] WARNING: model has transcripts priors but no "
                      f"reachable cellAdmix run.")
                print(f"  m.explain(complete=True) and (compare_transcripts=True) "
                      f"will skip silently.")
                print(f"  Expected location: "
                      f"{Path(self.bundle_path).resolve().parent / '_xesim_celladmix/runs/'}")
                print(f"  Re-run `xesim fit-model` on the bundle to "
                      f"populate it (cellAdmix is on by default; pass "
                      f"--celladmix-run PATH to reuse an existing fit).")
        else:
            self._celladmix_run_dir = None

        self._v37b = None
        self._encoder = None
        if not lazy_renderer:
            self._load_renderer()

    # -- Renderer loading -------------------------------------------------

    def _load_renderer(self) -> None:
        ck = torch.load(self.paths.renderer, map_location=self.device, weights_only=False)
        self._latent_dim = int(ck["latent_dim"])
        self._n_channels = int(ck.get("n_channels", 3))
        self._channel_names = ck.get("channel_names") or None
        self._renderer_variant = str(ck.get("renderer_variant", "v16"))
        # Phase 2.D v20+: optional per-pixel expected-intensity & stromal channels
        self._use_per_type_means = bool(ck.get("use_per_type_means", False))
        self._stromal_type_indices = ck.get("stromal_type_indices") or None
        if self._stromal_type_indices is not None:
            self._stromal_type_indices = tuple(self._stromal_type_indices)
        self._per_type_means_arr: "torch.Tensor | None" = None
        if self._use_per_type_means:
            ptm_path = self.paths.canonical / "per_type_channel_means.npy"
            if ptm_path.exists():
                import numpy as _np
                self._per_type_means_arr = _np.load(ptm_path).astype(_np.float32)
        # Total input channel count, prefer checkpoint's stored value
        n_extra = ((self._per_type_means_arr.shape[1] if self._per_type_means_arr is not None else 0)
                   + (1 if self._stromal_type_indices else 0))
        in_channels = int(ck.get("n_struct",
                                  8 + self.n_type_one_hot + n_extra)) + self._latent_dim
        self._v37b = V37bUNet(
            in_channels=in_channels,
            hidden=int(ck["hidden_v37b"]),
            out_channels=self._n_channels,
        ).to(self.device).eval()
        self._v37b.load_state_dict(ck["v37b"])
        self._encoder = CellEncoder(
            n_types=self.n_type_one_hot, latent_dim=self._latent_dim,
            in_channels=self._n_channels + 1,
        ).to(self.device).eval()
        self._encoder.load_state_dict(ck["encoder"])

    @property
    def v37b(self) -> V37bUNet:
        if self._v37b is None: self._load_renderer()
        return self._v37b  # type: ignore[return-value]

    @property
    def encoder(self) -> CellEncoder:
        if self._encoder is None: self._load_renderer()
        return self._encoder  # type: ignore[return-value]

    @property
    def latent_dim(self) -> int:
        if self._v37b is None: self._load_renderer()
        return self._latent_dim

    @property
    def pixel_size(self) -> float:
        return float(self.manifest["pixel_size"])

    @property
    def tile_px(self) -> int:
        return int(self.manifest["tile_px"])

    @property
    def bundle_path(self) -> str | None:
        return self.manifest.get("bundle")

    # -- Fitting (constructor variant) ------------------------------------

    @classmethod
    def fit(
        cls,
        bundle_path: str | Path,
        annotations_path: str | Path,
        out_dir: str | Path,
        *,
        num_crops: int = 128,
        crop_size_um: float = 64.0,
        steps: int = 8000,
        device: str | None = None,
        seed: int = 1,
        with_transcripts: bool = True,
        celladmix_run: str | Path | None = None,
        crop_selection: str = "density",
        stratified_alpha: float = 0.5,
        stratified_within_pick: str = "centroid",
    ) -> "XesimModel":
        """Fit a complete model from a Xenium bundle and produce a
        self-contained MODEL_DIR. Long-running (~30-60 min).

        ``crop_selection`` controls canonical-crop sampling. See
        ``canonicalize_bundle`` for the available strategies.
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        paths = _Paths(out_dir)
        for sub in (paths.canonical, paths.priors_dir, paths.exemplars.parent):
            sub.mkdir(parents=True, exist_ok=True)

        # 1) canonicalize
        print(f"[fit] canonicalize {bundle_path} → {paths.canonical} "
              f"(crop_selection={crop_selection})")
        canonicalize_bundle(
            bundle_path, paths.canonical, num_crops=num_crops,
            crop_size_um=crop_size_um, seed=seed,
            crop_selection=crop_selection,
            annotation_path=annotations_path,
            stratified_alpha=stratified_alpha,
            stratified_within_pick=stratified_within_pick,
        )
        # 2) attach annotations
        print(f"[fit] attach cell types from {annotations_path}")
        attach_cell_types(paths.canonical_manifest, Path(annotations_path), paths.cell_types)
        # Stash a copy of the full annotation file inside the model dir so
        # explain_region can resolve cell types for cells OUTSIDE the
        # training crops without depending on bundle layout. Without this,
        # ~93% of bundle cells fall back to "unknown" type at inference
        # and per-type-conditioned channels (especially membrane) render dim.
        _ann_src = Path(annotations_path)
        _ann_dst_dir = paths.root / "annotations"
        _ann_dst_dir.mkdir(parents=True, exist_ok=True)
        _ann_dst = _ann_dst_dir / "annotation.csv.gz"
        import shutil as _shutil
        _shutil.copyfile(_ann_src, _ann_dst)
        print(f"[fit] stashed annotation file → {_ann_dst}")

        # 3) fit priors
        print("[fit] mechanistic priors")
        # fit_mechanistic_priors expects a DIRECTORY to write into (creates
        # mechanistic_params.json + mechanistic_prior_report.json there).
        # Passing the .json path would mkdir it as a directory.
        fit_mechanistic_priors(paths.canonical_manifest, paths.priors_dir,
                                 cell_types_path=paths.cell_types)
        print("[fit] intensity scatter")
        fit_intensity_scatter(paths.canonical_manifest, paths.cell_types,
                                paths.mechanistic_params)
        print("[fit] tissue neighborhood")
        fit_tissue_neighborhood(paths.canonical_manifest, paths.cell_types,
                                  paths.tissue_neighborhood)
        print("[fit] sectioning states")
        fit_sectioning_states(paths.canonical_manifest, paths.cell_types,
                                paths.sectioning_states)
        print("[fit] polyA spectrum")
        fit_polya_spectrum(paths.canonical_manifest, paths.cell_types,
                              paths.polya_spectrum)
        print("[fit] cell shape exemplars")
        build_cell_shape_exemplars(paths.canonical_manifest, paths.cell_types,
                                     paths.exemplars,
                                     sectioning_states_path=paths.sectioning_states)

        # 4) train renderer — pass channel info from the canonical manifest
        # so the trained renderer matches the bundle's actual channel count.
        # Without this the trainer falls back to the legacy 3-channel default,
        # producing a renderer that's silently incompatible with the bundle.
        with open(paths.canonical_manifest) as _f:
            _canon_mf = json.load(_f)
        _ch_names = _canon_mf.get("image_channels") or None
        _n_channels = len(_ch_names) if _ch_names else 3
        print(f"[fit] train renderer ({steps} steps, {_n_channels}-channel: "
                f"{_ch_names})")
        train_renderer(paths.canonical, paths.renderer, steps=steps,
                          device=device, n_channels=_n_channels,
                          channel_names=_ch_names)

        # 5) transcript NMF priors (cellAdmix sister package, default on).
        # Three integration modes:
        #   - with_transcripts=False           → skip entirely
        #   - celladmix_run is a path          → reuse that cellAdmix dir
        #   - else                             → run cellAdmix on the bundle
        #                                         (cellAdmix's own cache will
        #                                         skip re-fit if a previous
        #                                         run exists at the default
        #                                         <bundle>/_xesim_celladmix/)
        if with_transcripts:
            from .transcripts import bundle_celladmix_dir
            if celladmix_run is not None:
                ca_dir_use = Path(celladmix_run)
                if not ca_dir_use.exists():
                    raise FileNotFoundError(
                        f"--celladmix-run not found: {ca_dir_use}")
                print(f"[fit] cellAdmix: using supplied run dir → {ca_dir_use}")
            else:
                ca_dir_use = bundle_celladmix_dir(bundle_path)
                if ca_dir_use.exists():
                    print(f"[fit] cellAdmix: reusing or extending existing "
                          f"run at {ca_dir_use}")
                else:
                    print(f"[fit] cellAdmix: running fresh on bundle "
                          f"({ca_dir_use})")
            fit_transcripts_nmf(
                bundle_path=bundle_path,
                annotation_path=annotations_path,
                model_dir=out_dir,
                annotation_col="merged_annotation",
                score_membrane=False,    # set True once we plumb the morphology image
                celladmix_dir=str(ca_dir_use),
            )

        # 5b) fit 3D nucleus priors (always — used by 2.5D rendering).
        # Quick (~5s) and writes a single nucleus_priors.json under
        # priors_3d/. The 2.5D rendering path auto-discovers this file.
        try:
            from .scene_2_5d.fit_priors import fit_nucleus_priors_from_bundle
            priors_3d_dir = out_dir / "priors_3d"
            priors_3d_dir.mkdir(parents=True, exist_ok=True)
            print(f"[fit] 3D nucleus priors → {priors_3d_dir}/nucleus_priors.json")
            fit_nucleus_priors_from_bundle(
                bundle_path=bundle_path,
                annotation_path=annotations_path,
                out_path=priors_3d_dir / "nucleus_priors.json",
            )
        except Exception as e:
            print(f"[fit] 3D nucleus priors skipped: {e}")

        # 5c) build stain-classifier latent bank for the kNN fallback
        # (4th tier after annotation, transcript-classifier, training
        # cid_to_type). Catches cells that fall through all earlier
        # fallbacks. Bank ~10k typed cells × 12-dim latent = ~0.5MB.
        try:
            from .stain_classifier import build_latent_bank, save_bank
            print(f"[fit] stain-classifier latent bank")
            # Load self-as-model to call encode_real
            self_model = cls.load(out_dir, device=device)
            lat, ty, cids = build_latent_bank(self_model, paths.canonical)
            save_bank(out_dir / "cell_latent_bank.npz", lat, ty, cids)
            print(f"  bank: {len(lat)} typed cells, "
                  f"{len(np.unique(ty))} types -> {out_dir}/cell_latent_bank.npz")
        except Exception as e:
            print(f"[fit] stain-classifier bank skipped: {e}")

        # 6) write manifest — pull renderer-side fields (tile_px,
        # channel_names, n_channels, renderer_variant, recon_weight_mode)
        # from the renderer checkpoint so the model.tile_px / channel_names
        # accessors find them. Without this the bundle build crashes on
        # `KeyError: 'tile_px'`.
        import torch as _torch
        try:
            ckpt = _torch.load(paths.renderer, map_location="cpu",
                                  weights_only=False)
            ckpt_keys = {k: ckpt.get(k) for k in
                            ("n_channels", "channel_names", "renderer_variant",
                             "use_per_type_means", "stromal_type_indices")
                            if k in ckpt}
        except Exception:
            ckpt_keys = {}
        # tile_px: training tile in pixels (cell crops are 64µm crops at
        # bundle pixel_size). The trainer doesn't write this directly, so
        # derive from crop_size_um / pixel_size.
        from .xenium import resolve_bundle as _rb
        from datetime import datetime, timezone
        psz = float(_rb(str(bundle_path)).pixel_size)
        manifest = {
            "schema_version": MODEL_MANIFEST_VERSION,
            "bundle": str(bundle_path),
            "annotations": str(annotations_path),
            "num_crops": num_crops,
            "crop_size_um": crop_size_um,
            "renderer_steps": steps,
            "renderer": "renderer.pt",
            "with_transcripts": bool(with_transcripts),
            "pixel_size": psz,
            "tile_px": int(round(crop_size_um / psz)),
            "recon_weight_mode": "uniform",
            "fit_timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "seed": int(seed),
            **ckpt_keys,
        }
        paths.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"[fit] model written → {out_dir}  (variant="
                f"{ckpt_keys.get('renderer_variant', '?')}, fit_timestamp="
                f"{manifest['fit_timestamp']})")
        return cls(out_dir, device=device)

    # -- Load ------------------------------------------------------------

    @classmethod
    def load(cls, model_dir: str | Path, *, device: str | None = None,
              lazy_renderer: bool = True) -> "XesimModel":
        return cls(model_dir, device=device, lazy_renderer=lazy_renderer)

    # -- Render ----------------------------------------------------------

    def precompute_render_inputs(self, scene: MechanisticScene, *,
                                     cell_latents: dict[int, np.ndarray] | None = None,
                                     seed: int = 0,
                                     latent_scale: float = 1.0,
                                     ) -> dict:
        """Compute the slowly-varying per-pixel inputs to render() ONCE
        for a whole-region scene. Returns a dict suitable for passing to
        `render(scene, precomputed=...)` so per-tile renders reuse the
        global computation.

        Cuts ~60% of bridge-tile rendering time on large scenes (the
        per-tile build_structural_channels + 3× distance transforms are
        the dominant cost; computing once globally and slicing is much
        cheaper).
        """
        compact, nuc_remap, cti, _ = _compact_scene(scene, self.name_to_idx)
        if self._v37b is None: self._load_renderer()
        struct_np, _ = build_structural_channels(
            compact, nuc_remap, cti, self.n_type_one_hot,
            per_type_channel_means=getattr(self, "_per_type_means_arr", None),
            stromal_type_indices=getattr(self, "_stromal_type_indices", None),
        )
        nz = np.unique(scene.cell_label); nz = nz[nz > 0]
        n_cells = int(len(nz))
        torch.manual_seed(int(seed))
        z = torch.randn(n_cells, self.latent_dim, device=self.device) * float(latent_scale)
        if cell_latents is not None:
            for i, orig_lbl in enumerate(nz):
                if int(orig_lbl) in cell_latents:
                    z[i] = torch.from_numpy(
                        np.asarray(cell_latents[int(orig_lbl)])).to(self.device)
        lut = torch.cat([torch.zeros(1, self.latent_dim, device=self.device), z], dim=0)
        return {"compact": compact, "struct_np": struct_np, "lut": lut}

    def render(self, scene: MechanisticScene, *,
                cell_latents: dict[int, np.ndarray] | None = None,
                seed: int | None = None,
                latent_scale: float = 1.0,
                background_mask_sigma: float | None = 3.0,
                autocast_dtype: "torch.dtype | None" = None,
                precomputed: dict | None = None,
                precomputed_slice: tuple[int, int, int, int] | None = None,
                ) -> np.ndarray:
        """Render a `MechanisticScene` through the trained renderer.

        cell_latents: optional dict of original-label → (latent_dim,) array.
        Labels are the labels in `scene.cell_label` (not the 1..N compacted
        indices). Cells not in the dict fall back to N(0, latent_scale²·I).
        When None, all latents are sampled from N(0, latent_scale²·I).

        latent_scale: scale factor on the random latent prior. >1 widens the
        prior to surface more tail behaviors at inference. Has no effect on
        cells whose latent is provided in ``cell_latents``.

        background_mask_sigma: if not None, post-render multiply the output
        by a Gaussian-blurred cell mask (σ in pixels). Suppresses non-cell
        emission ("purple background" in cell-free regions) that some
        renderers learn — particularly brighter ones trained with
        density-sampled crops or low-variance-channel up-weighting.
        Default 3.0 matches v21 baseline. Pass None only if the renderer
        is confirmed not to produce far-from-cell emission.
        """
        if self._v37b is None: self._load_renderer()
        if precomputed is not None:
            # Slice the global compact/struct/lut to the tile.
            if precomputed_slice is None:
                raise ValueError(
                    "render(precomputed=...) requires precomputed_slice=(y0,x0,h,w)")
            y0, x0, h_, w_ = precomputed_slice
            compact = precomputed["compact"][y0:y0+h_, x0:x0+w_].astype(np.int32, copy=False)
            struct_np = precomputed["struct_np"][:, y0:y0+h_, x0:x0+w_]
            lut = precomputed["lut"]
            struct_t = torch.from_numpy(struct_np).unsqueeze(0).to(self.device)
            compact_t = torch.from_numpy(compact).long().to(self.device)
        else:
            compact, nuc_remap, cti, _types = _compact_scene(scene, self.name_to_idx)
            struct_np, _ = build_structural_channels(
                compact, nuc_remap, cti, self.n_type_one_hot,
                per_type_channel_means=getattr(self, "_per_type_means_arr", None),
                stromal_type_indices=getattr(self, "_stromal_type_indices", None),
            )
            struct_t = torch.from_numpy(struct_np).unsqueeze(0).to(self.device)
            compact_t = torch.from_numpy(compact).long().to(self.device)
            nz = np.unique(scene.cell_label); nz = nz[nz > 0]
            n_cells = int(len(nz))

            if seed is not None: torch.manual_seed(int(seed))
            z = torch.randn(n_cells, self.latent_dim, device=self.device) * float(latent_scale)
            if cell_latents is not None:
                for i, orig_lbl in enumerate(nz):
                    if int(orig_lbl) in cell_latents:
                        z[i] = torch.from_numpy(np.asarray(cell_latents[int(orig_lbl)])).to(self.device)
            lut = torch.cat([torch.zeros(1, self.latent_dim, device=self.device), z], dim=0)

        with torch.no_grad():
            lat_block = lut[compact_t].permute(2, 0, 1).unsqueeze(0)
            cond = torch.cat([struct_t, lat_block], dim=1)
            if autocast_dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=autocast_dtype):
                    recon = self.v37b(cond)
                recon = recon.float()
            else:
                recon = self.v37b(cond)
        out = recon.squeeze(0).cpu().numpy()

        # Phase 2.D: post-render soft-mask to kill background leakage.
        # The renderer (v19/v21) was trained on dense canonical crops and
        # learned a non-zero "baseline" everywhere. In sparse tiles this
        # becomes visible aSMA blobs in cell-label==0 regions.
        if background_mask_sigma is not None and background_mask_sigma > 0:
            from scipy.ndimage import gaussian_filter
            cm = (compact > 0).astype(np.float32)
            soft = gaussian_filter(cm, sigma=float(background_mask_sigma))
            out = out * soft[None, :, :]
        return out

    def render_batch(self, scenes: "list[MechanisticScene]", *,
                       cell_latents: dict[int, np.ndarray] | None = None,
                       seed: int = 0,
                       latent_scale: float = 1.0,
                       background_mask_sigma: float | None = 3.0,
                       autocast_dtype: "torch.dtype | None" = None,
                       ) -> list[np.ndarray]:
        """Render a BATCH of equally-sized scenes in one GPU forward.

        Scenes must share the same `image_shape` so they can be stacked.
        Each scene's `cell_label` integer namespace is treated
        independently — labels are compacted per scene, then a global
        latent lookup applies to each scene's compacted ids.

        Cuts the renderer cost ~N for N tiles (the dominant cost in
        large-region rendering) by amortizing kernel launches and
        utilizing the GPU more fully.

        Returns a list of N `(C, H, W)` float32 arrays in the same
        order as `scenes`.
        """
        if not scenes:
            return []
        if self._v37b is None: self._load_renderer()
        shapes = {s.image_shape for s in scenes}
        if len(shapes) != 1:
            raise ValueError(f"All scenes in render_batch must share image_shape; "
                              f"got {shapes!r}")
        H, W = next(iter(shapes))

        compact_list, struct_list, nz_list = [], [], []
        for s in scenes:
            compact, nuc_remap, cti, _ = _compact_scene(s, self.name_to_idx)
            struct_np, _ = build_structural_channels(
                compact, nuc_remap, cti, self.n_type_one_hot,
                per_type_channel_means=getattr(self, "_per_type_means_arr", None),
                stromal_type_indices=getattr(self, "_stromal_type_indices", None),
            )
            compact_list.append(compact)
            struct_list.append(struct_np)
            nz = np.unique(s.cell_label); nz = nz[nz > 0]
            nz_list.append(nz)

        struct_t = torch.from_numpy(np.stack(struct_list, axis=0)).to(self.device)  # (N, S, H, W)

        # Per-scene latent block.
        torch.manual_seed(int(seed))
        lat_blocks = []
        for compact, nz in zip(compact_list, nz_list):
            n_cells = int(len(nz))
            z = torch.randn(n_cells, self.latent_dim, device=self.device) * float(latent_scale)
            if cell_latents is not None:
                for i, orig_lbl in enumerate(nz):
                    if int(orig_lbl) in cell_latents:
                        z[i] = torch.from_numpy(
                            np.asarray(cell_latents[int(orig_lbl)])).to(self.device)
            lut = torch.cat([torch.zeros(1, self.latent_dim, device=self.device), z], dim=0)
            compact_t = torch.from_numpy(compact).long().to(self.device)
            lat_block = lut[compact_t].permute(2, 0, 1)  # (D, H, W)
            lat_blocks.append(lat_block)
        lat_t = torch.stack(lat_blocks, dim=0)  # (N, D, H, W)

        cond = torch.cat([struct_t, lat_t], dim=1)  # (N, S+D, H, W)
        with torch.no_grad():
            if autocast_dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=autocast_dtype):
                    recon = self.v37b(cond)
                recon = recon.float()
            else:
                recon = self.v37b(cond)
        recon_cpu = recon.cpu().numpy()
        out_list = [recon_cpu[i] for i in range(recon_cpu.shape[0])]

        if background_mask_sigma is not None and background_mask_sigma > 0:
            from scipy.ndimage import gaussian_filter
            for i, compact in enumerate(compact_list):
                cm = (compact > 0).astype(np.float32)
                soft = gaussian_filter(cm, sigma=float(background_mask_sigma))
                out_list[i] = out_list[i] * soft[None, :, :]
        return out_list

    def encode_real(self, scene: MechanisticScene, real_rgb: np.ndarray
                      ) -> dict[int, np.ndarray]:
        """Encode each cell's appearance from a real RGB tile into the
        per-cell latent. Returns dict of original-label → (latent_dim,) np.ndarray.

        Used in `explain` mode so the renderer reproduces observed
        appearance rather than inventing new ones.
        """
        compact, _, _, per_cell_type = _compact_scene(scene, self.name_to_idx)
        # extract_cell_crops walks the unique labels of `compact` in order,
        # so the i-th crop corresponds to compact label (i+1), which
        # corresponds to the i-th original label in np.unique(scene.cell_label)[>0].
        cell_crops, base_masks = extract_cell_crops(real_rgb, compact)
        crops_in = cell_crops * base_masks[:, None]
        x_enc = np.concatenate([crops_in, base_masks[:, None]], axis=1)  # (N, 4, K, K)
        with torch.no_grad():
            x = torch.from_numpy(x_enc).to(self.device).float()
            t = torch.from_numpy(per_cell_type).to(self.device)
            mu, _ = self.encoder(x, t)
        nz = np.unique(scene.cell_label); nz = nz[nz > 0]
        return {int(orig_lbl): mu[i].cpu().numpy() for i, orig_lbl in enumerate(nz)}

    # -- Transcript accessors (Phase 2) ---------------------------------

    def _require_transcripts(self) -> dict[str, Any]:
        if self.transcripts_priors is None:
            raise RuntimeError(
                "model has no transcripts priors; refit with `xesim fit-model "
                "--with-transcripts`"
            )
        return self.transcripts_priors

    def cell_factor_fractions(
        self,
        *,
        only_biology: bool = False,
        include_annotation: bool = False,
    ) -> "pd.DataFrame":
        """Per-cell biology-factor fractions from the cellAdmix fit.

        Returns a DataFrame with `cell_id, transcript_count, dominant_factor,
        dominant_fraction, factor_0_fraction .. factor_{K-1}_fraction` (0-indexed).
        With `only_biology=True`, admixture-factor columns are dropped.
        With `include_annotation=True`, the annotated cell type is joined in
        as `_cell_type`.
        """
        self._require_transcripts()
        return load_cell_factor_fractions(
            self.paths.root, self.transcripts_priors,
            only_biology=only_biology, include_annotation=include_annotation,
            bundle_path=self.bundle_path,
        )

    def classify_cells_by_transcripts(
        self,
        cell_h: dict[str, np.ndarray] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Assign each cell a transcript-implied type by cosine similarity
        to the per-type alpha centroids in the priors.

        If `cell_h` is None, all cells from the cellAdmix fit are scored.
        Returns dict cell_id → {cell_type_transcripts, type_uncertainty,
        cosine_top, type_scores}.
        """
        self._require_transcripts()
        if cell_h is None:
            cf = self.cell_factor_fractions()
            cell_h = cell_h_dict_from_factor_fractions(
                self.transcripts_priors, cf,
            )
        return classify_cells_by_centroids(self.transcripts_priors, cell_h)

    def stamp_scene_with_transcripts(
        self,
        scene: "MechanisticScene",
        classifications: dict[str, dict[str, Any]] | None = None,
    ) -> "MechanisticScene":
        """Return a new scene whose cells have a `transcripts` block in
        provenance carrying the cellAdmix-implied cell type and uncertainty.

        If `classifications` is None, the model classifies all fit cells
        (one-time pass, cached if the caller reuses it across scenes).
        """
        from dataclasses import replace
        self._require_transcripts()
        if classifications is None:
            classifications = self.classify_cells_by_transcripts()
        new_cells = []
        for c in scene.cells:
            cls = classifications.get(c.cell_id)
            if cls is None:
                new_cells.append(c)
                continue
            new_prov = {**c.provenance,
                          "transcripts": {
                              "cell_type_transcripts": cls["cell_type_transcripts"],
                              "type_uncertainty": cls["type_uncertainty"],
                              "cosine_top": cls["cosine_top"],
                              "type_scores": cls["type_scores"],
                          }}
            new_cells.append(replace(c, provenance=new_prov))
        return replace(scene, cells=tuple(new_cells))

    def molecule_factor_assignments(
        self,
        *,
        columns: list[str] | None = None,
        with_gene_names: bool = True,
        with_cell_ids: bool = True,
        include_unassigned: bool = False,
        unassigned_min_qv: float = 20.0,
    ) -> "pd.DataFrame":
        """Per-molecule factor labels from the cellAdmix fit.

        Returns a DataFrame with (at least) `factor_label` (0-indexed) and
        `factor_margin`; by default also adds `cell_id`, `gene`, `x`, `y`,
        and any QV / nuclear-overlap columns cellAdmix recorded.

        `include_unassigned=True` additionally pulls bundle-level
        `cell_id='UNASSIGNED'` transcripts (qv >= `unassigned_min_qv`),
        each tagged with the gene's argmax-factor from `factors.parquet`.
        These are the orphans the transcript proposer needs to find
        missed cells.

        Results are memoized per (columns, flags, qv) tuple; multi-tile
        stitching loops pay the parquet read once.
        """
        self._require_transcripts()
        cache_key = (
            tuple(columns) if columns is not None else None,
            bool(with_gene_names), bool(with_cell_ids),
            bool(include_unassigned), float(unassigned_min_qv),
        )
        cache = getattr(self, "_molecule_assignments_cache", None)
        if cache is None:
            cache = {}
            self._molecule_assignments_cache = cache
        if cache_key in cache:
            return cache[cache_key]
        df = load_molecule_factor_assignments(
            self.paths.root, self.transcripts_priors,
            columns=columns,
            with_gene_names=with_gene_names,
            with_cell_ids=with_cell_ids,
            bundle_path=self.bundle_path,
            include_unassigned=include_unassigned,
            unassigned_min_qv=unassigned_min_qv,
        )
        cache[cache_key] = df
        return df

    # -- Explain (real tile → render with optional completion) ----------

    def explain(self, bundle_or_canonical: str | Path, *,
                  crop_ids: Iterable[str] | None = None,
                  num_crops: int | None = None,
                  complete: bool = False,
                  compare_transcripts: bool = False,
                  seed: int = 0) -> list[dict[str, Any]]:
        """Render real tiles through the renderer.

        For first release, `bundle_or_canonical` is expected to be a
        canonical-crop directory (the model's own canonical/ or a
        compatible one). True bundle-to-explain (re-canonicalize on the
        fly) is a follow-up; today the model's canonical crops cover the
        common case.

        complete=True runs the transcript-based latent cell proposer
        (Phase 4): each crop's orphan transcripts (those not covered by a
        segmented cell) are DBSCAN-clustered and each cluster of ≥3
        molecules becomes a proposed cell. Requires `--with-transcripts`.

        compare_transcripts=True stamps each cell's `provenance` with a
        `transcripts` dict carrying the cellAdmix-implied cell type and a
        cosine-margin uncertainty (requires `--with-transcripts` at fit
        time).
        """
        canonical_dir = Path(bundle_or_canonical)
        if not (canonical_dir / "manifest.json").exists():
            raise NotImplementedError(
                "explain currently expects a canonical-crop directory (a "
                "MODEL_DIR/canonical/ or any sibling). Bundle-to-explain "
                "(re-canonicalize on the fly) will be added in a follow-up. "
                "For now, run `xesim debug canonicalize` first."
            )
        if complete and self.transcripts_priors is None:
            raise NotImplementedError(
                "complete=True requires transcripts priors; refit with "
                "`xesim fit-model --with-transcripts`."
            )

        ct_path = canonical_dir / "cell_types.json"
        manifest = json.loads((canonical_dir / "manifest.json").read_text())
        crops = manifest.get("crops", [])
        if crop_ids is not None:
            wanted = set(crop_ids)
            crops = [c for c in crops if c.get("crop_id") in wanted]
        elif num_crops is not None:
            crops = crops[: int(num_crops)]

        # Phase 2.F: m.explain is a thin wrapper over explain_region. We use
        # canonical's stored crop bounds to know what regions to render; the
        # rendering reads from the real bundle directly (not the frozen npz)
        # and includes ALL bundle cells in the region.
        bundle_root = manifest.get("bundle", {}).get("root")
        if not bundle_root:
            raise RuntimeError(
                "canonical manifest does not record a bundle root; cannot "
                "explain via the bundle path. Re-canonicalize with the bundle "
                "path baked in.")
        from .scene_2d.explain_region import explain_region
        rng_proposer = np.random.default_rng(int(seed))
        outputs: list[dict[str, Any]] = []
        for crop in crops:
            crop_id = crop.get("crop_id")
            crop_box = crop["crop_box"]
            bounds = (crop_box["xmin"], crop_box["ymin"],
                       crop_box["xmax"], crop_box["ymax"])
            res = explain_region(
                self, bundle_root, region_bounds_um=bounds,
                annotation_path=None,
                add_ghosts=False,
                add_transcript_proposed=bool(complete),
                stamp_transcripts=bool(compare_transcripts),
                sample_molecules=False,
                rng=rng_proposer,
            )
            outputs.append({
                "crop_id": crop_id,
                "scene": res.scene.mech_scene,
                "render": res.image,
                "real": res.real_image,
                "completion": ({"n_added": res.n_transcript_proposed}
                                if complete else None),
            })
        return outputs

    # -- Generate (forward synth, optional guide) ------------------------

    def generate(self, *,
                  num_scenes: int = 1,
                  guide: tuple[str | Path, str] | None = None,
                  seed: int = 0,
                  target_num_cells: int = 70,
                  with_transcripts: bool | None = None,
                  admixture_rate_multiplier: float = 1.0,
                  ) -> list[dict[str, Any]]:
        """Sample new scenes.

        guide: optional ``(canonical_dir, crop_id)`` to anchor slice-guided
        generation. When None, fully forward synth from priors.

        with_transcripts: if True, also sample synthetic transcripts via
        the NMF priors and attach them as a `transcripts` DataFrame to
        each output item. Defaults to True when transcripts priors are
        loaded, False otherwise.

        admixture_rate_multiplier: Phase 8 controlled-admixture knob.
        Default 1.0 reproduces the observed bundle admixture per type.
        Set 0.0 to emit pure-biology cells (every molecule from the
        cell's native factors only), > 1.0 to inject extra admixture
        (capped at 0.95 mass per type for stability). Only takes effect
        in the no-guide forward path; slice-guided uses per-tile alpha.
        """
        if with_transcripts is None:
            with_transcripts = self.transcripts_priors is not None
        if with_transcripts and self.transcripts_priors is None:
            raise RuntimeError("with_transcripts=True but model has no "
                                  "transcripts priors; refit with --with-transcripts")
        outputs: list[dict[str, Any]] = []
        if guide is not None:
            guide_dir, guide_id = guide
            guide_dir = Path(guide_dir)
            gmanifest = json.loads((guide_dir / "manifest.json").read_text())
            crops = {c.get("crop_id"): c for c in gmanifest.get("crops", [])}
            if guide_id not in crops:
                raise ValueError(f"guide crop_id {guide_id!r} not in {guide_dir}")
            real_scene, _ = scene_from_canonical_crop(
                guide_dir / crops[guide_id]["npz_path"],
                guide_dir / "cell_types.json",
            )
            guide_obj = extract_guide(real_scene, self.type_names, scales=(4, 8, 16),
                                        source_id=guide_id)
            # Tile-aware Dirichlet centers for slice-guided transcripts (Phase 6):
            # condition each type's factor mix on the guide tile's actual cells.
            tile_alpha = None
            if with_transcripts:
                guide_ids = [c.cell_id for c in real_scene.cells]
                cf = self.cell_factor_fractions(include_annotation=True)
                tile_alpha = tile_aware_per_type_alpha(
                    self.transcripts_priors, cf, guide_cell_ids=guide_ids,
                )
            # Pull per-type efficiencies straight from the model's already-
            # loaded mechanistic_params (the function falls back to a
            # hardcoded dev-machine path otherwise, which breaks on any
            # other machine).
            per_type_eff = self.mechanistic_params.get(
                "per_type_efficiencies", None)
            for i in range(num_scenes):
                scene = sample_slice_guided_scene(
                    guide_obj, seed=seed + i + 1, type_names=self.type_names,
                    tissue_neighborhood=self.tissue,
                    guide_weights={4: 0.0, 8: 0.0, 16: 16.0},
                    type_jitter_p=0.05, type_marginal_weight=2.0,
                    n_cells_jitter_frac=0.10,
                    per_type_efficiencies=per_type_eff,
                    mechanistic_params_path=None,
                )
                render = self.render(scene, seed=seed + i + 1)
                item = {"scene": scene, "render": render,
                          "guide_crop_id": guide_id, "seed": seed + i + 1}
                if with_transcripts:
                    item["transcripts"] = sample_scene_transcripts(
                        self.transcripts_priors, scene,
                        rng=np.random.default_rng(seed + i + 1),
                        per_type_alpha_override=tile_alpha,
                    )
                outputs.append(item)
        else:
            # Plain forward synth via mechanistic_sampler.
            from .mechanistic_sampler import SyntheticSceneConfig, sample_synthetic_scene
            for i in range(num_scenes):
                cfg = SyntheticSceneConfig(
                    image_shape=(256, 256), target_num_cells=target_num_cells,
                    seed=seed + i + 1,
                )
                scene = sample_synthetic_scene(cfg, type_names=self.type_names,
                                                  tissue_neighborhood=self.tissue,
                                                  scene_id=f"synth_{seed + i + 1}")
                render = self.render(scene, seed=seed + i + 1)
                item = {"scene": scene, "render": render, "seed": seed + i + 1}
                if with_transcripts:
                    admix_alpha = None
                    if admixture_rate_multiplier != 1.0:
                        admix_alpha = apply_admixture_rate(
                            self.transcripts_priors, admixture_rate_multiplier,
                        )
                    item["transcripts"] = sample_scene_transcripts(
                        self.transcripts_priors, scene,
                        rng=np.random.default_rng(seed + i + 1),
                        per_type_alpha_override=admix_alpha,
                    )
                outputs.append(item)
        return outputs

    # -- Serialization ---------------------------------------------------

    @staticmethod
    def write(item: dict[str, Any], out_dir: str | Path, *,
                name: str | None = None) -> Path:
        """Write a single explain/generate output to disk.

        Always writes:
            scene.json, render.png, cell_label.npy, nucleus_label.npy

        When the item also carries a "real" array (i.e. came from explain),
        additionally writes:
            real.png       — the original RGB tile
            compare.png    — side-by-side real|render for visual diff
            quality.json   — per-tile error metrics (cell-masked L1,
                             per-channel L1, foreground spectrum L1 if
                             scipy is available)
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            name = item.get("crop_id") or f"scene_{item.get('seed', 0):04d}"
        d = out_dir / str(name)
        d.mkdir(parents=True, exist_ok=True)

        scene: MechanisticScene = item["scene"]
        render: np.ndarray = item["render"]
        real: np.ndarray | None = item.get("real")

        # scene.json (full serialization via the existing dataclass / dict)
        from dataclasses import asdict
        scene_dict = {
            "scene_id": scene.scene_id,
            "image_shape": list(scene.image_shape),
            "pixel_size": float(scene.pixel_size),
            "cells": [asdict(c) for c in scene.cells],
        }
        (d / "scene.json").write_text(json.dumps(scene_dict, indent=2, default=str))
        np.save(d / "cell_label.npy", scene.cell_label.astype(np.int32))
        np.save(d / "nucleus_label.npy", scene.nucleus_label.astype(np.int32))

        def _to_rgb_u8(arr: np.ndarray) -> np.ndarray:
            # Visual convention: (R=DAPI, G=membrane, B=polyA) — channel
            # order in the renderer tensor is already (DAPI, membrane, polyA).
            rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
            return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        Image.fromarray(_to_rgb_u8(render)).save(d / "render.png")

        if real is not None:
            real_u8 = _to_rgb_u8(real)
            render_u8 = _to_rgb_u8(render)
            Image.fromarray(real_u8).save(d / "real.png")
            # Side-by-side: real | render, with a 4-px white separator
            H, W = real_u8.shape[:2]
            sep_w = 4
            compare = np.full((H + 24, W * 2 + sep_w, 3), 32, dtype=np.uint8)
            compare[24:, :W] = real_u8
            compare[24:, W:W + sep_w] = 255
            compare[24:, W + sep_w:] = render_u8
            from PIL import ImageDraw, ImageFont
            im = Image.fromarray(compare)
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((8, 4), "real", fill=(255, 255, 255), font=font)
            draw.text((W + sep_w + 8, 4), "render", fill=(255, 255, 255), font=font)
            im.save(d / "compare.png")

            # Per-tile error metrics
            cell_mask = (scene.cell_label > 0).astype(np.float32)
            diff = np.abs(render - real)
            n_fg = float(cell_mask.sum()) + 1e-6
            cell_masked_l1 = float((diff * cell_mask[None]).sum() / (n_fg * 3))
            per_channel_l1 = {
                "dapi":    float((diff[0] * cell_mask).sum() / n_fg),
                "membrane": float((diff[1] * cell_mask).sum() / n_fg),
                "polya":   float((diff[2] * cell_mask).sum() / n_fg),
            }
            (d / "quality.json").write_text(json.dumps({
                "cell_masked_l1": cell_masked_l1,
                "per_channel_l1": per_channel_l1,
                "n_foreground_px": int(cell_mask.sum()),
                "n_cells": int(len(scene.cells)),
            }, indent=2))

        tx = item.get("transcripts")
        if tx is not None and len(tx) > 0:
            tx.to_parquet(d / "transcripts.parquet", index=False)

        return d


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _compact_scene(scene: MechanisticScene, name_to_idx: dict[str, int]
                     ) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    """Compact a MechanisticScene's labels to 1..N, build cti dict, and
    return per-cell type index array.

    Vectorized: builds a label-LUT and applies it via fancy indexing, so
    the cost is O(image_pixels + n_cells) rather than O(image_pixels *
    n_cells). For 1700+ anchor cells on a 2353×2353 image this cut the
    per-call time from ~17 s to <100 ms in profiling.
    """
    cl = scene.cell_label.astype(np.int32)
    nl = scene.nucleus_label.astype(np.int32)
    nz = np.unique(cl); nz = nz[nz > 0]
    cell_by_label = {int(c.label): c for c in scene.cells}
    cti: dict[int, int] = {}
    per_cell_type: list[int] = []
    if len(nz) == 0:
        return (np.zeros_like(cl, dtype=np.int32),
                np.zeros_like(cl, dtype=np.int32),
                cti, np.zeros((0,), dtype=np.int64))
    # LUT: old_label -> compact 1..N
    max_lbl = int(nz.max())
    lut = np.zeros(max_lbl + 1, dtype=np.int32)
    lut[nz] = np.arange(1, len(nz) + 1, dtype=np.int32)
    # Vectorized remap (cap out-of-range to 0 to be safe)
    cl_clipped = np.clip(cl, 0, max_lbl)
    nl_clipped = np.clip(nl, 0, max_lbl)
    compact = lut[cl_clipped]
    nuc_remap = lut[nl_clipped]
    for new_idx, old_lbl in enumerate(nz, start=1):
        c = cell_by_label.get(int(old_lbl))
        t = int(name_to_idx.get(c.cell_type if c else "unknown", 0))
        cti[new_idx] = t
        per_cell_type.append(t)
    return compact, nuc_remap, cti, np.array(per_cell_type, dtype=np.int64)
