# xeSim quickstart

The three commands cover everything most users will do.

> Prefer a worked example? See
> [`pancreas_vignette.md`](pancreas_vignette.md) for an end-to-end run
> on the public Xenium pancreas bundle (download, fit, render a tile,
> render the whole bundle) with real-vs-synth panels at each step.

## 0. Install

xeSim has two pieces: the Python package (this repo) and `cellAdmix-core`
(a separate C++ / Python build that supplies the transcript NMF priors).

```bash
# xeSim — include the [io,analysis] extras for tifffile, pandas, pyarrow,
# scipy, scikit-image, imagecodecs, zarr. Plain `pip install -e .` only
# pulls numpy + torch and most features will fail at runtime.
git clone <this-repo-url> xeSim
cd xeSim
pip install -e ".[io,analysis]"
```

```bash
# cellAdmix-core — required unless you always pass --no-transcripts to
# fit-model. Built from C++ via scikit-build-core; needs CMake + a system
# C++ toolchain. See cellAdmix-core/docs/install.md for the full list of
# system dependencies (Arrow, OpenMP, BLAS, etc.) and any platform-specific
# build flags.
git clone https://github.com/kharchenkolab/cellAdmix-core
python -m pip install -e cellAdmix-core/python --no-build-isolation
```

scikit-learn is required if you plan to use `--crop-selection stratified`
(installed automatically with `[analysis]`).

A CUDA GPU is required for `fit-model`. Inference (`explain`) works on
CPU but is much slower.

## 1. Get a fitted model

You need a Xenium bundle (`.zip` or extracted directory) plus a
cell-type annotation CSV with columns `cell_id` and `merged_annotation`.

```bash
xesim fit-model PATH/TO/BUNDLE \
    --annotations PATH/TO/cell_types.csv \
    --out my_model/
```

The default config fits 128 canonical crops and trains the renderer for
8000 steps (~30-60 min on a recent GPU). Override with `--num-crops N`
and `--steps N`. `cellAdmix-core` runs once on the bundle and the result
is cached at `<bundle_parent>/_xesim_celladmix/` — pass
`--celladmix-run PATH/TO/_xesim_celladmix/runs/fit_rank9_invsqrt_kl` to
reuse it when re-fitting, or `--no-transcripts` to skip cellAdmix
entirely.

### Canonical-crop selection (`--crop-selection`)

Which windows of the bundle anchor the renderer training crops:

| Strategy | What it does |
|---|---|
| `density` (default) | high-cell-density windows — biased toward dense epithelium / islets |
| `spread` | uniform spatial sampling across the bundle |
| `stratified` | **k-means on per-window cell-type composition + n_c^α quota** — covers common patterns *and* rare ones (T cells, fibroblasts in tumor-dominated samples). Needs `--annotations`. Tune with `--stratified-alpha` (default 0.5; 1=proportional ≈ spread, 0=uniform-per-cluster). Mirrors the H&E foundation-model recipe (StainStyleSampler 2025, Yottixel). |
| `random`, `grid` | alternatives, mostly for ablations |

The resulting `my_model/` is self-contained:

```
my_model/
├── manifest.json
├── canonical/                  # 128 canonical crops + cell_types.json + manifest.json
├── annotations/                # copy of the input cell-type annotation CSV
├── priors/
│   ├── mechanistic_params.json
│   ├── tissue_neighborhood.json
│   ├── sectioning_states.json
│   ├── polya_spectrum.json
│   ├── transcripts_nmf.json    # cellAdmix NMF priors + per-gene compartment posteriors
│   └── cell_factors.parquet    # per-cell NMF factor loadings
├── priors_3d/
│   └── nucleus_priors.json     # 3D nucleus shape priors (per cell type)
├── exemplars/                  # per-type real-mask shape exemplars
├── renderer.pt                 # V37bUNet + CellEncoder + discriminator state
├── diagnostics/                # training_loss.png, nucleus_prior_fit.png
└── training.log                # per-step renderer training metrics
```

## 2. Explain real tiles, regions, or whole bundles

The `explain` command operates on a real Xenium bundle and renders it
through the fitted model. Three scopes (mutually exclusive):

```bash
# Single tile, centered at (x, y) µm — diagnostic, fast
xesim explain PATH/TO/BUNDLE --model my_model/ --out look/ \
    --tile 2000,1500

# Arbitrary rectangular region — multi-tile stitched bundle
xesim explain PATH/TO/BUNDLE --model my_model/ --out patch_bundle/ \
    --region 1900,1400,2100,1600

# Whole bundle (FOV inferred from morphology image)
xesim explain PATH/TO/BUNDLE --model my_model/ --out full_synth_bundle/
```

Default output writer:

| Scope     | Default `--format` | What you get                                                  |
|-----------|--------------------|---------------------------------------------------------------|
| `--tile`  | `scene`            | `morphology.{png,npy}`, `transcripts.parquet`, `cells.parquet`, `summary.json` |
| `--region`, whole | `bundle`   | Xenium-compatible directory: morphology TIFF + transcripts/cells parquet + gene panel + ground-truth subdir |

Pass `--format bundle` to a `--tile` call if you want the Xenium-style
output for a single tile. (`--format scene` is not valid for multi-tile
scopes.)

### Augmentation flags

These work identically in every scope:

| Flag                          | What it does                                                       |
|-------------------------------|--------------------------------------------------------------------|
| `--add-ghosts` / `--no-ghosts` | density-calibrated extra cells for realism (default: on)          |
| `--add-transcript-proposed`   | recover missed cells from orphan transcript clusters              |
| `--stamp-transcripts`         | tag each cell's provenance with cellAdmix-implied type + uncertainty |
| `--annotation PATH`           | override the model's bundled cell-type annotation                 |

Both `--add-transcript-proposed` and `--stamp-transcripts` require the
model to have been fitted with `fit-model --with-transcripts` (so the
cellAdmix NMF prior is available).

### Common recipes

Spot-check one tile with all augmentations on:
```bash
xesim explain PATH/TO/BUNDLE --model my_model/ --out look/ \
    --tile 2000,1500 \
    --add-transcript-proposed --stamp-transcripts
```

Make a Xenium-compatible synth bundle covering a 200×200 µm region,
recovering missed cells from transcript evidence:
```bash
xesim explain PATH/TO/BUNDLE --model my_model/ --out patch_bundle/ \
    --region 1900,1400,2100,1600 \
    --add-transcript-proposed --intensity-calibration scale
```

Render the whole bundle (this is large; expect tens of minutes to hours):
```bash
xesim explain PATH/TO/BUNDLE --model my_model/ --out full_synth_bundle/ \
    --add-transcript-proposed --intensity-calibration histmatch
```

### Output: bundle format

Bundle outputs are valid Xenium directories (loadable by Xenium Explorer
and most downstream toolchains). The `ground_truth/` subdir adds the
synthetic provenance not present in real bundles:

```
patch_bundle/
├── transcripts.{parquet,csv.gz}     # synth transcripts
├── cell_boundaries.{parquet,csv.gz}
├── nucleus_boundaries.{parquet,csv.gz}
├── cells.{parquet,csv.gz}
├── morphology.ome.tif               # multi-channel pyramid
├── morphology_focus/                # per-channel TIFFs
├── gene_panel.json
├── experiment.xenium
└── ground_truth/
    ├── cells_synth.parquet          # cell_id → type / source / latent / provenance
    ├── molecule_provenance.parquet  # per-molecule cell_id and factor
    └── config.json
```

### Output: scene format (`--tile`)

Single-tile scene dirs are smaller and easier to inspect:

```
look/
├── morphology.png        # quick RGB visualization
├── morphology.npy        # raw (C, H, W) float32 render
├── transcripts.parquet
├── cells.parquet
└── summary.json
```

### 2.5D (z-stack) output

Pass `--scene-mode 2.5d` to render a **multi-z DAPI stack** (12 planes
by default, 3 µm step across 33 µm of depth) alongside the focal-plane
4-channel morphology. Per-cell 3D tilt + extent are sampled from the
fitted nucleus shape priors (`priors_3d/nucleus_priors.json`).

```bash
# Region (auto-stitches if area > 0.3 mm²)
xesim explain PATH/TO/BUNDLE --model my_model/ --out region_25d/ \
    --scene-mode 2.5d --region 1900,1400,2400,1900

# Whole bundle — adds morphology.ome.tif z-stack to the standard bundle
xesim explain PATH/TO/BUNDLE --model my_model/ --out full_25d/ \
    --scene-mode 2.5d --whole-bundle --num-workers 3
```

2.5D outputs are standard Xenium-format bundles **plus** the extra
DAPI z-stack in `morphology.ome.tif` and 3D per-molecule coordinates
(`x_location`, `y_location`, `z_location`) in `transcripts.parquet`.
Wall time for a typical pancreas-scale bundle is ~45–50 min at 3 GPU
workers; multi-z DAPI dominates the runtime vs the 2D path.

Stitch knobs (whole-bundle / large regions):

| Flag | Default | Purpose |
|---|---|---|
| `--stitch-tile-um` | 300 | per-tile size; smaller = more tiles, more I/O |
| `--stitch-overlap-um` | 50 | tile overlap; feather-blended at seams |

Per-cell focal-plane render comes from the same 2D `explain_region`
path (so 2D and 2.5D match exactly at the focal plane); the 2.5D-novel
contributions are the z-stack and 3D molecule coordinates.

## Python API

The same tasks are available as Python entry points:

```python
from xesim import XesimModel
from xesim.scene_2d.explain_region import explain_region
from xesim.scene_2d.scene_pipeline import build_scene

model = XesimModel.load("my_model/")

# Single-tile / single-region explain (mirror of `xesim explain --tile / --region`)
res = explain_region(
    model, "PATH/TO/BUNDLE",
    region_bounds_um=(1900, 1400, 1964, 1464),  # xmin,ymin,xmax,ymax µm
    add_ghosts=True,
    add_transcript_proposed=True,
    stamp_transcripts=True,
)
# res.scene (Scene2D), res.image (C,H,W float), res.cell_latents, res.n_*

# Multi-tile stitched render (mirror of `xesim explain --region` / whole bundle)
result = build_scene(
    model, "PATH/TO/BUNDLE",
    scene_bounds_um=(1900, 1400, 2100, 1600),
    add_ghosts=True,
    add_transcript_proposed=True,
)
# result['scenes'] list[Scene2D], result['stitched_image'] (C,H,W uint16)
```

The `MechanisticScene` and `MechanisticParams` types live in
`xesim.mechanistic_scene` — you can build a scene by hand and pass it
to `model.render(scene)` for full control.

The legacy `model.explain(canonical_dir, num_crops=N, complete=True,
compare_transcripts=True)` still exists as a thin wrapper over
`explain_region` for batch processing pre-extracted canonical crops; new
code should prefer `explain_region` directly.

## Inspection

```bash
xesim inspect-bundle PATH/TO/BUNDLE   # quick bundle stats
xesim inspect-model my_model/         # fitted-model manifest + types
```

## What's next

Cross-bundle slice-guided generation is planned. The bundle output
already writes per-channel float32 TIFFs (`morphology_focus/`)
alongside the multi-channel pyramid (`morphology.ome.tif`).
