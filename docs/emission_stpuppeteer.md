# STpuppeteer emission backend

`xesim explain` ships two molecular-emission backends. The default
**legacy** backend samples transcripts from cellAdmix-fit NMF priors that
mirror the real bundle's biology. The **STpuppeteer** backend instead
emits transcripts from a *configurable* per-cell-type model — you decide
which genes a cell type expresses, at what rate, with what leakage, and
the simulator produces exactly that. Use it when you want predictable,
controllable emission for benchmarking downstream tools, sweeping
parameters, or studying selective contamination effects.

## Quick-start: pancreas (377-gene panel)

These examples use the included pancreas reference config and the
`xesim_v21_model` fitted model. Adjust paths to your local setup.

**Single-tile 2D explain (~1 minute):**

```bash
xesim explain /workspace/Xenium_pancreas_membrane_377/data \
    --model /workspace/xeSim/tmp/xesim_v21_model \
    --out /tmp/pancreas_stp_tile \
    --tile 2000,1500 \
    --emission-backend stpuppeteer \
    --stpuppeteer-config xesim/emission_stpuppeteer/reference_configs/pancreas.yml \
    --device cpu
```

**300 × 300 µm 2.5D region (~4 minutes on CPU):**

```bash
xesim explain /workspace/Xenium_pancreas_membrane_377/data \
    --model /workspace/xeSim/tmp/xesim_v21_model \
    --out /tmp/pancreas_stp_25d \
    --region 1850,1350,2150,1650 \
    --scene-mode 2.5d \
    --emission-backend stpuppeteer \
    --stpuppeteer-config xesim/emission_stpuppeteer/reference_configs/pancreas.yml \
    --device cpu --num-workers 1
```

**Re-emit on an existing bundle with a different config (~20 seconds):**

```bash
xesim re-emit-molecules /tmp/pancreas_stp_25d \
    --out /tmp/pancreas_stp_25d_v2 \
    --stpuppeteer-config tuned_config.yml \
    --seed 1 \
    --diagnostic
```

`--out` must not exist; the command never overwrites. `--diagnostic`
writes per-cell-type stats to `<out>/diagnostics/emission_stpuppeteer.json`
and prints a compact summary to stdout.

## What you configure is what you get

The STpuppeteer backend exists for **controlled emission, not realism**.
There is no auto-rescaling against a target bundle, no calibration step,
no fit-to-real loop. Your config is the contract — synthesised counts
will reflect whatever you put in `expression_baseline`, program loadings,
and per-cell-type activations.

If you want realism-matched counts on a specific Xenium bundle, use the
legacy backend.

If you want to set, say, "Immune cells in this scene leak 60% of their
transcripts and the rest follow my fixed marker panel", the STpuppeteer
backend gives you that exactly.

## How the two simulators are wired together

xeSim borrows STpuppeteer's emission math and embeds it as a backend.
The bulk of the heavy lifting lives in STpuppeteer; xeSim is a thin
adapter that supplies xeSim's scene cells + geometry and writes out a
Xenium-compatible bundle.

The four STpuppeteer entry points xeSim calls are:

| STpuppeteer call | What it does | Where it's called from |
| --- | --- | --- |
| `SimulationConfig.from_yaml(path)` | Parses your YAML config into a typed Python object with all the dispersion knobs, programs, and per-cell-type activations resolved. | Once per `explain` / `re-emit-molecules` run. |
| `sample_counts_program_model(cell_gdf, cfg, rng)` | Draws the per-(cell, gene) integer count matrix. Uses STpuppeteer's LMC model: a per-cell program-activation latent `z_c ~ N(μ_k, σ²_k)`, gene mean `μ* = max(μ₀ + W·z_c, ε)`, NB-equivalent Poisson-Gamma sampling. | Inside `emission_stpuppeteer/counts.py`. |
| `classify_leakage(trs_df, cell_gdf, gpar_df, rng)` | Per-transcript Bernoulli leak decision via the union rule `p_eff = 1 - (1 - p_celltype)(1 - p_gene)`. | Inside `emission_stpuppeteer/counts.py`. |
| `build_program_gpar_df(cfg)` | Builds the per-gene parameter table (baseline μ, per-program loadings) consumed by `classify_leakage` and downstream diagnostics. | Inside `emission_stpuppeteer/counts.py`. |

xeSim provides:

- The **cell list**: anchors from the bundle annotation, plus any
  transcript-proposed cells. Ghost cells stay in the scene for stain
  rendering but are filtered out of emission (they have no biological
  identity to drive STpuppeteer's per-cell-type model).
- The **geometry**: the rasterised `cell_label[...]` array (post overlap
  resolution). xeSim's per-cell SDF placement code uses this to put
  each transcript into a specific voxel and tag its `landed_in_cell_id`
  for ground-truth provenance.
- The **bundle output**: morphology images, cells.parquet, boundaries,
  and the public Xenium-compatible transcripts.parquet.

Conceptually: STpuppeteer answers *"how many transcripts of which genes
should this cell type emit, and how leaky is it?"* — xeSim answers
*"where in the rendered scene does each transcript land, and what does
the bundle on disk look like?"*. The two halves meet at a flat
per-transcript record that flows from STpuppeteer's count sampler
through xeSim's per-cell bounded-SDF placement to the bundle writer.

## Configuration

The YAML schema is STpuppeteer's `SimulationConfig.from_yaml`. A typical
explicit-style config looks like:

```yaml
seed: 42
expression_baseline: 0.05         # μ₀: background expression floor
overdispersion_cv: 0.5            # φ₀: NB CV at unit mean
overdispersion_exponent: 0.3      # α: mean–CV power-law slope
leak_dist_factor: 1.0             # halo radius / cell radius

leakage_by_celltype:
  Epithelial: 0.10                # 10% of Epithelial transcripts leak
  Immune: 0.30                    # 30% for Immune

programs:
  - name: ProgEpi
    loading: {EPCAM: 10.0, KRT7: 8.0, SOX17: 7.0, TFF2: 6.0}
  - name: ProgImm
    loading: {PTPRC: 11.0, CD3D: 9.0, CD8A: 8.0}

cell_type_specs:
  Epithelial:
    program_activations: {ProgEpi: 1.0}
  Immune:
    program_activations: {ProgImm: 1.0}
```

**Cell-type names must match the bundle annotation.** If the bundle has
types your config doesn't enumerate, xeSim warns and substitutes the
most common configured type (auto-fallback). If *no* configured type
appears in the bundle, you get a hard error.

**Gene names must appear in the bundle's gene panel.** xeSim hard-errors
at config load if a program references a gene that isn't in
`gene_panel.json` — otherwise you'd be emitting transcripts the
downstream tooling can't decode.

A working reference for the pancreas-377 bundle ships at
`xesim/emission_stpuppeteer/reference_configs/pancreas.yml`. Copy it as a
starting point and tune.

## Recipes

### Iterating on configs without re-rendering

The rendering step in `xesim explain` is the slow part (minutes per
tile). Once you have a rendered bundle, use `re-emit-molecules` to swap
in new transcripts in seconds:

```bash
xesim re-emit-molecules my_explain_out --out new_emission \
    --stpuppeteer-config tuned_config.yml --seed 1
```

- `--out` must not already exist (the command never overwrites).
- The morphology image, cell boundaries, and gene panel are copied
  verbatim from the source bundle.
- Only `transcripts.parquet` and `ground_truth/molecule_provenance.parquet`
  are re-sampled.
- Same `--diagnostic [DIR]` flag as `explain` — emits the same
  per-cell-type stats JSON.

### Studying selective leakage

To ask "what does the bundle look like if Type X is leaky and the rest
aren't?", flip `leakage_by_celltype` per type and re-emit:

```yaml
leakage_by_celltype:
  Immune: 0.60
  Epithelial: 0.00
  Fibroblast / CAF: 0.00
  Endothelial: 0.00
  # ... rest at 0.0
```

`leak_dist_factor` controls the halo radius (multiple of cell radius).

### Tuning per-cell-type counts

The mean transcript count per cell is roughly
`Σ_g s_c · (μ₀ + W·z_c)_g · scale_factor`. To bump a type up, bump its
program activation or the gene loadings on its primary program. The
`--diagnostic` output shows per-cell-type mean / median / p95 counts so
you can see what the change did. There's no auto-rescale — you tune
the input, not a post-hoc multiplier.

### Diagnostics

Pass `--diagnostic` to either `explain` or `re-emit-molecules`:

```
[emission-diag] 28,152 transcripts across 7 cell types
[emission-diag] cell_type                     cells       tx    mean  median    own%   leak%
[emission-diag] ---------------------------------------------------------------------------
[emission-diag] Ductal/tumor epithelial         289    10498    36.3    32.0   95.6%    0.0%
[emission-diag] Endothelial                      32     1369    42.8    31.5   96.5%    0.0%
[emission-diag] Fibroblast / CAF                162     9615    59.4    46.5   97.7%    0.0%
[emission-diag] Immune                           79     5283    66.9    54.0   97.9%    0.0%
...
```

A JSON version (with cross-cell-type marker-specificity matrix) lands at
`<diagnostic_dir>/emission_stpuppeteer.json`. Diff it between runs to
see the effect of config changes.

## Known limitations

- **Ghosts don't emit.** xeSim's 2D scene composition can add synthetic
  ghost cells for visual realism. Those cells contribute to the
  rendered morphology image but don't have a biological cell type, so
  the STpuppeteer backend skips them at emission time. The 2D output
  will accordingly lack the out-of-plane overlap signal that ghosts
  produce in the legacy backend. 2.5D doesn't use ghosts.

- **Compartments are uniform.** Phase 1 doesn't model per-gene
  nuclear/perinuclear/cytoplasmic/distal preference — every transcript
  is placed uniformly inside its source cell or in the exterior halo
  for leaked transcripts. A future extension on the STpuppeteer side
  (`GeneSpec.compartment_preference`) will let configs specify
  per-gene compartment biology; xeSim's existing compartment EDT
  machinery will consume it.

- **2.5D `transcripts.parquet` hard-codes `overlaps_nucleus=True`.**
  This is a legacy quirk of the 2.5D bundle writer
  (`scene_2_5d/bundle_writer_25d.py:86`). The true per-transcript value
  is computed by the STpuppeteer emitter but discarded by the writer.

- **No ambient background.** Truly cell-free transcripts (rare in real
  Xenium) aren't modelled. A `cfg.ambient_rate_per_um2` STpuppeteer
  extension is queued as a future addition.

## File layout

```
xesim/emission_stpuppeteer/
├── emit.py              # public entry points (emit_2d, emit_3d)
├── config_loader.py     # YAML → SimulationConfig + xeSim validation
├── cell_adapter.py      # xeSim scene cells → STpuppeteer cell_gdf (skips ghosts)
├── counts.py            # calls STpuppeteer's sample_counts + classify_leakage
├── placement_2d.py      # per-cell SDF placement, 2D
├── placement_3d.py      # per-cell SDF placement, 2.5D
├── _placement_core.py   # shared per-cell-bounded-SDF helper
├── bundle_reader.py     # re-emit's bundle loader + polygon rasteriser
├── reemit.py            # re-emit-molecules orchestrator
├── diagnostics.py       # per-cell-type stats + marker-specificity table
└── reference_configs/
    └── pancreas.yml
```

See `tmp/cellAdmix-integration.md` in this repo for the full design rationale.
