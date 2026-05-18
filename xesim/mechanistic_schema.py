"""Schema constants for Plan3 mechanistic xeSim artifacts."""

MECHANISTIC_SCENE_TYPE = "xesim.mechanistic_scene.v0"
MECHANISTIC_PARAMS_TYPE = "xesim.mechanistic_params.v0"
MECHANISTIC_RENDER_TYPE = "xesim.mechanistic_render.v0"
MECHANISTIC_RENDER_MANIFEST_TYPE = "xesim.mechanistic_render_manifest.v0"
MECHANISTIC_REPORT_TYPE = "xesim.mechanistic_report.v0"
MECHANISTIC_REFINER_CORPUS_TYPE = "xesim.mechanistic_refiner_corpus.v0"
MECHANISTIC_REFINER_TRAINING_TYPE = "xesim.mechanistic_refiner_training.v0"
MECHANISTIC_REFINER_SAMPLES_TYPE = "xesim.mechanistic_refiner_samples.v0"
MECHANISTIC_REFINER_EVAL_TYPE = "xesim.mechanistic_refiner_eval.v0"

MECHANISTIC_CELL_SOURCES = (
    "observed_anchor",
    "observed_low_quality",
    "mined_partial",
    "inferred_latent",     # image-evidence proposer (DAPI peaks, membrane ridges)
    "tx_inferred",          # orphan-transcript-cluster proposer (no image evidence;
                            #   not rendered for stains, only emits molecules)
    "synthetic",
)

MECHANISTIC_SECTIONING_STATES = (
    "full",
    "nucleus_rich",
    "nucleus_poor",
    "membrane_only",
    "sliver",
    "unknown",
    # Plan3 v34: explicit out-of-plane sectioning states. The cell's
    # apex (above) or base (below) lies outside the section slab;
    # in-plane footprint reflects the basal or apical projection.
    "apex_above",
    "base_below",
    "fragment_top",
    "fragment_bottom",
)
