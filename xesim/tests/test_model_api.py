"""End-to-end tests for the public XesimModel API.

These tests are skipped unless the dev fixture ``tmp/xesim_v16_model``
exists. To enable them, repackage the existing artifacts into a model
directory once::

    # see misc/cli_plan.md for the layout
    # canonical/, priors/, exemplars/, renderer.pt, manifest.json
"""

from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path("/workspace/xeSim/tmp/xesim_v16_model")

requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "manifest.json").exists(),
    reason=f"no fitted model at {MODEL_DIR}",
)


@requires_model
def test_load():
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    assert m.latent_dim > 0
    assert len(m.type_names) > 1
    assert m.tissue is not None
    assert m.mechanistic_params is not None


@requires_model
def test_explain_smoke():
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    items = m.explain(MODEL_DIR / "canonical", num_crops=1)
    assert len(items) == 1
    item = items[0]
    assert item["scene"] is not None
    assert item["render"].shape == (3, 256, 256)
    assert 0.0 <= float(item["render"].min()) <= float(item["render"].max()) <= 1.0


@requires_model
def test_generate_forward_smoke():
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    items = m.generate(num_scenes=2, seed=42)
    assert len(items) == 2
    for item in items:
        assert item["scene"] is not None
        assert item["render"].shape == (3, 256, 256)


@requires_model
def test_generate_slice_guided_smoke():
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    items = m.generate(num_scenes=1,
                          guide=(MODEL_DIR / "canonical", "crop_00011"),
                          seed=7)
    assert len(items) == 1
    assert items[0]["guide_crop_id"] == "crop_00011"


@requires_model
def test_render_determinism():
    """Same seed → same render."""
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    items1 = m.generate(num_scenes=1, seed=99)
    items2 = m.generate(num_scenes=1, seed=99)
    # MechanisticScene is the same → renders should be near-identical
    # (small drift from non-deterministic conv kernels possible).
    diff = np.abs(items1[0]["render"] - items2[0]["render"]).mean()
    assert diff < 0.05, f"render-vs-same-seed L1 too high: {diff}"


@requires_model
def test_write_layout(tmp_path):
    from xesim import XesimModel
    m = XesimModel.load(MODEL_DIR)
    items = m.generate(num_scenes=1, seed=1)
    d = m.write(items[0], tmp_path, name="t")
    assert (d / "scene.json").exists()
    assert (d / "render.png").exists()
    assert (d / "cell_label.npy").exists()
    assert (d / "nucleus_label.npy").exists()
