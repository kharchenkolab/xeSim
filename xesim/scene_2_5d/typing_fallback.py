"""2.5D adapter for the unified cell-type resolver.

The 2.5D pipeline (``scene_2_5d.scene_first``) works on a `pd.DataFrame`
of observed cells whose ``cell_type`` column is populated from the
templates bank (which in turn reads from the annotation CSV). Cells the
annotation didn't cover come through as the literal string "unknown".

We don't reimplement the cascade here — we just hand the cell ids and
centroids to ``xesim.cell_type_resolver.resolve_from_model`` (the same
function the 2D explain path calls) and stamp the resolved types back
into the DataFrame. The TypeResolution dict is returned so the caller
can attach it to each CellRecord and propagate it to
``ground_truth/cells_synth.parquet`` via the 2.5D writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..cell_type_resolver import (
    TypeResolution, assert_all_resolved, resolve_from_model,
)

if TYPE_CHECKING:
    from ..model import XesimModel


def resolve_observed_cell_types(
    obs: "pd.DataFrame",
    model: "XesimModel",
    bundle_path: str | Path,
    *,
    annotation_path: str | Path | None = None,
    progress: bool = True,
) -> tuple["pd.DataFrame", dict[str, TypeResolution]]:
    """Resolve every observed cell's type via the unified resolver.

    Returns ``(obs, resolutions)``. ``obs`` is the input DataFrame with
    its ``cell_type`` column updated; ``resolutions`` maps every
    observed cell_id to a TypeResolution.
    """
    cell_ids: list[str] = obs["cell_id"].astype(str).tolist()
    # The templates bank may have pre-populated some types from the
    # annotation CSV; forward those as the tier-1 input so we don't
    # need to re-read the file when the resolver already knows them.
    annotation_map: dict[str, str] = {
        c: t for c, t in zip(cell_ids, obs["cell_type"].astype(str))
        if t and t.lower() != "unknown"
    }
    centroids_um = obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    resolutions, _ = resolve_from_model(
        cell_ids, centroids_um, model, bundle_path,
        annotation_map=annotation_map,
        annotation_path=annotation_path,
        progress=progress,
    )
    assert_all_resolved(resolutions, cell_ids)

    obs = obs.copy()
    obs["cell_type"] = [resolutions[c].cell_type for c in cell_ids]
    return obs, resolutions


# Back-compat alias.
def retype_unknown_observed_cells(
    obs: "pd.DataFrame", model: "XesimModel", bundle_path,
    *, progress: bool = True,
) -> "pd.DataFrame":
    obs, _ = resolve_observed_cell_types(obs, model, bundle_path,
                                            progress=progress)
    return obs


__all__ = ["resolve_observed_cell_types", "retype_unknown_observed_cells"]
