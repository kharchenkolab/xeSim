"""Multi-tier cell typing fallback for the 2.5D pipeline.

The templates bank only knows cell types from the bundle's annotation
file; cells not in annotation fall through as 'unknown' (~10% on
pancreas v21). This module re-types those cells using the same
chain the 2D explain path uses:

  tier 2: cellAdmix transcript classifier (model.cell_factor_fractions)
  tier 3: training cid_to_type from model
  tier 4: stain-based encoder-latent kNN (only if cell_latent_bank.npz exists)

(Tier 1 = the annotation file = the bank's source, so already applied.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..model import XesimModel


def retype_unknown_observed_cells(
    obs: "pd.DataFrame",
    model: "XesimModel",
    bundle_path: str | Path,
    *,
    progress: bool = True,
) -> "pd.DataFrame":
    """Apply tiers 2-4 of cell typing to rows where `cell_type == "unknown"`.

    Mutates `obs` in place (also returns it for chaining). Requires
    `obs` to have columns: cell_id, cell_type, centroid_x, centroid_y.
    """
    n_unk_pre = int((obs['cell_type'] == 'unknown').sum())
    if n_unk_pre == 0:
        return obs

    # Tier 2: cellAdmix transcript classifier
    try:
        if hasattr(model, 'cell_factor_fractions'):
            cf_full = model.cell_factor_fractions(
                only_biology=True, include_annotation=True)
            type_col = ('predicted_type' if 'predicted_type' in cf_full.columns
                        else 'annotation' if 'annotation' in cf_full.columns
                        else None)
            if type_col is not None:
                tx_class = dict(zip(
                    cf_full['cell_id'].astype(str), cf_full[type_col]))
                mask = obs['cell_type'] == 'unknown'
                obs.loc[mask, 'cell_type'] = obs.loc[mask, 'cell_id'].astype(
                    str).map(lambda c: tx_class.get(c, 'unknown'))
    except Exception:
        pass

    # Tier 3: model.cid_to_type
    if hasattr(model, 'cid_to_type'):
        cid_map = {c: model.type_names[i]
                   for c, i in model.cid_to_type.items()
                   if 0 <= i < len(model.type_names)}
        mask = obs['cell_type'] == 'unknown'
        obs.loc[mask, 'cell_type'] = obs.loc[mask, 'cell_id'].astype(
            str).map(lambda c: cid_map.get(c, 'unknown'))

    # Tier 4: stain-classifier kNN
    bank_path = (Path(model.paths.root) / 'cell_latent_bank.npz'
                 if hasattr(model, 'paths') else None)
    if bank_path is not None and bank_path.exists():
        still_unk = obs[obs['cell_type'] == 'unknown']
        if len(still_unk) > 0:
            try:
                from ..stain_classifier import (
                    classify_cells_by_centroid, load_bank)
                sb = load_bank(bank_path)
                triples = [(str(r.cell_id), float(r.centroid_x),
                            float(r.centroid_y))
                           for r in still_unk.itertuples()]
                preds = classify_cells_by_centroid(
                    model, bundle_path, triples, bank=sb)
                obs.loc[obs['cell_type'] == 'unknown', 'cell_type'] = (
                    obs.loc[obs['cell_type'] == 'unknown', 'cell_id']
                    .astype(str).map(lambda c: preds.get(c, 'unknown')))
            except Exception as e:
                if progress:
                    print(f"[retype_25d] stain-classifier skipped: {e}")

    if progress:
        n_unk_post = int((obs['cell_type'] == 'unknown').sum())
        print(f"[retype_25d] re-typed: {n_unk_pre} → {n_unk_post} "
              f"unknown observed cells")
    return obs


__all__ = ['retype_unknown_observed_cells']
