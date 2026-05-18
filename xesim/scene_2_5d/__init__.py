"""2.5D scene generation: Tilted-Template SDF approach.

See `misc/2.5D.md` for the full design. Key public API:

  z_attrs.load_or_fit_cells_z(bundle_path)
      Returns per-cell (z_center_um, z_extent_um, z_confidence).
      Persisted at <bundle_parent>/_xesim_z_attrs/cells_z.parquet.

  templates.build_template_bank(bundle_path, ...)
      Returns a per-type-and-zone bank of real Xenium contours.

  seeds.sample_unobserved_seeds(region, ...)
      Returns per-z-band synthetic cell seeds (xy, type, z_center, z_extent).

  tilt.sample_tilts(cells, ...)
      Per-cell tilt vector via MRF Gibbs.

  sdf_tess.assign_cell_label(scene, z, ...)
      Per-z 2D SDF assignment → cell_label.

  compose.compose_region_scene_25d(model, bundle, region, ...)
      Top-level driver.
"""
__all__ = [
    "z_attrs",
    "templates",
    "seeds",
    "tilt",
    "sdf_tess",
    "bodies_3d",
    "render_multi_z",
    "compose",
    "nucleus_prior",
    "section_prior",
    "fit_priors",
]
