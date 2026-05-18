"""2.5D placement via per-cell bounded SDF with anisotropic sampling.

Phase 1 will implement ``place_3d(trs_df, cell_label_3d, nucleus_label_3d,
z_slices_um, tile_origin_um, psz_um, z_step_um, leak_params, rng)`` —
structurally identical to ``place_2d`` with ``sampling=(z_step_um,
psz_um, psz_um)`` and a 3D voxel-to-µm conversion. Both wrappers share
a private ``_place_via_per_cell_sdf`` helper that handles any
dimensionality.

See §4.4 of the integration plan: the 2D and 2.5D code paths differ
only in array shape and the sampling tuple. The anisotropic ``sampling``
makes EDT distances respect z-step (3 µm typical) vs xy pixel size
(0.2125 µm typical).
"""
