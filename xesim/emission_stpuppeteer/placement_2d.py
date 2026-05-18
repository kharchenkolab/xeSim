"""2D placement via per-cell bounded SDF.

Phase 1 will implement ``place_2d(trs_df, cell_label, nucleus_label,
sampling, leak_params, rng)`` — a thin wrapper around the shared
``_place_via_per_cell_sdf`` helper with ``sampling=(psz_um, psz_um)``
and a 2D voxel-to-µm conversion.

See §4 of the integration plan for the algorithm: per-cell EDT on a
bounded bbox around each cell, with leaked transcripts sampled by
``exp(-d/λ)`` weights from owner-attributed voxels. Densities add
across cells; leaked transcripts can land inside neighbor cells'
volumes (``landed_in_cell_id`` recorded in provenance).
"""
