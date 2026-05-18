"""Adapter from xeSim scene cells to STpuppeteer's input cell_gdf.

Phase 1 will implement ``build_stpuppeteer_cell_gdf(scene_cells,
scene_label, mode, pixel_size_um, z_step_um=3.0)`` returning a minimal
pandas DataFrame with cell_id, celltype, scale columns — the only
inputs STpuppeteer's count sampler needs.
"""
