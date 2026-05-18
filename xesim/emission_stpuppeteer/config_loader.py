"""YAML → SimulationConfig loader, with xeSim-side validation hooks.

Phase 1 will implement ``load_stpuppeteer_config(path, bundle_annotation,
gene_panel)`` as a thin adapter over STpuppeteer's
``SimulationConfig.from_yaml`` (§6.6 of the integration plan), adding
xeSim-specific validation: cell-type name matching against the bundle
annotation, gene-name matching against the bundle's gene panel.
"""
