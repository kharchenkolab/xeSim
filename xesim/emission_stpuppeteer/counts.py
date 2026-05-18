"""Counts + leakage decisions via STpuppeteer.

Phase 1 will implement ``emit_decisions(cell_gdf, cfg, gene_names, rng)``
which calls STpuppeteer's ``sample_counts_program_model`` for the count
draw, expands to per-transcript rows, and applies the leakage Bernoulli
via ``classify_leakage`` (a small helper to factor out of STpuppeteer's
``simulate_transcript_locations`` — §6.2 of the integration plan).

Output is a long DataFrame with ``transcript_id, cell_id, gene, is_leaked,
compartment`` columns. Coordinates come later from the placement layer.
"""
