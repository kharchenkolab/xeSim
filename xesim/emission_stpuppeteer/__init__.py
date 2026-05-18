"""STpuppeteer-driven molecular emission backend for xeSim.

Configurable per-cell-type emission using STpuppeteer's LMC count model
and per-cell-independent halo-leakage placement. Selectable via
``xesim explain --emission-backend stpuppeteer --stpuppeteer-config c.yml``.

Phase 0: scaffolding only. The public ``emit_2d`` / ``emit_3d`` entry
points are stubs that raise ``NotImplementedError``. Phase 1 fills in
the counts + leakage + placement modules. See
``tmp/cellAdmix-integration.md`` for the full integration plan.
"""

from .emit import emit_2d, emit_3d

__all__ = ["emit_2d", "emit_3d"]
