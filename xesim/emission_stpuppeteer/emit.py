"""Public entry points for the STpuppeteer emission backend.

Phase 0: stubs. ``emit_2d`` and ``emit_3d`` accept any args/kwargs and
raise ``NotImplementedError`` with a pointer to the integration plan.
Phase 1 fills in the real implementations.
"""

from __future__ import annotations

import pandas as pd


_PHASE0_MSG = (
    "STpuppeteer emission backend is scaffolded but not yet implemented "
    "(Phase 0 ships the wiring only). See tmp/cellAdmix-integration.md "
    "for the implementation plan. Use --emission-backend legacy until "
    "Phase 1 lands."
)


def emit_2d(*args, **kwargs) -> pd.DataFrame:
    """2D STpuppeteer emission. Stub — not implemented until Phase 1."""
    raise NotImplementedError(_PHASE0_MSG)


def emit_3d(*args, **kwargs) -> pd.DataFrame:
    """2.5D STpuppeteer emission. Stub — not implemented until Phase 1."""
    raise NotImplementedError(_PHASE0_MSG)
