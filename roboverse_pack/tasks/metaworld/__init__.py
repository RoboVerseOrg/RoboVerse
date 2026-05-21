"""Meta-World v3 passthrough — 50 SAWYER manipulation tasks (assembly, pick-place, ...)."""

from __future__ import annotations

from ._passthrough import register_metaworld_passthrough

try:
    register_metaworld_passthrough()
except Exception:
    pass
