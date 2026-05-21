"""dm_control suite passthrough — 51 classic control + walker + swimmer + humanoid tasks."""

from __future__ import annotations

from ._passthrough import register_dm_control_passthrough

try:
    register_dm_control_passthrough()
except Exception:
    pass
