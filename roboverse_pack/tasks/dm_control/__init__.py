"""dm_control suite passthrough — 51 classic control + walker + swimmer + humanoid tasks."""

from __future__ import annotations

from loguru import logger as log

from ._passthrough import register_dm_control_passthrough

try:
    register_dm_control_passthrough()
except ImportError as exc:  # optional benchmark package absent: expected, keep quiet
    log.debug(f"[dm_control] passthrough registration skipped: {exc}")
except Exception as exc:  # anything else is a real bug and must not be hidden
    log.warning(f"[dm_control] passthrough registration failed: {exc!r}")
