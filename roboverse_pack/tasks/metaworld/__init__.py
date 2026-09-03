"""Meta-World v3 passthrough — 50 SAWYER manipulation tasks (assembly, pick-place, ...)."""

from __future__ import annotations

from loguru import logger as log

from ._passthrough import register_metaworld_passthrough

try:
    register_metaworld_passthrough()
except ImportError as exc:  # optional benchmark package absent: expected, keep quiet
    log.debug(f"[metaworld] passthrough registration skipped: {exc}")
except Exception as exc:  # anything else is a real bug and must not be hidden
    log.warning(f"[metaworld] passthrough registration failed: {exc!r}")
