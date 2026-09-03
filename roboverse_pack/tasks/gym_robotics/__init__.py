"""gymnasium-robotics passthrough — Fetch + Adroit Hand + Shadow Hand + Ant-Maze etc."""

from __future__ import annotations

from loguru import logger as log

from ._passthrough import register_gymnasium_robotics_passthrough

try:
    register_gymnasium_robotics_passthrough()
except ImportError as exc:  # optional benchmark package absent: expected, keep quiet
    log.debug(f"[gym_robotics] passthrough registration skipped: {exc}")
except Exception as exc:  # anything else is a real bug and must not be hidden
    log.warning(f"[gym_robotics] passthrough registration failed: {exc!r}")
