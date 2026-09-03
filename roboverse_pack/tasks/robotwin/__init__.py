"""RoboTwin integration — passthrough registration of native RoboTwin v2 tasks.

Mirrors the ManiSkill passthrough: registers ``RoboTwin/<name>`` ids that lazily
run native RoboTwin tasks (1:1 by construction). Registration is import-safe even
when RoboTwin's deps are not installed; see :mod:`._passthrough`.
"""

from __future__ import annotations

from loguru import logger as log

from roboverse_pack.tasks.robotwin._passthrough import list_robotwin_tasks, register_robotwin_passthrough

__all__ = ["list_robotwin_tasks", "register_robotwin_passthrough"]

try:
    register_robotwin_passthrough()
except ImportError as exc:  # optional benchmark package absent: expected, keep quiet
    log.debug(f"[robotwin] passthrough registration skipped: {exc}")
except Exception as exc:  # anything else is a real bug and must not be hidden
    log.warning(f"[robotwin] passthrough registration failed: {exc!r}")
