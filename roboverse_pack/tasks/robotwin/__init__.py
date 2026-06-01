"""RoboTwin integration — passthrough registration of native RoboTwin v2 tasks.

Mirrors the ManiSkill passthrough: registers ``RoboTwin/<name>`` ids that lazily
run native RoboTwin tasks (1:1 by construction). Registration is import-safe even
when RoboTwin's deps are not installed; see :mod:`._passthrough`.
"""

from __future__ import annotations

try:
    from roboverse_pack.tasks.robotwin._passthrough import register_robotwin_passthrough

    register_robotwin_passthrough()
except Exception:
    pass
