"""gymnasium-robotics passthrough — Fetch + Adroit Hand + Shadow Hand + Ant-Maze etc."""

from __future__ import annotations

from ._passthrough import register_gymnasium_robotics_passthrough

try:
    register_gymnasium_robotics_passthrough()
except Exception:
    pass
