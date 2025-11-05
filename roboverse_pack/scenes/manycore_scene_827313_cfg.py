from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class ManycoreScene827313Cfg(SceneCfg):
    """Config class for manycore scene."""

    name: str = "manycore_827313"
    usd_path: str = "roboverse_data/scenes/manycore/827313/start.usda"
    positions: list[tuple[float, float, float]] = [
        (-0.28508, -0.95154, -0.00891),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (-0.28508, -0.95154, -0.00891)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
