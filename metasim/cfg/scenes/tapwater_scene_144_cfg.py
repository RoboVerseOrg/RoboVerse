from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class TapwaterScene144Cfg(SceneCfg):
    """Config class for tapwater scene"""

    name: str = "tapwater_144"
    usd_path: str = "roboverse_data/scenes/arnold/tapwater_scene_144/usd/layout.usd"
    positions: list[tuple[float, float, float]] = [
        (-2.06621, 0.8603, -0.47689),
        (-0.87213, -0.42288, 0.00183),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (-2.06621, 0.8603, -0.47689)
    quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
