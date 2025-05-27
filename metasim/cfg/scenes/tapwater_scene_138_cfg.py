from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class TapwaterScene138Cfg(SceneCfg):
    """Config class for tapwater scene"""

    name: str = "tapwater_138"
    usd_path: str = "roboverse_data/scenes/arnold/tapwater_scene_138/usd/layout.usd"
    positions: list[tuple[float, float, float]] = [
        (1.22698, -4.16989, 0.00006),
        (1.22698, -6.92048, 0.00006),
        (-1.98215, -7.33935, -0.77046),
        (-1.98215, -4.58876, -0.77046),
        (4.55382, -0.66319, -0.3379),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (1.22698, -4.16989, 0.00006)
    quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
