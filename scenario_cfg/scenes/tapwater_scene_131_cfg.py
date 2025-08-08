from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class TapwaterScene131Cfg(SceneCfg):
    """Config class for tapwater scene"""

    name: str = "tapwater_131"
    usd_path: str = "roboverse_data/scenes/arnold/tapwater_scene_131/usd/layout.usd"
    positions: list[tuple[float, float, float]] = [
        (0.62354, -1.77833, -0.73064),
        (-3.8, -3.1, -0.5),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (0.62354, -1.77833, -0.73064)
    quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
