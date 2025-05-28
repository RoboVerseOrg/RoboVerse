from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class TapwaterScene155Cfg(SceneCfg):
    """Config class for tapwater scene"""

    name: str = "tapwater_155"
    usd_path: str = "roboverse_data/scenes/arnold/tapwater_scene_155/usd/layout.usd"
    positions: list[tuple[float, float, float]] = [
        (1.2855, -0.43902, -0.86244),
        (-1.40262, -1.28942, 0.0034),
        (-2.99159, 1.82148, 0.0034),
        (4.09718, -1.00533, 0.0034),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (1.2855, -0.43902, -0.86244)
    quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
