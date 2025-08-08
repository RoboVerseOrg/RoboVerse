from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class TapwaterScene152Cfg(SceneCfg):
    """Config class for tapwater scene"""

    name: str = "tapwater_152"
    usd_path: str = "roboverse_data/scenes/arnold/tapwater_scene_152/usd/layout.usd"
    positions: list[tuple[float, float, float]] = [
        (-3.6135, 2.12884, -0.87274),
        (-1.04886, 3.53126, 0.00084),
        (-6.47796, 2.80498, 0.00084),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (-3.6135, 2.12884, -0.87274)
    quat: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
