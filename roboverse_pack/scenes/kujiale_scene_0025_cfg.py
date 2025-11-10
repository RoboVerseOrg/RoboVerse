from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class KujialeScene0025Cfg(SceneCfg):
    """Config class for Kujiale scene 0025."""

    name: str = "kujiale_0025"
    usd_path: str = "third_party/InteriorAgent/kujiale_0025/025.usda"
    positions: list[tuple[float, float, float]] = [
        (2.4, 5.7, 0.000),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (2.4, 5.7, 0.000)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
