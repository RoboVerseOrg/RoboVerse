from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class KujialeScene0024Cfg(SceneCfg):
    """Config class for Kujiale scene 0024."""

    name: str = "kujiale_0024"
    usd_path: str = "third_party/InteriorAgent/kujiale_0024/024.usda"
    positions: list[tuple[float, float, float]] = [
        (1.5, 2.6, 0.000),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (1.5, 2.6, 0.000)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
