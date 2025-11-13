from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class KujialeScene0008Cfg(SceneCfg):
    """Config class for Kujiale scene 0003."""

    name: str = "kujiale_0008"
    usd_path: str = "third_party/InteriorAgent/kujiale_0008/008.usda"
    positions: list[tuple[float, float, float]] = [
        (-7.2, -1.5, 0.000),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (-7.2, -1.5, 0.000)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
