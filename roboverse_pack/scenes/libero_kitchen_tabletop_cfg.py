from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class LiberoKitchenTabletopCfg(SceneCfg):
    """LIBERO kitchen tabletop scene configuration."""

    name: str = "libero_kitchen_tabletop"
    mjcf_path: str = "roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml"
    positions: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),
    ]
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
