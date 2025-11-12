from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class LiberoSceneCfg(SceneCfg):
    """Config class for Kujiale scene 0003."""

    name: str = "libero"
    # /home/priosin/murphy/demos/LIBERO/libero/libero/assets/scenes/libero_tabletop_warm_style.xml
    mjcf_path: str = "/home/priosin/murphy/demos/LIBERO/libero/libero/assets/scenes/libero_coffee_table_warm_style.xml"
    positions: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.000),
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.000)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
