from __future__ import annotations

from typing import Literal

from metasim.utils import configclass


@configclass
class GroundCfg:
    """Global ground description used to assemble terrain tiles."""

    width: float = 20.0  # m
    length: float = 20.0  # m
    horizontal_scale: float = 0.1  # m
    vertical_scale: float = 0.1  # m
    margin: float = 10  # m
    max_mesh_triangles: int = 2000  # cap IsaacSim mesh complexity to avoid GPU OOM
    elements: dict[str, list[BaseTerrainCfg]] = None
    repeat_direction_gap: list[int, Literal["row", "column"], float] = (0, "row", 0.1)  # (repeat, repeat_direction)
    difficulty: list[float, float, Literal["linear"]] = [1.0, 4.0, "linear"]  # (difficulty, type)
    # For Isaacgym
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 1.0

    def __post_init__(self):
        self.num_rows: int = int(self.width / self.horizontal_scale)
        self.margin_num_rows: int = int(self.margin / self.horizontal_scale)
        self.num_cols: int = int(self.length / self.horizontal_scale)
        self.margin_num_cols: int = int(self.margin / self.horizontal_scale)
        if self.elements is None:
            self.elements = {
                "slope": [],
                "stair": [],
                "obstacle": [],
                "stone": [],
                "gap": [],
                "pit": [],
            }


@configclass
class BaseTerrainCfg:
    """Base parameters shared by all terrain primitives."""

    type: str = "base"
    origin: list[float] = [0, 0]  # [row, col] OR [width, length]
    size: list[float] = [1.0, 1.0]  # [width, length] OR [row, col]
    platform_size: float = 1.0


@configclass
class SlopeCfg(BaseTerrainCfg):
    """Config for a planar slope feature."""

    type: str = "slope"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    slope: float = 0.2  # radians
    random: bool = False
    platform_size: float = 1.0


@configclass
class StairCfg(BaseTerrainCfg):
    """Config for staircase features."""

    type: str = "stair"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    step: list[float] = [0.31, 0.05]
    platform_size: float = 1.0  # size of the platform at the top of the stairs when use pyramid_stairs_terrain


@configclass
class ObstacleCfg(BaseTerrainCfg):
    """Config for obstacle fields composed of random rectangles."""

    type: str = "obstacle"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    rectangle_params: list[int, float, float] = (1.0, 2.0, 20)  # (min_size, max_size, num_rectangles)
    max_height: float = 0.2  # height of the obstacles in meters
    platform_size: float = 1.0


@configclass
class StoneCfg(BaseTerrainCfg):
    """Config for stone-like protrusions."""

    type: str = "stone"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    stone_params: list[float, float] = (0.5, 1.0)
    max_height: float = 0.2  # height of the stones in meters
    platform_size: float = 1.0


@configclass
class GapCfg(BaseTerrainCfg):
    """Config for gaps that robots must traverse."""

    type: str = "gap"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    gap_size: float = 1.0  # size of the gap in meters
    platform_size: float = 1.0


@configclass
class PitCfg(BaseTerrainCfg):
    """Config for rectangular pits."""

    type: str = "pit"
    position: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    depth: float = 1.0
    platform_size: float = 1.0
