from __future__ import annotations
from typing import Literal

import yaml

from metasim.utils import configclass

@configclass
class BaseTerrainCfg:
    type: str = "base"
    origin: list[float] = [0, 0]  # [row, col] OR [width, length]
    size: list[float] = [1.0, 1.0]  # [width, length] OR [row, col]
    platform_size: float = 1.0


@configclass
class SlopeCfg(BaseTerrainCfg):
    type: str = "slope"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    slope: float = 0.2  # radians
    random: bool = False
    platform_size: float = 1.0


@configclass
class StairCfg(BaseTerrainCfg):
    type: str = "stair"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    step: list[float] = [0.31, 0.05]
    platform_size: float = 1.0  # size of the platform at the top of the stairs when use pyramid_stairs_terrain


@configclass
class ObstacleCfg(BaseTerrainCfg):
    type: str = "obstacle"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    rectangle_params: list[int, float, float] = (1.0, 2.0, 20)  # (min_size, max_size, num_rectangles)
    max_height: float = 0.2  # height of the obstacles in meters
    platform_size: float = 1.0


@configclass
class StoneCfg(BaseTerrainCfg):
    type: str = "stone"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    stone_params: list[float, float] = (0.5, 1.0)
    max_height: float = 0.2  # height of the stones in meters
    platform_size: float = 1.0


@configclass
class GapCfg(BaseTerrainCfg):
    type: str = "gap"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    gap_size: float = 1.0  # size of the gap in meters
    platform_size: float = 1.0


@configclass
class PitCfg(BaseTerrainCfg):
    type: str = "pit"
    position: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    depth: float = 1.0
    platform_size: float = 1.0


@configclass
class GroundCfg:
    width: float = 20.0  # m
    length: float = 20.0  # m
    horizontal_scale: float = 0.1  # m
    vertical_scale: float = 0.005  # m
    margin: float = 10  # m
    elements: dict[str, SlopeCfg | StairCfg | ObstacleCfg | StoneCfg | GapCfg | PitCfg] = None
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

    @classmethod
    def from_yaml(cls, yaml_file: str) -> GroundCfg:
        with open(yaml_file) as f:
            raw_data = yaml.safe_load(f)["terrain"]
        elements = {t: [] for t in ["slope", "stair", "obstacle", "stone", "gap", "pit"]}
        for elem in raw_data["elements"]:
            t = elem["type"]
            class_wrapper = globals().get(f"{t.capitalize()}Cfg")
            if class_wrapper is None:
                raise ValueError(f"Unknown terrain type: {t}")
            elements[t].append(class_wrapper(**elem))

        raw_data["elements"] = elements
        return cls(**raw_data)


if __name__ == "__main__":
    cfg = BaseTerrainCfg.from_yaml("terrain.yaml")
