from __future__ import annotations

from typing import Literal

import yaml

from metasim.utils import configclass


@configclass
class TerrainCfg:
    mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale = 0.1  # [m]
    vertical_scale = 0.005  # [m]
    border_size = 25  # [m]
    curriculum = True
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.0
    # rough terrain only:
    measure_heights = True
    include_act_obs_pair_buf = False
    # 1mx1.6m rectangle (without center line)
    measured_points_x = [
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
    # measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]

    selected = False  # select a unique terrain type and pass all arguments
    terrain_kwargs = None  # Dict of arguments for selected terrain
    max_init_terrain_level = 5  # starting curriculum state
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 10  # number of terrain rows (levels)
    num_cols = 20  # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    # terrain_proportions = [0.1, 0.2, 0.30, 0.30, 0.1]
    # trimesh only:
    slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces


@configclass
class BaseTerrainConfig:
    type: str = "base"
    origin: list[float] = [0, 0]  # [row, col] OR [width, length]
    size: list[float] = [1.0, 1.0]  # [width, length] OR [row, col]
    platform_size: float = 1.0


@configclass
class SlopeConfig(BaseTerrainConfig):
    type: str = "slope"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    slope: float = 0.2  # radians
    random: bool = False
    platform_size: float = 1.0


@configclass
class StairConfig(BaseTerrainConfig):
    type: str = "stair"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    step: list[float] = [0.31, 0.05]
    platform_size: float = 1.0  # size of the platform at the top of the stairs when use pyramid_stairs_terrain


@configclass
class ObstacleConfig(BaseTerrainConfig):
    type: str = "obstacle"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    rectangle_params: list[int, float, float] = (1.0, 2.0, 20)  # (min_size, max_size, num_rectangles)
    max_height: float = 0.2  # height of the obstacles in meters
    platform_size: float = 1.0


@configclass
class StoneConfig(BaseTerrainConfig):
    type: str = "stone"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    stone_params: list[float, float] = (0.5, 1.0)
    max_height: float = 0.2  # height of the stones in meters
    platform_size: float = 1.0


@configclass
class GapConfig(BaseTerrainConfig):
    type: str = "gap"
    origin: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    gap_size: float = 1.0  # size of the gap in meters
    platform_size: float = 1.0


@configclass
class PitConfig(BaseTerrainConfig):
    type: str = "pit"
    position: list[float] = [0, 0]
    size: list[float] = [1.0, 1.0]
    depth: float = 1.0
    platform_size: float = 1.0


@configclass
class TerrainConfig:
    width: float = 20.0  # m
    length: float = 20.0  # m
    horizontal_scale: float = 0.1  # m
    vertical_scale: float = 0.005  # m
    margin: float = 10  # m
    elements: dict[str, SlopeConfig | StairConfig | ObstacleConfig | StoneConfig | GapConfig | PitConfig] = None
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
    def from_yaml(cls, yaml_file: str) -> TerrainConfig:
        with open(yaml_file) as f:
            raw_data = yaml.safe_load(f)["terrain"]
        elements = {t: [] for t in ["slope", "stair", "obstacle", "stone", "gap", "pit"]}
        for elem in raw_data["elements"]:
            t = elem["type"]
            class_wrapper = globals().get(f"{t.capitalize()}Config")
            if class_wrapper is None:
                raise ValueError(f"Unknown terrain type: {t}")
            elements[t].append(class_wrapper(**elem))

        raw_data["elements"] = elements
        return cls(**raw_data)


if __name__ == "__main__":
    cfg = TerrainConfig.from_yaml("terrain.yaml")
