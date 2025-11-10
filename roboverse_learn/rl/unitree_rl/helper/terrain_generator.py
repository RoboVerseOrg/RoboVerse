from __future__ import annotations

import math
import numpy as np

from roboverse_learn.rl.unitree_rl.helper import terrain_utils
from roboverse_learn.rl.unitree_rl.configs.cfg_terrain import *

class TerrainGenerator:
    """Abstract base class for backend-specific terrain implementation."""

    def __init__(self, config: GroundCfg = None):
        if config is not None:
            self._parse_cfg(config)

    def _parse_cfg(self, config: GroundCfg):
        """Parse the terrain configuration."""
        self.config = config
        self.height_mat = np.zeros((config.num_rows, config.num_cols), dtype=np.int16)
        self.horizontal_scale = config.horizontal_scale
        self.vertical_scale = config.vertical_scale
        self.margin = config.margin

    def _make_sub_terrain(self, config: BaseTerrainCfg):
        terrain = terrain_utils.SubTerrain(
            config.type,
            width=math.ceil(config.size[0] / self.horizontal_scale),
            length=math.ceil(config.size[1] / self.horizontal_scale),
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )
        return terrain

    def _make_slope(self, config: SlopeCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pyramid_sloped_terrain(
            terrain,
            slope=config.slope * difficulty,
            platform_size=config.platform_size,
        )
        if config.random:
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=2.0 * self.horizontal_scale
            )
        return config.origin, terrain

    def _make_stair(self, config: StairCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pyramid_stairs_terrain(
            terrain,
            step_width=config.step[0],
            step_height=config.step[1] * difficulty,
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_obstacle(self, config: ObstacleCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.discrete_obstacles_terrain(
            terrain,
            max_height=config.max_height * difficulty,
            min_size=config.rectangle_params[0],
            max_size=config.rectangle_params[1],
            num_rects=config.rectangle_params[2],
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_stone(self, config: StoneCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.stepping_stones_terrain(
            terrain,
            stone_size=config.stone_params[0] / np.log(1 + difficulty),
            stone_distance=config.stone_params[1],
            max_height=config.max_height,
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_gap(self, config: GapCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.gap_terrain(terrain, gap_size=config.gap_size * difficulty, platform_size=config.platform_size)
        return config.origin, terrain

    def _make_pit(self, config: PitCfg, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pit_terrain(terrain, depth=config.depth * difficulty, platform_size=config.platform_size)
        return config.origin, terrain

    def _add_terrain_to_map(self, origin, terrain: terrain_utils.SubTerrain, matrix: np.ndarray = None):
        start_row = math.floor(origin[0] / self.horizontal_scale)
        start_col = math.floor(origin[1] / self.horizontal_scale)
        end_row = start_row + terrain.width
        end_col = start_col + terrain.length
        matrix[start_row:end_row, start_col:end_col] = terrain.height_field_raw
        return matrix

    def _repeat_terrain(
        self, repeat: int = 0, direction: str = "column", gap: float = 0.0, difficulty_list: list[float] | None = None
    ):
        """Repeat the terrain in the specified direction."""
        assert len(difficulty_list) == repeat, "Length of difficulty_list must match the number of repeats."
        if direction == "column":
            padding = np.zeros((self.config.num_rows, int(gap / self.horizontal_scale)), dtype=np.int16)
            for i in range(0, repeat):
                mat = self.generate_matrix(difficulty_list[i])
                extend_mat = np.concatenate((padding, mat), axis=1)
                self.height_mat = np.concatenate((self.height_mat, extend_mat), axis=1)
        elif direction == "row":
            padding = np.zeros((int(gap / self.horizontal_scale), self.config.num_cols), dtype=np.int16)
            extend_mat = np.concatenate((self.height_mat, padding), axis=0)
            for i in range(0, repeat):
                mat = self.generate_matrix(difficulty_list[i])
                extend_mat = np.concatenate((padding, mat), axis=0)
                self.height_mat = np.concatenate((self.height_mat, extend_mat), axis=0)
        else:
            raise ValueError("Direction must be either 'column' or 'row'.")
        self.config.num_rows = self.height_mat.shape[0]
        self.config.num_cols = self.height_mat.shape[1]
        return self.height_mat

    def generate_matrix(self, difficulty: float = 1.0) -> np.ndarray:
        matrix = np.zeros((self.config.num_rows, self.config.num_cols), dtype=np.int16)
        for t in self.config.elements.keys():
            func_name = f"_make_{t}"
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                for cfg in self.config.elements[t]:
                    origin, terrain = func(cfg, difficulty)
                    self._add_terrain_to_map(origin, terrain, matrix)
            else:
                raise NotImplementedError(f"Terrain type '{t}' is not implemented in {self.__class__.__name__}")
        return matrix

    def generate_terrain(self, config: TerrainConfig = None, type: str = "trimesh"):
        """Generate terrain based on the specified type and parameters."""
        if config is not None:
            self._parse_cfg(config)

        assert hasattr(self, "config"), "Terrain configuration must be set before generating terrain."
        difficulty_list = (
            np.linspace(
                self.config.difficulty[0], self.config.difficulty[1], num=self.config.repeat_direction_gap[0] + 1
            ).tolist()
            if self.config.difficulty[2] == "linear"
            else [self.config.difficulty[0]] * (self.config.repeat_direction_gap[0] + 1)
        )
        self.height_mat = self.generate_matrix(difficulty_list.pop(0))
        self.height_mat = self._repeat_terrain(*self.config.repeat_direction_gap, difficulty_list)
        row_padding_size = self.config.margin_num_rows
        col_padding_size = self.config.margin_num_cols
        self.height_mat_pad = np.pad(
            self.height_mat,
            ((row_padding_size, row_padding_size), (col_padding_size, col_padding_size)),
            mode="constant",
            constant_values=0,
        )
        if type == "trimesh":
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                height_field_raw=self.height_mat_pad,
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
                slope_threshold=0.1,
            )

            return vertices, triangles
        elif type == "heightfield":
            return self.height_mat_pad * self.vertical_scale

    @property
    def height_measure(self):
        """Get the height map of the generated terrain."""
        return self.height_mat * self.vertical_scale

    @property
    def height_measure_pad(self):
        """Get the padded height map of the generated terrain."""
        return self.height_mat_pad * self.vertical_scale
