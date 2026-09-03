from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _discretize(value: float, scale: float) -> int:
    """Project a metric value into height-field units."""
    return int(value / scale)


def _discretize_many(values: Sequence[float], scale: float) -> tuple[int, ...]:
    return tuple(_discretize(v, scale) for v in values)


def _downsample_shape(terrain: SubTerrain, spacing: float) -> tuple[int, int]:
    width = max(1, int(terrain.width * terrain.horizontal_scale / spacing))
    length = max(1, int(terrain.length * terrain.horizontal_scale / spacing))
    return width, length


def _center_slice(size: int, span: int) -> slice:
    span = min(span, size)
    offset = max((size - span) // 2, 0)
    return slice(offset, offset + span)


def random_uniform_terrain(
    terrain,
    min_height,
    max_height,
    step=1,
    downsampled_scale=None,
):
    """Fill the terrain with uniformly sampled noise and bilinear upsampling."""
    downsampled_scale = downsampled_scale or terrain.horizontal_scale
    min_h, max_h, step = _discretize_many((min_height, max_height, step), terrain.vertical_scale)
    heights = np.arange(min_h, max_h + step, step, dtype=np.int32)

    coarse_shape = _downsample_shape(terrain, downsampled_scale)
    coarse_height = np.random.choice(heights, size=coarse_shape)

    x_coarse = np.linspace(0.0, terrain.width * terrain.horizontal_scale, coarse_shape[0])
    y_coarse = np.linspace(0.0, terrain.length * terrain.horizontal_scale, coarse_shape[1])
    interpolator = RegularGridInterpolator((x_coarse, y_coarse), coarse_height, method="linear")

    x_fine = np.linspace(0.0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_fine = np.linspace(0.0, terrain.length * terrain.horizontal_scale, terrain.length)
    grid_x, grid_y = np.meshgrid(x_fine, y_fine, indexing="ij")
    samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    upsampled = np.rint(interpolator(samples)).reshape(terrain.width, terrain.length)

    terrain.height_field_raw += upsampled.astype(terrain.height_field_raw.dtype)
    return terrain


def sloped_terrain(terrain, slope=1):
    """Apply a constant slope along the x-axis."""
    gradient = np.arange(terrain.width, dtype=np.float32).reshape(terrain.width, 1)
    max_height = slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width
    scaled = (gradient * max_height / terrain.width).astype(terrain.height_field_raw.dtype)
    terrain.height_field_raw += scaled
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    """Create a pyramid-like profile with an optional plateau."""
    center_x = max(terrain.width // 2, 1)
    center_y = max(terrain.length // 2, 1)
    x_profile = 1.0 - np.abs(np.arange(terrain.width) - center_x) / center_x
    y_profile = 1.0 - np.abs(np.arange(terrain.length) - center_y) / center_y
    x_profile = np.clip(x_profile, 0.0, 1.0).reshape(terrain.width, 1)
    y_profile = np.clip(y_profile, 0.0, 1.0).reshape(1, terrain.length)

    max_height = slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2)
    terrain.height_field_raw += (max_height * x_profile * y_profile).astype(terrain.height_field_raw.dtype)

    platform_radius = _discretize(platform_size / 2.0, terrain.horizontal_scale)
    x1 = terrain.width // 2 - platform_radius
    x2 = terrain.width // 2 + platform_radius
    y1 = terrain.length // 2 - platform_radius
    y2 = terrain.length // 2 + platform_radius
    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.0):
    """Add randomly sized rectangular blocks or pits across the terrain."""
    max_height = _discretize(max_height, terrain.vertical_scale)
    min_size, max_size, platform_cells = _discretize_many((min_size, max_size, platform_size), terrain.horizontal_scale)

    height_range = np.array([-max_height, -max_height // 2, max_height // 2, max_height], dtype=np.int32)
    width_options = np.arange(min_size, max_size, 4, dtype=int)
    length_options = np.arange(min_size, max_size, 4, dtype=int)

    i_max, j_max = terrain.height_field_raw.shape
    for _ in range(num_rects):
        if len(width_options) == 0 or len(length_options) == 0:
            break
        width = int(np.random.choice(width_options))
        length = int(np.random.choice(length_options))
        if width >= i_max or length >= j_max:
            continue
        valid_i = np.arange(0, i_max - width, 4, dtype=int)
        valid_j = np.arange(0, j_max - length, 4, dtype=int)
        if len(valid_i) == 0 or len(valid_j) == 0:
            continue
        start_i = int(np.random.choice(valid_i))
        start_j = int(np.random.choice(valid_j))
        terrain.height_field_raw[start_i : start_i + width, start_j : start_j + length] = np.random.choice(height_range)

    platform_x = _center_slice(terrain.width, platform_cells)
    platform_y = _center_slice(terrain.length, platform_cells)
    terrain.height_field_raw[platform_x, platform_y] = 0
    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1.0):
    """Compose sinusoidal waves along both axes."""
    amplitude = _discretize(0.5 * amplitude, terrain.vertical_scale)
    if num_waves <= 0:
        return terrain

    freq = num_waves * np.pi * 2.0 / terrain.length
    x = np.arange(terrain.width, dtype=np.float32).reshape(terrain.width, 1)
    y = np.arange(terrain.length, dtype=np.float32).reshape(1, terrain.length)
    undulation = amplitude * (np.cos(y * freq) + np.sin(x * freq))
    terrain.height_field_raw += undulation.astype(terrain.height_field_raw.dtype)
    return terrain


def stairs_terrain(terrain, step_width, step_height):
    """Generate a staircase that increases height in the x direction."""
    step_width = _discretize(step_width, terrain.horizontal_scale)
    step_height = _discretize(step_height, terrain.vertical_scale)
    if step_width <= 0 or step_height == 0:
        return terrain

    num_steps = terrain.width // step_width
    if num_steps <= 0:
        return terrain

    heights = np.arange(1, num_steps + 1, dtype=np.int32) * step_height
    profile = np.repeat(heights, step_width)
    rows = min(profile.size, terrain.width)
    terrain.height_field_raw[:rows, :] += profile[:rows].reshape(rows, 1).astype(terrain.height_field_raw.dtype)
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """Inset the terrain step by step while increasing height."""
    step_width = _discretize(step_width, terrain.horizontal_scale)
    step_height = _discretize(step_height, terrain.vertical_scale)
    platform_cells = _discretize(platform_size, terrain.horizontal_scale)
    if step_width <= 0:
        return terrain

    x_min, x_max = 0, terrain.width
    y_min, y_max = 0, terrain.length
    height = 0
    while (x_max - x_min) > platform_cells and (y_max - y_min) > platform_cells:
        x_min += step_width
        y_min += step_width
        x_max -= step_width
        y_max -= step_width
        if x_min >= x_max or y_min >= y_max:
            break
        height += step_height
        terrain.height_field_raw[x_min:x_max, y_min:y_max] = height
    return terrain


def _fill_stepping_region(field: np.ndarray, stone_size: int, stone_distance: int, height_choices: np.ndarray):
    """Populate a 2D array with stepping stones along the second axis."""
    primary = 0
    while primary < field.shape[1]:
        primary_stop = min(field.shape[1], primary + stone_size)
        offset = np.random.randint(0, max(stone_size, 1)) if stone_size > 0 else 0
        first_stop = max(0, offset - stone_distance)
        if first_stop > 0:
            field[:first_stop, primary:primary_stop] = np.random.choice(height_choices)
        secondary = offset
        while secondary < field.shape[0]:
            secondary_stop = min(field.shape[0], secondary + stone_size)
            field[secondary:secondary_stop, primary:primary_stop] = np.random.choice(height_choices)
            secondary += stone_size + stone_distance
        primary += stone_size + stone_distance


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-10):
    """Scatter stepping stones separated by holes of uniform depth."""
    stone_size = _discretize(stone_size, terrain.horizontal_scale)
    stone_distance = _discretize(stone_distance, terrain.horizontal_scale)
    max_height = _discretize(max_height, terrain.vertical_scale)
    platform_cells = _discretize(platform_size, terrain.horizontal_scale)
    depth_cells = _discretize(depth, terrain.vertical_scale)

    height_choices = np.arange(-max_height - 1, max_height, 1, dtype=np.int32)
    terrain.height_field_raw[:, :] = depth_cells

    if stone_size <= 0:
        stone_size = 1
    if stone_distance < 0:
        stone_distance = 0

    if terrain.length >= terrain.width:
        _fill_stepping_region(terrain.height_field_raw, stone_size, stone_distance, height_choices)
    else:
        _fill_stepping_region(terrain.height_field_raw.swapaxes(0, 1), stone_size, stone_distance, height_choices)

    platform_x = _center_slice(terrain.width, platform_cells)
    platform_y = _center_slice(terrain.length, platform_cells)
    terrain.height_field_raw[platform_x, platform_y] = 0
    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """Convert a regular heightfield to triangle mesh vertices and faces."""
    hf = height_field_raw
    num_rows, num_cols = hf.shape

    y_coords = np.linspace(0.0, (num_cols - 1) * horizontal_scale, num_cols, dtype=np.float32)
    x_coords = np.linspace(0.0, (num_rows - 1) * horizontal_scale, num_rows, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords)

    if slope_threshold is not None:
        scaled = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros_like(hf, dtype=np.float32)
        move_y = np.zeros_like(hf, dtype=np.float32)
        move_corners = np.zeros_like(hf, dtype=np.float32)

        delta_x = hf[1:, :] - hf[:-1, :]
        move_x[:-1, :] += delta_x > scaled
        move_x[1:, :] -= (-delta_x) > scaled

        delta_y = hf[:, 1:] - hf[:, :-1]
        move_y[:, :-1] += delta_y > scaled
        move_y[:, 1:] -= (-delta_y) > scaled

        delta_diag = hf[1:, 1:] - hf[:-1, :-1]
        move_corners[:-1, :-1] += delta_diag > scaled
        move_corners[1:, 1:] -= (-delta_diag) > scaled

        xx += (move_x + np.where(move_x == 0, move_corners, 0)) * horizontal_scale
        yy += (move_y + np.where(move_y == 0, move_corners, 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.reshape(-1)
    vertices[:, 1] = yy.reshape(-1)
    vertices[:, 2] = hf.reshape(-1) * vertical_scale

    indices = np.arange(num_rows * num_cols, dtype=np.uint32).reshape(num_rows, num_cols)
    v00 = indices[:-1, :-1].reshape(-1)
    v01 = indices[:-1, 1:].reshape(-1)
    v10 = indices[1:, :-1].reshape(-1)
    v11 = indices[1:, 1:].reshape(-1)

    triangles = np.empty((2 * v00.size, 3), dtype=np.uint32)
    triangles[0::2] = np.stack((v00, v11, v01), axis=1)
    triangles[1::2] = np.stack((v00, v10, v11), axis=1)

    return vertices, triangles


@dataclass
class SubTerrain:
    """Container representing a patch of terrain before tiling."""

    terrain_name: str = "terrain"
    width: int = 256
    length: int = 256
    vertical_scale: float = 1.0
    horizontal_scale: float = 1.0
    height_field_raw: np.ndarray = field(init=False)

    def __post_init__(self):
        # Use float storage to allow sub-vertical-scale precision (smoother slopes without tiny scales)
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.float32)


def gap_terrain(terrain, gap_size, platform_size=1.0):
    """Create a square pit surrounded by a flat platform."""
    gap_cells = _discretize(gap_size, terrain.horizontal_scale)
    platform_cells = _discretize(platform_size, terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_cells) // 2
    x2 = x1 + gap_cells
    y1 = (terrain.width - platform_cells) // 2
    y2 = y1 + gap_cells

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    """Excavate a rectangular pit centered in the terrain."""
    depth_cells = _discretize(depth, terrain.vertical_scale)
    half_platform = max(1, _discretize(platform_size, terrain.horizontal_scale) // 2)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x_slice = slice(max(center_x - half_platform, 0), min(center_x + half_platform, terrain.length))
    y_slice = slice(max(center_y - half_platform, 0), min(center_y + half_platform, terrain.width))
    terrain.height_field_raw[x_slice, y_slice] = -depth_cells


class TerrainGenerator:
    """Generate composite terrains from a declarative configuration."""

    def __init__(self, config=None):
        self.config = None
        self.height_mat: np.ndarray | None = None
        self.height_mat_pad: np.ndarray | None = None
        self.horizontal_scale = 1.0
        self.vertical_scale = 1.0
        self.margin = 0.0
        if config is not None:
            self._parse_cfg(config)

    def _parse_cfg(self, config):
        self.config = config
        self.horizontal_scale = config.horizontal_scale
        self.vertical_scale = config.vertical_scale
        self.margin = config.margin
        self.height_mat = np.zeros((config.num_rows, config.num_cols), dtype=np.float32)
        return self.config

    def _make_sub_terrain(self, config):
        width = max(1, math.ceil(config.size[0] / self.horizontal_scale))
        length = max(1, math.ceil(config.size[1] / self.horizontal_scale))
        return SubTerrain(
            config.type,
            width=width,
            length=length,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )

    def _make_slope(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        pyramid_sloped_terrain(terrain, slope=config.slope * difficulty, platform_size=config.platform_size)
        if getattr(config, "random", False):
            random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=2.0 * self.horizontal_scale,
            )
        return config.origin, terrain

    def _make_stair(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        pyramid_stairs_terrain(
            terrain,
            step_width=config.step[0],
            step_height=config.step[1] * difficulty,
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_obstacle(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        discrete_obstacles_terrain(
            terrain,
            max_height=config.max_height * difficulty,
            min_size=config.rectangle_params[0],
            max_size=config.rectangle_params[1],
            num_rects=config.rectangle_params[2],
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_stone(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        stone_scale = max(np.log1p(difficulty), 1e-6)
        stepping_stones_terrain(
            terrain,
            stone_size=config.stone_params[0] / stone_scale,
            stone_distance=config.stone_params[1],
            max_height=config.max_height,
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_gap(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        gap_terrain(terrain, gap_size=config.gap_size * difficulty, platform_size=config.platform_size)
        return config.origin, terrain

    def _make_pit(self, config, difficulty: float = 1.0):
        terrain = self._make_sub_terrain(config)
        pit_terrain(terrain, depth=config.depth * difficulty, platform_size=config.platform_size)
        return config.origin, terrain

    def _add_terrain_to_map(self, origin, terrain, matrix: np.ndarray):
        start_row = math.floor(origin[0] / self.horizontal_scale)
        start_col = math.floor(origin[1] / self.horizontal_scale)
        end_row = start_row + terrain.width
        end_col = start_col + terrain.length
        matrix[start_row:end_row, start_col:end_col] = terrain.height_field_raw
        return matrix

    def _repeat_terrain(self, repeat: int, direction: str, gap: float, difficulty_list: list[float]):
        if repeat <= 0:
            return self.height_mat
        assert len(difficulty_list) == repeat, "Difficulty schedule must match repeat count."

        base_rows, base_cols = self.config.num_rows, self.config.num_cols
        gap_cells = _discretize(gap, self.horizontal_scale)
        axis = 1 if direction == "column" else 0
        gap_block = None
        if gap_cells > 0:
            gap_shape = (base_rows, gap_cells) if axis == 1 else (gap_cells, base_cols)
            gap_block = np.zeros(gap_shape, dtype=np.float32)

        blocks = [self.height_mat]
        for diff in difficulty_list:
            if gap_block is not None:
                blocks.append(gap_block.copy())
            blocks.append(self.generate_matrix(diff))

        self.height_mat = np.concatenate(blocks, axis=axis)
        self.config.num_rows = self.height_mat.shape[0]
        self.config.num_cols = self.height_mat.shape[1]
        return self.height_mat

    def generate_matrix(self, difficulty: float = 1.0) -> np.ndarray:
        """Assemble a single tiled height matrix for the given difficulty value."""
        assert self.config is not None, "Call _parse_cfg or generate_terrain with a config before use."
        composite = np.zeros((self.config.num_rows, self.config.num_cols), dtype=np.float32)
        for terrain_type, terrains in self.config.elements.items():
            builder = getattr(self, f"_make_{terrain_type}", None)
            if builder is None:
                raise NotImplementedError(f"Terrain type '{terrain_type}' is not supported.")
            for cfg in terrains:
                origin, sub = builder(cfg, difficulty)
                self._add_terrain_to_map(origin, sub, composite)
        return composite

    def generate_terrain(self, config=None, type: str = "trimesh"):
        """Generate terrain data (heightfield and/or mesh) for the configured ground."""
        if config is not None:
            self._parse_cfg(config)
        assert self.config is not None, "Terrain configuration must be set before generating terrain."

        repeats, direction, gap = self.config.repeat_direction_gap
        if self.config.difficulty[2] == "linear":
            schedule = np.linspace(self.config.difficulty[0], self.config.difficulty[1], repeats + 1).tolist()
        else:
            schedule = [self.config.difficulty[0]] * (repeats + 1)

        self.height_mat = self.generate_matrix(schedule.pop(0))
        if repeats:
            self._repeat_terrain(repeats, direction, gap, schedule)

        pad_rows = self.config.margin_num_rows
        pad_cols = self.config.margin_num_cols
        self.height_mat_pad = np.pad(
            self.height_mat,
            ((pad_rows, pad_rows), (pad_cols, pad_cols)),
            mode="constant",
            constant_values=0,
        )

        if type == "trimesh":
            vertices, triangles = convert_heightfield_to_trimesh(
                height_field_raw=self.height_mat_pad,
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
                slope_threshold=None,
            )
            return vertices, triangles
        if type == "heightfield":
            return self.height_mat_pad * self.vertical_scale
        if type == "both":
            vertices, triangles = convert_heightfield_to_trimesh(
                height_field_raw=self.height_mat_pad,
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
                slope_threshold=None,
            )
            return vertices, triangles, self.height_mat_pad * self.vertical_scale
        raise ValueError(f"Unknown terrain export type '{type}'.")

    @property
    def height_measure(self):
        """Return the current unpadded heightfield in metric units."""
        if self.height_mat is None:
            return None
        return self.height_mat * self.vertical_scale

    @property
    def height_measure_pad(self):
        """Return the padded heightfield in metric units used for mesh export."""
        if self.height_mat_pad is None:
            return None
        return self.height_mat_pad * self.vertical_scale
