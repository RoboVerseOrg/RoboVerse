"""Light Randomizer - Property editor for light properties.

The LightRandomizer modifies properties of existing lights.
Lights are Static Objects (Handler-created) but accessed directly via USD
because IsaacLab does not provide a Light API.

Key features:
- Intensity randomization
- Color randomization (RGB or color temperature)
- Position randomization
- Orientation randomization
- Supports Hybrid simulation (uses render_handler)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.randomization.core.isaacsim_adapter import IsaacSimAdapter
from metasim.randomization.core.object_registry import ObjectRegistry
from metasim.utils.configclass import configclass

# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class LightIntensityRandomCfg:
    """Light intensity randomization configuration.

    Attributes:
        intensity_range: Absolute intensity range (min, max)
        intensity_delta_range: Relative intensity delta range (offset from original)
        use_delta: If True, intensity_delta_range is used (relative offset), else intensity_range (absolute values)
        distribution: Random sampling distribution
        enabled: Whether to apply intensity randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different intensity (diverse lighting)
                 False: all envs get same intensity (consistent lighting)
    """

    intensity_range: tuple[float, float] | None = None
    intensity_delta_range: tuple[float, float] | None = None
    use_delta: bool = True  # Default: use delta (relative to original intensity)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same intensity for all envs


@configclass
class LightColorRandomCfg:
    """Light color randomization configuration.

    Attributes:
        color_range: Absolute RGB color ranges ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        color_delta_range: Relative RGB color delta ranges (for micro-adjustments)
        temperature_range: Absolute color temperature range in Kelvin
        temperature_delta_range: Relative color temperature delta range in Kelvin
        use_temperature: Use color temperature instead of RGB
        use_delta: Use delta (relative) mode instead of absolute (for both RGB and temperature)
        distribution: Random sampling distribution
        enabled: Whether to apply color randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different color (diverse lighting)
                 False: all envs get same color (consistent lighting)
    """

    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    color_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    temperature_range: tuple[float, float] | None = None
    temperature_delta_range: tuple[float, float] | None = None
    use_temperature: bool = False
    use_delta: bool = True  # Default: use delta (relative to original color)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same color for all envs


@configclass
class LightPositionRandomCfg:
    """Light position randomization configuration.

    Attributes:
        position_range: Absolute position ranges ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        position_delta_range: Relative position delta ranges
        use_delta: Use delta (relative to original) mode instead of absolute
        distribution: Random sampling distribution
        enabled: Whether to apply position randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different position (diverse lighting)
                 False: all envs get same position (consistent lighting)
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    position_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True  # Default: use delta (relative to original)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same position for all envs


@configclass
class LightOrientationRandomCfg:
    """Light orientation randomization configuration.

    Attributes:
        angle_range: Absolute angle ranges in degrees ((roll_min, roll_max), (pitch_min, pitch_max), (yaw_min, yaw_max))
        angle_delta_range: Relative angle delta ranges in degrees
        use_delta: Use delta (relative to original) mode instead of absolute
        distribution: Random sampling distribution
        enabled: Whether to apply orientation randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different orientation (diverse lighting)
                 False: all envs get same orientation (consistent lighting)
    """

    angle_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    angle_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True  # Default: use delta (relative to original)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same orientation for all envs


@configclass
class LightRandomCfg:
    """Light randomization configuration.

    Attributes:
        light_name: Name of light to randomize (must exist in ObjectRegistry)
        intensity: Intensity randomization configuration
        color: Color randomization configuration
        position: Position randomization configuration
        orientation: Orientation randomization configuration
        env_ids: Environment IDs to apply randomization (None = all, but lights are usually shared)
    """

    light_name: str = dataclasses.MISSING
    intensity: LightIntensityRandomCfg | None = None
    color: LightColorRandomCfg | None = None
    position: LightPositionRandomCfg | None = None
    orientation: LightOrientationRandomCfg | None = None
    env_ids: list[int] | None = None

    def __post_init__(self):
        configs = [cfg for cfg in [self.intensity, self.color, self.position, self.orientation] if cfg]
        if not configs:
            logger.warning(f"No light configurations for {self.light_name}. Creating default intensity config.")
            self.intensity = LightIntensityRandomCfg(intensity_range=(100.0, 1000.0), enabled=True)
            configs = [self.intensity]

        enabled_configs = [cfg for cfg in configs if getattr(cfg, "enabled", True)]
        if not enabled_configs:
            raise ValueError("At least one light randomization type must be enabled")


# =============================================================================
# Light Randomizer Implementation
# =============================================================================


class LightRandomizer(BaseRandomizerType):
    """Light property randomizer.

    Responsibilities:
    - Modify light properties (intensity, color, position, orientation)
    - NOT responsible for: Creating/deleting lights

    Characteristics:
    - Uses ObjectRegistry to find lights
    - Uses IsaacSimAdapter for light property modification
    - Direct USD access (IsaacLab has no Light API)
    - Hybrid support: uses render_handler

    Usage:
        randomizer = LightRandomizer(
            LightRandomCfg(
                light_name="ceiling_light",
                intensity=LightIntensityRandomCfg(
                    intensity_range=(5000, 20000)
                )
            ),
            seed=42
        )
        randomizer.bind_handler(handler)
        randomizer()  # Apply light randomization
    """

    REQUIRES_HANDLER = "render"  # Use render_handler for Hybrid

    def __init__(self, cfg: LightRandomCfg, seed: int | None = None):
        """Initialize light randomizer.

        Args:
            cfg: Light randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.cfg = cfg
        self.registry: ObjectRegistry | None = None
        self.adapter: IsaacSimAdapter | None = None
        self._original_positions: dict[str, tuple] = {}
        self._original_orientations: dict[str, tuple] = {}
        self._original_intensities: dict[str, float] = {}  # Store original intensities
        self._original_colors: dict[str, tuple] = {}  # Store original colors

    def __call__(self, env_ids: list[int] | None = None):
        """Execute light randomization.

        Args:
            env_ids: Environment IDs to randomize. If None, uses self.cfg.env_ids.
                     If both are None, randomizes all environments.
        """
        # Use provided env_ids, or fall back to config, or all environments
        if env_ids is None:
            env_ids = self.cfg.env_ids
        if env_ids is None:
            env_ids = list(range(self._actual_handler.num_envs))

        # Get light prim paths from Registry (filtered by env_ids)
        try:
            prim_paths = self.registry.get_prim_paths(self.cfg.light_name, env_ids)
        except ValueError as e:
            logger.error(f"LightRandomizer: {e}")
            return

        # Apply randomization (per_env now means "different value for each environment")
        if self.cfg.intensity and self.cfg.intensity.enabled:
            self._randomize_intensity(prim_paths)

        if self.cfg.color and self.cfg.color.enabled:
            self._randomize_color(prim_paths)

        if self.cfg.position and self.cfg.position.enabled:
            self._randomize_position(prim_paths)

        if self.cfg.orientation and self.cfg.orientation.enabled:
            self._randomize_orientation(prim_paths)

        self._mark_visual_dirty()

        # Flush visual updates for instant switching
        self._flush_visual_updates()

    # -------------------------------------------------------------------------
    # Randomization Methods
    # -------------------------------------------------------------------------

    def _randomize_intensity(self, prim_paths: list[str]):
        """Randomize light intensity.

        Args:
            prim_paths: List of light prim paths
        """
        if not self.cfg.intensity.intensity_delta_range and not self.cfg.intensity.intensity_range:
            return

        if self.cfg.intensity.per_env:
            # Mode A: Different intensity for each environment
            for prim_path in prim_paths:
                # Save original intensity on first call
                if prim_path not in self._original_intensities:
                    try:
                        current_intensity = self.adapter.get_light_intensity(prim_path)
                        self._original_intensities[prim_path] = current_intensity
                    except Exception as e:
                        logger.warning(f"Failed to get intensity for {prim_path}: {e}")
                        self._original_intensities[prim_path] = 1000.0  # Default fallback

                original_intensity = self._original_intensities[prim_path]

                # Generate random value
                rand_value = self._generate_random_value(
                    (
                        self.cfg.intensity.intensity_delta_range
                        if self.cfg.intensity.use_delta
                        else self.cfg.intensity.intensity_range
                    ),
                    self.cfg.intensity.distribution,
                )

                # Apply based on mode
                if self.cfg.intensity.use_delta:
                    # Delta mode: relative to original
                    intensity = original_intensity + rand_value
                else:
                    # Absolute mode: direct value
                    intensity = rand_value

                # Ensure non-negative
                intensity = max(0.0, intensity)

                try:
                    self.adapter.set_light_intensity(prim_path, intensity)
                except Exception as e:
                    logger.warning(f"Failed to set light intensity for {prim_path}: {e}")
        else:
            # Mode B: Same intensity for all environments
            # Save original intensity (use first prim as representative)
            if prim_paths and prim_paths[0] not in self._original_intensities:
                try:
                    current_intensity = self.adapter.get_light_intensity(prim_paths[0])
                    # Use same original for all prims (they're usually same type)
                    for prim_path in prim_paths:
                        if prim_path not in self._original_intensities:
                            self._original_intensities[prim_path] = current_intensity
                except Exception as e:
                    logger.warning(f"Failed to get intensity: {e}")
                    for prim_path in prim_paths:
                        self._original_intensities[prim_path] = 1000.0

            # Get original (use first as representative if multiple lights)
            original_intensity = self._original_intensities[prim_paths[0]] if prim_paths else 1000.0

            # Generate random value once
            rand_value = self._generate_random_value(
                (
                    self.cfg.intensity.intensity_delta_range
                    if self.cfg.intensity.use_delta
                    else self.cfg.intensity.intensity_range
                ),
                self.cfg.intensity.distribution,
            )

            # Apply based on mode
            if self.cfg.intensity.use_delta:
                intensity = original_intensity + rand_value
            else:
                intensity = rand_value

            intensity = max(0.0, intensity)

            # Apply to all prims
            for prim_path in prim_paths:
                try:
                    self.adapter.set_light_intensity(prim_path, intensity)
                except Exception as e:
                    logger.warning(f"Failed to set light intensity for {prim_path}: {e}")

    def _randomize_color(self, prim_paths: list[str]):
        """Randomize light color.

        Args:
            prim_paths: List of light prim paths
        """
        if self.cfg.color.use_delta:
            # Delta mode: relative to original color
            if self.cfg.color.per_env:
                # Mode A: Different delta for each environment
                for prim_path in prim_paths:
                    # Save original color on first call
                    if prim_path not in self._original_colors:
                        current_color = self.adapter.get_light_color(prim_path)
                        self._original_colors[prim_path] = current_color

                    original_color = self._original_colors[prim_path]

                    if self.cfg.color.use_temperature and (
                        self.cfg.color.temperature_delta_range or self.cfg.color.temperature_range
                    ):
                        # Temperature delta mode
                        # Convert original color to temperature first
                        original_temp = self._rgb_to_temperature(original_color)
                        temp_delta = self._generate_random_value(
                            (
                                self.cfg.color.temperature_delta_range
                                if self.cfg.color.use_delta
                                else self.cfg.color.temperature_range
                            ),
                            self.cfg.color.distribution,
                        )
                        new_temp = original_temp + temp_delta
                        color = self._temperature_to_rgb(new_temp)
                    elif self.cfg.color.color_delta_range:
                        # RGB delta mode
                        color = tuple(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    original_color[i]
                                    + self._generate_random_value(
                                        self.cfg.color.color_delta_range[i], self.cfg.color.distribution
                                    ),
                                ),
                            )
                            for i in range(3)
                        )
                    else:
                        return

                    try:
                        self.adapter.set_light_color(prim_path, color)
                    except Exception as e:
                        logger.warning(f"Failed to set light color for {prim_path}: {e}")
            else:
                # Mode B: Same delta for all environments
                # Use first prim as reference for original color
                if prim_paths and prim_paths[0] not in self._original_colors:
                    current_color = self.adapter.get_light_color(prim_paths[0])
                    # Store same original for all prims
                    for prim_path in prim_paths:
                        self._original_colors[prim_path] = current_color

                if not prim_paths:
                    return

                original_color = self._original_colors[prim_paths[0]]

                if self.cfg.color.use_temperature and (
                    self.cfg.color.temperature_delta_range or self.cfg.color.temperature_range
                ):
                    original_temp = self._rgb_to_temperature(original_color)
                    temp_delta = self._generate_random_value(
                        (
                            self.cfg.color.temperature_delta_range
                            if self.cfg.color.use_delta
                            else self.cfg.color.temperature_range
                        ),
                        self.cfg.color.distribution,
                    )
                    new_temp = original_temp + temp_delta
                    color = self._temperature_to_rgb(new_temp)
                elif self.cfg.color.color_delta_range:
                    color = tuple(
                        max(
                            0.0,
                            min(
                                1.0,
                                original_color[i]
                                + self._generate_random_value(
                                    self.cfg.color.color_delta_range[i], self.cfg.color.distribution
                                ),
                            ),
                        )
                        for i in range(3)
                    )
                else:
                    return

                for prim_path in prim_paths:
                    try:
                        self.adapter.set_light_color(prim_path, color)
                    except Exception as e:
                        logger.warning(f"Failed to set light color for {prim_path}: {e}")
        else:
            # Absolute mode: set absolute color
            if self.cfg.color.per_env:
                # Mode A: Different color for each environment
                for prim_path in prim_paths:
                    if self.cfg.color.use_temperature and (
                        self.cfg.color.temperature_delta_range or self.cfg.color.temperature_range
                    ):
                        temp = self._generate_random_value(
                            (
                                self.cfg.color.temperature_delta_range
                                if self.cfg.color.use_delta
                                else self.cfg.color.temperature_range
                            ),
                            self.cfg.color.distribution,
                        )
                        color = self._temperature_to_rgb(temp)
                    elif self.cfg.color.color_range:
                        color = tuple(
                            self._generate_random_value(r, self.cfg.color.distribution)
                            for r in self.cfg.color.color_range
                        )
                    else:
                        return

                    try:
                        self.adapter.set_light_color(prim_path, color)
                    except Exception as e:
                        logger.warning(f"Failed to set light color for {prim_path}: {e}")
            else:
                # Mode B: Same color for all environments
                if self.cfg.color.use_temperature and (
                    self.cfg.color.temperature_delta_range or self.cfg.color.temperature_range
                ):
                    temp = self._generate_random_value(
                        (
                            self.cfg.color.temperature_delta_range
                            if self.cfg.color.use_delta
                            else self.cfg.color.temperature_range
                        ),
                        self.cfg.color.distribution,
                    )
                    color = self._temperature_to_rgb(temp)
                elif self.cfg.color.color_range:
                    color = tuple(
                        self._generate_random_value(r, self.cfg.color.distribution) for r in self.cfg.color.color_range
                    )
                else:
                    return

                for prim_path in prim_paths:
                    try:
                        self.adapter.set_light_color(prim_path, color)
                    except Exception as e:
                        logger.warning(f"Failed to set light color for {prim_path}: {e}")

    def _randomize_position(self, prim_paths: list[str]):
        """Randomize light position.

        Args:
            prim_paths: List of light prim paths
        """
        if not (self.cfg.position.position_delta_range or self.cfg.position.position_range):
            return

        # Determine which range to use
        active_position_range = (
            self.cfg.position.position_delta_range if self.cfg.position.use_delta else self.cfg.position.position_range
        )

        if self.cfg.position.per_env:
            # Mode A: Different position for each environment
            for prim_path in prim_paths:
                position_offset = tuple(
                    self._generate_random_value(r, self.cfg.position.distribution) for r in active_position_range
                )
                self._apply_position_offset(prim_path, position_offset)
        else:
            # Mode B: Same position for all environments
            position_offset = tuple(
                self._generate_random_value(r, self.cfg.position.distribution) for r in active_position_range
            )
            for prim_path in prim_paths:
                self._apply_position_offset(prim_path, position_offset)

    def _randomize_orientation(self, prim_paths: list[str]):
        """Randomize light orientation.

        Args:
            prim_paths: List of light prim paths
        """
        if not (self.cfg.orientation.angle_delta_range or self.cfg.orientation.angle_range):
            return

        # Determine which range to use
        active_angle_range = (
            self.cfg.orientation.angle_delta_range
            if self.cfg.orientation.use_delta
            else self.cfg.orientation.angle_range
        )

        if self.cfg.orientation.per_env:
            # Mode A: Different orientation for each environment
            for prim_path in prim_paths:
                orientation_offset = tuple(
                    self._generate_random_value(r, self.cfg.orientation.distribution) for r in active_angle_range
                )
                self._apply_orientation_offset(prim_path, orientation_offset)
        else:
            # Mode B: Same orientation for all environments
            orientation_offset = tuple(
                self._generate_random_value(r, self.cfg.orientation.distribution) for r in active_angle_range
            )
            for prim_path in prim_paths:
                self._apply_orientation_offset(prim_path, orientation_offset)

    def _apply_position_offset(self, prim_path: str, position_offset: tuple[float, float, float]):
        """Apply position offset to light.

        Args:
            prim_path: Light prim path
            position_offset: Position offset (x, y, z)
        """
        # Save original position if not saved
        if prim_path not in self._original_positions:
            try:
                current_pos, _, _ = self.adapter.get_transform(prim_path)
                self._original_positions[prim_path] = current_pos
            except Exception as e:
                logger.warning(f"Failed to get position for {prim_path}: {e}")
                return

        original_pos = self._original_positions[prim_path]

        # Compute new position
        if self.cfg.position.use_delta:
            new_pos = tuple(original_pos[i] + position_offset[i] for i in range(3))
        else:
            new_pos = position_offset

        # Apply new position
        try:
            self.adapter.set_transform(prim_path, position=new_pos)
        except Exception as e:
            logger.warning(f"Failed to set position for {prim_path}: {e}")

    def _apply_orientation_offset(self, prim_path: str, orientation_offset: tuple[float, float, float]):
        """Apply orientation offset to light.

        Args:
            prim_path: Light prim path
            orientation_offset: Orientation offset in degrees (roll, pitch, yaw)
        """
        # Save original orientation if not saved
        if prim_path not in self._original_orientations:
            try:
                _, current_rot, _ = self.adapter.get_transform(prim_path)
                self._original_orientations[prim_path] = current_rot
            except Exception as e:
                logger.warning(f"Failed to get orientation for {prim_path}: {e}")
                return

        original_rot = self._original_orientations[prim_path]

        # Compute new orientation
        if self.cfg.orientation.use_delta:
            # Convert degrees to radians
            roll_rad = math.radians(orientation_offset[0])
            pitch_rad = math.radians(orientation_offset[1])
            yaw_rad = math.radians(orientation_offset[2])

            # Create rotation quaternion from delta
            from metasim.utils.math import quat_from_euler_xyz, quat_mul

            delta_quat = quat_from_euler_xyz(
                torch.tensor([roll_rad]), torch.tensor([pitch_rad]), torch.tensor([yaw_rad])
            )
            original_quat = torch.tensor([original_rot], dtype=torch.float32)

            # Apply delta rotation
            new_quat = quat_mul(delta_quat, original_quat)
            new_rot = tuple(new_quat[0].tolist())
        else:
            # Absolute orientation (convert from Euler to quaternion)
            from metasim.utils.math import quat_from_euler_xyz

            roll_rad = math.radians(orientation_offset[0])
            pitch_rad = math.radians(orientation_offset[1])
            yaw_rad = math.radians(orientation_offset[2])

            new_quat = quat_from_euler_xyz(torch.tensor([roll_rad]), torch.tensor([pitch_rad]), torch.tensor([yaw_rad]))
            new_rot = tuple(new_quat[0].tolist())

        # Apply new orientation
        try:
            self.adapter.set_transform(prim_path, rotation=new_rot)
        except Exception as e:
            logger.warning(f"Failed to set orientation for {prim_path}: {e}")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _generate_random_value(self, value_range: tuple[float, float], distribution: str) -> float:
        """Generate a single random value."""
        if distribution == "uniform":
            return self.rng.uniform(value_range[0], value_range[1])
        elif distribution == "log_uniform":
            log_min = math.log(value_range[0])
            log_max = math.log(value_range[1])
            return math.exp(self.rng.uniform(log_min, log_max))
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            val = self.rng.gauss(mean, std)
            return max(value_range[0], min(value_range[1], val))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _rgb_to_temperature(self, rgb: tuple[float, float, float]) -> float:
        """Estimate color temperature from RGB (approximation).

        Args:
            rgb: RGB tuple (0-1 range)

        Returns:
            Estimated color temperature in Kelvin
        """
        # This is an approximation. For more accurate conversion,
        # you'd need a lookup table or more complex algorithm.
        # Here we use a simple heuristic based on the color balance.

        r, g, b = rgb

        # Warm colors (red-dominant) -> lower temperature
        # Cool colors (blue-dominant) -> higher temperature

        if b > r:
            # Cool light (bluish)
            ratio = b / max(r, 0.001)
            # Map ratio to temperature (6500K - 10000K)
            temp = 6500 + (ratio - 1.0) * 2000
        else:
            # Warm light (reddish)
            ratio = r / max(b, 0.001)
            # Map ratio to temperature (2000K - 6500K)
            temp = 6500 - (ratio - 1.0) * 1500

        return max(1000, min(40000, temp))

    def _temperature_to_rgb(self, temp_kelvin: float) -> tuple[float, float, float]:
        """Convert color temperature to RGB.

        Args:
            temp_kelvin: Color temperature in Kelvin (1000-40000)

        Returns:
            RGB tuple (0-1 range)
        """
        # Clamp temperature
        temp = max(1000, min(40000, temp_kelvin)) / 100.0

        # Calculate red
        if temp <= 66:
            red = 1.0
        else:
            red = temp - 60
            red = 329.698727446 * (red**-0.1332047592)
            red = max(0, min(255, red)) / 255.0

        # Calculate green
        if temp <= 66:
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
            green = max(0, min(255, green)) / 255.0
        else:
            green = temp - 60
            green = 288.1221695283 * (green**-0.0755148492)
            green = max(0, min(255, green)) / 255.0

        # Calculate blue
        if temp >= 66:
            blue = 1.0
        elif temp <= 19:
            blue = 0.0
        else:
            blue = temp - 10
            blue = 138.5177312231 * math.log(blue) - 305.0447927307
            blue = max(0, min(255, blue)) / 255.0

        return (red, green, blue)

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> tuple:
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (w, x, y, z)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.tensor([w, x, y, z])

    def _flush_visual_updates(self):
        """Flush visual updates to ensure light changes are visible instantly.

        This is critical for real-time light switching to be visible.
        Respects global defer flag for atomic multi-randomizer operations.
        """
        # Check global defer flag (set by apply_randomization for 22→1 flush optimization)
        if (
            hasattr(self._actual_handler, "_defer_all_visual_flushes")
            and self._actual_handler._defer_all_visual_flushes
        ):
            return  # Skip flush, will be done by apply_randomization

        if hasattr(self._actual_handler, "flush_visual_updates"):
            self._actual_handler.flush_visual_updates()
