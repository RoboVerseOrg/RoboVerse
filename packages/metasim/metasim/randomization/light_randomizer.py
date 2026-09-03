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
        intensity_range: Intensity range (min, max)
        distribution: Random sampling distribution
        enabled: Whether to apply intensity randomization
    """

    intensity_range: tuple[float, float] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightColorRandomCfg:
    """Light color randomization configuration.

    Attributes:
        color_range: RGB color ranges ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        temperature_range: Color temperature range in Kelvin
        use_temperature: Use color temperature instead of RGB
        distribution: Random sampling distribution
        enabled: Whether to apply color randomization
    """

    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    temperature_range: tuple[float, float] | None = None
    use_temperature: bool = False
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightPositionRandomCfg:
    """Light position randomization configuration.

    Attributes:
        position_range: Position ranges ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        relative_to_origin: Whether positions are relative to original position
        distribution: Random sampling distribution
        enabled: Whether to apply position randomization
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    relative_to_origin: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightOrientationRandomCfg:
    """Light orientation randomization configuration.

    Attributes:
        angle_range: Angle ranges in degrees ((roll_min, roll_max), (pitch_min, pitch_max), (yaw_min, yaw_max))
        relative_to_origin: Whether angles are relative to original orientation
        distribution: Random sampling distribution
        enabled: Whether to apply orientation randomization
    """

    angle_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    relative_to_origin: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


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

    def __call__(self):
        """Execute light randomization."""
        # Get light prim paths from Registry
        try:
            prim_paths = self.registry.get_prim_paths(self.cfg.light_name)
        except ValueError as e:
            logger.error(f"LightRandomizer: {e}")
            return

        # Apply randomization to each light prim
        for prim_path in prim_paths:
            if self.cfg.intensity and self.cfg.intensity.enabled:
                self._randomize_intensity(prim_path)

            if self.cfg.color and self.cfg.color.enabled:
                self._randomize_color(prim_path)

            if self.cfg.position and self.cfg.position.enabled:
                self._randomize_position(prim_path)

            if self.cfg.orientation and self.cfg.orientation.enabled:
                self._randomize_orientation(prim_path)

        self._mark_visual_dirty()

        # Flush visual updates for instant switching
        self._flush_visual_updates()

    # -------------------------------------------------------------------------
    # Randomization Methods
    # -------------------------------------------------------------------------

    def _randomize_intensity(self, prim_path: str):
        """Randomize light intensity.

        Args:
            prim_path: Light prim path
        """
        if not self.cfg.intensity.intensity_range:
            return

        intensity = self._generate_random_value(self.cfg.intensity.intensity_range, self.cfg.intensity.distribution)

        try:
            self.adapter.set_light_intensity(prim_path, intensity)
        except Exception as e:
            logger.warning(f"Failed to set light intensity for {prim_path}: {e}")

    def _randomize_color(self, prim_path: str):
        """Randomize light color.

        Args:
            prim_path: Light prim path
        """
        if self.cfg.color.use_temperature and self.cfg.color.temperature_range:
            # Color temperature mode
            temp = self._generate_random_value(self.cfg.color.temperature_range, self.cfg.color.distribution)
            color = self._temperature_to_rgb(temp)
        elif self.cfg.color.color_range:
            # RGB mode
            color = tuple(
                self._generate_random_value(r, self.cfg.color.distribution) for r in self.cfg.color.color_range
            )
        else:
            return

        try:
            self.adapter.set_light_color(prim_path, color)
        except Exception as e:
            logger.warning(f"Failed to set light color for {prim_path}: {e}")

    def _randomize_position(self, prim_path: str):
        """Randomize light position.

        Args:
            prim_path: Light prim path
        """
        if not self.cfg.position.position_range:
            return

        # Get original position (for relative mode)
        if prim_path not in self._original_positions:
            try:
                pos, _, _ = self.adapter.get_transform(prim_path)
                self._original_positions[prim_path] = pos
            except Exception:
                self._original_positions[prim_path] = (0.0, 0.0, 0.0)

        original_pos = self._original_positions[prim_path]

        # Generate random position
        if self.cfg.position.relative_to_origin:
            new_pos = tuple(
                original_pos[i] + self._generate_random_value(r, self.cfg.position.distribution)
                for i, r in enumerate(self.cfg.position.position_range)
            )
        else:
            new_pos = tuple(
                self._generate_random_value(r, self.cfg.position.distribution) for r in self.cfg.position.position_range
            )

        try:
            self.adapter.set_transform(prim_path, position=new_pos)
        except Exception as e:
            logger.warning(f"Failed to set light position for {prim_path}: {e}")

    def _randomize_orientation(self, prim_path: str):
        """Randomize light orientation.

        Args:
            prim_path: Light prim path
        """
        if not self.cfg.orientation.angle_range:
            return

        # Get original orientation (for relative mode)
        if prim_path not in self._original_orientations:
            try:
                _, rot, _ = self.adapter.get_transform(prim_path)
                self._original_orientations[prim_path] = rot
            except Exception:
                self._original_orientations[prim_path] = (1.0, 0.0, 0.0, 0.0)

        # Generate random Euler angles
        angles = [
            self._generate_random_value(r, self.cfg.orientation.distribution) for r in self.cfg.orientation.angle_range
        ]

        # Convert to radians and quaternion
        roll_rad = angles[0] * (math.pi / 180.0)
        pitch_rad = angles[1] * (math.pi / 180.0)
        yaw_rad = angles[2] * (math.pi / 180.0)

        new_rot = self._euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

        if self.cfg.orientation.relative_to_origin:
            # Compose with original rotation
            original_rot = self._original_orientations[prim_path]
            import torch

            new_rot_tensor = self._quaternion_multiply(torch.tensor(original_rot), torch.tensor(new_rot))
            new_rot = tuple(new_rot_tensor.tolist())

        try:
            self.adapter.set_transform(prim_path, rotation=new_rot)
        except Exception as e:
            logger.warning(f"Failed to set light orientation for {prim_path}: {e}")

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
