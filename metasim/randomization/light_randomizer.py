from __future__ import annotations

import dataclasses
from typing import Any, Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass


@configclass
class LightIntensityRandomCfg:
    """Configuration for light intensity randomization.

    Args:
        intensity_range: Range for intensity randomization (min, max)
        distribution: Type of distribution for random sampling
        enabled: Whether to apply intensity randomization
    """

    intensity_range: tuple[float, float] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightColorRandomCfg:
    """Configuration for light color randomization.

    Args:
        color_range: RGB color ranges as ((r_min,r_max), (g_min,g_max), (b_min,b_max))
        temperature_range: Color temperature range in Kelvin (alternative to color_range)
        use_temperature: Whether to use color temperature instead of RGB
        distribution: Type of distribution for random sampling
        enabled: Whether to apply color randomization
    """

    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    temperature_range: tuple[float, float] | None = None
    use_temperature: bool = False
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightPositionRandomCfg:
    """Configuration for light position randomization.

    Args:
        position_range: Position ranges as ((x_min,x_max), (y_min,y_max), (z_min,z_max))
        relative_to_origin: Whether positions are relative to original position
        distribution: Type of distribution for random sampling
        enabled: Whether to apply position randomization
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    relative_to_origin: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightOrientationRandomCfg:
    """Configuration for light orientation randomization.

    Args:
        angle_range: Angle ranges in degrees as ((roll_min,roll_max), (pitch_min,pitch_max), (yaw_min,yaw_max))
        relative_to_origin: Whether angles are relative to original orientation
        distribution: Type of distribution for random sampling
        enabled: Whether to apply orientation randomization
    """

    angle_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    relative_to_origin: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class LightRandomCfg:
    """Unified configuration for light randomization.

    Args:
        light_name: Name of the light to randomize
        intensity: Intensity randomization configuration (optional)
        color: Color randomization configuration (optional)
        position: Position randomization configuration (optional)
        orientation: Orientation randomization configuration (optional)
        env_ids: List of environment IDs to apply randomization to (None = all)
        randomization_mode: How to apply multiple randomization types
    """

    light_name: str = dataclasses.MISSING
    intensity: LightIntensityRandomCfg | None = None
    color: LightColorRandomCfg | None = None
    position: LightPositionRandomCfg | None = None
    orientation: LightOrientationRandomCfg | None = None
    env_ids: list[int] | None = None
    randomization_mode: Literal["combined", "intensity_only", "color_only", "position_only", "orientation_only"] = (
        "combined"
    )

    def __post_init__(self):
        """Validate configuration."""
        available_configs = [
            cfg for cfg in [self.intensity, self.color, self.position, self.orientation] if cfg is not None
        ]
        if not available_configs:
            # If no configurations provided, create a default intensity configuration
            logger.warning(
                f"No light configurations provided for {self.light_name}. Creating default intensity configuration."
            )
            self.intensity = LightIntensityRandomCfg(intensity_range=(100.0, 1000.0), enabled=True)
            available_configs = [self.intensity]

        enabled_configs = [cfg for cfg in available_configs if getattr(cfg, "enabled", True)]
        if not enabled_configs:
            raise ValueError("At least one light randomization type must be enabled")


class LightRandomizer(BaseRandomizerType):
    """Light randomizer supporting intensity, color, position, and orientation.

    Supports multiple randomization modes and distributions with reproducible seeding.
    """

    def __init__(self, cfg: LightRandomCfg, seed: int | None = None):
        self.cfg = cfg
        super().__init__(seed=seed)

    def set_seed(self, seed: int | None) -> None:
        """Set or update RNG seed."""
        super().set_seed(seed)

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Bind the handler to the randomizer."""
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            super().bind_handler(handler, *args, **kwargs)
            # Import IsaacSim specific modules only when needed
            try:
                global omni, prim_utils
                import omni

                try:
                    import omni.isaac.core.utils.prims as prim_utils
                except ModuleNotFoundError:
                    import isaacsim.core.utils.prims as prim_utils

                self.stage = omni.usd.get_context().get_stage()
            except ImportError as e:
                raise ImportError(f"Failed to import IsaacSim modules: {e}") from e
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for LightRandomizer")

    def _get_light_prim(self, light_name: str):
        """Get light prim from the scene."""
        # First try the direct path using the configured name
        light_paths = [
            f"/World/{light_name}",  # Direct name path (new preferred approach)
        ]

        # Also try index-based paths for backward compatibility
        if light_name.startswith("light_"):
            light_index = light_name.replace("light_", "")
        else:
            light_index = light_name

        # Add index-based paths as fallback
        light_paths.extend([
            f"/World/DistantLight_{light_index}",
            f"/World/SphereLight_{light_index}",
            f"/World/CylinderLight_{light_index}",
            f"/World/DiskLight_{light_index}",
            f"/World/DomeLight_{light_index}",
            "/World/DefaultLight",  # Fallback default light
        ])

        for path in light_paths:
            prim = prim_utils.get_prim_at_path(path)
            if prim and prim.IsValid():
                # Determine light type from path or prim type
                light_type = self._get_light_type_from_path_or_prim(path, prim)
                return prim, path, light_type

        raise ValueError(f"Light {light_name} not found in scene. Tried paths: {light_paths}")

    def _get_light_type_from_path_or_prim(self, path: str, prim) -> str:
        """Determine light type from USD path or prim type."""
        # First try to determine from path
        if "DistantLight" in path:
            return "distant"
        elif "SphereLight" in path:
            return "sphere"
        elif "CylinderLight" in path:
            return "cylinder"
        elif "DiskLight" in path:
            return "disk"
        elif "DomeLight" in path:
            return "dome"

        # If path doesn't contain type info, get it from the prim itself
        prim_type = prim.GetTypeName()
        if prim_type == "DistantLight":
            return "distant"
        elif prim_type == "SphereLight":
            return "sphere"
        elif prim_type == "CylinderLight":
            return "cylinder"
        elif prim_type == "DiskLight":
            return "disk"
        elif prim_type == "DomeLight":
            return "dome"
        elif prim_type == "Light":
            return "generic"
        else:
            return "unknown"

    def _get_env_ids(self) -> list[int]:
        """Get environment IDs to operate on."""
        return self.cfg.env_ids or list(range(self.handler.num_envs))

    def _generate_random_value(self, value_range: tuple[float, float], distribution: str = "uniform") -> float:
        """Generate a single random value using reproducible RNG."""
        if distribution == "uniform":
            return self._rng.uniform(value_range[0], value_range[1])
        elif distribution == "log_uniform":
            log_min = torch.log(torch.tensor(value_range[0])).item()
            log_max = torch.log(torch.tensor(value_range[1])).item()
            return torch.exp(torch.tensor(self._rng.uniform(log_min, log_max))).item()
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            val = self._rng.gauss(mean, std)
            return max(value_range[0], min(value_range[1], val))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _kelvin_to_rgb(self, temperature: float) -> tuple[float, float, float]:
        """Convert color temperature in Kelvin to RGB values."""
        # Simplified conversion - for more accurate conversion, use proper color science
        temp = temperature / 100.0

        # Calculate red
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red**-0.1332047592)
            red = max(0, min(255, red))

        # Calculate green
        if temp <= 66:
            green = temp
            green = 99.4708025861 * torch.log(torch.tensor(green)).item() - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green**-0.0755148492)
        green = max(0, min(255, green))

        # Calculate blue
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * torch.log(torch.tensor(blue)).item() - 305.0447927307
            blue = max(0, min(255, blue))

        return (red / 255.0, green / 255.0, blue / 255.0)

    def randomize_intensity(self) -> None:
        """Randomize light intensity."""
        if not self.cfg.intensity or not self.cfg.intensity.enabled:
            return

        try:
            light_prim, light_path, light_type = self._get_light_prim(self.cfg.light_name)

            if self.cfg.intensity.intensity_range:
                new_intensity = self._generate_random_value(
                    self.cfg.intensity.intensity_range, self.cfg.intensity.distribution
                )

                # Set intensity attribute (common to all light types)
                intensity_attr = light_prim.GetAttribute("inputs:intensity")
                if intensity_attr:
                    intensity_attr.Set(new_intensity)

                else:
                    logger.warning(f"Could not find intensity attribute for {light_type} light {self.cfg.light_name}")

        except Exception as e:
            logger.warning(f"Failed to randomize intensity for light {self.cfg.light_name}: {e}")

    def randomize_color(self) -> None:
        """Randomize light color."""
        if not self.cfg.color or not self.cfg.color.enabled:
            return

        try:
            light_prim, light_path, light_type = self._get_light_prim(self.cfg.light_name)

            if self.cfg.color.use_temperature and self.cfg.color.temperature_range:
                # Use color temperature
                temperature = self._generate_random_value(self.cfg.color.temperature_range, self.cfg.color.distribution)
                rgb = self._kelvin_to_rgb(temperature)

            elif self.cfg.color.color_range:
                # Use RGB values
                r = self._generate_random_value(self.cfg.color.color_range[0], self.cfg.color.distribution)
                g = self._generate_random_value(self.cfg.color.color_range[1], self.cfg.color.distribution)
                b = self._generate_random_value(self.cfg.color.color_range[2], self.cfg.color.distribution)
                rgb = (r, g, b)

            else:
                return

            # Set color attribute (common to all light types)
            color_attr = light_prim.GetAttribute("inputs:color")
            if color_attr:
                from pxr import Gf

                color_attr.Set(Gf.Vec3f(*rgb))
            else:
                logger.warning(f"Could not find color attribute for {light_type} light {self.cfg.light_name}")

        except Exception as e:
            logger.warning(f"Failed to randomize color for light {self.cfg.light_name}: {e}")

    def randomize_position(self) -> None:
        """Randomize light position (not applicable to distant lights)."""
        if not self.cfg.position or not self.cfg.position.enabled:
            return

        try:
            light_prim, light_path, light_type = self._get_light_prim(self.cfg.light_name)

            # Skip position randomization for distant lights (they don't have meaningful position)
            if light_type == "distant":
                #                 logger.debug(f"Skipping position randomization for distant light {self.cfg.light_name}")
                return

            if self.cfg.position.position_range:
                if self.cfg.position.relative_to_origin:
                    # Get current position and add offset
                    translate_attr = light_prim.GetAttribute("xformOp:translate")
                    if translate_attr:
                        current_pos = translate_attr.Get()
                        if current_pos is None:
                            current_pos = (0.0, 0.0, 0.0)
                    else:
                        current_pos = (0.0, 0.0, 0.0)

                    # Generate random offset
                    x_offset = self._generate_random_value(
                        self.cfg.position.position_range[0], self.cfg.position.distribution
                    )
                    y_offset = self._generate_random_value(
                        self.cfg.position.position_range[1], self.cfg.position.distribution
                    )
                    z_offset = self._generate_random_value(
                        self.cfg.position.position_range[2], self.cfg.position.distribution
                    )

                    new_pos = (current_pos[0] + x_offset, current_pos[1] + y_offset, current_pos[2] + z_offset)
                else:
                    # Use absolute positioning - directly sample from the range
                    x_pos = self._generate_random_value(
                        self.cfg.position.position_range[0], self.cfg.position.distribution
                    )
                    y_pos = self._generate_random_value(
                        self.cfg.position.position_range[1], self.cfg.position.distribution
                    )
                    z_pos = self._generate_random_value(
                        self.cfg.position.position_range[2], self.cfg.position.distribution
                    )

                    new_pos = (x_pos, y_pos, z_pos)

                # Set position
                translate_attr = light_prim.GetAttribute("xformOp:translate")
                if not translate_attr:
                    from pxr import Sdf

                    translate_attr = light_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)

                from pxr import Gf

                translate_attr.Set(Gf.Vec3d(*new_pos))
                # logger.info(
                #     f"Set {light_type} light '{self.cfg.light_name}' position to ({new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f})"
                # )

        except Exception as e:
            logger.warning(f"Failed to randomize position for light {self.cfg.light_name}: {e}")

    def randomize_orientation(self) -> None:
        """Randomize light orientation (mainly for distant and area lights)."""
        if not self.cfg.orientation or not self.cfg.orientation.enabled:
            return

        try:
            light_prim, light_path, light_type = self._get_light_prim(self.cfg.light_name)

            if self.cfg.orientation.angle_range:
                if self.cfg.orientation.relative_to_origin:
                    # Get current rotation
                    rotate_attr = light_prim.GetAttribute("xformOp:rotateXYZ")
                    if rotate_attr:
                        current_rot = rotate_attr.Get()
                        if current_rot is None:
                            current_rot = (0.0, 0.0, 0.0)
                    else:
                        current_rot = (0.0, 0.0, 0.0)
                else:
                    current_rot = (0.0, 0.0, 0.0)

                # Generate random rotation offset
                roll_offset = self._generate_random_value(
                    self.cfg.orientation.angle_range[0], self.cfg.orientation.distribution
                )
                pitch_offset = self._generate_random_value(
                    self.cfg.orientation.angle_range[1], self.cfg.orientation.distribution
                )
                yaw_offset = self._generate_random_value(
                    self.cfg.orientation.angle_range[2], self.cfg.orientation.distribution
                )

                new_rot = (current_rot[0] + roll_offset, current_rot[1] + pitch_offset, current_rot[2] + yaw_offset)

                # Set rotation
                rotate_attr = light_prim.GetAttribute("xformOp:rotateXYZ")
                if not rotate_attr:
                    from pxr import Sdf

                    rotate_attr = light_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3)

                from pxr import Gf

                rotate_attr.Set(Gf.Vec3d(*new_rot))
        #                 logger.debug(f"Set {light_type} light orientation to {new_rot}")

        except Exception as e:
            logger.warning(f"Failed to randomize orientation for light {self.cfg.light_name}: {e}")

    def get_light_properties(self) -> dict:
        """Get current light properties for logging."""
        try:
            light_prim, light_path, light_type = self._get_light_prim(self.cfg.light_name)

            properties = {"light_path": light_path, "light_type": light_type}

            # Get intensity
            intensity_attr = light_prim.GetAttribute("inputs:intensity")
            if intensity_attr:
                properties["intensity"] = intensity_attr.Get()

            # Get color
            color_attr = light_prim.GetAttribute("inputs:color")
            if color_attr:
                properties["color"] = color_attr.Get()

            # Get position (skip for distant lights)
            if light_type != "distant":
                translate_attr = light_prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    properties["position"] = translate_attr.Get()

            # Get rotation
            rotate_attr = light_prim.GetAttribute("xformOp:rotateXYZ")
            if rotate_attr:
                properties["rotation"] = rotate_attr.Get()

            # Get light-specific properties
            if light_type in ["sphere", "disk"]:
                radius_attr = light_prim.GetAttribute("inputs:radius")
                if radius_attr:
                    properties["radius"] = radius_attr.Get()
            elif light_type == "cylinder":
                radius_attr = light_prim.GetAttribute("inputs:radius")
                length_attr = light_prim.GetAttribute("inputs:length")
                if radius_attr:
                    properties["radius"] = radius_attr.Get()
                if length_attr:
                    properties["length"] = length_attr.Get()

            return properties
        except Exception as e:
            logger.warning(f"Failed to get light properties: {e}")
            return {}

    def __call__(self) -> None:
        """Execute light randomization based on configuration."""
        did_update = False
        try:
            enabled_types = self._get_enabled_light_types()
            if not enabled_types:
                return

            if self.cfg.randomization_mode == "combined":
                did_update = self._apply_combined_randomization(enabled_types)
            elif self.cfg.randomization_mode == "intensity_only":
                if "intensity" in enabled_types:
                    self.randomize_intensity()
                    did_update = True
            elif self.cfg.randomization_mode == "color_only":
                if "color" in enabled_types:
                    self.randomize_color()
                    did_update = True
            elif self.cfg.randomization_mode == "position_only":
                if "position" in enabled_types:
                    self.randomize_position()
                    did_update = True
            elif self.cfg.randomization_mode == "orientation_only":
                if "orientation" in enabled_types:
                    self.randomize_orientation()
                    did_update = True
            else:
                raise ValueError(f"Unknown randomization mode: {self.cfg.randomization_mode}")

        except Exception as e:
            logger.error(f"Light randomization failed for {self.cfg.light_name}: {e}")
            raise
        else:
            if did_update:
                self._mark_visual_dirty()

    def _get_enabled_light_types(self) -> list[str]:
        """Get list of enabled light randomization types."""
        enabled = []
        if self.cfg.intensity and self.cfg.intensity.enabled:
            enabled.append("intensity")
        if self.cfg.color and self.cfg.color.enabled:
            enabled.append("color")
        if self.cfg.position and self.cfg.position.enabled:
            enabled.append("position")
        if self.cfg.orientation and self.cfg.orientation.enabled:
            enabled.append("orientation")
        return enabled

    def _apply_combined_randomization(self, enabled_types: list[str]) -> bool:
        """Apply all enabled randomization types."""
        updated = False
        if "intensity" in enabled_types:
            self.randomize_intensity()
            updated = True
        if "color" in enabled_types:
            self.randomize_color()
            updated = True
        if "position" in enabled_types:
            self.randomize_position()
            updated = True
        if "orientation" in enabled_types:
            self.randomize_orientation()
            updated = True

        return updated
