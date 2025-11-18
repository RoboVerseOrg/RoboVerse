"""Camera Randomizer - Property editor for camera properties.

The CameraRandomizer modifies properties of existing cameras.
Cameras are Static Objects (Handler-created) and accessed through Handler API.

Key features:
- Position randomization
- Orientation randomization
- Look-at target randomization
- Intrinsics randomization (focal length, FOV, etc.)
- Supports Hybrid simulation (uses render_handler)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass

# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class CameraPositionRandomCfg:
    """Camera position randomization configuration.

    Attributes:
        position_range: Absolute position ranges ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        delta_range: Relative delta ranges for micro-adjustments
        use_delta: Use delta (relative) mode instead of absolute
        distribution: Random sampling distribution
        enabled: Whether to apply position randomization
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraOrientationRandomCfg:
    """Camera orientation randomization configuration.

    Attributes:
        rotation_delta: Rotation delta ranges in degrees ((pitch_min, pitch_max), (yaw_min, yaw_max), (roll_min, roll_max))
        distribution: Random sampling distribution
        enabled: Whether to apply orientation randomization
    """

    rotation_delta: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraLookAtRandomCfg:
    """Camera look-at target randomization configuration.

    Attributes:
        look_at_range: Look-at point ranges ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        look_at_delta: Look-at delta ranges for micro-adjustments
        use_delta: Use delta (relative) mode instead of absolute look-at points
        distribution: Random sampling distribution
        enabled: Whether to apply look-at randomization
    """

    look_at_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    look_at_delta: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraIntrinsicsRandomCfg:
    """Camera intrinsics randomization configuration.

    Attributes:
        focal_length_range: Focal length range (min, max) in cm
        fov_range: Field of view range (min, max) in degrees (alternative to focal_length)
        use_fov: Use FOV instead of focal length
        distribution: Random sampling distribution
        enabled: Whether to apply intrinsics randomization
    """

    focal_length_range: tuple[float, float] | None = None
    fov_range: tuple[float, float] | None = None
    use_fov: bool = False
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraImageRandomCfg:
    """Camera image properties randomization configuration.

    Attributes:
        width_range: Image width range (min, max) in pixels
        height_range: Image height range (min, max) in pixels
        aspect_ratio_range: Aspect ratio range (min, max)
        use_aspect_ratio: Use aspect ratio instead of independent width/height
        distribution: Random sampling distribution
        enabled: Whether to apply image properties randomization
    """

    width_range: tuple[int, int] | None = None
    height_range: tuple[int, int] | None = None
    aspect_ratio_range: tuple[float, float] | None = None
    use_aspect_ratio: bool = False
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraRandomCfg:
    """Camera randomization configuration.

    Attributes:
        camera_name: Name of camera to randomize (must exist in Handler)
        position: Position randomization configuration
        orientation: Orientation randomization configuration
        intrinsics: Intrinsics randomization configuration
    """

    camera_name: str = "default_camera"
    position: CameraPositionRandomCfg | None = None
    orientation: CameraOrientationRandomCfg | None = None
    look_at: CameraLookAtRandomCfg | None = None
    intrinsics: CameraIntrinsicsRandomCfg | None = None
    image: CameraImageRandomCfg | None = None


# =============================================================================
# Camera Randomizer Implementation
# =============================================================================


class CameraRandomizer(BaseRandomizerType):
    """Camera property randomizer.

    Responsibilities:
    - Modify camera properties (position, orientation, intrinsics)
    - NOT responsible for: Creating/deleting cameras

    Characteristics:
    - Accesses cameras through Handler API (handler.cameras, handler.scene.sensors)
    - Cameras are well-supported by IsaacLab
    - Hybrid support: uses render_handler

    Usage:
        randomizer = CameraRandomizer(
            CameraRandomCfg(
                camera_name="main_camera",
                position=CameraPositionRandomCfg(
                    delta_range=[(-0.1, 0.1), (-0.1, 0.1), (0, 0)],
                    use_delta=True
                )
            ),
            seed=42
        )
        randomizer.bind_handler(handler)
        randomizer()  # Apply camera randomization
    """

    REQUIRES_HANDLER = "render"  # Use render_handler for Hybrid

    def __init__(self, cfg: CameraRandomCfg, seed: int | None = None):
        """Initialize camera randomizer.

        Args:
            cfg: Camera randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.cfg = cfg
        self._original_positions: dict[str, tuple] = {}

    def bind_handler(self, handler):
        """Bind handler.

        Args:
            handler: SimHandler instance (automatically uses render_handler for Hybrid)
        """
        super().bind_handler(handler)

    def __call__(self):
        """Execute camera randomization."""
        # Find camera in Handler
        camera_cfg = None
        for cam in self._actual_handler.cameras:
            if cam.name == self.cfg.camera_name:
                camera_cfg = cam
                break

        if not camera_cfg:
            logger.error(f"Camera '{self.cfg.camera_name}' not found in Handler.cameras")
            return

        # Get camera instance from Handler.scene.sensors
        try:
            camera_inst = self._actual_handler.scene.sensors[self.cfg.camera_name]
        except (AttributeError, KeyError):
            logger.error(f"Camera '{self.cfg.camera_name}' not found in Handler.scene.sensors")
            return

        # Randomize position
        if self.cfg.position and self.cfg.position.enabled:
            self._randomize_position(camera_cfg, camera_inst)

        # Randomize orientation
        if self.cfg.orientation and self.cfg.orientation.enabled:
            self._randomize_orientation(camera_cfg, camera_inst)

        # Randomize intrinsics
        if self.cfg.intrinsics and self.cfg.intrinsics.enabled:
            self._randomize_intrinsics(camera_cfg)

        self._mark_visual_dirty()

    # -------------------------------------------------------------------------
    # Randomization Methods
    # -------------------------------------------------------------------------

    def _randomize_position(self, camera_cfg, camera_inst):
        """Randomize camera position.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance from Handler.scene.sensors
        """
        # Save original position
        if self.cfg.camera_name not in self._original_positions:
            self._original_positions[self.cfg.camera_name] = camera_cfg.pos

        original_pos = self._original_positions[self.cfg.camera_name]

        if self.cfg.position.use_delta and self.cfg.position.delta_range:
            # Delta mode: small adjustments
            new_pos = tuple(
                original_pos[i]
                + self._generate_random_value(self.cfg.position.delta_range[i], self.cfg.position.distribution)
                for i in range(3)
            )
        elif self.cfg.position.position_range:
            # Absolute mode
            new_pos = tuple(
                self._generate_random_value(r, self.cfg.position.distribution) for r in self.cfg.position.position_range
            )
        else:
            return

        # Update camera configuration
        camera_cfg.pos = new_pos

        # Update camera instance
        position_tensor = torch.tensor(new_pos, device=self._actual_handler.device).unsqueeze(0)
        position_tensor = position_tensor.repeat(self._actual_handler.num_envs, 1)
        look_at_tensor = torch.tensor(camera_cfg.look_at, device=self._actual_handler.device).unsqueeze(0)
        look_at_tensor = look_at_tensor.repeat(self._actual_handler.num_envs, 1)

        camera_inst.set_world_poses_from_view(position_tensor, look_at_tensor)

    def _randomize_orientation(self, camera_cfg, camera_inst):
        """Randomize camera orientation.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance
        """
        if not self.cfg.orientation.rotation_delta:
            return

        # Generate random rotation deltas (in degrees)
        pitch_delta = self._generate_random_value(
            self.cfg.orientation.rotation_delta[0], self.cfg.orientation.distribution
        )
        yaw_delta = self._generate_random_value(
            self.cfg.orientation.rotation_delta[1], self.cfg.orientation.distribution
        )
        roll_delta = self._generate_random_value(
            self.cfg.orientation.rotation_delta[2], self.cfg.orientation.distribution
        )

        # This would require modifying the camera's orientation
        # For now, log a warning that this needs Handler support
        logger.warning("Camera orientation randomization not yet fully implemented")

    def _randomize_intrinsics(self, camera_cfg):
        """Randomize camera intrinsics.

        Args:
            camera_cfg: Camera configuration
        """
        if self.cfg.intrinsics.use_fov and self.cfg.intrinsics.fov_range:
            # FOV mode
            new_fov = self._generate_random_value(self.cfg.intrinsics.fov_range, self.cfg.intrinsics.distribution)
            # Convert FOV to focal length
            # focal_length = (sensor_width / 2) / tan(FOV / 2)
            # For standard sensor (24mm): focal_length (mm) = 12 / tan(FOV_rad / 2)
            fov_rad = new_fov * (math.pi / 180.0)
            camera_cfg.focal_length = 1.2 / math.tan(fov_rad / 2.0)  # Convert to cm

        elif self.cfg.intrinsics.focal_length_range:
            # Focal length mode
            camera_cfg.focal_length = self._generate_random_value(
                self.cfg.intrinsics.focal_length_range, self.cfg.intrinsics.distribution
            )

        # Note: Changing intrinsics at runtime may require camera recreation
        logger.warning("Camera intrinsics randomization changes may not take effect until camera recreation")

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
