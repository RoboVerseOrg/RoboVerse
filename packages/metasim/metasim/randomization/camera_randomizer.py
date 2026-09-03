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
from metasim.randomization.core.isaacsim_adapter import IsaacSimAdapter
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
        horizontal_aperture_range: Horizontal aperture range (min, max) in cm
        focus_distance_range: Focus distance range (min, max) in meters
        clipping_range: Clipping plane ranges ((near_min, near_max), (far_min, far_max)) in meters
        distribution: Random sampling distribution
        enabled: Whether to apply intrinsics randomization
    """

    focal_length_range: tuple[float, float] | None = None
    fov_range: tuple[float, float] | None = None
    use_fov: bool = False
    horizontal_aperture_range: tuple[float, float] | None = None
    focus_distance_range: tuple[float, float] | None = None
    clipping_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraRandomCfg:
    """Camera randomization configuration.

    Attributes:
        camera_name: Name of camera to randomize (must exist in Handler)
        position: Position randomization configuration
        orientation: Orientation randomization configuration (mutually exclusive with look_at)
        look_at: Look-at target randomization configuration (mutually exclusive with orientation)
        intrinsics: Intrinsics randomization configuration

    Note:
        Orientation and look-at are mutually exclusive camera control modes:
        - orientation: Direct pitch/yaw/roll rotation (free camera)
        - look-at: Point camera at a target (orbit camera)
        If both are enabled, look-at takes precedence and orientation is skipped.
    """

    camera_name: str = "default_camera"
    position: CameraPositionRandomCfg | None = None
    orientation: CameraOrientationRandomCfg | None = None
    look_at: CameraLookAtRandomCfg | None = None
    intrinsics: CameraIntrinsicsRandomCfg | None = None


# =============================================================================
# Camera Randomizer Implementation
# =============================================================================


class CameraRandomizer(BaseRandomizerType):
    """Camera property randomizer.

    Responsibilities:
    - Modify camera properties (position, orientation/look-at, intrinsics)
    - NOT responsible for: Creating/deleting cameras

    Characteristics:
    - Accesses cameras through Handler API (handler.cameras, handler.scene.sensors)
    - Cameras are well-supported by IsaacLab
    - Hybrid support: uses render_handler

    Camera Control Modes:
    The randomizer supports two mutually exclusive camera control modes:

    1. Free Rotation Mode (orientation):
       - Direct control via pitch/yaw/roll deltas
       - Best for: First-person views, free-floating cameras
       - Example: Surveillance camera with small angular adjustments

    2. Look-at Mode (look_at):
       - Camera points at a target point in space
       - Best for: Orbit cameras, object-focused views
       - Example: Camera orbiting around a workspace center

    If both are configured and enabled, look-at takes precedence as it provides
    more intuitive control for most robotic scenarios.

    Usage:
        randomizer = CameraRandomizer(
            CameraRandomCfg(
                camera_name="main_camera",
                position=CameraPositionRandomCfg(
                    delta_range=[(-0.1, 0.1), (-0.1, 0.1), (0, 0)],
                    use_delta=True
                ),
                look_at=CameraLookAtRandomCfg(
                    look_at_delta=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
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
        self.adapter: IsaacSimAdapter | None = None
        self._original_positions: dict[str, tuple] = {}

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

        # Randomize position (always independent)
        if self.cfg.position and self.cfg.position.enabled:
            self._randomize_position(camera_cfg, camera_inst)

        # Orientation control: look-at takes precedence over orientation
        # These two modes are mutually exclusive as they both control camera direction
        if self.cfg.look_at and self.cfg.look_at.enabled:
            # Look-at mode: camera points at a target point
            self._randomize_look_at(camera_cfg, camera_inst)

            if self.cfg.orientation and self.cfg.orientation.enabled:
                logger.debug(
                    f"Camera '{self.cfg.camera_name}': look-at is enabled, "
                    f"orientation randomization will be skipped (mutually exclusive)"
                )
        elif self.cfg.orientation and self.cfg.orientation.enabled:
            # Free rotation mode: direct pitch/yaw/roll control
            self._randomize_orientation(camera_cfg, camera_inst)

        # Randomize intrinsics (always independent)
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

        try:
            # from omni.isaac.lab.utils import math
            from metasim.utils.math import quat_from_euler_xyz, quat_mul

            # Get current orientation
            current_rot = camera_inst.data.quat_w_world

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

            # Convert degrees to radians
            roll_rad = math.radians(roll_delta)
            pitch_rad = math.radians(pitch_delta)
            yaw_rad = math.radians(yaw_delta)

            # Create delta rotation quaternion
            delta_rotation = quat_from_euler_xyz(
                torch.tensor([roll_rad], device=self._actual_handler.device),
                torch.tensor([pitch_rad], device=self._actual_handler.device),
                torch.tensor([yaw_rad], device=self._actual_handler.device),
            )
            delta_rotation = delta_rotation.repeat(self._actual_handler.num_envs, 1)

            # Apply rotation delta
            new_rot = quat_mul(delta_rotation, current_rot)

            # Set new orientation
            camera_inst.set_world_poses(orientations=new_rot, convention="world")

        except Exception as e:
            logger.warning(f"Failed to randomize camera orientation: {e}")

    def _randomize_look_at(self, camera_cfg, camera_inst):
        """Randomize camera look-at target.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance
        """
        # Save original look-at
        if not hasattr(self, "_original_look_at"):
            self._original_look_at = {}
        if self.cfg.camera_name not in self._original_look_at:
            self._original_look_at[self.cfg.camera_name] = camera_cfg.look_at

        original_look_at = self._original_look_at[self.cfg.camera_name]

        if self.cfg.look_at.use_delta and self.cfg.look_at.look_at_delta:
            # Delta mode: small adjustments to look-at point
            new_look_at = tuple(
                original_look_at[i]
                + self._generate_random_value(self.cfg.look_at.look_at_delta[i], self.cfg.look_at.distribution)
                for i in range(3)
            )
        elif self.cfg.look_at.look_at_range:
            # Absolute mode: specify exact look-at point
            new_look_at = tuple(
                self._generate_random_value(r, self.cfg.look_at.distribution) for r in self.cfg.look_at.look_at_range
            )
        else:
            return

        # Update camera configuration
        camera_cfg.look_at = new_look_at

        # Get current position
        current_pos = camera_inst.data.pos_w[:1]

        # Apply new look-at
        look_at_tensor = torch.tensor(new_look_at, device=self._actual_handler.device).unsqueeze(0)
        look_at_tensor = look_at_tensor.repeat(self._actual_handler.num_envs, 1)

        camera_inst.set_world_poses_from_view(current_pos.repeat(self._actual_handler.num_envs, 1), look_at_tensor)

    def _randomize_intrinsics(self, camera_cfg):
        """Randomize camera intrinsics.

        Args:
            camera_cfg: Camera configuration
        """
        if not self.adapter:
            logger.debug("IsaacSimAdapter not available for intrinsics randomization")
            return

        try:
            from pxr import UsdGeom

            # Get camera prim path from instance
            camera_inst = self._actual_handler.scene.sensors[self.cfg.camera_name]
            camera_prim_path_pattern = camera_inst.cfg.prim_path

            # Get stage from adapter
            stage = self.adapter.stage

            # Randomize for each environment
            for env_idx in range(self._actual_handler.num_envs):
                # Construct proper prim path (handle both specific and pattern paths)
                if "/env_0/" in camera_prim_path_pattern:
                    # Specific path like "/World/envs/env_0/main_camera"
                    env_prim_path = camera_prim_path_pattern.replace("/env_0/", f"/env_{env_idx}/")
                elif "env_.*" in camera_prim_path_pattern:
                    # Pattern path like "/World/envs/env_.*/main_camera"
                    env_prim_path = camera_prim_path_pattern.replace("env_.*", f"env_{env_idx}")
                else:
                    # Fallback: assume single environment or shared camera
                    env_prim_path = camera_prim_path_pattern

                prim = stage.GetPrimAtPath(env_prim_path)
                if not prim or not prim.IsValid():
                    continue

                camera = UsdGeom.Camera(prim)
                if not camera:
                    continue

                # Randomize FOV or focal length
                if self.cfg.intrinsics.use_fov and self.cfg.intrinsics.fov_range:
                    new_fov = self._generate_random_value(
                        self.cfg.intrinsics.fov_range, self.cfg.intrinsics.distribution
                    )
                    fov_rad = new_fov * (math.pi / 180.0)

                    # Convert FOV to focal length
                    # For standard horizontal aperture (20.955mm)
                    aperture = 20.955
                    focal_length = aperture / (2.0 * math.tan(fov_rad / 2.0))

                    camera.CreateFocalLengthAttr().Set(focal_length)
                    camera_cfg.focal_length = focal_length / 10.0  # Convert mm to cm for config

                elif self.cfg.intrinsics.focal_length_range:
                    focal_length_cm = self._generate_random_value(
                        self.cfg.intrinsics.focal_length_range, self.cfg.intrinsics.distribution
                    )
                    focal_length_mm = focal_length_cm * 10.0
                    camera.CreateFocalLengthAttr().Set(focal_length_mm)
                    camera_cfg.focal_length = focal_length_cm

                # Randomize horizontal aperture
                if self.cfg.intrinsics.horizontal_aperture_range:
                    aperture_cm = self._generate_random_value(
                        self.cfg.intrinsics.horizontal_aperture_range, self.cfg.intrinsics.distribution
                    )
                    aperture_mm = aperture_cm * 10.0
                    camera.CreateHorizontalApertureAttr().Set(aperture_mm)

                # Randomize focus distance
                if self.cfg.intrinsics.focus_distance_range:
                    focus_distance = self._generate_random_value(
                        self.cfg.intrinsics.focus_distance_range, self.cfg.intrinsics.distribution
                    )
                    camera.CreateFocusDistanceAttr().Set(focus_distance * 100.0)  # Convert m to cm

                # Randomize clipping range
                if self.cfg.intrinsics.clipping_range:
                    near_clip = self._generate_random_value(
                        self.cfg.intrinsics.clipping_range[0], self.cfg.intrinsics.distribution
                    )
                    far_clip = self._generate_random_value(
                        self.cfg.intrinsics.clipping_range[1], self.cfg.intrinsics.distribution
                    )
                    from pxr import Gf

                    camera.CreateClippingRangeAttr().Set(
                        Gf.Vec2f(near_clip * 100.0, far_clip * 100.0)
                    )  # Convert m to cm

        except ImportError:
            logger.debug("USD modules not available for intrinsics randomization")
        except Exception as e:
            logger.warning(f"Failed to randomize camera intrinsics: {e}")

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
