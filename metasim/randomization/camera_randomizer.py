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
        position_delta_range: Relative delta ranges for micro-adjustments
        use_delta: Use delta (relative) mode instead of absolute
        distribution: Random sampling distribution
        enabled: Whether to apply position randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different random value (diverse views)
                 False: all envs get same random value (consistent relative view)
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    position_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same random value for all envs (consistent view)


@configclass
class CameraOrientationRandomCfg:
    """Camera orientation randomization configuration.

    Attributes:
        rotation_range: Absolute rotation ranges in degrees ((pitch_min, pitch_max), (yaw_min, yaw_max), (roll_min, roll_max))
        rotation_delta_range: Rotation delta ranges in degrees ((pitch_min, pitch_max), (yaw_min, yaw_max), (roll_min, roll_max))
        use_delta: Use delta (relative) mode instead of absolute
        distribution: Random sampling distribution
        enabled: Whether to apply orientation randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different random value (diverse views)
                 False: all envs get same random value (consistent relative view)
    """

    rotation_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    rotation_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same random value for all envs (consistent view)


@configclass
class CameraLookAtRandomCfg:
    """Camera look-at target randomization configuration.

    Attributes:
        look_at_range: Look-at point ranges ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        look_at_delta_range: Look-at delta ranges for micro-adjustments
        use_delta: Use delta (relative) mode instead of absolute look-at points
        distribution: Random sampling distribution
        enabled: Whether to apply look-at randomization
        per_env: Whether to generate different random values for each environment
                 True: each env gets different random value (diverse views)
                 False: all envs get same random value (consistent relative view)
    """

    look_at_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    look_at_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same random value for all envs (consistent view)


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
        per_env: Whether to generate different random values for each environment
                 True: each env gets different random intrinsics (diverse camera properties)
                 False: all envs get same random intrinsics (consistent camera properties)
    """

    focal_length_range: tuple[float, float] | None = None
    fov_range: tuple[float, float] | None = None
    use_fov: bool = False
    horizontal_aperture_range: tuple[float, float] | None = None
    focus_distance_range: tuple[float, float] | None = None
    clipping_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True
    per_env: bool = False  # Default: same random value for all envs


@configclass
class CameraRandomCfg:
    """Camera randomization configuration.

    Attributes:
        camera_name: Name of camera to randomize (must exist in Handler)
        position: Position randomization configuration
        orientation: Orientation randomization configuration (mutually exclusive with look_at)
        look_at: Look-at target randomization configuration (mutually exclusive with orientation)
        intrinsics: Intrinsics randomization configuration
        env_ids: Environment IDs to apply randomization (None = all)
                 Note: Only effective when per_env=True in sub-configs

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
    env_ids: list[int] | None = None


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
        self._original_look_at: dict[str, tuple] = {}
        self._original_orientation: dict[str, torch.Tensor] = {}  # Store original quaternions

    def __call__(self, env_ids: list[int] | None = None):
        """Execute camera randomization.

        Args:
            env_ids: Environment IDs to randomize. If None, uses self.cfg.env_ids.
                     If both are None, randomizes all environments.
        """
        # Use provided env_ids, or fall back to config, or all environments
        if env_ids is None:
            env_ids = self.cfg.env_ids
        if env_ids is None:
            env_ids = list(range(self._actual_handler.num_envs))

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
            self._randomize_position(camera_cfg, camera_inst, env_ids)

        # Orientation control: look-at takes precedence over orientation
        # These two modes are mutually exclusive as they both control camera direction
        if self.cfg.look_at and self.cfg.look_at.enabled:
            # Look-at mode: camera points at a target point
            self._randomize_look_at(camera_cfg, camera_inst, env_ids)

            if self.cfg.orientation and self.cfg.orientation.enabled:
                logger.debug(
                    f"Camera '{self.cfg.camera_name}': look-at is enabled, "
                    f"orientation randomization will be skipped (mutually exclusive)"
                )
        elif self.cfg.orientation and self.cfg.orientation.enabled:
            # Free rotation mode: direct pitch/yaw/roll control
            self._randomize_orientation(camera_cfg, camera_inst, env_ids)

        # Randomize intrinsics (always independent)
        if self.cfg.intrinsics and self.cfg.intrinsics.enabled:
            self._randomize_intrinsics(camera_cfg, env_ids)

        self._mark_visual_dirty()

    # -------------------------------------------------------------------------
    # Randomization Methods
    # -------------------------------------------------------------------------

    def _randomize_position(self, camera_cfg, camera_inst, env_ids):
        """Randomize camera position for specified environments.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance from Handler.scene.sensors
            env_ids: List of environment IDs to randomize
        """
        # Save original position
        if self.cfg.camera_name not in self._original_positions:
            self._original_positions[self.cfg.camera_name] = camera_cfg.pos

        original_pos = self._original_positions[self.cfg.camera_name]
        num_envs = self._actual_handler.num_envs

        # Get current camera poses for all environments
        current_pos_w, current_quat_w = camera_inst.data.pos_w.clone(), camera_inst.data.quat_w_ros.clone()

        # Generate random value(s) based on per_env setting (only for specified env_ids)
        if self.cfg.position.per_env:
            # Mode A: Different random value for each environment (diverse views)
            for env_id in env_ids:
                if self.cfg.position.use_delta and self.cfg.position.position_delta_range:
                    # Delta mode: small adjustments
                    new_pos = tuple(
                        original_pos[i]
                        + self._generate_random_value(
                            self.cfg.position.position_delta_range[i], self.cfg.position.distribution
                        )
                        for i in range(3)
                    )
                elif self.cfg.position.position_range:
                    # Absolute mode
                    new_pos = tuple(
                        self._generate_random_value(r, self.cfg.position.distribution)
                        for r in self.cfg.position.position_range
                    )
                else:
                    new_pos = original_pos

                # Update tensor for this env_id (local coords)
                new_pos_tensor = torch.tensor(new_pos, device=self._actual_handler.device)
                # Add env_origins offset for world positioning
                if hasattr(self._actual_handler.scene, "env_origins"):
                    new_pos_tensor = new_pos_tensor + self._actual_handler.scene.env_origins[env_id]
                current_pos_w[env_id] = new_pos_tensor
        else:
            # Mode B: Same random value for all specified environments (consistent relative view)
            if self.cfg.position.use_delta and self.cfg.position.position_delta_range:
                # Generate delta once
                delta = tuple(
                    self._generate_random_value(
                        self.cfg.position.position_delta_range[i], self.cfg.position.distribution
                    )
                    for i in range(3)
                )
                # Apply same delta
                new_pos = tuple(original_pos[i] + delta[i] for i in range(3))
            elif self.cfg.position.position_range:
                # Generate position once
                new_pos = tuple(
                    self._generate_random_value(r, self.cfg.position.distribution)
                    for r in self.cfg.position.position_range
                )
            else:
                new_pos = original_pos

            # Apply same position to all specified env_ids
            new_pos_tensor = torch.tensor(new_pos, device=self._actual_handler.device)
            for env_id in env_ids:
                if hasattr(self._actual_handler.scene, "env_origins"):
                    current_pos_w[env_id] = new_pos_tensor + self._actual_handler.scene.env_origins[env_id]
                else:
                    current_pos_w[env_id] = new_pos_tensor

        # Update camera configuration with local coordinates (not world coordinates!)
        # camera_cfg.pos should always store local coordinates for consistency
        if hasattr(self._actual_handler.scene, "env_origins"):
            # Convert world coords back to local coords: world - env_origin
            local_pos = current_pos_w[env_ids[0]] - self._actual_handler.scene.env_origins[env_ids[0]]
            camera_cfg.pos = tuple(local_pos.cpu().numpy())
        else:
            camera_cfg.pos = tuple(current_pos_w[env_ids[0]].cpu().numpy())

        # Compute look-at targets for all environments
        look_at_base = torch.tensor(camera_cfg.look_at, device=self._actual_handler.device)
        look_at_tensor = look_at_base.unsqueeze(0).repeat(num_envs, 1)
        if hasattr(self._actual_handler.scene, "env_origins"):
            look_at_tensor = look_at_tensor + self._actual_handler.scene.env_origins

        # Set updated poses
        camera_inst.set_world_poses_from_view(current_pos_w, look_at_tensor)

    def _randomize_orientation(self, camera_cfg, camera_inst, env_ids):
        """Randomize camera orientation for specified environments.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance
            env_ids: List of environment IDs to randomize
        """
        if not self.cfg.orientation.rotation_delta_range and not self.cfg.orientation.rotation_range:
            return

        try:
            from metasim.utils.math import quat_from_euler_xyz, quat_mul

            # Save original orientation on first call
            if self.cfg.camera_name not in self._original_orientation:
                current_rot = camera_inst.data.quat_w_world
                self._original_orientation[self.cfg.camera_name] = current_rot.clone()

            original_rot = self._original_orientation[self.cfg.camera_name]
            num_envs = self._actual_handler.num_envs

            # Get current orientations for all environments
            new_rot = camera_inst.data.quat_w_world.clone()

            # Generate rotation based on mode (only for specified env_ids)
            if self.cfg.orientation.use_delta and self.cfg.orientation.rotation_delta_range:
                # Delta mode: relative to original orientation
                if self.cfg.orientation.per_env:
                    # Mode A: Different delta for each environment
                    for env_id in env_ids:
                        pitch_delta = self._generate_random_value(
                            self.cfg.orientation.rotation_delta_range[0], self.cfg.orientation.distribution
                        )
                        yaw_delta = self._generate_random_value(
                            self.cfg.orientation.rotation_delta_range[1], self.cfg.orientation.distribution
                        )
                        roll_delta = self._generate_random_value(
                            self.cfg.orientation.rotation_delta_range[2], self.cfg.orientation.distribution
                        )

                        roll_rad = math.radians(roll_delta)
                        pitch_rad = math.radians(pitch_delta)
                        yaw_rad = math.radians(yaw_delta)

                        # Create delta rotation for this env
                        delta_rotation = quat_from_euler_xyz(
                            torch.tensor([roll_rad], device=self._actual_handler.device),
                            torch.tensor([pitch_rad], device=self._actual_handler.device),
                            torch.tensor([yaw_rad], device=self._actual_handler.device),
                        )

                        # Apply to original orientation
                        new_rot[env_id] = quat_mul(delta_rotation, original_rot[env_id].unsqueeze(0)).squeeze(0)
                else:
                    # Mode B: Same delta for all environments
                    pitch_delta = self._generate_random_value(
                        self.cfg.orientation.rotation_delta_range[0], self.cfg.orientation.distribution
                    )
                    yaw_delta = self._generate_random_value(
                        self.cfg.orientation.rotation_delta_range[1], self.cfg.orientation.distribution
                    )
                    roll_delta = self._generate_random_value(
                        self.cfg.orientation.rotation_delta_range[2], self.cfg.orientation.distribution
                    )

                    roll_rad = math.radians(roll_delta)
                    pitch_rad = math.radians(pitch_delta)
                    yaw_rad = math.radians(yaw_delta)

                    # Create single delta rotation
                    delta_rotation = quat_from_euler_xyz(
                        torch.tensor([roll_rad], device=self._actual_handler.device),
                        torch.tensor([pitch_rad], device=self._actual_handler.device),
                        torch.tensor([yaw_rad], device=self._actual_handler.device),
                    )

                    # Apply same delta to all specified env_ids
                    for env_id in env_ids:
                        new_rot[env_id] = quat_mul(delta_rotation, original_rot[env_id].unsqueeze(0)).squeeze(0)
            elif self.cfg.orientation.rotation_range:
                # Absolute mode: set absolute orientation
                if self.cfg.orientation.per_env:
                    # Mode A: Different absolute orientation for each environment
                    for env_id in env_ids:
                        pitch = self._generate_random_value(
                            self.cfg.orientation.rotation_range[0], self.cfg.orientation.distribution
                        )
                        yaw = self._generate_random_value(
                            self.cfg.orientation.rotation_range[1], self.cfg.orientation.distribution
                        )
                        roll = self._generate_random_value(
                            self.cfg.orientation.rotation_range[2], self.cfg.orientation.distribution
                        )

                        roll_rad = math.radians(roll)
                        pitch_rad = math.radians(pitch)
                        yaw_rad = math.radians(yaw)

                        abs_rotation = quat_from_euler_xyz(
                            torch.tensor([roll_rad], device=self._actual_handler.device),
                            torch.tensor([pitch_rad], device=self._actual_handler.device),
                            torch.tensor([yaw_rad], device=self._actual_handler.device),
                        )
                        new_rot[env_id] = abs_rotation.squeeze(0)
                else:
                    # Mode B: Same absolute orientation for all environments
                    pitch = self._generate_random_value(
                        self.cfg.orientation.rotation_range[0], self.cfg.orientation.distribution
                    )
                    yaw = self._generate_random_value(
                        self.cfg.orientation.rotation_range[1], self.cfg.orientation.distribution
                    )
                    roll = self._generate_random_value(
                        self.cfg.orientation.rotation_range[2], self.cfg.orientation.distribution
                    )

                    roll_rad = math.radians(roll)
                    pitch_rad = math.radians(pitch)
                    yaw_rad = math.radians(yaw)

                    abs_rotation = quat_from_euler_xyz(
                        torch.tensor([roll_rad], device=self._actual_handler.device),
                        torch.tensor([pitch_rad], device=self._actual_handler.device),
                        torch.tensor([yaw_rad], device=self._actual_handler.device),
                    )

                    # Apply same orientation to all specified env_ids
                    for env_id in env_ids:
                        new_rot[env_id] = abs_rotation.squeeze(0)
            else:
                return

            # Set new orientation for all environments (only modified ones changed)
            camera_inst.set_world_poses(orientations=new_rot, convention="world")

        except Exception as e:
            logger.warning(f"Failed to randomize camera orientation: {e}")

    def _randomize_look_at(self, camera_cfg, camera_inst, env_ids):
        """Randomize camera look-at target for specified environments.

        Args:
            camera_cfg: Camera configuration
            camera_inst: Camera instance
            env_ids: List of environment IDs to randomize
        """
        # Save original look-at
        if not hasattr(self, "_original_look_at"):
            self._original_look_at = {}
        if self.cfg.camera_name not in self._original_look_at:
            self._original_look_at[self.cfg.camera_name] = camera_cfg.look_at

        original_look_at = self._original_look_at[self.cfg.camera_name]
        num_envs = self._actual_handler.num_envs

        # Get current position
        current_pos = camera_inst.data.pos_w.clone()

        # Get current look-at (we'll modify only specified env_ids)
        # Start with current look-at from camera_cfg, expanded to all envs
        look_at_base = torch.tensor(camera_cfg.look_at, device=self._actual_handler.device)
        look_at_tensor = look_at_base.unsqueeze(0).repeat(num_envs, 1)
        if hasattr(self._actual_handler.scene, "env_origins"):
            # Remove env_origins first to work in local coords
            look_at_local = look_at_tensor - self._actual_handler.scene.env_origins
        else:
            look_at_local = look_at_tensor

        # Generate random value(s) based on per_env setting (only for specified env_ids)
        if self.cfg.look_at.per_env:
            # Mode A: Different random value for each environment (diverse views)
            for env_id in env_ids:
                if self.cfg.look_at.use_delta and self.cfg.look_at.look_at_delta_range:
                    # Delta mode: small adjustments to look-at point
                    new_look_at = tuple(
                        original_look_at[i]
                        + self._generate_random_value(
                            self.cfg.look_at.look_at_delta_range[i], self.cfg.look_at.distribution
                        )
                        for i in range(3)
                    )
                elif self.cfg.look_at.look_at_range:
                    # Absolute mode: specify exact look-at point
                    new_look_at = tuple(
                        self._generate_random_value(r, self.cfg.look_at.distribution)
                        for r in self.cfg.look_at.look_at_range
                    )
                else:
                    new_look_at = original_look_at

                # Update this env's look-at (local coords)
                look_at_local[env_id] = torch.tensor(new_look_at, device=self._actual_handler.device)
        else:
            # Mode B: Same random value for all specified environments (consistent relative view)
            if self.cfg.look_at.use_delta and self.cfg.look_at.look_at_delta_range:
                # Generate delta once
                delta = tuple(
                    self._generate_random_value(self.cfg.look_at.look_at_delta_range[i], self.cfg.look_at.distribution)
                    for i in range(3)
                )
                # Apply same delta
                new_look_at = tuple(original_look_at[i] + delta[i] for i in range(3))
            elif self.cfg.look_at.look_at_range:
                # Generate look_at once
                new_look_at = tuple(
                    self._generate_random_value(r, self.cfg.look_at.distribution)
                    for r in self.cfg.look_at.look_at_range
                )
            else:
                new_look_at = original_look_at

            # Apply same look-at to all specified env_ids
            new_look_at_tensor = torch.tensor(new_look_at, device=self._actual_handler.device)
            for env_id in env_ids:
                look_at_local[env_id] = new_look_at_tensor

        # Update camera configuration (use first randomized env's value as representative)
        camera_cfg.look_at = tuple(look_at_local[env_ids[0]].cpu().numpy())

        # Convert back to world coords and apply
        if hasattr(self._actual_handler.scene, "env_origins"):
            look_at_tensor = look_at_local + self._actual_handler.scene.env_origins
        else:
            look_at_tensor = look_at_local

        camera_inst.set_world_poses_from_view(current_pos, look_at_tensor)

    def _randomize_intrinsics(self, camera_cfg, env_ids):
        """Randomize camera intrinsics for specified environments.

        Args:
            camera_cfg: Camera configuration
            env_ids: List of environment IDs to randomize
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

            # Generate random values (once if per_env=False, per-env if per_env=True)
            if not self.cfg.intrinsics.per_env:
                # Mode A: Same random value for all specified environments
                shared_fov = None
                shared_focal_length = None
                shared_aperture = None
                shared_focus_distance = None
                shared_clipping = None

                if self.cfg.intrinsics.use_fov and self.cfg.intrinsics.fov_range:
                    shared_fov = self._generate_random_value(
                        self.cfg.intrinsics.fov_range, self.cfg.intrinsics.distribution
                    )
                elif self.cfg.intrinsics.focal_length_range:
                    shared_focal_length = self._generate_random_value(
                        self.cfg.intrinsics.focal_length_range, self.cfg.intrinsics.distribution
                    )

                if self.cfg.intrinsics.horizontal_aperture_range:
                    shared_aperture = self._generate_random_value(
                        self.cfg.intrinsics.horizontal_aperture_range, self.cfg.intrinsics.distribution
                    )

                if self.cfg.intrinsics.focus_distance_range:
                    shared_focus_distance = self._generate_random_value(
                        self.cfg.intrinsics.focus_distance_range, self.cfg.intrinsics.distribution
                    )

                if self.cfg.intrinsics.clipping_range:
                    shared_clipping = (
                        self._generate_random_value(
                            self.cfg.intrinsics.clipping_range[0], self.cfg.intrinsics.distribution
                        ),
                        self._generate_random_value(
                            self.cfg.intrinsics.clipping_range[1], self.cfg.intrinsics.distribution
                        ),
                    )

            # Apply to each specified environment
            for env_idx in env_ids:
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

                # Mode B: Different random value for each environment
                if self.cfg.intrinsics.per_env:
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
                else:
                    # Mode A: Use shared random values
                    if shared_fov is not None:
                        fov_rad = shared_fov * (math.pi / 180.0)
                        aperture = 20.955
                        focal_length = aperture / (2.0 * math.tan(fov_rad / 2.0))
                        camera.CreateFocalLengthAttr().Set(focal_length)
                        camera_cfg.focal_length = focal_length / 10.0

                    elif shared_focal_length is not None:
                        focal_length_mm = shared_focal_length * 10.0
                        camera.CreateFocalLengthAttr().Set(focal_length_mm)
                        camera_cfg.focal_length = shared_focal_length

                    if shared_aperture is not None:
                        aperture_mm = shared_aperture * 10.0
                        camera.CreateHorizontalApertureAttr().Set(aperture_mm)

                    if shared_focus_distance is not None:
                        camera.CreateFocusDistanceAttr().Set(shared_focus_distance * 100.0)

                    if shared_clipping is not None:
                        from pxr import Gf

                        camera.CreateClippingRangeAttr().Set(
                            Gf.Vec2f(shared_clipping[0] * 100.0, shared_clipping[1] * 100.0)
                        )

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
