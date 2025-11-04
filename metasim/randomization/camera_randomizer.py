from __future__ import annotations

import math
import random
import torch
from typing import Any, Literal

from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass
from metasim.utils import math


@configclass
class CameraPositionRandomCfg:
    """Configuration for camera position randomization.

    Args:
        position_range: Position ranges as ((x_min,x_max), (y_min,y_max), (z_min,z_max)) for absolute positioning
        delta_range: Delta ranges as ((dx_min,dx_max), (dy_min,dy_max), (dz_min,dz_max)) for relative micro-adjustments
        use_delta: Whether to use delta-based (micro-adjustment) mode instead of absolute positioning
        distribution: Type of distribution for random sampling
        enabled: Whether to apply position randomization
    """

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_delta: bool = True  # Default to micro-adjustment mode
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraOrientationRandomCfg:
    """Configuration for camera orientation (rotation) randomization.

    Args:
        rotation_delta: Rotation delta ranges as ((pitch_min,pitch_max), (yaw_min,yaw_max), (roll_min,roll_max)) in degrees
        distribution: Type of distribution for random sampling
        enabled: Whether to apply orientation randomization
    """

    rotation_delta: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraLookAtRandomCfg:
    """Configuration for camera look-at target randomization.

    Args:
        look_at_range: Look-at point ranges as ((x_min,x_max), (y_min,y_max), (z_min,z_max)) for absolute targeting
        look_at_delta: Look-at delta ranges as ((dx_min,dx_max), (dy_min,dy_max), (dz_min,dz_max)) for relative micro-adjustments
        spherical_range: Spherical coordinate ranges as ((radius_min,radius_max), (theta_min,theta_max), (phi_min,phi_max))
        use_spherical: Whether to use spherical coordinates instead of direct look_at randomization
        use_delta: Whether to use delta-based (micro-adjustment) mode instead of absolute look-at points
        distribution: Type of distribution for random sampling
        enabled: Whether to apply look-at randomization
    """

    look_at_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    look_at_delta: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    spherical_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    use_spherical: bool = False
    use_delta: bool = True  # Default to micro-adjustment mode
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraIntrinsicsRandomCfg:
    """Configuration for camera intrinsics randomization.

    Args:
        focal_length_range: Range for focal length randomization (min, max) in cm
        horizontal_aperture_range: Range for horizontal aperture randomization (min, max) in cm
        focus_distance_range: Range for focus distance randomization (min, max) in m
        clipping_range: Range for clipping distances randomization ((near_min,near_max), (far_min,far_max)) in m
        fov_range: Range for field of view randomization (min, max) in degrees (alternative to focal_length)
        use_fov: Whether to use FOV instead of focal length for randomization
        distribution: Type of distribution for random sampling
        enabled: Whether to apply intrinsics randomization
    """

    focal_length_range: tuple[float, float] | None = None
    horizontal_aperture_range: tuple[float, float] | None = None
    focus_distance_range: tuple[float, float] | None = None
    clipping_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    fov_range: tuple[float, float] | None = None
    use_fov: bool = False
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class CameraImageRandomCfg:
    """Configuration for camera image properties randomization.

    Args:
        width_range: Range for image width randomization (min, max) in pixels
        height_range: Range for image height randomization (min, max) in pixels
        aspect_ratio_range: Range for aspect ratio randomization (min, max)
        use_aspect_ratio: Whether to use aspect ratio instead of independent width/height
        distribution: Type of distribution for random sampling
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
    """Configuration for camera randomization.

    Args:
        camera_name: Name of the camera to randomize
        position: Position randomization configuration
        orientation: Orientation (rotation) randomization configuration
        look_at: Look-at target randomization configuration
        intrinsics: Intrinsics randomization configuration
        image: Image properties randomization configuration
        randomization_mode: How to apply randomization
    """

    camera_name: str = "default_camera"
    env_ids: list[int] | None = None
    position: CameraPositionRandomCfg | None = None
    orientation: CameraOrientationRandomCfg | None = None
    init_position: torch.Tensor | None = None
    init_look_at: torch.Tensor | None = None
    look_at: CameraLookAtRandomCfg | None = None
    intrinsics: CameraIntrinsicsRandomCfg | None = None
    image: CameraImageRandomCfg | None = None
    randomization_mode: Literal[
        "combined", "position_only", "orientation_only", "look_at_only", "intrinsics_only", "image_only"
    ] = "combined"


class CameraRandomizer(BaseRandomizerType):
    """Camera randomizer for domain randomization.

    This randomizer can modify camera position, orientation, and intrinsic parameters
    to provide visual domain randomization for training robust vision models.
    """

    def __init__(self, cfg: CameraRandomCfg, seed: int | None = None, device: torch.device | str = "cpu"):
        """Initialize camera randomizer.

        Args:
            cfg: Camera randomization configuration
            seed: Random seed for reproducible randomization
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Setup deterministic random number generator
        if seed is not None:
            # Create unique seed for this camera
            camera_seed = seed + sum(ord(c) for c in cfg.camera_name)
            self._rng = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(camera_seed)

        else:
            self._rng = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")

        self.handler = None
        self.current_pos = self.cfg.init_position
        self.current_look_at = self.cfg.init_look_at

    def bind_handler(self, handler):
        """Bind simulation handler."""
        self.handler = handler
    
    def _get_env_ids(self) -> list[int]:
        """Get environment IDs to operate on."""
        return self.cfg.env_ids or list(range(self.handler.num_envs))


    def __call__(self, env_ids: list[int] | None = None):
        """Apply camera randomization."""
        if self.handler is None:
            logger.warning("Camera randomizer not bound to handler, skipping randomization")
            return

        try:
            camera_prim = self._get_camera_prim()
            if camera_prim is None:
                logger.warning(f"Camera '{self.cfg.camera_name}' not found in scene")
                return

            # Apply randomization based on mode - clean dispatch table
            mode_dispatch = {
                "combined": self._randomize_all,
                "position_only": self._randomize_position,
                "orientation_only": self._randomize_orientation,
                "look_at_only": self._randomize_look_at,
                "intrinsics_only": self._randomize_intrinsics,
                "image_only": self._randomize_image_properties,
            }
            
            env_ids = env_ids if env_ids is not None else self._get_env_ids()

            randomize_func = mode_dispatch.get(self.cfg.randomization_mode)
            if randomize_func:
                randomize_func(camera_prim, env_ids)
            else:
                logger.warning(f"Unknown randomization mode: {self.cfg.randomization_mode}, using combined")
                self._randomize_all(camera_prim, env_ids)

        except Exception as e:
            logger.error(f"Camera randomization failed for '{self.cfg.camera_name}': {e}")

    def _get_camera_prim(self):
        """Get camera prim from scene."""
        try:
            camera_inst = self.handler.scene.sensors[self.cfg.camera_name]
            return camera_inst.cfg.prim_path

            return None
        except Exception as e:
            logger.error(f"Error finding camera '{self.cfg.camera_name}': {e}")
            return None

    def _randomize_all(self, camera_prim, env_ids: list[int]):
        """Apply all enabled randomization types in proper order to avoid conflicts."""
        # Apply position first (if enabled)
        if self.cfg.position and self.cfg.position.enabled:
            self._randomize_position(env_ids)

        # Apply rotation randomization - choose ONE rotation method to avoid conflicts:
        # Priority: look_at > orientation (look_at overrides orientation for more predictable behavior)
        if self.cfg.look_at and self.cfg.look_at.enabled:
            self._randomize_look_at(env_ids)
        elif self.cfg.orientation and self.cfg.orientation.enabled:
            self._randomize_orientation(env_ids)

        # Apply camera properties (independent of transform)
        if self.cfg.intrinsics and self.cfg.intrinsics.enabled:
            self._randomize_intrinsics(camera_prim, env_ids)
        if self.cfg.image and self.cfg.image.enabled:
            self._randomize_image_properties(camera_prim, env_ids)

    def _randomize_position(self, env_ids: list[int]):    
        """Randomize camera position ONLY (independent of orientation)."""
        if not self.cfg.position or not self.cfg.position.enabled:
            return

        try:
            camera_inst = self.handler.scene.sensors[self.cfg.camera_name]
            original_pos = self.cfg.init_position

            if self.cfg.position.use_delta and self.cfg.position.delta_range:
                # Delta-based micro-adjustment mode (默认)
                delta_range = self.cfg.position.delta_range
                dx = self._sample_value(delta_range[0], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)
                dy = self._sample_value(delta_range[1], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)
                dz = self._sample_value(delta_range[2], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)

                # Apply delta to current position
                new_pos = original_pos.clone()
                new_pos[env_ids, 0] += dx
                new_pos[env_ids, 1] += dy
                new_pos[env_ids, 2] += dz

            elif self.cfg.position.position_range:
                # Absolute positioning mode
                position_range = self.cfg.position.position_range
                new_x = self._sample_value(position_range[0], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)
                new_y = self._sample_value(position_range[1], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)
                new_z = self._sample_value(position_range[2], self.cfg.position.distribution, size=len(env_ids)).to(original_pos.device)
                
                new_pos = original_pos
                new_pos[env_ids, 0] = new_x
                new_pos[env_ids, 1] = new_y
                new_pos[env_ids, 2] = new_z

            else:
                logger.warning(f"No position range configured for camera '{self.cfg.camera_name}'")
                return

            # Apply new position only, preserving existing rotation
            camera_inst.set_world_poses(positions=new_pos[env_ids], env_ids=env_ids, convention='world')
            self.current_pos = new_pos

        except Exception as e:
            logger.error(f"Failed to randomize camera position: {e}")

    def _randomize_orientation(self, env_ids: list[int]):
        """Randomize camera orientation by adding rotation deltas (independent of position and look-at)."""
        if not self.cfg.orientation or not self.cfg.orientation.enabled:
            return

        try:
            camera_inst = self.handler.scene.sensors[self.cfg.camera_name]
            current_rot = camera_inst.data.quat_w_world[env_ids]

            # Sample rotation deltas
            rotation_delta = self.cfg.orientation.rotation_delta
            delta_pitch = self._sample_value(rotation_delta[0], self.cfg.orientation.distribution, size=len(env_ids)).to(current_rot.device)
            delta_yaw = self._sample_value(rotation_delta[1], self.cfg.orientation.distribution, size=len(env_ids)).to(current_rot.device)
            delta_roll = self._sample_value(rotation_delta[2], self.cfg.orientation.distribution, size=len(env_ids)).to(current_rot.device)

            delta_rotation = math.quat_from_euler_xyz(delta_roll, delta_pitch, delta_yaw)  # Note order: roll, pitch, yaw
            new_rot = math.quat_mul(delta_rotation, current_rot)
            
            camera_inst.set_world_poses(orientations=new_rot, env_ids=env_ids, convention='world')

        except Exception as e:
            logger.error(f"Failed to randomize camera orientation: {e}")

    def _randomize_look_at(self, env_ids: list[int]):
        """Randomize camera by moving it in a spherical orbit around the original look-at target."""
        if not self.cfg.look_at or not self.cfg.look_at.enabled:
            return

        try:
            camera_inst = self.handler.scene.sensors[self.cfg.camera_name]
            current_pos = self.current_pos[env_ids]
            original_look_at = self.cfg.init_look_at[env_ids]
            
            if self.cfg.look_at.use_spherical and self.cfg.look_at.spherical_range:
                assert False, "Spherical look-at randomization not implemented yet"

            # Default look-at behavior: small orbital movement around ORIGINAL position
            if self.cfg.look_at.use_delta and self.cfg.look_at.look_at_delta:

                delta_range = self.cfg.look_at.look_at_delta
                dx = self._sample_value(delta_range[0], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)
                dy = self._sample_value(delta_range[1], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)
                dz = self._sample_value(delta_range[2], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)

                # Apply delta to current position
                new_look_at = original_look_at.clone()
                new_look_at[:, 0] += dx
                new_look_at[:, 1] += dy
                new_look_at[:, 2] += dz

            elif self.cfg.look_at.look_at_range:
                # Absolute positioning mode
                look_at_range = self.cfg.look_at.look_at_range
                new_x = self._sample_value(look_at_range[0], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)
                new_y = self._sample_value(look_at_range[1], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)
                new_z = self._sample_value(look_at_range[2], self.cfg.look_at.distribution, size=len(env_ids)).to(original_look_at.device)
                
                new_look_at = original_look_at
                new_look_at[:, 0] = new_x
                new_look_at[:, 1] = new_y
                new_look_at[:, 2] = new_z

            else:
                logger.warning(f"No position range configured for camera '{self.cfg.camera_name}'")
                return

            # Apply new position only, preserving existing rotation
            camera_inst.set_world_poses_from_view(current_pos, new_look_at, env_ids=env_ids)

        except Exception as e:
            logger.error(f"Failed to randomize camera look-at: {e}")
            import traceback

            logger.error(traceback.format_exc())


    def _randomize_intrinsics(self, camera_prim, env_ids: list[int]):
        """Randomize camera intrinsics."""
        if not self.cfg.intrinsics or not self.cfg.intrinsics.enabled:
            return

        try:
            from pxr import UsdGeom
            
            for env_id in env_ids:
                prim = camera_prim.replace("env_.*", f"env_{env_id}")
                camera = UsdGeom.Camera(camera_prim)
                if not camera:
                    return

                if self.cfg.intrinsics.use_fov and self.cfg.intrinsics.fov_range:
                    self._randomize_fov(camera)
                elif self.cfg.intrinsics.focal_length_range:
                    self._randomize_focal_length(camera)

                if self.cfg.intrinsics.horizontal_aperture_range:
                    self._randomize_aperture(camera)

                if self.cfg.intrinsics.focus_distance_range:
                    self._randomize_focus_distance(camera)

                if self.cfg.intrinsics.clipping_range:
                    self._randomize_clipping_range(camera)

        except Exception as e:
            logger.error(f"Failed to randomize camera intrinsics: {e}")

    def _randomize_fov(self, camera):
        """Randomize field of view by adjusting focal length and aperture."""
        fov_range = self.cfg.intrinsics.fov_range
        if not fov_range:
            return

        # Sample new FOV
        new_fov = self._sample_value(fov_range, self.cfg.intrinsics.distribution)

        # Convert FOV to focal length (using standard 35mm equivalent)
        # FOV = 2 * atan(aperture / (2 * focal_length))
        # focal_length = aperture / (2 * tan(FOV/2))
        aperture = 20.955  # Default horizontal aperture in cm
        focal_length = aperture / (2 * math.tan(math.radians(new_fov / 2)))

        # Set focal length to achieve desired FOV
        camera.CreateFocalLengthAttr().Set(focal_length)

    def _randomize_focal_length(self, camera):
        """Randomize focal length."""
        focal_range = self.cfg.intrinsics.focal_length_range
        if not focal_range:
            return

        # Sample new focal length
        new_focal_length = self._sample_value(focal_range, self.cfg.intrinsics.distribution)

        # Set focal length (in cm, matching metasim camera config)
        camera.CreateFocalLengthAttr().Set(new_focal_length)

    def _randomize_aperture(self, camera):
        """Randomize horizontal aperture."""
        aperture_range = self.cfg.intrinsics.horizontal_aperture_range
        if not aperture_range:
            return

        # Sample new aperture
        new_aperture = self._sample_value(aperture_range, self.cfg.intrinsics.distribution)

        # Set horizontal aperture (in cm, matching metasim camera config)
        camera.CreateHorizontalApertureAttr().Set(new_aperture)

    def _randomize_focus_distance(self, camera):
        """Randomize focus distance."""
        focus_range = self.cfg.intrinsics.focus_distance_range
        if not focus_range:
            return

        # Sample new focus distance
        new_focus_distance = self._sample_value(focus_range, self.cfg.intrinsics.distribution)

        # Set focus distance (in m, matching metasim camera config)
        camera.CreateFocusDistanceAttr().Set(new_focus_distance)

    def _randomize_clipping_range(self, camera):
        """Randomize clipping range."""
        clipping_range = self.cfg.intrinsics.clipping_range
        if not clipping_range:
            return

        # Sample new clipping distances
        near_range, far_range = clipping_range
        new_near = self._sample_value(near_range, self.cfg.intrinsics.distribution)
        new_far = self._sample_value(far_range, self.cfg.intrinsics.distribution)

        # Ensure far > near
        if new_far <= new_near:
            new_far = new_near + 0.1  # Minimum separation

        # Set clipping range (in m, matching metasim camera config)
        from pxr import Gf

        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(new_near, new_far))

    def _randomize_image_properties(self, camera_prim):
        """Randomize image properties - using FOV changes as visual proxy."""
        if not self.cfg.image or not self.cfg.image.enabled:
            return

        try:
            from pxr import UsdGeom

            # Since direct width/height changes are not supported in USD camera,
            # we'll use FOV changes as a proxy for "image property" changes
            # This creates a visible effect similar to changing aspect ratio

            camera = UsdGeom.Camera(camera_prim)
            if not camera:
                return

            # Randomize FOV as proxy for image changes
            if self.cfg.image.aspect_ratio_range:
                # Use aspect ratio to modify horizontal aperture
                aspect_min, aspect_max = self.cfg.image.aspect_ratio_range
                new_aspect = self._sample_value((aspect_min, aspect_max), self.cfg.image.distribution)

                # Get current focal length
                focal_attr = camera.GetFocalLengthAttr()
                focal_length = focal_attr.Get() if focal_attr else 24.0

                # Calculate new aperture based on aspect ratio change
                base_aperture = 20.955  # Standard 35mm
                new_aperture = base_aperture * new_aspect

                # Apply new aperture
                camera.CreateHorizontalApertureAttr().Set(new_aperture)

                # logger.info(
                #     f"Set camera '{self.cfg.camera_name}' aspect ratio to {new_aspect:.2f} (aperture: {new_aperture:.1f}cm)"
                # )

            elif self.cfg.image.use_aspect_ratio and self.cfg.image.width_range and self.cfg.image.height_range:
                # Simulate resolution change through FOV adjustment
                width_min, width_max = self.cfg.image.width_range
                height_min, height_max = self.cfg.image.height_range

                new_width = self._sample_value((width_min, width_max), self.cfg.image.distribution)
                new_height = self._sample_value((height_min, height_max), self.cfg.image.distribution)

                # Use width/height ratio to adjust FOV
                aspect_ratio = new_width / new_height
                fov_multiplier = aspect_ratio / (16 / 9)  # Normalize to 16:9

                # Adjust focal length to simulate resolution change
                current_focal = camera.GetFocalLengthAttr().Get() if camera.GetFocalLengthAttr() else 24.0
                new_focal = current_focal * fov_multiplier
                new_focal = max(8.0, min(100.0, new_focal))  # Clamp to reasonable range

                camera.CreateFocalLengthAttr().Set(new_focal)

                # logger.info(
                #     f"Set camera '{self.cfg.camera_name}' virtual resolution to {new_width:.0f}x{new_height:.0f} (focal: {new_focal:.1f}cm)"
                # )

            else:
                logger.warning(f"Image randomization for '{self.cfg.camera_name}' has no configured ranges")

        except Exception as e:
            logger.error(f"Failed to randomize image properties: {e}")

    def _sample_value(self, value_range: tuple[float, float], distribution: str, size: int | None = None) -> float:
        """Sample a value from the given range using specified distribution."""
        min_val, max_val = value_range

        if distribution == "uniform":
            if size is None:
                return torch.empty((), device=self.device).uniform_(min_val, max_val, generator=self._rng).item()
            return torch.empty(size, device=self.device).uniform_(min_val, max_val, generator=self._rng)
        elif distribution == "log_uniform":
            log_min = math.log(max(min_val, 1e-8))
            log_max = math.log(max_val)
            if size is None:
                val = torch.empty((), device=self.device).uniform_(log_min, log_max, generator=self._rng)
                return torch.exp(val).item()
            vals = torch.empty(size, device=self.device).uniform_(log_min, log_max, generator=self._rng)
            return torch.exp(vals)
        elif distribution == "gaussian":
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 3-sigma range
            if size is None:
                value = torch.empty((), device=self.device).normal_(mean, std, generator=self._rng).item()
                return max(min_val, min(max_val, value))
            value = torch.empty(size, device=self.device).normal_(mean, std, generator=self._rng)
            return torch.clamp(value, min=min_val, max=max_val)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def get_camera_properties(self) -> dict[str, Any]:
        """Get current camera properties for debugging/logging."""
        try:
            camera_prim = self._get_camera_prim().replace("env_.*", "env_0")
            if camera_prim is None:
                return {}

            from pxr import UsdGeom

            properties = {}

            # Get transform
            xformable = UsdGeom.Xformable(camera_prim)
            if xformable:
                ops = xformable.GetOrderedXformOps()
                for op in ops:
                    if "translate" in op.GetOpName():
                        properties["position"] = list(op.Get())
                    elif "rotate" in op.GetOpName():
                        properties["rotation"] = list(op.Get())

            # Get camera properties
            camera = UsdGeom.Camera(camera_prim)
            if camera:
                focal_attr = camera.GetFocalLengthAttr()
                if focal_attr:
                    properties["focal_length"] = focal_attr.Get()

                aperture_attr = camera.GetHorizontalApertureAttr()
                if aperture_attr:
                    properties["horizontal_aperture"] = aperture_attr.Get()

                focus_distance_attr = camera.GetFocusDistanceAttr()
                if focus_distance_attr:
                    properties["focus_distance"] = focus_distance_attr.Get()

                clipping_attr = camera.GetClippingRangeAttr()
                if clipping_attr:
                    clipping_range = clipping_attr.Get()
                    properties["clipping_range"] = [clipping_range[0], clipping_range[1]]

                # Calculate FOV from focal length and aperture
                focal = properties.get("focal_length", 24.0)
                aperture = properties.get("horizontal_aperture", 20.955)
                if focal > 0:
                    import math

                    fov = 2 * math.atan(aperture / (2 * focal)) * 180 / math.pi
                    properties["horizontal_fov"] = fov

            return properties

        except Exception as e:
            logger.error(f"Failed to get camera properties: {e}")
            return {}
