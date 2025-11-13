from __future__ import annotations

import math
from typing import Any, Literal

from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass


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
    position: CameraPositionRandomCfg | None = None
    orientation: CameraOrientationRandomCfg | None = None
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

    def __init__(self, cfg: CameraRandomCfg, seed: int | None = None):
        """Initialize camera randomizer.

        Args:
            cfg: Camera randomization configuration
            seed: Random seed for reproducible randomization
        """
        self.cfg = cfg
        super().__init__(seed=seed)

        self.handler = None

    def set_seed(self, seed: int | None) -> None:
        """Set or update RNG seed."""
        super().set_seed(seed)

    def bind_handler(self, handler):
        """Bind simulation handler."""
        self.handler = handler

    def _get_current_transform(self, xformable):
        """Get current position and complete transform state from USD scene."""
        import math

        from pxr import Gf, UsdGeom

        # Store all transform operations to preserve them exactly
        ops = xformable.GetOrderedXformOps()

        # Extract position and compute actual final transform
        position = None
        all_ops = []

        for op in ops:
            op_type = op.GetOpType()
            op_value = op.Get()
            all_ops.append((op_type, op_value))

            if op_type == UsdGeom.XformOp.TypeTranslate:
                position = op_value

        # Get the ACTUAL final transform matrix (this combines all ops)
        world_transform = xformable.ComputeLocalToWorldTransform(0.0)
        final_translation = world_transform.ExtractTranslation()
        final_rotation_matrix = world_transform.ExtractRotationMatrix()

        # Convert final rotation matrix to Euler angles for comparison
        m = final_rotation_matrix
        sy = math.sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0])

        if sy > 1e-6:
            x = math.atan2(m[2][1], m[2][2])
            y = math.atan2(-m[2][0], sy)
            z = math.atan2(m[1][0], m[0][0])
        else:
            x = math.atan2(-m[1][2], m[1][1])
            y = math.atan2(-m[2][0], sy)
            z = 0

        final_rotation = Gf.Vec3f(
            math.degrees(x),  # pitch
            math.degrees(y),  # yaw
            math.degrees(z),  # roll
        )

        if position is None:
            position = Gf.Vec3d(final_translation)
            logger.warning("No translate op found, using computed position")

        # Return position from ops but store all ops for preservation
        return position, final_rotation, all_ops

    def _update_position_only_preserve_all(self, xformable, new_position):
        """Update ONLY the translate operation, preserving all other transform operations."""
        try:
            from pxr import UsdGeom

            # Find and update only the translate operation
            ops = xformable.GetOrderedXformOps()
            translate_op = None

            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op:
                old_position = translate_op.Get()
                translate_op.Set(new_position)

            else:
                logger.error("No translate operation found! Cannot update position safely.")
                raise ValueError("No translate operation found in transform stack")

        except Exception as e:
            logger.error(f"Failed to update position only: {e}")
            raise

    def _add_rotation_delta(self, xformable, delta_rotation):
        """Add rotation delta to existing rotation using quaternion composition."""
        try:
            import math

            from pxr import Gf, UsdGeom

            # Get current quaternion directly from TypeOrient
            ops = xformable.GetOrderedXformOps()
            current_quat = None

            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    current_quat = op.Get()
                    break

            if current_quat is None:
                logger.error("No TypeOrient operation found for rotation delta")
                return

            # Debug info removed for cleaner output

            # Convert small delta rotation to quaternion using USD rotation utilities
            pitch_rad = math.radians(delta_rotation[0])
            yaw_rad = math.radians(delta_rotation[1])
            roll_rad = math.radians(delta_rotation[2])

            # Create individual rotations and combine them (XYZ order)
            rotation_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(pitch_rad))
            rotation_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(yaw_rad))
            rotation_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(roll_rad))

            combined_rotation = rotation_x * rotation_y * rotation_z
            delta_quat = combined_rotation.GetQuat()

            # Compose quaternions: new_quat = current_quat * delta_quat
            new_quat = current_quat * delta_quat

            #             logger.info(f"Delta quaternion: {delta_quat}")
            #             logger.info(f"New quaternion: {new_quat}")

            # Update the TypeOrient operation
            self._update_orient_only(xformable, new_quat)

        except Exception as e:
            logger.error(f"Failed to add rotation delta: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _update_orient_only(self, xformable, new_quaternion):
        """Update ONLY the TypeOrient operation, preserving all other transform operations."""
        try:
            from pxr import UsdGeom

            # Find and update only the TypeOrient operation
            ops = xformable.GetOrderedXformOps()
            orient_op = None

            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
                    break

            if orient_op:
                old_quat = orient_op.Get()
                orient_op.Set(new_quaternion)
            #                 logger.info(f"Updated TypeOrient: {old_quat} → {new_quaternion}")
            else:
                logger.error("No TypeOrient operation found! Cannot update orientation safely.")
                raise ValueError("No TypeOrient operation found in transform stack")

        except Exception as e:
            logger.error(f"Failed to update orientation only: {e}")
            raise

    def __call__(self):
        """Apply camera randomization."""
        if self.handler is None:
            logger.warning("Camera randomizer not bound to handler, skipping randomization")
            return

        did_update = False
        try:
            camera_prim = self._get_camera_prim()
            if camera_prim is None:
                logger.warning(f"Camera '{self.cfg.camera_name}' not found in scene")
                return

            mode = self.cfg.randomization_mode
            if mode == "combined":
                did_update = self._randomize_all(camera_prim)
            elif mode == "position_only" and self.cfg.position and self.cfg.position.enabled:
                self._randomize_position(camera_prim)
                did_update = True
            elif mode == "orientation_only" and self.cfg.orientation and self.cfg.orientation.enabled:
                self._randomize_orientation(camera_prim)
                did_update = True
            elif mode == "look_at_only" and self.cfg.look_at and self.cfg.look_at.enabled:
                self._randomize_look_at(camera_prim)
                did_update = True
            elif mode == "intrinsics_only" and self.cfg.intrinsics and self.cfg.intrinsics.enabled:
                self._randomize_intrinsics(camera_prim)
                did_update = True
            elif mode == "image_only" and self.cfg.image and self.cfg.image.enabled:
                self._randomize_image_properties(camera_prim)
                did_update = True
            else:
                if mode != "combined":
                    logger.warning(f"Unknown or disabled randomization mode '{mode}', applying combined")
                did_update = self._randomize_all(camera_prim)

        except Exception as e:
            logger.error(f"Camera randomization failed for '{self.cfg.camera_name}': {e}")
        else:
            if did_update:
                self._mark_visual_dirty()

    def _get_camera_prim(self):
        """Get camera prim from scene."""
        try:
            # Try to find camera in the scene
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            camera_path = f"/World/{self.cfg.camera_name}"

            camera_prim = stage.GetPrimAtPath(camera_path)
            if camera_prim and camera_prim.IsValid():
                return camera_prim

            # Alternative search pattern
            for prim in stage.Traverse():
                if prim.GetName() == self.cfg.camera_name and prim.GetTypeName() == "Camera":
                    return prim

            return None
        except Exception as e:
            logger.error(f"Error finding camera '{self.cfg.camera_name}': {e}")
            return None

    def _randomize_all(self, camera_prim) -> bool:
        """Apply all enabled randomization types in proper order to avoid conflicts."""
        updated = False

        if self.cfg.position and self.cfg.position.enabled:
            self._randomize_position(camera_prim)
            updated = True

        if self.cfg.look_at and self.cfg.look_at.enabled:
            self._randomize_look_at(camera_prim)
            updated = True
        elif self.cfg.orientation and self.cfg.orientation.enabled:
            self._randomize_orientation(camera_prim)
            updated = True

        if self.cfg.intrinsics and self.cfg.intrinsics.enabled:
            self._randomize_intrinsics(camera_prim)
            updated = True
        if self.cfg.image and self.cfg.image.enabled:
            self._randomize_image_properties(camera_prim)
            updated = True

        return updated

    def _randomize_position(self, camera_prim):
        """Randomize camera position ONLY (independent of orientation)."""
        if not self.cfg.position or not self.cfg.position.enabled:
            return

        try:
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(camera_prim)
            if not xformable:
                return

            # Get current position
            current_pos, _, _ = self._get_current_transform(xformable)

            if self.cfg.position.use_delta and self.cfg.position.delta_range:
                # Delta-based micro-adjustment mode (默认)
                delta_range = self.cfg.position.delta_range
                dx = self._sample_value(delta_range[0], self.cfg.position.distribution)
                dy = self._sample_value(delta_range[1], self.cfg.position.distribution)
                dz = self._sample_value(delta_range[2], self.cfg.position.distribution)

                # Apply delta to current position
                new_pos = Gf.Vec3d(current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)

                # logger.info(f"Adjusted camera '{self.cfg.camera_name}' position by ({dx:+.2f}, {dy:+.2f}, {dz:+.2f})")
                # logger.info(
                #     f"  from ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
                # )

            elif self.cfg.position.position_range:
                # Absolute positioning mode
                position_range = self.cfg.position.position_range
                new_pos = Gf.Vec3d(
                    self._sample_value(position_range[0], self.cfg.position.distribution),
                    self._sample_value(position_range[1], self.cfg.position.distribution),
                    self._sample_value(position_range[2], self.cfg.position.distribution),
                )

                # logger.info(
                #     f"Set camera '{self.cfg.camera_name}' to absolute position ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
                # )

            else:
                logger.warning(f"No position range configured for camera '{self.cfg.camera_name}'")
                return

            # Apply new position only, preserving existing rotation
            self._update_position_only_preserve_all(xformable, new_pos)

        except Exception as e:
            logger.error(f"Failed to randomize camera position: {e}")

    def _randomize_orientation(self, camera_prim):
        """Randomize camera orientation by adding rotation deltas (independent of position and look-at)."""
        if not self.cfg.orientation or not self.cfg.orientation.enabled:
            return

        try:
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(camera_prim)
            if not xformable:
                return

            if not self.cfg.orientation.rotation_delta:
                logger.warning(f"No rotation delta configured for camera '{self.cfg.camera_name}' orientation")
                return

            # Sample rotation deltas
            rotation_delta = self.cfg.orientation.rotation_delta
            delta_pitch = self._sample_value(rotation_delta[0], self.cfg.orientation.distribution)
            delta_yaw = self._sample_value(rotation_delta[1], self.cfg.orientation.distribution)
            delta_roll = self._sample_value(rotation_delta[2], self.cfg.orientation.distribution)

            delta_rotation = Gf.Vec3f(delta_pitch, delta_yaw, delta_roll)

            # logger.info(
            #     f"Applying rotation delta to camera '{self.cfg.camera_name}': pitch={delta_pitch:+.1f}°, yaw={delta_yaw:+.1f}°, roll={delta_roll:+.1f}°"
            # )

            # Apply rotation delta to existing rotation
            self._add_rotation_delta(xformable, delta_rotation)

        except Exception as e:
            logger.error(f"Failed to randomize camera orientation: {e}")

    def _randomize_look_at(self, camera_prim):
        """Randomize camera by moving it in a spherical orbit around the original look-at target."""
        if not self.cfg.look_at or not self.cfg.look_at.enabled:
            return

        try:
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(camera_prim)
            if not xformable:
                return

            # Get original camera configuration
            original_pos, original_look_at = self._get_original_camera_config(camera_prim)
            # logger.debug(
            #     f"Original config: pos=({original_pos[0]:.2f}, {original_pos[1]:.2f}, {original_pos[2]:.2f}), target=({original_look_at[0]:.2f}, {original_look_at[1]:.2f}, {original_look_at[2]:.2f})"
            # )

            if self.cfg.look_at.use_spherical and self.cfg.look_at.spherical_range:
                # Spherical coordinate mode - move camera in orbit around target
                self._apply_spherical_look_at(xformable, original_pos)
                return

            # Default look-at behavior: small orbital movement around ORIGINAL position
            if self.cfg.look_at.use_delta and self.cfg.look_at.look_at_delta:
                # Small orbital movement around ORIGINAL position (not current)
                delta_range = self.cfg.look_at.look_at_delta
                dx = self._sample_value(delta_range[0], self.cfg.look_at.distribution)
                dy = self._sample_value(delta_range[1], self.cfg.look_at.distribution)
                dz = self._sample_value(delta_range[2], self.cfg.look_at.distribution)

                # Move camera position (small orbital adjustment from ORIGINAL position)
                new_pos = Gf.Vec3d(original_pos[0] + dx, original_pos[1] + dy, original_pos[2] + dz)

                # logger.info(f"Camera '{self.cfg.camera_name}' orbital move by ({dx:+.2f}, {dy:+.2f}, {dz:+.2f})")
                # logger.info(
                #     f"  from original ({original_pos[0]:.2f}, {original_pos[1]:.2f}, {original_pos[2]:.2f}) to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
                # )

            elif self.cfg.look_at.look_at_range:
                # Absolute positioning mode
                look_at_range = self.cfg.look_at.look_at_range
                new_pos = Gf.Vec3d(
                    self._sample_value(look_at_range[0], self.cfg.look_at.distribution),
                    self._sample_value(look_at_range[1], self.cfg.look_at.distribution),
                    self._sample_value(look_at_range[2], self.cfg.look_at.distribution),
                )

                # logger.info(
                #     f"Camera '{self.cfg.camera_name}' moved to absolute position ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
                # )

            else:
                logger.warning(f"No look-at range configured for camera '{self.cfg.camera_name}'")
                return

            # Calculate target quaternion for look-at using robust orthogonal basis method
            target_quat = self._calculate_look_at_quaternion(new_pos, original_look_at)

            # Apply new position first
            self._update_position_only_preserve_all(xformable, new_pos)

            # Apply absolute quaternion rotation (no delta calculation needed)
            self._update_orient_only(xformable, target_quat)

            # logger.info(
            #     f"Camera '{self.cfg.camera_name}' moved to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}) looking at ({original_look_at[0]:.2f}, {original_look_at[1]:.2f}, {original_look_at[2]:.2f})"
            # )

        except Exception as e:
            logger.error(f"Failed to randomize camera look-at: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _get_original_camera_config(self, camera_prim):
        """Get the original camera configuration (position and look-at target)."""
        try:
            # Try to get both position and look-at from scenario configuration
            if hasattr(self, "handler") and hasattr(self.handler, "scenario"):
                cameras = getattr(self.handler.scenario, "cameras", [])
                for camera_cfg in cameras:
                    if hasattr(camera_cfg, "name") and camera_cfg.name == self.cfg.camera_name:
                        from pxr import Gf

                        # Get original position
                        original_pos = Gf.Vec3d(1.5, -1.5, 1.5)  # Default
                        if hasattr(camera_cfg, "pos"):
                            pos = camera_cfg.pos
                            original_pos = Gf.Vec3d(pos[0], pos[1], pos[2])

                        # Get original look-at target
                        original_look_at = Gf.Vec3d(0.0, 0.0, 0.0)  # Default
                        if hasattr(camera_cfg, "look_at"):
                            look_at = camera_cfg.look_at
                            original_look_at = Gf.Vec3d(look_at[0], look_at[1], look_at[2])

                        return original_pos, original_look_at

            # Default fallback
            from pxr import Gf

            default_pos = Gf.Vec3d(1.5, -1.5, 1.5)
            default_target = Gf.Vec3d(0.0, 0.0, 0.0)
            # logger.debug(
            #     f"Using default camera config: pos=({default_pos[0]}, {default_pos[1]}, {default_pos[2]}), target=({default_target[0]}, {default_target[1]}, {default_target[2]})"
            # )
            return default_pos, default_target

        except Exception as e:
            logger.warning(f"Failed to get original camera config: {e}, using defaults")
            from pxr import Gf

            return Gf.Vec3d(1.5, -1.5, 1.5), Gf.Vec3d(0.0, 0.0, 0.0)

    def _calculate_look_at_quaternion(self, eye_pos, target_pos):
        """Calculate quaternion for camera to look at target using orthogonal basis method."""
        import math

        from pxr import Gf

        # Calculate forward vector (camera -Z direction points toward target)
        forward = target_pos - eye_pos
        forward_length = math.sqrt(forward[0] ** 2 + forward[1] ** 2 + forward[2] ** 2)
        if forward_length < 1e-6:
            # Return identity quaternion if target is at eye position
            return Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0))

        forward = forward / forward_length

        # World up vector (typically Z-axis up)
        world_up = Gf.Vec3d(0, 0, 1)

        # Calculate right vector (camera +X direction)
        right = forward.GetCross(world_up)
        right_length = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2)
        if right_length < 1e-6:
            # Forward is parallel to world up, use arbitrary right vector
            right = Gf.Vec3d(1, 0, 0)
        else:
            right = right / right_length

        # Calculate up vector (camera +Y direction)
        up = right.GetCross(forward)

        # Create rotation matrix from orthogonal basis
        # For USD camera: +X=right, +Y=up, -Z=forward (camera looks down -Z)
        # So our forward vector should be negated for -Z
        rotation_matrix = Gf.Matrix3d(
            right[0],
            right[1],
            right[2],  # First row: +X basis
            up[0],
            up[1],
            up[2],  # Second row: +Y basis
            -forward[0],
            -forward[1],
            -forward[2],  # Third row: -Z basis
        )

        # Convert rotation matrix to quaternion using SetRotate
        transform_matrix = Gf.Matrix4d(1.0)  # Identity 4x4 matrix
        transform_matrix.SetRotate(rotation_matrix)
        quat = transform_matrix.ExtractRotation().GetQuat()

        # logger.debug(
        #     f"Look-at quaternion: eye=({eye_pos[0]:.2f}, {eye_pos[1]:.2f}, {eye_pos[2]:.2f}) -> target=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) = {quat}"
        # )

        return quat

    def _apply_spherical_look_at(self, xformable, current_pos):
        """Apply spherical coordinate-based look-at positioning."""
        spherical_range = self.cfg.look_at.spherical_range
        if not spherical_range:
            return

        import math

        from pxr import Gf

        # Sample spherical coordinates relative to current position
        radius = self._sample_value(spherical_range[0], self.cfg.look_at.distribution)
        theta = self._sample_value(spherical_range[1], self.cfg.look_at.distribution)  # azimuth
        phi = self._sample_value(spherical_range[2], self.cfg.look_at.distribution)  # elevation

        # Convert to Cartesian coordinates (offset from current position)
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)

        # Calculate look-at point in spherical coordinates around current position
        look_x = current_pos[0] + radius * math.cos(phi_rad) * math.cos(theta_rad)
        look_y = current_pos[1] + radius * math.cos(phi_rad) * math.sin(theta_rad)
        look_z = current_pos[2] + radius * math.sin(phi_rad)

        target = Gf.Vec3d(look_x, look_y, look_z)
        target_quat = self._calculate_look_at_quaternion(current_pos, target)

        # Apply new rotation only, preserving existing position
        self._update_orient_only(xformable, target_quat)

        # logger.info(
        #     f"Set camera '{self.cfg.camera_name}' spherical look-at: r={radius:.2f}m, θ={theta:.1f}°, φ={phi:.1f}°"
        # )

    def _randomize_intrinsics(self, camera_prim):
        """Randomize camera intrinsics."""
        if not self.cfg.intrinsics or not self.cfg.intrinsics.enabled:
            return

        try:
            from pxr import UsdGeom

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

    def _sample_value(self, value_range: tuple[float, float], distribution: str) -> float:
        """Sample a value from the given range using specified distribution."""
        min_val, max_val = value_range

        if distribution == "uniform":
            return self._rng.uniform(min_val, max_val)
        elif distribution == "log_uniform":
            log_min = math.log(max(min_val, 1e-8))
            log_max = math.log(max_val)
            return math.exp(self._rng.uniform(log_min, log_max))
        elif distribution == "gaussian":
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 3-sigma range
            value = self._rng.gauss(mean, std)
            return max(min_val, min(max_val, value))  # Clip to range
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def get_camera_properties(self) -> dict[str, Any]:
        """Get current camera properties for debugging/logging."""
        try:
            camera_prim = self._get_camera_prim()
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
