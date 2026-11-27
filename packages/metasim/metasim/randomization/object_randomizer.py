"""Object Randomizer - Property editor for object physics and pose.

The ObjectRandomizer modifies physics properties and pose of existing objects.
It supports both Static Objects (Handler-managed) and Dynamic Objects (Scene-managed)
with intelligent handling based on object capabilities.

Key features:
- Physics randomization: mass, friction, restitution (Static Objects only), through the handler's
  body-property verbs (``get/set_body_masses`` ...), so it works on every backend that implements them
- Pose randomization: position, rotation (all objects); Static Objects through ``get_states`` /
  ``set_states``, Dynamic Objects through USD (Isaac Sim)
- Intelligent degradation: warns when Dynamic Objects receive physics randomization
- Supports Hybrid simulation (uses physics_handler for Static, but needs special handling)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.randomization.core.isaacsim_adapter import IsaacSimAdapter
from metasim.randomization.core.object_registry import ObjectRegistry
from metasim.utils.configclass import configclass
from metasim.utils.log import warn_once

# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class PhysicsRandomCfg:
    """Physics property randomization configuration.

    Attributes:
        enabled: Whether to enable physics randomization
        mass_range: Mass randomization range (kg)
        friction_range: Friction coefficient range
        restitution_range: Restitution (bounciness) range
        distribution: Random sampling distribution
        operation: Operation to apply (add, scale, abs)
    """

    enabled: bool = False
    mass_range: tuple[float, float] | None = None
    friction_range: tuple[float, float] | None = None
    restitution_range: tuple[float, float] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    operation: Literal["add", "scale", "abs"] = "scale"


@configclass
class PoseRandomCfg:
    """Pose randomization configuration.

    Attributes:
        enabled: Whether to enable pose randomization
        position_range: Position range per axis [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        rotation_range: Rotation range in degrees (min, max)
        rotation_axes: Which axes to randomize rotation around (x, y, z)
        distribution: Random sampling distribution
        operation: Operation to apply (add, abs)
        keep_on_ground: Keep object z >= 0
    """

    enabled: bool = False
    position_range: list[tuple[float, float]] | None = None
    rotation_range: tuple[float, float] | None = None
    rotation_axes: tuple[bool, bool, bool] = (True, True, True)
    distribution: Literal["uniform", "gaussian"] = "uniform"
    operation: Literal["add", "abs"] = "add"
    keep_on_ground: bool = True


@configclass
class ObjectRandomCfg:
    """Object randomization configuration.

    Attributes:
        obj_name: Name of object to randomize (must exist in ObjectRegistry)
        body_name: Only this body's physics (an articulation's body name; a rigid object is one body, named
            like the object). None = every body
        env_ids: Environment IDs to apply randomization (None = all)
        physics: Physics property randomization configuration
        pose: Pose randomization configuration
    """

    obj_name: str = ""
    body_name: str | None = None
    env_ids: list[int] | None = None
    physics: PhysicsRandomCfg = PhysicsRandomCfg()
    pose: PoseRandomCfg = PoseRandomCfg()


# =============================================================================
# Object Randomizer Implementation
# =============================================================================


class ObjectRandomizer(BaseRandomizerType):
    """Object property randomizer for all objects.

    Responsibilities:
    - Modify physics properties (mass, friction, restitution) for Static Objects
    - Modify pose (position, rotation) for all objects
    - Intelligent handling: Static (via Handler) vs Dynamic (via USD)
    - NOT responsible for: Creating/deleting objects, modifying materials

    Characteristics:
    - Uses ObjectRegistry to find objects (supports all types)
    - Static Objects: Uses Handler API for physics and pose
    - Dynamic Objects: Uses IsaacSimAdapter for pose only (warns for physics)
    - Hybrid support: uses physics_handler for Static Objects

    Usage:
        # For Static Objects (full support)
        randomizer = ObjectRandomizer(
            ObjectRandomCfg(
                obj_name="box_base",
                physics=PhysicsRandomCfg(
                    enabled=True,
                    mass_range=(0.1, 1.0)
                ),
                pose=PoseRandomCfg(
                    enabled=True,
                    position_range=[(0, 0.2), (0, 0.2), (0, 0)]
                )
            ),
            seed=42
        )

        # For Dynamic Objects (pose only, physics warns)
        randomizer = ObjectRandomizer(
            ObjectRandomCfg(
                obj_name="table",  # Created by SceneRandomizer
                pose=PoseRandomCfg(
                    enabled=True,
                    position_range=[(0, 0.1), (0, 0.1), (0, 0)]
                )
            ),
            seed=43
        )
    """

    REQUIRES_HANDLER = "physics"  # Use physics_handler for Hybrid

    def __init__(self, cfg: ObjectRandomCfg, seed: int | None = None):
        """Initialize object randomizer.

        Args:
            cfg: Object randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.cfg = cfg
        self.registry: ObjectRegistry | None = None
        self.adapter: IsaacSimAdapter | None = None

    def __call__(self):
        """Execute object randomization with intelligent handling."""
        # Get object metadata from Registry
        obj_meta = self.registry.get(self.cfg.obj_name)
        if not obj_meta:
            raise ValueError(
                f"Object '{self.cfg.obj_name}' not found in registry. Available objects: {self.registry.list_objects()}"
            )

        env_ids = self.cfg.env_ids or list(range(self._actual_handler.num_envs))

        # Physics randomization (only for Static Objects with physics)
        if self.cfg.physics.enabled:
            if obj_meta.has_physics and obj_meta.lifecycle == "static":
                self._randomize_physics(torch.tensor(env_ids, dtype=torch.long))
            else:
                if not obj_meta.has_physics:
                    logger.warning(
                        f"[ObjectRandomizer] Object '{self.cfg.obj_name}' has no physics (pure visual). "
                        f"Physics randomization will be skipped."
                    )
                elif obj_meta.lifecycle == "dynamic":
                    logger.warning(
                        f"[ObjectRandomizer] Object '{self.cfg.obj_name}' is a Dynamic Object. "
                        f"Physics randomization will be skipped. Use SceneRandomizer to manage transforms."
                    )

        # Pose randomization (all objects supported)
        if self.cfg.pose.enabled:
            if obj_meta.lifecycle == "static":
                self._randomize_pose_static(torch.tensor(env_ids, dtype=torch.long))
            else:
                self._randomize_pose_dynamic(obj_meta, env_ids)
                # USD writes bypass set_states, so the handler's cached state would stay pre-randomization
                # (see BaseSimHandler.invalidate_state_caches).
                self._actual_handler.invalidate_state_caches()

    # -------------------------------------------------------------------------
    # Physics Randomization (Static Objects only, via Handler API)
    # -------------------------------------------------------------------------

    def _randomize_physics(self, env_ids: torch.Tensor):
        """Randomize mass, friction and restitution through the handler's body-property verbs.

        Mass is drawn per (env, body) and combined with the current value by ``operation``; friction and
        restitution are drawn per env and set on every body that has collision shapes. ``body_name``
        limits the write to that body. A property the engine does not model (restitution on MuJoCo and
        Newton) warns once and changes nothing.

        Args:
            env_ids: Environment IDs to randomize
        """
        handler = self._actual_handler
        phys = self.cfg.physics
        name = self.cfg.obj_name
        ids = [int(i) for i in env_ids.tolist()]
        column = self._body_column(name)
        for prop, getter, setter, value_range in (
            ("mass", handler.get_body_masses, handler.set_body_masses, phys.mass_range),
            ("friction", handler.get_body_friction, handler.set_body_friction, phys.friction_range),
            ("restitution", handler.get_body_restitution, handler.set_body_restitution, phys.restitution_range),
        ):
            if not value_range:
                continue
            current = getter(name)[ids].to(torch.float64)
            new = current.clone()
            if prop == "mass":
                rand = self._generate_random_tensor(tuple(current.shape), phys.distribution, value_range).to(new)
                if phys.operation == "add":
                    new = current + rand
                elif phys.operation == "scale":
                    new = current * rand
                elif phys.operation == "abs":
                    new = rand
                else:
                    raise ValueError(f"Unsupported operation: {phys.operation}")
            else:
                rand = self._generate_random_tensor((len(ids),), phys.distribution, value_range).to(new)
                new = torch.where(torch.isnan(current), current, rand[:, None].expand_as(current))
            if column is not None:
                new = torch.where(torch.arange(new.shape[1]) == column, new, current)
            setter(name, new, env_ids=ids)

    def _body_column(self, obj_name: str) -> int | None:
        """Column of ``cfg.body_name`` in the handler's body-property tensors, or ``None`` for every body."""
        if self.cfg.body_name is None:
            return None
        names = self._actual_handler.get_body_names(obj_name) or [obj_name]  # a rigid object is one body
        if self.cfg.body_name not in names:
            raise ValueError(f"{obj_name!r} has no body {self.cfg.body_name!r}; bodies: {names}.")
        return names.index(self.cfg.body_name)

    # -------------------------------------------------------------------------
    # Pose Randomization (All Objects)
    # -------------------------------------------------------------------------

    def _randomize_pose_static(self, env_ids: torch.Tensor):
        """Randomize the root pose of a handler-managed object through ``get_states`` / ``set_states``.

        Positions are in the env frame, like every handler state.

        Args:
            env_ids: Environment IDs to randomize (tensor)
        """
        handler = self._actual_handler
        name = self.cfg.obj_name
        states = handler.get_states(mode="tensor")
        group = states.robots if name in states.robots else states.objects
        if name not in group:
            raise ValueError(f"{name!r} is neither a robot nor an object of the handler's state.")
        ids = env_ids.to(device=group[name].root_state.device, dtype=torch.long)
        num_envs = ids.shape[0]

        root_state = group[name].root_state[ids]
        current_pos = root_state[:, 0:3]
        current_rot = root_state[:, 3:7]

        # Randomize position
        new_pos = current_pos.clone()
        if self.cfg.pose.position_range:
            for axis in range(3):
                if axis < len(self.cfg.pose.position_range):
                    rand_offset = self._generate_random_tensor(
                        (num_envs,), self.cfg.pose.distribution, self.cfg.pose.position_range[axis]
                    )
                    rand_offset = rand_offset.to(current_pos.device)

                    if self.cfg.pose.operation == "add":
                        new_pos[:, axis] += rand_offset
                    else:  # abs
                        new_pos[:, axis] = rand_offset

            # Keep on ground if requested
            if self.cfg.pose.keep_on_ground:
                new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=0.0)

        # Randomize rotation
        new_rot = current_rot.clone()
        if self.cfg.pose.rotation_range:
            # Generate random Euler angles for all enabled axes (batch)
            roll = torch.zeros(num_envs, device=current_rot.device)
            pitch = torch.zeros(num_envs, device=current_rot.device)
            yaw = torch.zeros(num_envs, device=current_rot.device)

            if self.cfg.pose.rotation_axes[0]:  # roll (x-axis)
                roll = self._generate_random_tensor(
                    (num_envs,), self.cfg.pose.distribution, self.cfg.pose.rotation_range
                ) * (math.pi / 180.0)
                roll = roll.to(current_rot.device)

            if self.cfg.pose.rotation_axes[1]:  # pitch (y-axis)
                pitch = self._generate_random_tensor(
                    (num_envs,), self.cfg.pose.distribution, self.cfg.pose.rotation_range
                ) * (math.pi / 180.0)
                pitch = pitch.to(current_rot.device)

            if self.cfg.pose.rotation_axes[2]:  # yaw (z-axis)
                yaw = self._generate_random_tensor(
                    (num_envs,), self.cfg.pose.distribution, self.cfg.pose.rotation_range
                ) * (math.pi / 180.0)
                yaw = yaw.to(current_rot.device)

            # Convert to quaternion (batch)
            rand_quat = self._euler_to_quaternion_batch(roll, pitch, yaw)

            if self.cfg.pose.operation == "add":
                new_rot = self._quaternion_multiply(current_rot, rand_quat)
            else:  # abs
                new_rot = rand_quat

        # Set new pose
        new_root_state = root_state.clone()
        new_root_state[:, 0:3] = new_pos
        new_root_state[:, 3:7] = new_rot

        group[name].root_state[ids] = new_root_state
        handler.set_states(states, env_ids=ids.tolist())
        after = handler.get_states(mode="tensor")
        reached = (after.robots if name in after.robots else after.objects)[name].root_state[ids, :7].cpu()
        wanted = new_root_state[:, :7].cpu()
        same_rot = (reached[:, 3:7] * wanted[:, 3:7]).sum(dim=1).abs() > 1 - 1e-4  # q and -q are one rotation
        if not (torch.allclose(reached[:, :3], wanted[:, :3], atol=1e-4) and same_rot.all()):
            warn_once(
                (type(handler).__name__, "static_pose", name),
                f"ObjectRandomizer: {type(handler).__name__} did not move {name!r} (a static object it builds as "
                "a shape on the world, e.g. a GEOM / XFORM primitive on Newton); its pose randomization has no effect.",
            )
        self._mark_visual_dirty()

    def _randomize_pose_dynamic(self, obj_meta, env_ids: list[int]):
        """Randomize pose for Dynamic Objects via USD.

        Args:
            obj_meta: Object metadata
            env_ids: Environment IDs to randomize
        """
        prim_paths = self.registry.get_prim_paths(self.cfg.obj_name, env_ids)

        for prim_path in prim_paths:
            # Get current transform
            try:
                current_pos, current_rot, current_scale = self.adapter.get_transform(prim_path)
            except Exception:
                current_pos = (0.0, 0.0, 0.0)
                current_rot = (1.0, 0.0, 0.0, 0.0)

            # Randomize position
            new_pos = None
            if self.cfg.pose.position_range:
                if self.cfg.pose.operation == "add":
                    new_pos = tuple(
                        current_pos[i] + self.rng.uniform(r[0], r[1])
                        for i, r in enumerate(self.cfg.pose.position_range)
                    )
                else:  # abs
                    new_pos = tuple(self.rng.uniform(r[0], r[1]) for r in self.cfg.pose.position_range)

                if self.cfg.pose.keep_on_ground:
                    new_pos = (new_pos[0], new_pos[1], max(0.0, new_pos[2]))

            # Randomize rotation
            new_rot = None
            if self.cfg.pose.rotation_range:
                # Generate random Euler angles
                roll = self.rng.uniform(*self.cfg.pose.rotation_range) if self.cfg.pose.rotation_axes[0] else 0.0
                pitch = self.rng.uniform(*self.cfg.pose.rotation_range) if self.cfg.pose.rotation_axes[1] else 0.0
                yaw = self.rng.uniform(*self.cfg.pose.rotation_range) if self.cfg.pose.rotation_axes[2] else 0.0

                # Convert to radians and then to quaternion
                roll_rad = roll * (math.pi / 180.0)
                pitch_rad = pitch * (math.pi / 180.0)
                yaw_rad = yaw * (math.pi / 180.0)

                new_rot = self._euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

                if self.cfg.pose.operation == "add":
                    # Compose with current rotation
                    import torch

                    current_rot_tensor = torch.tensor(current_rot)
                    new_rot_tensor = torch.tensor(new_rot)
                    composed = self._quaternion_multiply(current_rot_tensor, new_rot_tensor)
                    new_rot = tuple(composed.tolist())

            # Apply transform
            self.adapter.set_transform(prim_path, position=new_pos, rotation=new_rot)

        self._mark_visual_dirty()

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _generate_random_tensor(
        self, shape: tuple[int, ...], distribution: str, range_vals: tuple[float, float]
    ) -> torch.Tensor:
        """Generate random tensor using reproducible RNG.

        Args:
            shape: Tensor shape
            distribution: Distribution type
            range_vals: Value range (min, max)

        Returns:
            Random tensor
        """
        if distribution == "uniform":
            if len(shape) == 1:
                rand_vals = [self.rng.uniform(range_vals[0], range_vals[1]) for _ in range(shape[0])]
            else:
                rand_vals = [
                    [self.rng.uniform(range_vals[0], range_vals[1]) for _ in range(shape[1])] for _ in range(shape[0])
                ]
            return torch.tensor(rand_vals, dtype=torch.float32)

        elif distribution == "log_uniform":
            log_min = math.log(range_vals[0])
            log_max = math.log(range_vals[1])
            if len(shape) == 1:
                rand_vals = [math.exp(self.rng.uniform(log_min, log_max)) for _ in range(shape[0])]
            else:
                rand_vals = [
                    [math.exp(self.rng.uniform(log_min, log_max)) for _ in range(shape[1])] for _ in range(shape[0])
                ]
            return torch.tensor(rand_vals, dtype=torch.float32)

        elif distribution == "gaussian":
            mean = (range_vals[0] + range_vals[1]) / 2
            std = (range_vals[1] - range_vals[0]) / 6
            if len(shape) == 1:
                rand_vals = [max(range_vals[0], min(range_vals[1], self.rng.gauss(mean, std))) for _ in range(shape[0])]
            else:
                rand_vals = [
                    [max(range_vals[0], min(range_vals[1], self.rng.gauss(mean, std))) for _ in range(shape[1])]
                    for _ in range(shape[0])
                ]
            return torch.tensor(rand_vals, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> tuple:
        """Convert Euler angles to quaternion.

        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            yaw: Yaw angle (radians)

        Returns:
            Quaternion (w, x, y, z)
        """
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

    def _euler_to_quaternion_batch(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles to quaternions (batch).

        Args:
            roll: Roll angles (radians)
            pitch: Pitch angles (radians)
            yaw: Yaw angles (radians)

        Returns:
            Quaternions [w, x, y, z]
        """
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions.

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Product quaternion [w, x, y, z]
        """
        # Ensure both quaternions on same device
        q2 = q2.to(q1.device)

        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)
