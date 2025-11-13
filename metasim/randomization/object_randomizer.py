"""Unified Object Randomizer for comprehensive object property randomization."""

from __future__ import annotations

from typing import Any, Literal

import torch

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass


@configclass
class PhysicsRandomCfg:
    """Configuration for physics property randomization."""

    enabled: bool = False
    """Whether to enable physics randomization."""

    mass_range: tuple[float, float] | None = None
    """Mass randomization range. If None, mass won't be randomized."""

    friction_range: tuple[float, float] | None = None
    """Friction randomization range. If None, friction won't be randomized.

    Note: MaterialRandomizer can also modify friction. Consider using MaterialRandomizer
    if you need complex material properties (PBR, MDL) along with friction changes.
    """

    restitution_range: tuple[float, float] | None = None
    """Restitution (bounce) randomization range. If None, restitution won't be randomized.

    Note: MaterialRandomizer can also modify restitution. Consider using MaterialRandomizer
    if you need complex material properties (PBR, MDL) along with restitution changes.
    """

    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    """Distribution type for physics randomization."""

    operation: Literal["add", "scale", "abs"] = "scale"
    """Operation to apply: add (current + random), scale (current * random), abs (random only)."""


@configclass
class PoseRandomCfg:
    """Configuration for pose (position and rotation) randomization."""

    enabled: bool = False
    """Whether to enable pose randomization."""

    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    """Position randomization range for (x, y, z). If None, position won't be randomized."""

    rotation_range: tuple[float, float] | None = None
    """Rotation randomization range in degrees around each axis. If None, rotation won't be randomized."""

    rotation_axes: tuple[bool, bool, bool] = (True, True, True)
    """Which axes to randomize rotation around (x, y, z)."""

    distribution: Literal["uniform", "gaussian"] = "uniform"
    """Distribution type for pose randomization."""

    operation: Literal["add", "abs"] = "add"
    """Operation to apply: add (current + random), abs (random only)."""

    keep_on_ground: bool = True
    """Whether to keep object on ground (z >= 0) for position randomization."""


@configclass
class ObjectRandomCfg:
    """Unified configuration for object randomization."""

    obj_name: str = ""
    """Name of the object to randomize."""

    body_name: str | None = None
    """Specific body name within the object. If None, applies to all bodies."""

    env_ids: list[int] | None = None
    """Environment IDs to apply randomization to. If None, applies to all environments."""

    physics: PhysicsRandomCfg = PhysicsRandomCfg()
    """Physics property randomization configuration."""

    pose: PoseRandomCfg = PoseRandomCfg()
    """Pose randomization configuration."""


class ObjectRandomizer(BaseRandomizerType):
    """Unified object randomizer for comprehensive object property randomization.

    This randomizer can handle:
    - Physics properties: mass, friction, restitution
    - Pose properties: position, rotation
    - Applies to both articulated objects (robots) and rigid objects

    Note: For friction and restitution, MaterialRandomizer provides equivalent functionality
    plus visual material properties (PBR, MDL). Consider your workflow:
    - Use ObjectRandomizer for object-centric workflows (mass + pose + basic physics)
    - Use MaterialRandomizer for material-centric workflows (visual + physics properties)
    - Both can be used together: ObjectRandomizer for mass/pose, MaterialRandomizer for materials
    """

    def __init__(self, cfg: ObjectRandomCfg, seed: int | None = None):
        self.cfg = cfg
        super().__init__(seed=seed)

    def set_seed(self, seed: int | None) -> None:
        """Set or update RNG seed."""
        super().set_seed(seed)

    def _generate_random_tensor(
        self, shape: tuple[int, ...], distribution: str, range_vals: tuple[float, float]
    ) -> torch.Tensor:
        """Generate random tensor using our reproducible RNG."""
        if distribution == "uniform":
            # Generate uniform random values using our RNG
            if len(shape) == 1:
                rand_vals = [self._rng.uniform(range_vals[0], range_vals[1]) for _ in range(shape[0])]
                return torch.tensor(rand_vals, dtype=torch.float32)
            else:
                rand_vals = [
                    [self._rng.uniform(range_vals[0], range_vals[1]) for _ in range(shape[1])] for _ in range(shape[0])
                ]
                return torch.tensor(rand_vals, dtype=torch.float32)
        elif distribution == "log_uniform":
            # Generate log-uniform values
            log_min, log_max = torch.log(torch.tensor(range_vals[0])), torch.log(torch.tensor(range_vals[1]))
            if len(shape) == 1:
                rand_vals = [
                    torch.exp(torch.tensor(self._rng.uniform(0.0, 1.0)) * (log_max - log_min) + log_min).item()
                    for _ in range(shape[0])
                ]
                return torch.tensor(rand_vals, dtype=torch.float32)
            else:
                rand_vals = [
                    [
                        torch.exp(torch.tensor(self._rng.uniform(0.0, 1.0)) * (log_max - log_min) + log_min).item()
                        for _ in range(shape[1])
                    ]
                    for _ in range(shape[0])
                ]
                return torch.tensor(rand_vals, dtype=torch.float32)
        elif distribution == "gaussian":
            # Generate Gaussian values
            mean = (range_vals[0] + range_vals[1]) / 2
            std = (range_vals[1] - range_vals[0]) / 6
            if len(shape) == 1:
                rand_vals = [
                    max(range_vals[0], min(range_vals[1], self._rng.gauss(mean, std))) for _ in range(shape[0])
                ]
                return torch.tensor(rand_vals, dtype=torch.float32)
            else:
                rand_vals = [
                    [max(range_vals[0], min(range_vals[1], self._rng.gauss(mean, std))) for _ in range(shape[1])]
                    for _ in range(shape[0])
                ]
                return torch.tensor(rand_vals, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Bind the handler to the randomizer."""
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            super().bind_handler(handler, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for ObjectRandomizer")

    def _get_body_names(self, obj_name: str) -> list[str]:
        """Get body names for an object."""
        if hasattr(self.handler, "_get_body_names"):
            return self.handler._get_body_names(obj_name)
        else:
            # Fallback implementation
            if obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[obj_name]
                # This is a simplified approach - actual implementation may vary
                return [f"body_{i}" for i in range(obj_inst.root_physx_view.get_masses().shape[1])]
            return []

    # Physics Property Methods
    def get_mass(self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None) -> torch.Tensor:
        """Get the mass of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return masses[env_ids, body_idx]
            else:
                return masses[env_ids, :]
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            masses = obj_inst.root_physx_view.get_masses()
            return masses[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_mass(
        self, obj_name: str, mass: torch.Tensor, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> None:
        """Set the mass of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                masses[env_ids, body_idx] = mass
            else:
                masses[env_ids, :] = mass

            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            masses = obj_inst.root_physx_view.get_masses()
            masses[env_ids] = mass
            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def get_friction(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """Get the friction coefficient of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return friction[env_ids, body_idx]
            else:
                return friction[env_ids, :]
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction
            return friction[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_friction(
        self, obj_name: str, friction: torch.Tensor, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> None:
        """Set the friction coefficient of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                materials[env_ids, body_idx, 0] = friction  # Static friction
                materials[env_ids, body_idx, 1] = friction  # Dynamic friction
            else:
                materials[env_ids, :, 0] = friction  # Static friction
                materials[env_ids, :, 1] = friction  # Dynamic friction

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            # Rigid objects can have 3D materials array like articulations
            if len(materials.shape) == 3:
                # Format: [num_envs, num_bodies, num_properties] - same as articulations
                materials[env_ids, :, 0] = friction  # Static friction
                materials[env_ids, :, 1] = friction  # Dynamic friction
            else:
                # Format: [num_envs, num_properties] - 2D format
                materials[env_ids, 0] = friction  # Static friction
                materials[env_ids, 1] = friction  # Dynamic friction
            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def get_restitution(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """Get the restitution (bounce) coefficient of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            restitution = materials[..., 2]  # Third component is restitution

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return restitution[env_ids, body_idx]
            else:
                return restitution[env_ids, :]
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            restitution = materials[..., 2]  # Third component is restitution
            return restitution[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_restitution(
        self, obj_name: str, restitution: torch.Tensor, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> None:
        """Set the restitution (bounce) coefficient of an object or specific body."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()

            if body_name is not None:
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                materials[env_ids, body_idx, 2] = restitution
            else:
                materials[env_ids, :, 2] = restitution

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            # Rigid objects can have 3D materials array like articulations
            if len(materials.shape) == 3:
                # Format: [num_envs, num_bodies, num_properties] - same as articulations
                materials[env_ids, :, 2] = restitution
            else:
                # Format: [num_envs, num_properties] - 2D format
                materials[env_ids, 2] = restitution
            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        else:
            raise ValueError(f"Object {obj_name} not found")

    # Pose Methods
    def get_pose(self, obj_name: str, env_ids: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the pose (position and rotation) of an object.

        Returns:
            tuple: (position, rotation) where position is (N, 3) and rotation is (N, 4) quaternion
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            # Use data properties for articulations (same as rigid objects)
            pos = obj_inst.data.root_pos_w[env_ids] - self.handler.scene.env_origins[env_ids]
            rot = obj_inst.data.root_quat_w[env_ids]
            return pos, rot
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            # Use data properties for rigid objects
            pos = obj_inst.data.root_pos_w[env_ids] - self.handler.scene.env_origins[env_ids]
            rot = obj_inst.data.root_quat_w[env_ids]
            return pos, rot
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_pose(
        self, obj_name: str, position: torch.Tensor, rotation: torch.Tensor, env_ids: list[int] | None = None
    ) -> None:
        """Set the pose (position and rotation) of an object.

        Args:
            obj_name: Name of the object to set pose for
            position: (N, 3) position tensor
            rotation: (N, 4) quaternion rotation tensor
            env_ids: List of environment IDs to apply changes to. If None, applies to all environments.
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            # Use same low-level API for articulations
            pose = torch.concat(
                [
                    position.to(self.handler.device, dtype=torch.float32) + self.handler.scene.env_origins[env_ids],
                    rotation.to(self.handler.device, dtype=torch.float32),
                ],
                dim=-1,
            )
            obj_inst.write_root_pose_to_sim(pose, env_ids=torch.tensor(env_ids, device=self.handler.device))
            obj_inst.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.handler.device, dtype=torch.float32),
                env_ids=torch.tensor(env_ids, device=self.handler.device),
            )
            obj_inst.write_data_to_sim()
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            # Use proper pose setting method for rigid objects
            pose = torch.concat(
                [
                    position.to(self.handler.device, dtype=torch.float32) + self.handler.scene.env_origins[env_ids],
                    rotation.to(self.handler.device, dtype=torch.float32),
                ],
                dim=-1,
            )
            obj_inst.write_root_pose_to_sim(pose, env_ids=torch.tensor(env_ids, device=self.handler.device))
            obj_inst.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.handler.device, dtype=torch.float32),
                env_ids=torch.tensor(env_ids, device=self.handler.device),
            )
            obj_inst.write_data_to_sim()
        else:
            raise ValueError(f"Object {obj_name} not found")

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> torch.Tensor:
        """Convert Euler angles (in radians) to quaternion [w, x, y, z]."""
        cr = torch.cos(torch.tensor(roll * 0.5))
        sr = torch.sin(torch.tensor(roll * 0.5))
        cp = torch.cos(torch.tensor(pitch * 0.5))
        sp = torch.sin(torch.tensor(pitch * 0.5))
        cy = torch.cos(torch.tensor(yaw * 0.5))
        sy = torch.sin(torch.tensor(yaw * 0.5))

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.tensor([w, x, y, z])

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    # Randomization Methods
    def randomize_physics(self) -> None:
        """Randomize physics properties based on configuration."""
        if not self.cfg.physics.enabled:
            return

        env_ids = self.cfg.env_ids or list(range(self.handler.num_envs))

        # Randomize mass
        if self.cfg.physics.mass_range is not None:
            current_mass = self.get_mass(self.cfg.obj_name, self.cfg.body_name, env_ids)
            if len(current_mass.shape) == 1:
                shape = (len(env_ids),)
            else:
                shape = (len(env_ids), current_mass.shape[1])

            rand_values = self._generate_random_tensor(
                shape, self.cfg.physics.distribution, self.cfg.physics.mass_range
            )
            # Ensure device compatibility
            rand_values = rand_values.to(current_mass.device)

            if self.cfg.physics.operation == "add":
                new_mass = current_mass + rand_values
            elif self.cfg.physics.operation == "scale":
                new_mass = current_mass * rand_values
            elif self.cfg.physics.operation == "abs":
                new_mass = rand_values
            else:
                raise ValueError(f"Unsupported operation: {self.cfg.physics.operation}")

            self.set_mass(self.cfg.obj_name, new_mass, self.cfg.body_name, env_ids)

        # Randomize friction
        if self.cfg.physics.friction_range is not None:
            current_friction = self.get_friction(self.cfg.obj_name, self.cfg.body_name, env_ids)
            if len(current_friction.shape) == 1:
                shape = (len(env_ids),)
            else:
                shape = (len(env_ids), current_friction.shape[1])

            rand_values = self._generate_random_tensor(
                shape, self.cfg.physics.distribution, self.cfg.physics.friction_range
            )
            # Ensure device compatibility
            rand_values = rand_values.to(current_friction.device)

            if self.cfg.physics.operation == "add":
                new_friction = current_friction + rand_values
            elif self.cfg.physics.operation == "scale":
                new_friction = current_friction * rand_values
            elif self.cfg.physics.operation == "abs":
                new_friction = rand_values
            else:
                raise ValueError(f"Unsupported operation: {self.cfg.physics.operation}")

            self.set_friction(self.cfg.obj_name, new_friction, self.cfg.body_name, env_ids)

        # Randomize restitution
        if self.cfg.physics.restitution_range is not None:
            current_restitution = self.get_restitution(self.cfg.obj_name, self.cfg.body_name, env_ids)
            if len(current_restitution.shape) == 1:
                shape = (len(env_ids),)
            else:
                shape = (len(env_ids), current_restitution.shape[1])

            rand_values = self._generate_random_tensor(
                shape, self.cfg.physics.distribution, self.cfg.physics.restitution_range
            )
            # Ensure device compatibility
            rand_values = rand_values.to(current_restitution.device)

            if self.cfg.physics.operation == "add":
                new_restitution = current_restitution + rand_values
            elif self.cfg.physics.operation == "scale":
                new_restitution = current_restitution * rand_values
            elif self.cfg.physics.operation == "abs":
                new_restitution = rand_values
            else:
                raise ValueError(f"Unsupported operation: {self.cfg.physics.operation}")

            self.set_restitution(self.cfg.obj_name, new_restitution, self.cfg.body_name, env_ids)

    def randomize_pose(self) -> None:
        """Randomize pose (position and rotation) based on configuration."""
        if not self.cfg.pose.enabled:
            return

        env_ids = self.cfg.env_ids or list(range(self.handler.num_envs))
        current_pos, current_rot = self.get_pose(self.cfg.obj_name, env_ids)

        new_pos = current_pos.clone()
        new_rot = current_rot.clone()

        # Randomize position
        if self.cfg.pose.position_range is not None:
            for axis, (min_val, max_val) in enumerate(self.cfg.pose.position_range):
                rand_values = self._generate_random_tensor(
                    (len(env_ids),), self.cfg.pose.distribution, (min_val, max_val)
                )
                # Ensure device compatibility
                rand_values = rand_values.to(current_pos.device)

                if self.cfg.pose.operation == "add":
                    new_pos[:, axis] = current_pos[:, axis] + rand_values
                elif self.cfg.pose.operation == "abs":
                    new_pos[:, axis] = rand_values
                else:
                    raise ValueError(f"Unsupported position operation: {self.cfg.pose.operation}")

            # Keep on ground if specified
            if self.cfg.pose.keep_on_ground:
                new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=0.0)

        # Randomize rotation
        if self.cfg.pose.rotation_range is not None:
            min_angle, max_angle = self.cfg.pose.rotation_range
            # Convert degrees to radians
            min_rad = torch.deg2rad(torch.tensor(min_angle))
            max_rad = torch.deg2rad(torch.tensor(max_angle))

            for env_idx in range(len(env_ids)):
                # Generate random rotations for each enabled axis
                rotations = []
                for axis_idx, enabled in enumerate(self.cfg.pose.rotation_axes):
                    if enabled:
                        angle = self._rng.uniform(min_rad.item(), max_rad.item())
                        rotations.append(angle)
                    else:
                        rotations.append(0.0)

                # Create quaternion from Euler angles
                delta_quat = self._euler_to_quaternion(rotations[0], rotations[1], rotations[2])
                # Ensure device compatibility
                delta_quat = delta_quat.to(current_rot.device)

                if self.cfg.pose.operation == "add":
                    # Multiply current rotation with delta rotation
                    new_rot[env_idx] = self._quaternion_multiply(
                        current_rot[env_idx].unsqueeze(0), delta_quat.unsqueeze(0)
                    ).squeeze(0)
                elif self.cfg.pose.operation == "abs":
                    # Set absolute rotation
                    new_rot[env_idx] = delta_quat
                else:
                    raise ValueError(f"Unsupported rotation operation: {self.cfg.pose.operation}")

        self.set_pose(self.cfg.obj_name, new_pos, new_rot, env_ids)

    def __call__(self) -> None:
        """Execute object randomization based on configuration."""
        pose_updated = False

        if self.cfg.physics.enabled:
            self.randomize_physics()

        if self.cfg.pose.enabled:
            self.randomize_pose()
            pose_updated = True

        if pose_updated:
            self._mark_visual_dirty()

    # Getter methods for backward compatibility and debugging
    def get_properties(self) -> dict[str, Any]:
        """Get current object properties for debugging/logging."""
        env_ids = self.cfg.env_ids or list(range(self.handler.num_envs))
        properties = {}

        try:
            if self.cfg.physics.enabled:
                if self.cfg.physics.mass_range is not None:
                    properties["mass"] = self.get_mass(self.cfg.obj_name, self.cfg.body_name, env_ids)
                if self.cfg.physics.friction_range is not None:
                    properties["friction"] = self.get_friction(self.cfg.obj_name, self.cfg.body_name, env_ids)
                if self.cfg.physics.restitution_range is not None:
                    properties["restitution"] = self.get_restitution(self.cfg.obj_name, self.cfg.body_name, env_ids)

            if self.cfg.pose.enabled:
                pos, rot = self.get_pose(self.cfg.obj_name, env_ids)
                properties["position"] = pos
                properties["rotation"] = rot

        except Exception as e:
            properties["error"] = str(e)

        return properties
