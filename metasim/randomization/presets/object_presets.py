"""Object randomization presets for common use cases."""

from __future__ import annotations

from ..object_randomizer import ObjectRandomCfg, PhysicsRandomCfg, PoseRandomCfg


class ObjectPresets:
    """Predefined object randomization configurations for common use cases."""

    @staticmethod
    def physics_only(
        obj_name: str,
        mass_delta_range: tuple[float, float] | None = None,
        friction_delta_range: tuple[float, float] | None = None,
        restitution_delta_range: tuple[float, float] | None = None,
        distribution: str = "uniform",
        body_name: str | None = None,
        env_ids: list[int] | None = None,
    ) -> ObjectRandomCfg:
        """Create configuration for physics-only randomization.

        Args:
            obj_name: Name of the object to randomize
            mass_delta_range: Delta range for mass randomization (kg). None to disable.
            friction_delta_range: Delta range for friction randomization. None to disable.
            restitution_delta_range: Delta range for restitution randomization. None to disable.
            distribution: Distribution type ("uniform", "log_uniform", "gaussian")
            body_name: Specific body name or None for all bodies
            env_ids: Environment IDs or None for all environments
        """
        return ObjectRandomCfg(
            obj_name=obj_name,
            body_name=body_name,
            env_ids=env_ids,
            physics=PhysicsRandomCfg(
                enabled=True,
                mass_delta_range=mass_delta_range,
                friction_delta_range=friction_delta_range,
                restitution_delta_range=restitution_delta_range,
                distribution=distribution,
                use_delta=True,
                use_scale=True,
            ),
            pose=PoseRandomCfg(enabled=False),
        )

    @staticmethod
    def pose_only(
        obj_name: str,
        position_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        rotation_delta_range: tuple[float, float] | None = None,
        rotation_axes: tuple[bool, bool, bool] = (True, True, True),
        distribution: str = "uniform",
        keep_on_ground: bool = True,
        env_ids: list[int] | None = None,
    ) -> ObjectRandomCfg:
        """Create configuration for pose-only randomization.

        Args:
            obj_name: Name of the object to randomize
            position_delta_range: Delta range for position randomization [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            rotation_delta_range: Delta range for rotation randomization in degrees (min, max)
            rotation_axes: Which axes to randomize rotation around (x, y, z)
            distribution: Distribution type ("uniform", "gaussian")
            keep_on_ground: Whether to keep object on ground (z >= 0)
            env_ids: Environment IDs or None for all environments
        """
        return ObjectRandomCfg(
            obj_name=obj_name,
            env_ids=env_ids,
            physics=PhysicsRandomCfg(enabled=False),
            pose=PoseRandomCfg(
                enabled=True,
                position_delta_range=position_delta_range,
                rotation_delta_range=rotation_delta_range,
                rotation_axes=rotation_axes,
                distribution=distribution,
                keep_on_ground=keep_on_ground,
                use_delta=True,
            ),
        )

    @staticmethod
    def combined(
        obj_name: str,
        mass_delta_range: tuple[float, float] | None = None,
        friction_delta_range: tuple[float, float] | None = None,
        restitution_delta_range: tuple[float, float] | None = None,
        position_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        rotation_delta_range: tuple[float, float] | None = None,
        rotation_axes: tuple[bool, bool, bool] = (True, True, True),
        physics_distribution: str = "uniform",
        pose_distribution: str = "uniform",
        keep_on_ground: bool = True,
        body_name: str | None = None,
        env_ids: list[int] | None = None,
    ) -> ObjectRandomCfg:
        """Create configuration for combined physics and pose randomization.

        Args:
            obj_name: Name of the object to randomize
            mass_delta_range: Delta range for mass randomization (kg). None to disable.
            friction_delta_range: Delta range for friction randomization. None to disable.
            restitution_delta_range: Delta range for restitution randomization. None to disable.
            position_delta_range: Delta range for position randomization [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            rotation_delta_range: Delta range for rotation randomization in degrees (min, max)
            rotation_axes: Which axes to randomize rotation around (x, y, z)
            physics_distribution: Distribution type for physics ("uniform", "log_uniform", "gaussian")
            pose_distribution: Distribution type for pose ("uniform", "gaussian")
            keep_on_ground: Whether to keep object on ground (z >= 0)
            body_name: Specific body name or None for all bodies
            env_ids: Environment IDs or None for all environments
        """
        physics_enabled = any([mass_delta_range, friction_delta_range, restitution_delta_range])
        pose_enabled = any([position_delta_range, rotation_delta_range])

        return ObjectRandomCfg(
            obj_name=obj_name,
            body_name=body_name,
            env_ids=env_ids,
            physics=PhysicsRandomCfg(
                enabled=physics_enabled,
                mass_delta_range=mass_delta_range,
                friction_delta_range=friction_delta_range,
                restitution_delta_range=restitution_delta_range,
                distribution=physics_distribution,
                use_delta=True,
                use_scale=True,
            ),
            pose=PoseRandomCfg(
                enabled=pose_enabled,
                position_delta_range=position_delta_range,
                rotation_delta_range=rotation_delta_range,
                rotation_axes=rotation_axes,
                distribution=pose_distribution,
                keep_on_ground=keep_on_ground,
                use_delta=True,
            ),
        )

    # Common use case presets
    @staticmethod
    def light_object(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for light objects (small variations in mass and friction, position jitter)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.8, 1.2),  # ±20% mass variation
            friction_delta_range=(0.7, 1.3),  # ±30% friction variation
            position_delta_range=[(-0.05, 0.05), (-0.05, 0.05), (0.0, 0.02)],  # Small position jitter
            rotation_delta_range=(-10, 10),  # ±10 degree rotation
            physics_distribution="gaussian",
            pose_distribution="gaussian",
        )

    @staticmethod
    def heavy_object(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for heavy objects (larger mass variations, less position jitter)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.5, 2.0),  # 2x-4x mass variation
            friction_delta_range=(0.5, 1.5),  # ±50% friction variation
            restitution_delta_range=(0.1, 0.3),  # Low bounce for heavy objects
            position_delta_range=[(-0.02, 0.02), (-0.02, 0.02), (0.0, 0.01)],  # Minimal position jitter
            rotation_delta_range=(-5, 5),  # ±5 degree rotation
            physics_distribution="log_uniform",
            pose_distribution="uniform",
        )

    @staticmethod
    def bouncy_object(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for bouncy objects (high restitution, varied physics)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.6, 1.4),  # ±40% mass variation
            friction_delta_range=(0.3, 0.8),  # Lower friction for bounciness
            restitution_delta_range=(0.6, 0.9),  # High bounce
            position_delta_range=[(-0.1, 0.1), (-0.1, 0.1), (0.0, 0.05)],  # More position variation
            rotation_delta_range=(-20, 20),  # ±20 degree rotation
            physics_distribution="uniform",
            pose_distribution="uniform",
        )

    @staticmethod
    def robot_base(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for robot base randomization (mainly mass and friction, minimal pose)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.7, 1.3),  # ±30% mass variation for payload simulation
            friction_delta_range=(0.8, 1.2),  # ±20% friction variation
            position_delta_range=[(-0.01, 0.01), (-0.01, 0.01), (0.0, 0.0)],  # Minimal base movement
            rotation_delta_range=(-2, 2),  # ±2 degree rotation (base alignment errors)
            rotation_axes=(False, False, True),  # Only Z-axis rotation
            physics_distribution="gaussian",
            pose_distribution="gaussian",
        )

    @staticmethod
    def grasping_target(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for objects that will be grasped (varied position and moderate physics)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.5, 1.5),  # Varied mass for grasping challenge
            friction_delta_range=(0.4, 1.6),  # Wide friction range for grip variation
            position_delta_range=[(-0.15, 0.15), (-0.15, 0.15), (0.0, 0.1)],  # Varied starting positions
            rotation_delta_range=(-45, 45),  # ±45 degree rotation for orientation challenge
            physics_distribution="log_uniform",
            pose_distribution="uniform",
            keep_on_ground=True,
        )

    @staticmethod
    def manipulation_tool(obj_name: str, env_ids: list[int] | None = None) -> ObjectRandomCfg:
        """Preset for manipulation tools (moderate variations, realistic constraints)."""
        return ObjectPresets.combined(
            obj_name=obj_name,
            env_ids=env_ids,
            mass_delta_range=(0.8, 1.2),  # ±20% mass variation
            friction_delta_range=(0.9, 1.1),  # ±10% friction variation (tools should be predictable)
            position_delta_range=[(-0.03, 0.03), (-0.03, 0.03), (0.0, 0.01)],  # Small position variation
            rotation_delta_range=(-15, 15),  # ±15 degree rotation
            physics_distribution="gaussian",
            pose_distribution="gaussian",
        )
