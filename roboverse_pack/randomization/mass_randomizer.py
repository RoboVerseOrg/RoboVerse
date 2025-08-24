from __future__ import annotations
from metasim.sim.randomizaer.base import BaseRandomizerType
import torch
from typing import Literal
from typing import Any


class MassRandomCfg:
    enabled: bool = False
    obj_name: str | None = None
    range: tuple[float, float] = (0.0, 0.0)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    operation: Literal["add", "scale", "abs"] = "scale"
    body_name: str | None = None
    env_ids: list[int] | None = None


class FrictionRandomCfg:
    enabled: bool = False
    obj_name: str | None = None
    range: tuple[float, float] = (0.0, 0.0)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    operation: Literal["add", "scale", "abs"] = "scale"
    body_name: str | None = None
    env_ids: list[int] | None = None


class MassRandomizer(BaseRandomizerType):
    """Mass and friction randomizer for domain randomization."""

    def __init__(self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None):
        super().__init__()
        self.obj_name = obj_name
        self.body_name = body_name
        self.env_ids = env_ids

    def bind_handler(self, handler, *args: Any, **kwargs):
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            super().bind_handler(handler, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for MassRandomizer")

    def get_body_mass(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """Get the mass of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            body_name (str, optional): Name of the specific body. If None, returns mass of all bodies
            env_ids (list[int], optional): List of environment ids. If None, returns for all environments

        Returns:
            torch.Tensor: Mass values with shape (num_envs, num_bodies) or (num_envs,) if body_name is specified
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                # Get specific body mass
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return masses[env_ids, body_idx]
            else:
                # Get all body masses
                return masses[env_ids, :]
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            masses = obj_inst.root_physx_view.get_masses()
            return masses[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_body_mass(
        self,
        obj_name: str,
        mass: torch.Tensor,
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        """Set the mass of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            mass (torch.Tensor): Mass values to set
            body_name (str, optional): Name of the specific body. If None, sets mass for all bodies
            env_ids (list[int], optional): List of environment ids. If None, sets for all environments
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                # Set specific body mass
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                masses[env_ids, body_idx] = mass.to(device)
            else:
                # Set all body masses
                masses[env_ids, :] = mass.to(device)

            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids, device=device))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            # TODO: check why obj_inst is cpu
            masses = obj_inst.root_physx_view.get_masses()
            masses[env_ids] = mass.to(masses.device)
            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids, device=device))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def get_body_friction(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """Get the friction coefficient of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            body_name (str, optional): Name of the specific body. If None, returns friction of all bodies
            env_ids (list[int], optional): List of environment ids. If None, returns for all environments

        Returns:
            torch.Tensor: Friction values with shape (num_envs, num_bodies) or (num_envs,) if body_name is specified
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction

            if body_name is not None:
                # Get specific body friction
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return friction[env_ids, body_idx]
            else:
                # Get all body friction
                return friction[env_ids, :]
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction
            return friction[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def set_body_friction(
        self,
        obj_name: str,
        friction: torch.Tensor,
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        """Set the friction coefficient of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            friction (torch.Tensor): Friction values to set
            body_name (str, optional): Name of the specific body. If None, sets friction for all bodies
            env_ids (list[int], optional): List of environment ids. If None, sets for all environments
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        if obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()

            if body_name is not None:
                # Set specific body friction
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                materials[env_ids, body_idx, 0] = friction.to(device)  # Static friction
                materials[env_ids, body_idx, 1] = friction.to(device)  # Dynamic friction
            else:
                # Set all body friction
                materials[env_ids, :, 0] = friction.to(device)  # Static friction
                materials[env_ids, :, 1] = friction.to(device)  # Dynamic friction

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids, device=device))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            materials[env_ids, 0] = friction.to(device)  # Static friction
            materials[env_ids, 1] = friction.to(device)  # Dynamic friction
            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids, device=device))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def randomize_body_mass(
        self,
        obj_name: str,
        mass_range: tuple[float, float],
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        operation: str = "scale",
        distribution: str = "uniform",
        device: str = "cpu",
    ) -> None:
        """Randomize mass using handler's get/set APIs."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        current_mass = self.get_body_mass(obj_name, body_name, env_ids)

        if distribution == "uniform":
            rand_values = torch.rand(len(env_ids), device=device) * (mass_range[1] - mass_range[0]) + mass_range[0]
        elif distribution == "log_uniform":
            log_min, log_max = torch.log(torch.tensor(mass_range[0])), torch.log(torch.tensor(mass_range[1]))
            rand_values = torch.exp(torch.rand(len(env_ids), device=device) * (log_max - log_min) + log_min)
        elif distribution == "gaussian":
            mean = (mass_range[0] + mass_range[1]) / 2
            std = (mass_range[1] - mass_range[0]) / 6
            rand_values = torch.normal(mean, std, size=(len(env_ids),), device=device)
            rand_values = torch.clamp(rand_values, mass_range[0], mass_range[1])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        if operation == "add":
            new_mass = current_mass + (
                rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
            )
        elif operation == "scale":
            new_mass = current_mass * (
                rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
            )
        elif operation == "abs":
            new_mass = rand_values.unsqueeze(-1) if body_name is None and current_mass.ndim == 2 else rand_values
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        self.set_body_mass(obj_name, new_mass, body_name, env_ids, device)

    def randomize_body_friction(
        self,
        obj_name: str,
        friction_range: tuple[float, float],
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        operation: str = "scale",
        distribution: str = "uniform",
        device: str = "cpu",
    ) -> None:
        """Randomize friction using handler's get/set APIs."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        current_friction = self.get_body_friction(obj_name, body_name, env_ids)

        if distribution == "uniform":
            rand_values = (
                torch.rand(len(env_ids), device=device) * (friction_range[1] - friction_range[0]) + friction_range[0]
            )
        elif distribution == "log_uniform":
            log_min, log_max = torch.log(torch.tensor(friction_range[0])), torch.log(torch.tensor(friction_range[1]))
            rand_values = torch.exp(torch.rand(len(env_ids), device=device) * (log_max - log_min) + log_min)
        elif distribution == "gaussian":
            mean = (friction_range[0] + friction_range[1]) / 2
            std = (friction_range[1] - friction_range[0]) / 6
            rand_values = torch.normal(mean, std, size=(len(env_ids),), device=device)
            rand_values = torch.clamp(rand_values, friction_range[0], friction_range[1])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        if operation == "add":
            new_friction = current_friction + (
                rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
            )
        elif operation == "scale":
            new_friction = current_friction * (
                rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
            )
        elif operation == "abs":
            new_friction = (
                rand_values.unsqueeze(-1) if body_name is None and current_friction.ndim == 2 else rand_values
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        self.set_body_friction(obj_name, new_friction, body_name, env_ids, device)

    def __call__(self, cfg: MassRandomCfg):
        """Execute mass randomization based on configuration."""
        if not cfg.enabled:
            return

        # Use object name from config or fallback to instance variable
        obj_name = getattr(cfg, "obj_name", self.obj_name)
        body_name = cfg.body_name or self.body_name
        env_ids = cfg.env_ids or self.env_ids
        device = self.handler.device if hasattr(self.handler, "device") else "cpu"

        # Randomize mass
        if hasattr(cfg, "mass") and cfg.mass.enabled:
            self.randomize_body_mass(
                obj_name=obj_name,
                mass_range=cfg.mass.range,
                body_name=body_name,
                env_ids=env_ids,
                operation=cfg.mass.operation,
                distribution=cfg.mass.distribution,
                device=device,
            )
        else:
            # Direct mass randomization if mass config is not nested
            self.randomize_body_mass(
                obj_name=obj_name,
                mass_range=cfg.range,
                body_name=body_name,
                env_ids=env_ids,
                operation=cfg.operation,
                distribution=cfg.distribution,
                device=device,
            )

        # Randomize friction if friction config is provided
        if hasattr(cfg, "friction") and cfg.friction.enabled:
            self.randomize_body_friction(
                obj_name=obj_name,
                friction_range=cfg.friction.range,
                body_name=body_name,
                env_ids=env_ids,
                operation=cfg.friction.operation,
                distribution=cfg.friction.distribution,
                device=device,
            )
