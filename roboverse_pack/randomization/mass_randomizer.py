from __future__ import annotations
from metasim.sim.randomizaer.base import BaseRandomizerType
import torch
from typing import Literal
from typing import Any
from metasim.utils.configclass import configclass


@configclass
class MassRandomCfg:
    obj_name: str | None = None
    range: tuple[float, float] = (0.0, 0.0)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    operation: Literal["add", "scale", "abs"] = "scale"
    body_name: str | None = None
    env_ids: list[int] | None = None


class MassRandomizer(BaseRandomizerType):
    """Mass randomizer for domain randomization."""

    def __init__(self, cfg: MassRandomCfg|None=None):
        super().__init__()
        if cfg is None:
            raise ValueError("MassRandomizer requires a MassRandomCfg before called")
        self.cfg = cfg

    def bind_handler(self, handler, *args: Any, **kwargs):
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            super().bind_handler(handler, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for MassRandomizer")

    def _get_body_names(self, obj_name: str) -> list[str]:
        """Get body names for an object."""
        if hasattr(self.handler, "_get_body_names"):
            return self._get_body_names(obj_name)
        else:
            # Fallback implementation
            if obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[obj_name]
                # This is a simplified approach - actual implementation may vary
                return [f"body_{i}" for i in range(obj_inst.root_physx_view.get_masses().shape[1])]
            return []

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
                masses[env_ids, body_idx] = mass
            else:
                # Set all body masses
                masses[env_ids, :] = mass

            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            # TODO: check why obj_inst is cpu
            masses = obj_inst.root_physx_view.get_masses()
            masses[env_ids] = mass
            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids))
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
    ) -> None:
        """Randomize mass using handler's get/set APIs."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        current_mass = self.get_body_mass(obj_name, body_name, env_ids)
        num_bodies = current_mass.shape[1]

        if distribution == "uniform":
            rand_values = torch.rand([len(env_ids), num_bodies]) * (mass_range[1] - mass_range[0]) + mass_range[0]
        elif distribution == "log_uniform":
            log_min, log_max = torch.log(torch.tensor(mass_range[0])), torch.log(torch.tensor(mass_range[1]))
            rand_values = torch.exp(torch.rand([len(env_ids), num_bodies]) * (log_max - log_min) + log_min)
        elif distribution == "gaussian":
            mean = (mass_range[0] + mass_range[1]) / 2
            std = (mass_range[1] - mass_range[0]) / 6
            rand_values = torch.normal(mean, std, size=(len(env_ids), num_bodies))
            rand_values = torch.clamp(rand_values, mass_range[0], mass_range[1])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        if operation == "add":
            new_mass = current_mass + rand_values
        elif operation == "scale":
            new_mass = current_mass * rand_values
        elif operation == "abs":
            new_mass = rand_values
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        self.set_body_mass(obj_name, new_mass, body_name, env_ids)

    def __call__(self):
        """Execute mass randomization based on configuration."""

        # Randomize mass
        self.randomize_body_mass(
            obj_name=self.cfg.obj_name,
            mass_range=self.cfg.range,
            body_name=self.cfg.body_name,
            env_ids=self.cfg.env_ids,
            operation=self.cfg.operation,
            distribution=self.cfg.distribution,
        )
