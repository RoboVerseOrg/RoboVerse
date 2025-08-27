from __future__ import annotations

import random
from typing import Any, Literal

import torch

from metasim.sim.randomizaer.base import BaseRandomizerType
from metasim.utils.configclass import configclass


@configclass
class FrictionRandomCfg:
    """Configuration for the friction randomizer."""

    obj_name: str | None = None
    range: tuple[float, float] = (0.0, 0.0)
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    operation: Literal["add", "scale", "abs"] = "scale"
    body_name: str | None = None
    env_ids: list[int] | None = None


class FrictionRandomizer(BaseRandomizerType):
    """Friction randomizer for domain randomization."""

    def __init__(self, cfg: FrictionRandomCfg | None = None, seed: int | None = None):
        super().__init__()
        if cfg is None:
            raise ValueError("FrictionRandomizer requires a cFrictionRandomCfg before called")
        self.cfg = cfg

        # Set up reproducible random state
        if seed is not None:
            # Use provided seed + simple string-to-number conversion for uniqueness
            name_sum = sum(ord(c) for c in (cfg.obj_name or "friction"))
            self._seed = seed + name_sum
        else:
            self._seed = random.randint(0, 2**32 - 1)

        self._rng = random.Random(self._seed)

    def _generate_random_tensor(self, shape, distribution: str, range_vals: tuple[float, float]):
        """Generate random tensor using our reproducible RNG."""
        if distribution == "uniform":
            # Generate uniform random values using our RNG
            rand_vals = [
                [self._rng.uniform(range_vals[0], range_vals[1]) for _ in range(shape[1])] for _ in range(shape[0])
            ]
            return torch.tensor(rand_vals, dtype=torch.float32)
        elif distribution == "log_uniform":
            # Generate log-uniform values
            log_min, log_max = torch.log(torch.tensor(range_vals[0])), torch.log(torch.tensor(range_vals[1]))
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
            raise ValueError(f"Unsupported handler type: {type(handler)} for FrictionRandomizer")

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
                materials[env_ids, body_idx, 0] = friction  # Static friction
                materials[env_ids, body_idx, 1] = friction  # Dynamic friction
            else:
                # Set all body friction
                materials[env_ids, :, 0] = friction  # Static friction
                materials[env_ids, :, 1] = friction  # Dynamic friction

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        elif obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            materials[env_ids, 0] = friction  # Static friction
            materials[env_ids, 1] = friction  # Dynamic friction
            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def randomize_body_friction(
        self,
        obj_name: str,
        friction_range: tuple[float, float],
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        operation: str = "scale",
        distribution: str = "uniform",
    ) -> None:
        """Randomize friction using handler's get/set APIs."""
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        current_friction = self.get_body_friction(obj_name, body_name, env_ids)
        num_bodies = current_friction.shape[1]

        # Use our reproducible random tensor generation
        rand_values = self._generate_random_tensor([len(env_ids), num_bodies], distribution, friction_range)

        if operation == "add":
            new_friction = current_friction + rand_values
        elif operation == "scale":
            new_friction = current_friction * rand_values
        elif operation == "abs":
            new_friction = rand_values
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        self.set_body_friction(obj_name, new_friction, body_name, env_ids)

    def __call__(self):
        """Execute friction randomization based on configuration."""
        # Randomize friction
        self.randomize_body_friction(
            obj_name=self.cfg.obj_name,
            friction_range=self.cfg.range,
            body_name=self.cfg.body_name,
            env_ids=self.cfg.env_ids,
            operation=self.cfg.operation,
            distribution=self.cfg.distribution,
        )
