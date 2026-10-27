# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab (https://github.com/isaac-sim/IsaacLab).
# Changes: only the helpers under the 'adapted from Isaac Lab' marker (randomize_prop_by_op, _validate_scale_range)
# and the bucketed sampling in MaterialRandomizer are Isaac Lab-derived; the randomizer classes are RoboVerse code.
# Full license: roboverse_pack/randomization/LICENSE.isaaclab

"""Physical-property randomizers (body materials and masses) for the humanoid and tracking tasks."""

from __future__ import annotations

from typing import Literal

import torch
from loguru import logger as log

from metasim.sim.base import BaseQueryType, BaseSimHandler
from metasim.utils.math import sample_gaussian, sample_log_uniform, sample_uniform

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass


class MaterialRandomizer(BaseQueryType):
    """Randomize material properties (friction, restitution) for selected bodies."""

    handler: BaseSimHandler

    def __init__(
        self,
        obj_name: str,
        body_names: list[str] | str | None = None,
        static_friction_range: list | tuple = (1.0, 1.0),
        dynamic_friction_range: list | tuple = (1.0, 1.0),
        restitution_range: list | tuple = (0.0, 0.0),
        num_buckets: int = 1,
        make_consistent: bool = False,
    ):
        super().__init__()
        self.obj_name = obj_name
        self.set_body_names = [body_names] if isinstance(body_names, str) else body_names
        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.num_buckets = num_buckets
        self.make_consistent = make_consistent

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Attach to the simulator handler and initialize randomization buffers."""
        super().bind_handler(handler, *args, **kwargs)
        self.simulator_name = handler.scenario.simulator
        self.initialize()

    def __call__(self, env_ids=None, **kwargs):
        """Apply material randomization for provided environment ids."""
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs, device="cpu")
        else:
            env_ids = torch.tensor(env_ids).cpu()
        self.randomize(env_ids)

    def initialize(self):
        """Sample material buckets and cache body indices for randomization."""
        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [
            self.static_friction_range,
            self.dynamic_friction_range,
            self.restitution_range,
        ]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = sample_uniform(ranges[:, 0], ranges[:, 1], (self.num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        if self.make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.body_names = self.handler.get_body_names(self.obj_name, sort=False)
        self.set_body_ids = (
            torch.tensor(
                [self.body_names.index(_name) for _name in self.set_body_names],
                dtype=torch.int,
                device="cpu",
            )
            if self.set_body_names is not None
            else torch.arange(len(self.body_names), dtype=torch.int, device="cpu")
        )

        self.all_robot_names = [robot.name for robot in self.handler.robots]
        self.all_object_names = [obj.name for obj in self.handler.objects]

        self.set_shape_indices = self._get_set_shape_indices()
        if self.simulator_name == "newton":
            self._newton_shape_indices = self._build_newton_shape_indices()

    def _get_set_shape_indices(self):
        num_shapes_per_body = None
        expected_shapes = 0
        if self.simulator_name == "isaacsim":
            if self.obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[self.obj_name]
                # obtain number of shapes per body (needed for indexing the material properties correctly)
                # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
                #  per body. We use the physics simulation view to obtain the number of shapes per body.
                num_shapes_per_body = []
                for link_path in obj_inst.root_physx_view.link_paths[0]:
                    link_physx_view = obj_inst._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                    num_shapes_per_body.append(link_physx_view.max_shapes)
                # ensure the parsing is correct
                expected_shapes = obj_inst.root_physx_view.max_shapes
            elif self.obj_name in self.handler.scene.rigid_objects:
                obj_inst = self.handler.scene.rigid_objects[self.obj_name]
                expected_shapes = obj_inst.root_physx_view.max_shapes
        elif self.simulator_name == "isaacgym":
            if self.obj_name in self.all_robot_names:
                _tmp_handle = self.handler._robot_handles[0]
            elif self.obj_name in self.all_object_names:
                _tmp_handle = self.handler._obj_handles[0][self.handler.objects.index(self.obj_name)]
            body_shape_indices = self.handler.gym.get_actor_rigid_body_shape_indices(self.handler._envs[0], _tmp_handle)
            num_shapes_per_body = []
            for body_shape in body_shape_indices:
                num_shapes_per_body.append(body_shape.count)
            expected_shapes = len(self.handler.gym.get_actor_rigid_shape_properties(self.handler._envs[0], _tmp_handle))
        elif self.simulator_name == "mujoco":
            model = self.handler.physics.model
            num_shapes_per_body = [0] * model.nbody
            # geom_bodyid[j] = geom j belongs to body geom_bodyid[j]
            for geom_bodyid in model.geom_bodyid:
                num_shapes_per_body[geom_bodyid] += 1
            expected_shapes = model.ngeom
        elif self.simulator_name == "newton":
            return []
        else:
            return []
        if num_shapes_per_body is not None and sum(num_shapes_per_body) != expected_shapes:
            raise ValueError(
                "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {sum(num_shapes_per_body)}."
            )
        # update material buffer with new samples
        if num_shapes_per_body is not None:
            set_shape_indices = []
            # sample material properties from the given ranges
            for body_id in self.set_body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(num_shapes_per_body[:body_id])
                end_idx = start_idx + num_shapes_per_body[body_id]
                set_shape_indices.extend(list(range(start_idx, end_idx)))
                # assign the new materials
        else:
            # assign all the materials
            set_shape_indices = list(range(expected_shapes))

        return set_shape_indices

    def _build_newton_shape_indices(self) -> dict[int, list[int]]:
        """Build per-env shape index lists for Newton."""
        model = getattr(self.handler, "_model", None)
        if model is None or not hasattr(model, "body_shapes"):
            return {}

        body_names = self.body_names
        target_names = self.set_body_names if self.set_body_names is not None else body_names
        target_names = [name for name in target_names if name in body_names]

        shape_indices_by_env: dict[int, list[int]] = {}
        for env_id in range(self.handler.num_envs):
            body_ids = self.handler._get_body_indices(env_id, self.obj_name)
            if not body_ids:
                continue
            name_to_body = {model.body_key[idx].split("/")[-1]: idx for idx in body_ids}  # as get_body_names
            shape_indices: list[int] = []
            for name in target_names:
                body_id = name_to_body.get(name)
                if body_id is None:
                    continue
                shape_indices.extend(list(model.body_shapes.get(body_id, [])))
            if shape_indices:
                shape_indices_by_env[env_id] = sorted(set(shape_indices))

        return shape_indices_by_env

    def randomize(self, env_ids: torch.Tensor):
        """Randomize material properties across supported simulators."""
        if self.simulator_name == "isaacsim":
            self._randomize_isaacsim(env_ids)
        elif self.simulator_name == "isaacgym":
            self._randomize_isaacgym(env_ids)
        elif self.simulator_name == "mujoco":
            self._randomize_mujoco(env_ids)
        elif self.simulator_name == "newton":
            self._randomize_newton(env_ids)
        else:
            log.warning(
                f"Material randomization not implemented for simulator: {self.simulator_name}. This randomization step will be skipped."
            )

    def _randomize_isaacsim(self, env_ids: torch.Tensor):
        if self.obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[self.obj_name]
        elif self.obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[self.obj_name]
        else:
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: {self.obj_name}."
            )

        # retrieve material buffer from the physics simulation
        materials = obj_inst.root_physx_view.get_material_properties()
        # randomly assign material IDs to the geometries
        total_num_shapes = obj_inst.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, self.num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # update material buffer with new samples
        materials[env_ids] = material_samples[:, self.set_shape_indices]

        # apply to simulation
        obj_inst.root_physx_view.set_material_properties(materials, env_ids)

    def _randomize_isaacgym(self, env_ids: torch.Tensor):
        """Randomize friction properties for IsaacGym simulator."""
        # Sample friction values for each environment
        if self.obj_name in self.all_robot_names:
            # For robot, get actor handle and modify rigid shape properties
            _all_handles = self.handler._robot_handles
        elif self.obj_name in self.all_object_names:
            # For objects, find the corresponding object handle
            _all_handles = [
                self.handler._obj_handles[i][self.handler.objects.index(self.obj_name)]
                for i in range(self.handler.num_envs)
            ]
        else:
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: {self.obj_name}."
            )
        max_body_shape = len(self.handler.gym.get_actor_rigid_shape_properties(self.handler._envs[0], _all_handles[0]))
        bucket_ids = torch.randint(0, self.num_buckets, (len(env_ids), max_body_shape), device="cpu")
        material_samples = self.material_buckets[bucket_ids]  # static friction, dynamic friction and restitution

        roll_friction_factor = 0.05
        spin_friction_factor = 0.02
        for i, env_id in enumerate(env_ids):
            env = self.handler._envs[env_id]
            # For objects, find the corresponding object handle
            _tmp_handle = _all_handles[env_id]
            # Get current rigid shape properties
            shape_props = self.handler.gym.get_actor_rigid_shape_properties(env, _tmp_handle)

            for _id in self.set_shape_indices:
                shape_prop = shape_props[_id]
                shape_prop.friction = material_samples[i, _id, 0]
                shape_prop.rolling_friction = roll_friction_factor * material_samples[i, _id, 1]
                shape_prop.torsion_friction = spin_friction_factor * material_samples[i, _id, 1]
                shape_prop.restitution = material_samples[i, _id, 2]

            # Apply the modified properties
            self.handler.gym.set_actor_rigid_shape_properties(env, _tmp_handle, shape_props)

    def _randomize_mujoco(self, env_ids: torch.Tensor):
        """Randomize friction and restitution for MuJoCo simulator."""
        assert self.handler.num_envs == 1, "MuJoCo handler only supports single environment."
        model = self.handler.physics.model

        bucket_ids = torch.randint(0, self.num_buckets, (len(env_ids), model.ngeom), device="cpu")
        material_samples = self.material_buckets[bucket_ids]  # static friction, dynamic friction and restitution

        static_friction = material_samples[env_ids, self.set_shape_indices, 0]
        solimp_value = 0.1 * static_friction
        model.geom_solimp[self.set_shape_indices, 0] = solimp_value

        # model.geom_friction --> friction for (slide, spin, roll)
        dynamic_friction = material_samples[env_ids, self.set_shape_indices, 1]
        model.geom_friction[self.set_shape_indices, 0] = dynamic_friction  # slide friction
        model.geom_friction[self.set_shape_indices, 1] = 0.01 * dynamic_friction  # spin friction
        model.geom_friction[self.set_shape_indices, 2] = 0.01 * dynamic_friction  # roll friction

        # restitution and damping calculation
        restitution_scale = 1.0  # from 0.5 - 2.0
        restitution = material_samples[env_ids, self.set_shape_indices, 2] * restitution_scale + 1e-6
        damping = (-torch.log(restitution) / torch.sqrt(torch.pi**2 + torch.log(restitution) ** 2)).clamp(
            min=0.0, max=1.0
        )

        # solref：timeconst & damping ratio
        model.geom_solref[self.set_shape_indices, 1] = damping

    def _randomize_newton(self, env_ids: torch.Tensor):
        """Randomize material properties for Newton simulator."""
        model = getattr(self.handler, "_model", None)
        if model is None or model.shape_material_mu is None:
            return

        shape_mu = model.shape_material_mu.numpy()
        shape_restitution = (
            model.shape_material_restitution.numpy() if model.shape_material_restitution is not None else None
        )
        shape_torsion = (
            model.shape_material_torsional_friction.numpy()
            if model.shape_material_torsional_friction is not None
            else None
        )
        shape_rolling = (
            model.shape_material_rolling_friction.numpy() if model.shape_material_rolling_friction is not None else None
        )

        roll_friction_factor = 0.05
        spin_friction_factor = 0.02

        for env_id in env_ids.tolist():
            shape_indices = self._newton_shape_indices.get(int(env_id), [])
            if not shape_indices:
                continue
            bucket_ids = torch.randint(0, self.num_buckets, (len(shape_indices),), device="cpu")
            material_samples = self.material_buckets[bucket_ids].cpu().numpy()

            shape_mu[shape_indices] = material_samples[:, 0]
            if shape_restitution is not None:
                shape_restitution[shape_indices] = material_samples[:, 2]
            if shape_torsion is not None:
                shape_torsion[shape_indices] = spin_friction_factor * material_samples[:, 1]
            if shape_rolling is not None:
                shape_rolling[shape_indices] = roll_friction_factor * material_samples[:, 1]

        model.shape_material_mu.assign(shape_mu)
        if shape_restitution is not None:
            model.shape_material_restitution.assign(shape_restitution)
        if shape_torsion is not None:
            model.shape_material_torsional_friction.assign(shape_torsion)
        if shape_rolling is not None:
            model.shape_material_rolling_friction.assign(shape_rolling)


class MassRandomizer(BaseQueryType):
    """Randomize body masses of a robot or articulated object.

    Each call samples around the masses the bodies had when the randomizer was bound (so repeated calls do
    not drift) and writes them with ``handler.set_body_masses``, which also scales each body's inertia by
    the mass ratio when ``recompute_inertia`` is set. Runs on every backend that implements body masses.
    """

    handler: BaseSimHandler

    def __init__(
        self,
        obj_name: str,
        body_names: list[str] | str | None = None,
        mass_distribution_params: list | tuple = (-1.0, 3.0),
        operation: Literal["add", "scale", "abs"] = "add",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
    ):
        super().__init__()
        self.obj_name = obj_name
        self.set_body_names = [body_names] if isinstance(body_names, str) else body_names
        self.mass_distribution_params = mass_distribution_params
        self.operation = operation
        self.distribution = distribution
        self.recompute_inertia = recompute_inertia

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Attach to the simulator handler and record the default masses."""
        super().bind_handler(handler, *args, **kwargs)
        self.initialize()

    def initialize(self):
        """Validate configuration, resolve the bodies and record their default masses."""
        if self.operation == "scale":
            _validate_scale_range(self.mass_distribution_params, "mass_distribution_params", allow_zero=False)
        elif self.operation not in ("abs", "add"):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_mass' does not support operation: '{self.operation}'."
            )
        names = self.handler.get_body_names(self.obj_name)
        targets = names if self.set_body_names is None else self.set_body_names
        missing = [name for name in targets if name not in names]
        if missing:
            raise ValueError(f"MassRandomizer: {self.obj_name!r} has no bodies {missing}; it has {names}.")
        self.set_body_ids = torch.tensor([names.index(name) for name in targets], dtype=torch.long)
        self.default_masses = self.handler.get_body_masses(self.obj_name).clone()
        if self.distribution in ("uniform", "log_uniform"):
            # a bounded sample that can give a body a negative mass is a config error: say so now, not mid-run
            low = float(self.mass_distribution_params[0])
            lightest = float(self.default_masses[:, self.set_body_ids].min())
            worst = {"add": lightest + low, "scale": lightest * low, "abs": low}[self.operation]
            if worst < 0:
                raise ValueError(
                    f"MassRandomizer({self.obj_name!r}): {self.operation} {tuple(self.mass_distribution_params)} can "
                    f"give a body of {lightest:.3g} kg a negative mass ({worst:.3g} kg)."
                )

    def __call__(self, env_ids: torch.Tensor | None = None, **kwargs):
        """Apply mass randomization for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs)
        self.randomize(torch.as_tensor(env_ids).cpu().long().reshape(-1))

    def randomize(self, env_ids: torch.Tensor):
        """Sample masses for ``env_ids`` from the defaults and write them to the simulator."""
        masses = self.handler.get_body_masses(self.obj_name)
        rows, cols = env_ids[:, None], self.set_body_ids
        masses[rows, cols] = self.default_masses[rows, cols]
        masses = randomize_prop_by_op(
            masses,
            self.mass_distribution_params,
            env_ids,
            self.set_body_ids,
            operation=self.operation,
            distribution=self.distribution,
        )
        # a gaussian sample is unbounded: keep the mass physical
        masses[rows, cols] = masses[rows, cols].clamp(min=1e-6)
        self.handler.set_body_masses(
            self.obj_name, masses[env_ids], env_ids=env_ids, scale_inertia=self.recompute_inertia
        )


# helper functions adapted from Isaac Lab


def randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = sample_uniform
    elif distribution == "log_uniform":
        dist_fn = sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")
