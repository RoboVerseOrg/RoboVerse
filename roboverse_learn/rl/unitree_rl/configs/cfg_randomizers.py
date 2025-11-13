from __future__ import annotations
from typing import Literal
from copy import deepcopy
from loguru import logger as log
import torch

from metasim.sim.base import BaseSimHandler, BaseQueryType
from metasim.utils.math import sample_uniform, sample_log_uniform, sample_gaussian

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass


class MaterialRandomizer(BaseQueryType):
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
        self.set_body_names = (
            [body_names] if isinstance(body_names, str) else body_names
        )
        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.num_buckets = num_buckets
        self.make_consistent = make_consistent

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator_name = handler.scenario.simulator
        self.initialize()

    def __call__(self, env_ids=None):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs, device="cpu")
        else:
            env_ids = torch.tensor(env_ids).cpu()
        self.randomize(env_ids)

    def initialize(self):
        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [
            self.static_friction_range,
            self.dynamic_friction_range,
            self.restitution_range,
        ]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = sample_uniform(
            ranges[:, 0], ranges[:, 1], (self.num_buckets, 3), device="cpu"
        )

        # ensure dynamic friction is always less than static friction
        if self.make_consistent:
            self.material_buckets[:, 1] = torch.min(
                self.material_buckets[:, 0], self.material_buckets[:, 1]
            )

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

    def _get_set_shape_indices(self):
        num_shapes_per_body = None
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
        elif self.simulator_name == "isaacgym":
            if self.obj_name in self.all_robot_names:
                _tmp_handle = self.handler._robot_handles[0]
            elif self.obj_name in self.all_object_names:
                _tmp_handle = self.handler._obj_handles[0][
                    self.handler.objects.index(self.obj_name)
                ]
            body_shape_indices = self.handler.gym.get_actor_rigid_body_shape_indices(
                self.handler._envs[0], _tmp_handle
            )
            num_shapes_per_body = []
            for body_shape in body_shape_indices:
                num_shapes_per_body.append(body_shape.count)
            expected_shapes = len(
                self.handler.gym.get_actor_rigid_shape_properties(
                    self.handler._envs[0], _tmp_handle
                )
            )
        elif self.simulator_name == "mujoco":
            model = self.handler.physics.model
            num_shapes_per_body = [0] * model.nbody
            # geom_bodyid[j] = geom j belongs to body geom_bodyid[j]
            for geom_bodyid in model.geom_bodyid:
                num_shapes_per_body[geom_bodyid] += 1
            expected_shapes = model.ngeom
        if (
            num_shapes_per_body is not None
            and sum(num_shapes_per_body) != expected_shapes
        ):
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

    def randomize(self, env_ids: torch.Tensor):
        if self.simulator_name == "isaacsim":
            self._randomize_isaacsim(env_ids)
        elif self.simulator_name == "isaacgym":
            self._randomize_isaacgym(env_ids)
        elif self.simulator_name == "mujoco":
            self._randomize_mujoco(env_ids)
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
        bucket_ids = torch.randint(
            0, self.num_buckets, (len(env_ids), total_num_shapes), device="cpu"
        )
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
        max_body_shape = len(
            self.handler.gym.get_actor_rigid_shape_properties(
                self.handler._envs[0], _all_handles[0]
            )
        )
        bucket_ids = torch.randint(
            0, self.num_buckets, (len(env_ids), max_body_shape), device="cpu"
        )
        material_samples = self.material_buckets[
            bucket_ids
        ]  # static friction, dynamic friction and restitution

        roll_friction_factor = 0.05
        spin_friction_factor = 0.02
        for i, env_id in enumerate(env_ids):
            env = self.handler._envs[env_id]
            # For objects, find the corresponding object handle
            _tmp_handle = _all_handles[env_id]
            # Get current rigid shape properties
            shape_props = self.handler.gym.get_actor_rigid_shape_properties(
                env, _tmp_handle
            )

            for _id in self.set_shape_indices:
                shape_prop = shape_props[_id]
                shape_prop.friction = material_samples[i, _id, 0]
                shape_prop.rolling_friction = (
                    roll_friction_factor * material_samples[i, _id, 1]
                )
                shape_prop.torsion_friction = (
                    spin_friction_factor * material_samples[i, _id, 1]
                )
                shape_prop.restitution = material_samples[i, _id, 2]

            # Apply the modified properties
            self.handler.gym.set_actor_rigid_shape_properties(
                env, _tmp_handle, shape_props
            )

    def _randomize_mujoco(self, env_ids: torch.Tensor):
        """Randomize friction and restitution for MuJoCo simulator."""
        assert (
            self.handler.num_envs == 1
        ), "MuJoCo handler only supports single environment."
        model = self.handler.physics.model

        bucket_ids = torch.randint(
            0, self.num_buckets, (len(env_ids), model.ngeom), device="cpu"
        )
        material_samples = self.material_buckets[
            bucket_ids
        ]  # static friction, dynamic friction and restitution

        static_friction = material_samples[env_ids, self.set_shape_indices, 0]
        solimp_value = 0.1 * static_friction
        model.geom_solimp[self.set_shape_indices, 0] = solimp_value

        # model.geom_friction --> friction for (slide, spin, roll)
        dynamic_friction = material_samples[env_ids, self.set_shape_indices, 1]
        model.geom_friction[self.set_shape_indices, 0] = (
            dynamic_friction  # slide friction
        )
        model.geom_friction[self.set_shape_indices, 1] = (
            0.01 * dynamic_friction
        )  # spin friction
        model.geom_friction[self.set_shape_indices, 2] = (
            0.01 * dynamic_friction
        )  # roll friction

        # restitution and damping calculation
        restitution_scale = 1.0  # from 0.5 - 2.0
        restitution = (
            material_samples[env_ids, self.set_shape_indices, 2] * restitution_scale
            + 1e-6
        )
        damping = (
            -torch.log(restitution)
            / torch.sqrt(torch.pi**2 + torch.log(restitution) ** 2)
        ).clamp(min=0.0, max=1.0)

        # solref：timeconst & damping ratio
        model.geom_solref[self.set_shape_indices, 1] = damping


class MassRandomizer(BaseQueryType):
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
        self.set_body_names = (
            [body_names] if isinstance(body_names, str) else body_names
        )
        self.mass_distribution_params = mass_distribution_params
        self.operation = operation
        self.distribution = distribution
        self.recompute_inertia = recompute_inertia

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator_name = handler.scenario.simulator
        self.initialize()

    def initialize(self):
        # check for valid operation
        if self.operation == "scale":
            _validate_scale_range(
                self.mass_distribution_params,
                "mass_distribution_params",
                allow_zero=False,
            )
        elif self.operation not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support operation:"
                f" '{self.operation}'."
            )

        self.all_robot_names = [robot.name for robot in self.handler.robots]
        self.all_object_names = [obj.name for obj in self.handler.objects]
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
        self.default_masses = deepcopy(self._get_masses())

    def __call__(self, env_ids: torch.Tensor | None = None):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs, device="cpu")
        else:
            env_ids = torch.tensor(env_ids).cpu()
        self.randomize(env_ids)

    def _get_masses(self):
        if self.simulator_name == "isaacsim":
            return self._get_masses_isaacsim()
        elif self.simulator_name == "isaacgym":
            return self._get_masses_isaacgym()
        elif self.simulator_name == "mujoco":
            return self._get_masses_mujoco()

    def _get_masses_isaacsim(self):
        if self.obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[self.obj_name]
        elif self.obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[self.obj_name]
        else:
            raise ValueError(f"Not found: {self.obj_name}.")
        masses = obj_inst.root_physx_view.get_masses()
        return masses

    def _get_masses_isaacgym(self):
        """
        Get masses for IsaacGym simulator.
        Note that the isaacgym handler only support 1 robot and multiple objects, currently.
        """
        # Initialize masses tensor
        masses = torch.zeros(
            (self.handler.num_envs, len(self.body_names)),
            dtype=torch.float32,
            device=self.handler.device,
        )

        # Get masses from first environment (they should be the same across environments initially)
        for (
            env_id,
            env,
        ) in enumerate(self.handler._envs):
            if self.obj_name in self.all_robot_names:
                _tmp_handle = self.handler._robot_handles[0]
            elif self.obj_name in self.all_object_names:
                # Find the object handle
                _tmp_handle = self.handler._obj_handles[env_id][
                    self.handler.objects.index(self.obj_name)
                ]
            else:
                raise ValueError(f"Not found: {self.obj_name}.")

            body_props = self.handler.gym.get_actor_rigid_body_properties(
                env, _tmp_handle
            )
            for i, prop in enumerate(body_props):
                masses[env_id, i] = prop.mass

        return masses

    def _get_masses_mujoco(self):
        """Get masses for MuJoCo simulator."""
        assert (
            self.handler.num_envs == 1
        ), "MuJoCo handler only supports single environment."
        model = self.handler.physics.model
        body_masses = model.body_mass
        return torch.tensor(
            body_masses,
            dtype=torch.float32,
            device=self.handler.device,
        ).unsqueeze(
            0
        )  # shape: (1, num_bodies)

    def _set_masses(self, masses: torch.Tensor, env_ids: torch.Tensor):
        if self.simulator_name == "isaacsim":
            self._set_masses_isaacsim(masses, env_ids)
        elif self.simulator_name == "isaacgym":
            self._set_masses_isaacgym(masses, env_ids)
        elif self.simulator_name == "mujoco":
            self._set_masses_mujoco(masses, env_ids)

    def _set_masses_isaacsim(self, masses: torch.Tensor, env_ids: torch.Tensor):
        if self.obj_name in self.handler.scene.articulations:
            obj_inst = self.handler.scene.articulations[self.obj_name]
        elif self.obj_name in self.handler.scene.rigid_objects:
            obj_inst = self.handler.scene.rigid_objects[self.obj_name]
        obj_inst.root_physx_view.set_masses(masses, env_ids)

    def _set_masses_isaacgym(self, masses: torch.Tensor, env_ids: torch.Tensor):
        """Set masses for IsaacGym simulator."""
        for env_id in env_ids:
            env = self.handler._envs[env_id]
            if self.obj_name in self.all_robot_names:
                _tmp_handle = self.handler._robot_handles[env_id]
            elif self.obj_name in self.all_object_names:
                # Find the object handle
                _tmp_handle = self.handler._obj_handles[env_id][
                    self.handler.objects.index(self.obj_name)
                ]
            else:
                raise ValueError(f"Not found: {self.obj_name}.")

            # Get current body properties
            body_props = self.handler.gym.get_actor_rigid_body_properties(
                env, _tmp_handle
            )

            # Update masses for specified bodies
            for body_idx in self.set_body_ids:
                if body_idx < len(body_props):
                    body_props[body_idx].mass = float(masses[env_id, body_idx])

            # Apply the modified properties
            self.handler.gym.set_actor_rigid_body_properties(
                env, _tmp_handle, body_props
            )

    def _set_masses_mujoco(self, masses: torch.Tensor, env_ids: torch.Tensor):
        model = self.handler.physics.model
        model.body_mass[self.set_body_ids] = masses[env_ids, self.set_body_ids].cpu()

    def _recompute_inertias(self, ratios: torch.Tensor, env_ids: torch.Tensor):
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        if self.simulator_name == "isaacsim":
            if self.obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[self.obj_name]
                inertias = obj_inst.root_physx_view.get_inertias()
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[env_ids[:, None], self.set_body_ids] = (
                    obj_inst.data.default_inertia[env_ids[:, None], self.set_body_ids]
                    * ratios[..., None]
                )
            elif self.obj_name in self.handler.scene.rigid_objects:
                obj_inst = self.handler.scene.rigid_objects[self.obj_name]
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[env_ids] = obj_inst.data.default_inertia[env_ids] * ratios
            # set the inertia tensors into the physics simulation
            obj_inst.root_physx_view.set_inertias(inertias, env_ids)
        elif self.simulator_name == "isaacgym":
            # For IsaacGym, inertia recomputation is handled automatically by the physics engine
            # when mass is changed, so we don't need to manually update inertias

            # a little delay refresh in isaacym handler
            # self.gym.refresh_mass_matrix_tensors(self.sim)
            pass
        elif self.simulator_name == "mujoco":
            model = self.handler.physics.model
            model.body_inertia[self.set_body_ids] = (
                model.body_inertia[self.set_body_ids] * ratios.squeeze(0).numpy()
            )  # only single env

    def randomize(self, env_ids: torch.Tensor):
        if self.simulator_name not in ("isaacsim", "isaacgym", "mujoco"):
            log.warning(
                f"Mass randomization not implemented for simulator: {self.simulator_name}. This randomization step will be skipped."
            )
            return
        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self._get_masses()  # shape: (num_envs, num_bodies)
        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_ids[:, None], self.set_body_ids] = self.default_masses[
            env_ids[:, None], self.set_body_ids
        ].clone()
        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = _randomize_prop_by_op(
            masses,
            self.mass_distribution_params,
            env_ids,
            self.set_body_ids,
            operation=self.operation,
            distribution=self.distribution,
        )
        self._set_masses(masses, env_ids)
        # recompute inertia tensors if needed
        if self.recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = (
                masses[env_ids[:, None], self.set_body_ids]
                / self.default_masses[env_ids[:, None], self.set_body_ids]
            )
            self._recompute_inertias(ratios, env_ids)


##########################################################################################
# Private Helper Functions
# FROM NVIDIA ISAAC LAB
##########################################################################################


def _randomize_prop_by_op(
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
        data[dim_0_ids, dim_1_ids] += dist_fn(
            *distribution_parameters, (n_dim_0, n_dim_1), device=data.device
        )
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(
            *distribution_parameters, (n_dim_0, n_dim_1), device=data.device
        )
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(
            *distribution_parameters, (n_dim_0, n_dim_1), device=data.device
        )
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
    """
    Validates a (low, high) tuple used in scale-based randomization.

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
        raise TypeError(
            f"{name}: expected (low, high) to be a tuple of numbers, got {params}."
        )
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(
            f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low})."
        )
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(
            f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low})."
        )
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")


##########################################################################################
# Private Helper Functions
# FROM NVIDIA ISAAC LAB
##########################################################################################
