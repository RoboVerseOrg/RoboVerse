# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab (https://github.com/isaac-sim/IsaacLab).
# Changes: only the helpers under the 'adapted from Isaac Lab' marker (randomize_prop_by_op, _validate_scale_range)
# and the bucketed material sampling idea are Isaac Lab-derived; the randomizer classes are RoboVerse code.
# Full license: roboverse_pack/randomization/LICENSE.isaaclab

"""Physical-property randomizers (body materials and masses) for the humanoid and tracking tasks."""

from __future__ import annotations

from typing import Literal

import torch

from metasim.sim.base import BaseQueryType, BaseSimHandler
from metasim.utils.log import warn_once
from metasim.utils.math import sample_gaussian, sample_log_uniform, sample_uniform


class MaterialRandomizer(BaseQueryType):
    """Randomize the friction and restitution of a robot's or articulated object's bodies.

    ``num_buckets`` materials are sampled once at bind; each call gives every selected body of every env in
    ``env_ids`` one of them, written with ``handler.set_body_friction`` / ``set_body_restitution``. Friction is
    one sliding coefficient per body (see ``BaseSimHandler.get_body_friction``). On an engine without a
    restitution coefficient (MuJoCo, Newton) ``restitution_range`` is ignored with a warning.

    Args:
        obj_name: The robot or articulated object.
        body_names: Bodies to randomize; every body with a collision shape when ``None``.
        friction_range: ``(low, high)`` of the uniform friction sample; friction is left alone when ``None``.
        restitution_range: ``(low, high)`` of the uniform restitution sample; left alone when ``None``.
        num_buckets: Number of distinct materials.
        static_friction_range: Deprecated; the interface has one coefficient. Used only when neither
            ``friction_range`` nor ``dynamic_friction_range`` is given.
        dynamic_friction_range: Deprecated alias of ``friction_range``.
        make_consistent: Deprecated, no effect (there is no separate static coefficient to keep consistent).
    """

    handler: BaseSimHandler

    def __init__(
        self,
        obj_name: str,
        body_names: list[str] | str | None = None,
        *,
        friction_range: tuple[float, float] | None = None,
        restitution_range: tuple[float, float] | None = None,
        num_buckets: int = 1,
        static_friction_range: tuple[float, float] | None = None,
        dynamic_friction_range: tuple[float, float] | None = None,
        make_consistent: bool = False,
    ):
        super().__init__()
        if static_friction_range is not None or dynamic_friction_range is not None or make_consistent:
            warn_once(
                ("MaterialRandomizer", "static/dynamic"),
                "MaterialRandomizer: static_friction_range / dynamic_friction_range / make_consistent are "
                "deprecated; bodies have one friction coefficient. Pass friction_range (the dynamic range is used).",
            )
        if friction_range is None:
            friction_range = dynamic_friction_range if dynamic_friction_range is not None else static_friction_range
        if num_buckets < 1:
            raise ValueError(f"MaterialRandomizer: num_buckets must be >= 1, got {num_buckets}.")
        for label, rng in (("friction_range", friction_range), ("restitution_range", restitution_range)):
            if rng is not None and not (0 <= rng[0] <= rng[1] and (label != "restitution_range" or rng[1] <= 1)):
                bound = "0 <= low <= high <= 1" if label == "restitution_range" else "0 <= low <= high"
                raise ValueError(f"MaterialRandomizer: {label} must satisfy {bound}, got {tuple(rng)}.")
        self.obj_name = obj_name
        self.set_body_names = [body_names] if isinstance(body_names, str) else body_names
        self.friction_range = friction_range
        self.restitution_range = restitution_range
        self.num_buckets = num_buckets

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Attach to the simulator handler, resolve the bodies and sample the material buckets."""
        super().bind_handler(handler, *args, **kwargs)
        self.initialize()

    def initialize(self):
        """Resolve the bodies (by name, among those with a collision shape) and sample the buckets."""
        names = self.handler.get_body_names(self.obj_name)
        has_shape = ~torch.isnan(self.handler.get_body_friction(self.obj_name)[0])
        if self.set_body_names is None:
            ids = [i for i in range(len(names)) if has_shape[i]]
        else:
            missing = [n for n in self.set_body_names if n not in names]
            if missing:
                raise ValueError(f"MaterialRandomizer: {self.obj_name!r} has no bodies {missing}; it has {names}.")
            shapeless = [n for n in self.set_body_names if not has_shape[names.index(n)]]
            if shapeless:
                raise ValueError(
                    f"MaterialRandomizer: bodies {shapeless} of {self.obj_name!r} have no collision shape."
                )
            ids = [names.index(n) for n in self.set_body_names]
        self.set_body_ids = torch.tensor(ids, dtype=torch.long)
        self.buckets = {}
        for prop, rng in (("friction", self.friction_range), ("restitution", self.restitution_range)):
            if rng is not None:
                self.buckets[prop] = sample_uniform(float(rng[0]), float(rng[1]), (self.num_buckets,), device="cpu")

    def __call__(self, env_ids=None, **kwargs):
        """Apply material randomization for the given environment ids (all when ``None``)."""
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs)
        self.randomize(torch.as_tensor(env_ids).cpu().long().reshape(-1))

    def randomize(self, env_ids: torch.Tensor):
        """Give every selected body of ``env_ids`` a material drawn from the buckets."""
        rows, cols = env_ids[:, None], self.set_body_ids
        for prop, buckets in self.buckets.items():
            get = getattr(self.handler, f"get_body_{prop}")
            values = get(self.obj_name)
            values[rows, cols] = buckets[torch.randint(0, self.num_buckets, (len(env_ids), len(cols)))]
            getattr(self.handler, f"set_body_{prop}")(self.obj_name, values[env_ids], env_ids=env_ids)


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
        # a gaussian sample is unbounded: keep the mass physical (a massless body stays massless)
        sampled = masses[rows, cols]
        masses[rows, cols] = torch.where(self.default_masses[rows, cols] > 0, sampled.clamp(min=1e-6), sampled)
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
