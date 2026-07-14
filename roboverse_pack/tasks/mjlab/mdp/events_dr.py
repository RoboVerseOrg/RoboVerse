# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: domain-randomization terms ported from mjlab's `envs/mdp/dr/` (geom friction, body COM offset, body
#   mass, velocity push, encoder bias); rewritten to mutate the compiled `MjModel` / handler state directly under
#   RoboVerse's event-term convention.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Domain-randomization event functions — mjlab parity ports.

These get registered as ``EventTerm(mode="interval"|"startup", func=...)``
on the task's events cfg. They mutate env state in-place via the handler.

Currently provided:
  - ``push_by_setting_velocity`` — interval root-velocity perturbation
    (mjlab ``mdp.events.push_by_setting_velocity``).
  - ``geom_friction``  — startup geom friction randomization
    (mjlab ``mdp.dr.geom_friction``); operates on ``physics.model.geom_friction``.
  - ``body_com_offset`` — startup body COM perturbation
    (mjlab ``mdp.dr.body_com_offset``); operates on ``physics.model.body_ipos``.
  - ``body_mass``      — startup body mass randomization
    (mjlab ``mdp.dr.body_mass``); operates on ``physics.model.body_mass``.
  - ``encoder_bias``   — startup joint encoder bias offset; stored on env so
    ``joint_pos_rel`` obs adds it (mjlab parity via the encoder-bias channel).

mjlab source: src/mjlab/envs/mdp/dr/ (geom.py, body.py, joint.py).

Implementation notes:
  - All "startup" DR fns are called once at env construction via the
    manager's ``_apply_setup_events`` path; in mjlab parlance they
    mutate ``model`` (the compiled MjModel) directly.
  - We do the same here — ``mp.geom_friction`` etc. are writable arrays
    on the compiled MjModel; in-place mutation persists for the lifetime
    of the env instance.
"""

from __future__ import annotations

import re

import numpy as np
import torch


def push_by_setting_velocity(
    env,
    env_states,
    *,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Apply a random velocity impulse to the robot's root (push_robot equivalent).

    ``velocity_range``: dict of components → ``(lo, hi)`` uniform-sample range.
    Keys: ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``.

    Implementation: read current TensorState, mutate root_state[:, 7:13]
    (lin_vel xyz + ang_vel xyz), call handler.set_states.
    """
    if velocity_range is None:
        return
    if not env.scenario.robots:
        return
    robot_name = env.scenario.robots[0].name
    states = env.handler.get_states(mode="tensor")
    if robot_name not in states.robots:
        return
    root = states.robots[robot_name].root_state  # (N, 13) pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    N = root.shape[0]
    device = root.device

    def _u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(N, device=device).uniform_(lo, hi)

    perturb = torch.zeros((N, 6), device=device)
    for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        if key in velocity_range:
            perturb[:, i] = _u(*velocity_range[key])

    new_root = root.clone()
    new_root[:, 7:13] += perturb  # add to lin_vel + ang_vel

    # Write back via handler.set_states. The simplest path is to build a
    # TensorState delta. For Newton, we can use the same set_states pipeline.
    try:
        from metasim.types import TensorState
    except Exception:
        return

    # Build minimal updated TensorState (just the robot, just root_state).
    # Use dataclasses.replace so we stay forward-compatible with new RobotState
    # fields (body_names / joint_*_target) instead of reconstructing positionally.
    import dataclasses

    robot_state = states.robots[robot_name]
    if not hasattr(robot_state, "__dataclass_fields__"):
        return
    new_robot_state = dataclasses.replace(robot_state, root_state=new_root)

    # Replace and call set_states
    try:
        new_state = TensorState(
            robots={robot_name: new_robot_state},
            objects=states.objects if hasattr(states, "objects") else {},
            cameras=states.cameras if hasattr(states, "cameras") else {},
            sensors=states.sensors if hasattr(states, "sensors") else {},
        )
    except TypeError:
        # TensorState signature may differ; skip silently.
        return
    try:
        env.handler.set_states(new_state)
    except Exception:
        pass  # not all handlers support full set_states; quietly skip


# ---------------------------------------------------------------------------
# DR helpers
# ---------------------------------------------------------------------------


def _resolve_geom_ids(env, asset_cfg=None, geom_names: tuple[str, ...] | None = None) -> list[int]:
    """Resolve geom IDs from a regex/name list.

    mjlab DR fns take ``asset_cfg.geom_names`` (a tuple of patterns); we
    accept either an ``asset_cfg`` with ``geom_names`` or an explicit list.
    """
    import mujoco

    handler = env.handler
    if not hasattr(handler, "physics"):
        return []
    mp = handler.physics.model.ptr if hasattr(handler.physics.model, "ptr") else handler.physics.model
    n = mp.ngeom
    all_names = []
    for i in range(n):
        nm = mujoco.mj_id2name(mp, mujoco.mjtObj.mjOBJ_GEOM, i)
        all_names.append(nm if nm else f"<geom{i}>")
    if asset_cfg is not None:
        patterns = getattr(asset_cfg, "geom_names", None) or geom_names or ()
    else:
        patterns = geom_names or ()
    if not patterns:
        return list(range(n))
    if isinstance(patterns, str):
        patterns = (patterns,)
    out: list[int] = []
    for pat in patterns:
        # Exact match first, then regex.
        if pat in all_names:
            out.append(all_names.index(pat))
            continue
        rx = re.compile(pat)
        for i, name in enumerate(all_names):
            if rx.fullmatch(name):
                out.append(i)
    return sorted(set(out))


def _resolve_body_ids(env, asset_cfg=None, body_names: tuple[str, ...] | None = None) -> list[int]:
    import mujoco

    handler = env.handler
    if not hasattr(handler, "physics"):
        return []
    mp = handler.physics.model.ptr if hasattr(handler.physics.model, "ptr") else handler.physics.model
    n = mp.nbody
    all_names = []
    for i in range(n):
        nm = mujoco.mj_id2name(mp, mujoco.mjtObj.mjOBJ_BODY, i)
        all_names.append(nm if nm else f"<body{i}>")
    if asset_cfg is not None:
        patterns = getattr(asset_cfg, "body_names", None) or body_names or ()
    else:
        patterns = body_names or ()
    if not patterns:
        return list(range(n))
    if isinstance(patterns, str):
        patterns = (patterns,)
    out: list[int] = []
    for pat in patterns:
        if pat in all_names:
            out.append(all_names.index(pat))
            continue
        rx = re.compile(pat)
        for i, name in enumerate(all_names):
            if rx.fullmatch(name):
                out.append(i)
    return sorted(set(out))


def _sample_uniform(rng: np.random.Generator, lo, hi, shape):
    return rng.uniform(lo, hi, size=shape)


# ---------------------------------------------------------------------------
# geom_friction — mjlab parity
# ---------------------------------------------------------------------------


def geom_friction(
    env,
    *,
    asset_cfg=None,
    ranges: tuple[float, float] = (0.3, 1.2),
    operation: str = "abs",
    axes: tuple[int, ...] | list[int] = (0,),
    shared_random: bool = False,
    distribution: str = "uniform",
    **_unused,
) -> None:
    """Mjlab ``dr.geom_friction`` port — startup-mode DR.

    Mutates ``physics.model.geom_friction[gid, axis] = sample`` for each
    selected geom and each axis in ``axes`` (default: tangential friction
    only). ``operation="abs"`` replaces; ``"add"`` / ``"mul"`` modify the
    existing value (mjlab convention).

    NB: writes to the compiled ``MjModel``; effect persists across episodes
    until env destruction.
    """
    if not hasattr(env.handler, "physics"):
        return
    gids = _resolve_geom_ids(env, asset_cfg)
    if not gids:
        return
    mp = env.handler.physics.model.ptr if hasattr(env.handler.physics.model, "ptr") else env.handler.physics.model
    rng = np.random.default_rng()
    lo, hi = ranges
    if distribution == "log_uniform":
        log_lo, log_hi = np.log(max(lo, 1e-10)), np.log(max(hi, 1e-10))

        def sample_fn(shape):
            return np.exp(rng.uniform(log_lo, log_hi, size=shape))

    else:

        def sample_fn(shape):
            return rng.uniform(lo, hi, size=shape)

    if shared_random:
        # Same sample for every selected geom (good for "all foot geoms same μ").
        sample = sample_fn((len(axes),))
        for gid in gids:
            for k, ax in enumerate(axes):
                v = float(sample[k])
                if operation == "abs":
                    mp.geom_friction[gid, ax] = v
                elif operation == "add":
                    mp.geom_friction[gid, ax] += v
                elif operation == "mul":
                    mp.geom_friction[gid, ax] *= v
    else:
        for gid in gids:
            for ax in axes:
                v = float(sample_fn((1,))[0])
                if operation == "abs":
                    mp.geom_friction[gid, ax] = v
                elif operation == "add":
                    mp.geom_friction[gid, ax] += v
                elif operation == "mul":
                    mp.geom_friction[gid, ax] *= v


# ---------------------------------------------------------------------------
# body_com_offset — mjlab parity (mutates body_ipos = COM offset in body frame)
# ---------------------------------------------------------------------------


def body_com_offset(
    env,
    *,
    asset_cfg=None,
    operation: str = "add",
    ranges: dict[int, tuple[float, float]] | None = None,
    **_unused,
) -> None:
    """Mjlab ``dr.body_com_offset`` port — perturbs each body's local COM.

    ``ranges`` is a dict ``{axis: (lo, hi)}`` per the mjlab cfg.
    Axis 0 = x, 1 = y, 2 = z (body-local frame). ``operation="add"``
    adds to the existing ``body_ipos[bid, axis]``.
    """
    if not hasattr(env.handler, "physics") or not ranges:
        return
    bids = _resolve_body_ids(env, asset_cfg)
    if not bids:
        return
    mp = env.handler.physics.model.ptr if hasattr(env.handler.physics.model, "ptr") else env.handler.physics.model
    rng = np.random.default_rng()
    for bid in bids:
        for ax, (lo, hi) in ranges.items():
            v = float(rng.uniform(lo, hi))
            if operation == "abs":
                mp.body_ipos[bid, ax] = v
            elif operation == "add":
                mp.body_ipos[bid, ax] += v
            elif operation == "mul":
                mp.body_ipos[bid, ax] *= v


# ---------------------------------------------------------------------------
# body_mass — mjlab parity
# ---------------------------------------------------------------------------


def body_mass(
    env,
    *,
    asset_cfg=None,
    operation: str = "mul",
    ranges: tuple[float, float] = (0.9, 1.1),
    **_unused,
) -> None:
    """Mjlab ``dr.body_mass`` port — randomize per-body mass."""
    if not hasattr(env.handler, "physics"):
        return
    bids = _resolve_body_ids(env, asset_cfg)
    if not bids:
        return
    mp = env.handler.physics.model.ptr if hasattr(env.handler.physics.model, "ptr") else env.handler.physics.model
    rng = np.random.default_rng()
    lo, hi = ranges
    for bid in bids:
        v = float(rng.uniform(lo, hi))
        if operation == "abs":
            mp.body_mass[bid] = v
        elif operation == "add":
            mp.body_mass[bid] += v
        elif operation == "mul":
            mp.body_mass[bid] *= v


# ---------------------------------------------------------------------------
# encoder_bias — mjlab parity. Stored on env so joint_pos obs reads it.
# ---------------------------------------------------------------------------


def encoder_bias(
    env,
    *,
    asset_cfg=None,
    bias_range: tuple[float, float] = (-0.015, 0.015),
    **_unused,
) -> None:
    """Mjlab ``dr.encoder_bias`` port — per-joint encoder calibration offset.

    Stored on ``env._encoder_bias`` as a ``(num_envs, num_joints)`` tensor.
    The observation function :func:`observations.joint_pos_rel` adds this
    bias to its output if present, matching mjlab's behavior. Joint
    resolution follows ``asset_cfg.joint_names``.
    """
    if not hasattr(env.handler, "physics"):
        return
    # Resolve number of joints from the handler (single-env path).
    mp = env.handler.physics.model.ptr if hasattr(env.handler.physics.model, "ptr") else env.handler.physics.model
    if asset_cfg is not None and getattr(asset_cfg, "joint_names", None):
        from .scene_entity import resolve_joint_ids

        joint_ids = resolve_joint_ids(env, asset_cfg)
        n = joint_ids.numel()
    else:
        n = int(mp.njnt)
    rng = np.random.default_rng()
    bias = rng.uniform(*bias_range, size=(env.num_envs, n)).astype(np.float32)
    env._encoder_bias = torch.tensor(bias, device=env.device, dtype=torch.float32)
