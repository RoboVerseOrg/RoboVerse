"""Domain-randomization event functions — mjlab parity ports.

These get registered as ``EventTerm(mode="setup"|"post_step", func=...)``
on the task's events cfg. They mutate env state in-place via the handler.

Currently provided:
  - ``push_by_setting_velocity`` — interval root-velocity perturbation
    (mjlab ``mdp.events.push_by_setting_velocity``). Backend-agnostic:
    goes through ``handler.set_states``.
  - ``geom_friction``  — startup geom/shape friction randomization
    (mjlab ``mdp.dr.geom_friction``).
  - ``body_com_offset`` — startup body COM perturbation
    (mjlab ``mdp.dr.body_com_offset``).
  - ``body_mass``      — startup body mass randomization
    (mjlab ``mdp.dr.body_mass``).
  - ``encoder_bias``   — startup joint encoder bias offset; stored on env so
    ``joint_pos_rel`` obs adds it (mjlab parity via the encoder-bias channel).

mjlab source: src/mjlab/envs/mdp/dr/ (geom.py, body.py, joint.py).

Backend support:
  - **mujoco** — mutates the compiled ``MjModel`` reached through
    ``handler.physics.model`` (``geom_friction`` / ``body_ipos`` /
    ``body_mass``). Single-env only (``MujocoHandler`` rejects
    ``num_envs > 1``).
  - **newton** — mutates the finalized ``newton.Model`` reached through
    ``handler._model``: ``shape_material_{mu,torsional_friction,rolling_friction}``
    for friction, ``body_mass`` (+ ``body_inv_mass`` and inertia rescale)
    for mass, and ``body_com`` for the COM offset. Newton keeps one model
    row per body *per env*, so randomization here is genuinely per-env.
    This is the same access pattern used by
    ``roboverse_pack/randomization/humanoid.py``.
  - **anything else (mjx, isaacsim, …)** — raises ``NotImplementedError``.
    These backends do not expose per-env writable model fields through the
    handler, and a DR term that quietly does nothing is worse than no DR
    term at all: the task cfg would claim randomization the policy never
    saw. Drop the offending ``EventTerm`` from the task's events cfg to run
    such a backend without randomization, deliberately.

Implementation notes:
  - All "startup" DR fns are called once at env construction via the
    manager's ``_apply_setup_events`` path; in mjlab parlance they
    mutate ``model`` (the compiled MjModel) directly. We do the same,
    so the perturbation persists for the lifetime of the env instance.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NoReturn

import numpy as np
import torch

from metasim.types import TensorState

from .scene_entity import _match_names, resolve_joint_ids

# ---------------------------------------------------------------------------
# backend dispatch
# ---------------------------------------------------------------------------

_MUJOCO = "mujoco"
_NEWTON = "newton"


def _sim_name(env) -> str:
    """Return the backend name driving ``env``.

    ``MujocoHandler`` is the only handler that exposes a dm_control
    ``physics`` object, so that check is authoritative; every other backend
    is identified by ``scenario.simulator``.
    """
    handler = env.handler
    if hasattr(handler, "physics"):
        return _MUJOCO
    scenario = getattr(handler, "scenario", None) or getattr(env, "scenario", None)
    name = getattr(scenario, "simulator", None)
    return str(name) if name else type(handler).__name__


def _unsupported(event: str, sim: str) -> NoReturn:
    """Raise for a DR event the backend cannot apply. Never return silently."""
    raise NotImplementedError(
        f"mjlab DR event '{event}' is not implemented for the '{sim}' backend: it has no writable"
        f" per-env model fields reachable from the handler. Supported backends are"
        f" '{_MUJOCO}' (num_envs=1 only) and '{_NEWTON}'. Randomization must not silently no-op —"
        f" a policy would train with none of the randomization its task cfg declares. Either run"
        f" the task on a supported backend, or remove the '{event}' EventTerm from the task's"
        f" events cfg."
    )


def _mujoco_model(env):
    """Return the raw ``MjModel`` (``.ptr``-unwrapped) behind the MuJoCo handler."""
    model = env.handler.physics.model
    return model.ptr if hasattr(model, "ptr") else model


def _newton_model(env, event: str):
    """Return the finalized ``newton.Model``, raising if the handler has none."""
    model = getattr(env.handler, "_model", None)
    if model is None:
        raise RuntimeError(
            f"DR event '{event}': the Newton handler has no finalized model yet"
            " (handler.launch() must run before setup events)."
        )
    return model


def _asset_name(env, asset_cfg) -> str:
    """Entity name the DR event applies to (defaults to the first robot)."""
    name = getattr(asset_cfg, "name", None)
    if name:
        return str(name)
    if not env.scenario.robots:
        raise ValueError("DR event needs an asset_cfg with a name: the scenario declares no robots.")
    return env.scenario.robots[0].name


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def _make_sampler(ranges: tuple[float, float], distribution: str) -> Callable[[tuple[int, ...]], np.ndarray]:
    """Return ``shape -> samples`` for a uniform / log-uniform range."""
    rng = np.random.default_rng()
    lo, hi = ranges
    if distribution == "log_uniform":
        log_lo, log_hi = np.log(max(lo, 1e-10)), np.log(max(hi, 1e-10))
        return lambda shape: np.exp(rng.uniform(log_lo, log_hi, size=shape))
    if distribution != "uniform":
        raise ValueError(f"Unknown distribution '{distribution}'; expected 'uniform' or 'log_uniform'.")
    return lambda shape: rng.uniform(lo, hi, size=shape)


def _apply_op(current: float, sample: float, operation: str, event: str) -> float:
    """Combine an existing model value with a sample per the mjlab op convention."""
    if operation == "abs":
        return sample
    if operation == "add":
        return current + sample
    if operation == "mul":
        return current * sample
    raise ValueError(f"DR event '{event}': unknown operation '{operation}'; expected 'abs', 'add' or 'mul'.")


# ---------------------------------------------------------------------------
# MuJoCo id resolution
# ---------------------------------------------------------------------------


def _mujoco_names(mp, obj_type, count: int, kind: str) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(mp, obj_type, i) or f"<{kind}{i}>" for i in range(count)]


def _resolve_geom_ids(env, asset_cfg=None, geom_names: tuple[str, ...] | None = None) -> list[int]:
    """Resolve MuJoCo geom IDs from a regex/name list.

    mjlab DR fns take ``asset_cfg.geom_names`` (a tuple of patterns); we
    accept either an ``asset_cfg`` with ``geom_names`` or an explicit list.
    A pattern that matches nothing raises (``_match_names``) — a DR term
    that quietly selects zero geoms is the bug this module used to have.
    """
    import mujoco

    mp = _mujoco_model(env)
    all_names = _mujoco_names(mp, mujoco.mjtObj.mjOBJ_GEOM, mp.ngeom, "geom")
    patterns = getattr(asset_cfg, "geom_names", None) or geom_names or ()
    return sorted(set(_match_names(patterns, all_names)))


def _resolve_body_ids(env, asset_cfg=None, body_names: tuple[str, ...] | None = None) -> list[int]:
    """Resolve MuJoCo body IDs from a regex/name list."""
    import mujoco

    mp = _mujoco_model(env)
    all_names = _mujoco_names(mp, mujoco.mjtObj.mjOBJ_BODY, mp.nbody, "body")
    patterns = getattr(asset_cfg, "body_names", None) or body_names or ()
    return sorted(set(_match_names(patterns, all_names)))


# ---------------------------------------------------------------------------
# Newton id resolution
# ---------------------------------------------------------------------------


def _newton_body_ids(env, event: str, asset_cfg, patterns) -> dict[int, list[int]]:
    """Map ``env_id -> newton body indices`` matching ``patterns`` for the asset.

    Newton flattens every env's bodies into one model, so the selection has
    to be redone per env; ``handler._get_body_indices`` gives the body rows
    that belong to ``asset`` in that env, and ``model.body_key`` names them.
    """
    model = _newton_model(env, event)
    handler = env.handler
    name = _asset_name(env, asset_cfg)

    out: dict[int, list[int]] = {}
    for env_id in range(env.num_envs):
        body_ids = handler._get_body_indices(env_id, name)
        if not body_ids:
            continue
        names = [model.body_key[i] for i in body_ids]
        sel = _match_names(patterns or (), names)
        if sel:
            out[env_id] = [body_ids[i] for i in sel]
    if not out:
        raise ValueError(
            f"DR event '{event}': no Newton bodies of asset '{name}' matched {tuple(patterns or ())}."
            " Check the asset_cfg body names against handler.get_body_names()."
        )
    return out


def _newton_shape_ids(env, event: str, asset_cfg) -> dict[int, list[int]]:
    """Map ``env_id -> newton shape indices`` for the geoms selected by ``asset_cfg``.

    MuJoCo geom names have no direct Newton counterpart, so match the
    patterns against ``model.shape_key`` first (Newton keys collision shapes
    after the MJCF/URDF geom they came from) and fall back to *all* shapes of
    the selected bodies when the asset_cfg only names bodies.
    """
    model = _newton_model(env, event)
    handler = env.handler
    name = _asset_name(env, asset_cfg)
    geom_patterns = getattr(asset_cfg, "geom_names", None) or ()
    body_patterns = getattr(asset_cfg, "body_names", None) or ()

    shape_key = getattr(model, "shape_key", None)
    out: dict[int, list[int]] = {}
    for env_id in range(env.num_envs):
        body_ids = handler._get_body_indices(env_id, name)
        if not body_ids:
            continue
        if body_patterns:
            names = [model.body_key[i] for i in body_ids]
            body_ids = [body_ids[i] for i in _match_names(body_patterns, names)]
        shapes: list[int] = []
        for bid in body_ids:
            shapes.extend(model.body_shapes.get(bid, []))
        if geom_patterns:
            if shape_key is None:
                raise RuntimeError(
                    f"DR event '{event}': the Newton model exposes no 'shape_key', so geom names"
                    f" {tuple(geom_patterns)} cannot be resolved. Select bodies via"
                    " asset_cfg.body_names instead."
                )
            names = [shape_key[s] for s in shapes]
            shapes = [shapes[i] for i in _match_names(geom_patterns, names)]
        if shapes:
            out[env_id] = sorted(set(shapes))
    if not out:
        raise ValueError(
            f"DR event '{event}': no Newton shapes of asset '{name}' matched geom names"
            f" {tuple(geom_patterns)} / body names {tuple(body_patterns)}."
        )
    return out


# ---------------------------------------------------------------------------
# push_by_setting_velocity — backend-agnostic (goes through set_states)
# ---------------------------------------------------------------------------


def _mujoco_free_joint_dofadr(env) -> int:
    """Qvel offset of the scene's single free joint (the floating base)."""
    import mujoco

    mp = _mujoco_model(env)
    free = [j for j in range(mp.njnt) if int(mp.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if len(free) != 1:
        raise RuntimeError(
            f"push_by_setting_velocity: expected exactly one free joint to push in the scene MJCF,"
            f" found {len(free)}. Register the robot as a RobotCfg so the push can address it"
            " through the handler's TensorState."
        )
    return int(mp.jnt_dofadr[free[0]])


def push_by_setting_velocity(
    env,
    env_states,
    *,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Apply a random velocity impulse to the robot's root (push_robot equivalent).

    ``velocity_range``: dict of components → ``(lo, hi)`` uniform-sample range.
    Keys: ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``.

    Two paths, because mjlab tasks come in two shapes:

    - the robot is a registered ``RobotCfg`` (Newton path, and any backend
      that reports it in ``TensorState``) → mutate ``root_state[:, 7:13]``
      (lin_vel xyz + ang_vel xyz) and write it back with ``handler.set_states``;
    - the robot lives in a self-contained scene MJCF with no ``RobotCfg``
      (the MuJoCo path of the Go1 velocity tasks, which sets
      ``scenario.robots = []``) → add the perturbation to the floating base's
      six free-joint dofs in ``physics.data.qvel``, which is exactly what
      mjlab's ``push_by_setting_velocity`` writes.

    Neither path swallows failures: a push that never lands would make the
    task's robustness numbers meaningless while the cfg claims the robot is
    being pushed.
    """
    if not velocity_range:
        raise ValueError(
            "push_by_setting_velocity requires a non-empty velocity_range"
            " (e.g. {'x': (-0.5, 0.5), 'yaw': (-0.78, 0.78)})."
        )
    unknown = set(velocity_range) - {"x", "y", "z", "roll", "pitch", "yaw"}
    if unknown:
        raise ValueError(f"push_by_setting_velocity: unknown velocity_range keys {sorted(unknown)}.")

    N = env.num_envs
    perturb = torch.zeros((N, 6), device=env.device)
    for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        if key in velocity_range:
            perturb[:, i] = torch.empty(N, device=env.device).uniform_(*velocity_range[key])

    robot_name = env.scenario.robots[0].name if env.scenario.robots else None
    states = env.handler.get_states(mode="tensor")

    if robot_name is not None and robot_name in states.robots:
        root = states.robots[robot_name].root_state  # (N, 13) pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        new_root = root.clone()
        new_root[:, 7:13] += perturb.to(root.device)
        # Build a minimal updated TensorState (just the robot, just root_state).
        # dataclasses.replace keeps us forward-compatible with new RobotState
        # fields (body_names / joint_*_target) instead of reconstructing positionally.
        new_robot_state = dataclasses.replace(states.robots[robot_name], root_state=new_root)
        env.handler.set_states(
            TensorState(
                robots={robot_name: new_robot_state},
                objects=getattr(states, "objects", {}) or {},
                cameras=getattr(states, "cameras", {}) or {},
            )
        )
        return

    sim = _sim_name(env)
    if sim == _MUJOCO:
        # Scene-MJCF task: no RobotCfg, so the base is only reachable as the
        # free joint's dofs (qvel[adr:adr+6] = lin_vel world, ang_vel local).
        adr = _mujoco_free_joint_dofadr(env)
        qvel = env.handler.physics.data.qvel
        qvel[adr : adr + 6] += perturb[0].detach().cpu().numpy()
        return

    raise RuntimeError(
        f"push_by_setting_velocity: robot '{robot_name}' is not reported by the '{sim}' handler's"
        f" TensorState (has {sorted(states.robots)}), so the push cannot be applied. Register the"
        " robot as a RobotCfg on the scenario, or remove the push_robot EventTerm."
    )


# ---------------------------------------------------------------------------
# geom_friction — mjlab parity
# ---------------------------------------------------------------------------

# MuJoCo packs friction as (slide, torsion, roll) per geom; Newton keeps the
# three coefficients in separate shape-material arrays.
_NEWTON_FRICTION_FIELDS = {
    0: "shape_material_mu",
    1: "shape_material_torsional_friction",
    2: "shape_material_rolling_friction",
}


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

    Randomizes the friction of the selected geoms/shapes for each axis in
    ``axes`` (default: tangential friction only). ``operation="abs"``
    replaces; ``"add"`` / ``"mul"`` modify the existing value (mjlab
    convention). ``shared_random`` draws a single sample per axis shared by
    every selected geom (e.g. "all four feet get the same μ") — on Newton it
    is shared per env, so envs still differ.

    NB: writes to the compiled model; the effect persists across episodes
    until env destruction.

    Raises:
        NotImplementedError: on a backend with no writable friction field.
    """
    sim = _sim_name(env)
    sample_fn = _make_sampler(ranges, distribution)

    if sim == _MUJOCO:
        gids = _resolve_geom_ids(env, asset_cfg)
        mp = _mujoco_model(env)
        shared = sample_fn((len(axes),)) if shared_random else None
        for gid in gids:
            for k, ax in enumerate(axes):
                v = float(shared[k]) if shared is not None else float(sample_fn((1,))[0])
                mp.geom_friction[gid, ax] = _apply_op(float(mp.geom_friction[gid, ax]), v, operation, "geom_friction")
        return

    if sim == _NEWTON:
        model = _newton_model(env, "geom_friction")
        shapes_by_env = _newton_shape_ids(env, "geom_friction", asset_cfg)
        arrays = {}
        for ax in axes:
            field = _NEWTON_FRICTION_FIELDS.get(int(ax))
            arr = getattr(model, field, None) if field else None
            if arr is None:
                raise NotImplementedError(
                    f"geom_friction: the Newton model has no friction field for axis {ax}"
                    f" (expected '{field}'). MuJoCo friction axes are (slide, torsion, roll)."
                )
            arrays[int(ax)] = (arr, arr.numpy())
        for shapes in shapes_by_env.values():
            shared = sample_fn((len(axes),)) if shared_random else None
            for k, ax in enumerate(axes):
                _arr, host = arrays[int(ax)]
                for sid in shapes:
                    v = float(shared[k]) if shared is not None else float(sample_fn((1,))[0])
                    host[sid] = _apply_op(float(host[sid]), v, operation, "geom_friction")
        for arr, host in arrays.values():
            arr.assign(host)
        return

    _unsupported("geom_friction", sim)


# ---------------------------------------------------------------------------
# body_com_offset — mjlab parity (mutates the body-frame COM)
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
    adds to the existing COM (MuJoCo ``body_ipos``, Newton ``body_com``).

    Raises:
        NotImplementedError: on a backend with no writable COM field.
    """
    if not ranges:
        raise ValueError("body_com_offset requires ranges, e.g. {0: (-0.025, 0.025)} (axis -> (lo, hi)).")
    sim = _sim_name(env)
    rng = np.random.default_rng()

    if sim == _MUJOCO:
        bids = _resolve_body_ids(env, asset_cfg)
        mp = _mujoco_model(env)
        for bid in bids:
            for ax, (lo, hi) in ranges.items():
                v = float(rng.uniform(lo, hi))
                mp.body_ipos[bid, ax] = _apply_op(float(mp.body_ipos[bid, ax]), v, operation, "body_com_offset")
        return

    if sim == _NEWTON:
        model = _newton_model(env, "body_com_offset")
        com = getattr(model, "body_com", None)
        if com is None:
            raise NotImplementedError("body_com_offset: the Newton model exposes no 'body_com' array to perturb.")
        bodies_by_env = _newton_body_ids(env, "body_com_offset", asset_cfg, getattr(asset_cfg, "body_names", ()))
        host = com.numpy()
        for bids in bodies_by_env.values():
            for bid in bids:
                for ax, (lo, hi) in ranges.items():
                    v = float(rng.uniform(lo, hi))
                    host[bid][ax] = _apply_op(float(host[bid][ax]), v, operation, "body_com_offset")
        com.assign(host)
        return

    _unsupported("body_com_offset", sim)


# ---------------------------------------------------------------------------
# body_mass — mjlab parity
# ---------------------------------------------------------------------------


def body_mass(
    env,
    *,
    asset_cfg=None,
    operation: str = "mul",
    ranges: tuple[float, float] = (0.9, 1.1),
    distribution: str = "uniform",
    recompute_inertia: bool = True,
    **_unused,
) -> None:
    """Mjlab ``dr.body_mass`` port — randomize per-body mass.

    On Newton the derived quantities are kept consistent with the new mass:
    ``body_inv_mass`` is rewritten and (when ``recompute_inertia``) the
    body's inertia tensor is rescaled by the mass ratio, mirroring
    ``roboverse_pack/randomization/humanoid.py``.

    Raises:
        NotImplementedError: on a backend with no writable mass field.
    """
    sim = _sim_name(env)
    sample_fn = _make_sampler(ranges, distribution)

    if sim == _MUJOCO:
        bids = _resolve_body_ids(env, asset_cfg)
        mp = _mujoco_model(env)
        for bid in bids:
            old = float(mp.body_mass[bid])
            new = _apply_op(old, float(sample_fn((1,))[0]), operation, "body_mass")
            mp.body_mass[bid] = new
            if recompute_inertia and old > 0.0:
                mp.body_inertia[bid] *= new / old
        return

    if sim == _NEWTON:
        model = _newton_model(env, "body_mass")
        bodies_by_env = _newton_body_ids(env, "body_mass", asset_cfg, getattr(asset_cfg, "body_names", ()))
        mass_arr = model.body_mass
        mass = mass_arr.numpy()
        inv_mass_arr = getattr(model, "body_inv_mass", None)
        inv_mass = inv_mass_arr.numpy() if inv_mass_arr is not None else None
        inertia_arr = getattr(model, "body_inertia", None) if recompute_inertia else None
        inertia = inertia_arr.numpy() if inertia_arr is not None else None
        inv_inertia_arr = getattr(model, "body_inv_inertia", None) if recompute_inertia else None
        inv_inertia = inv_inertia_arr.numpy() if inv_inertia_arr is not None else None

        for bids in bodies_by_env.values():
            for bid in bids:
                old = float(mass[bid])
                new = _apply_op(old, float(sample_fn((1,))[0]), operation, "body_mass")
                mass[bid] = new
                if inv_mass is not None:
                    inv_mass[bid] = 1.0 / new if new > 0.0 else 0.0
                if old > 0.0 and new > 0.0:
                    ratio = new / old
                    if inertia is not None:
                        inertia[bid] = inertia[bid] * ratio
                    if inv_inertia is not None:
                        inv_inertia[bid] = inv_inertia[bid] / ratio

        mass_arr.assign(mass)
        if inv_mass is not None:
            inv_mass_arr.assign(inv_mass)
        if inertia is not None:
            inertia_arr.assign(inertia)
        if inv_inertia is not None:
            inv_inertia_arr.assign(inv_inertia)
        return

    _unsupported("body_mass", sim)


# ---------------------------------------------------------------------------
# encoder_bias — mjlab parity. Stored on env so joint_pos obs reads it.
# ---------------------------------------------------------------------------


def _num_joints(env, asset_cfg) -> int:
    """Number of joints the encoder bias covers (backend-agnostic)."""
    if asset_cfg is not None and getattr(asset_cfg, "joint_names", None):
        return int(resolve_joint_ids(env, asset_cfg).numel())
    name = _asset_name(env, asset_cfg)
    joint_names = env.handler.get_joint_names(name, sort=True)
    if not joint_names:
        raise ValueError(
            f"encoder_bias: the handler reports no joints for asset '{name}';"
            " pass an asset_cfg with explicit joint_names."
        )
    return len(joint_names)


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

    Backend-agnostic: joints are resolved through ``handler.get_joint_names``
    (via ``scene_entity.resolve_joint_ids``), which every backend implements.
    """
    n = _num_joints(env, asset_cfg)
    rng = np.random.default_rng()
    bias = rng.uniform(*bias_range, size=(env.num_envs, n)).astype(np.float32)
    env._encoder_bias = torch.tensor(bias, device=env.device, dtype=torch.float32)
