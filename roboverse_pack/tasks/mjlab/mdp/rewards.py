"""Reward term functions ported from mjlab.

Signature: ``func(env: ManagerBasedRVEnv, env_states: TensorState, **params) -> torch.Tensor``
Returns shape ``(num_envs,)``. The manager loop multiplies by
``weight * step_dt`` before summing across terms.

Coverage of mjlab reward families:

- velocity/mdp/rewards (locomotion): track_linear_velocity, track_angular_velocity,
  upright, variable_posture (3-regime), body_angular_velocity_penalty,
  joint_pos_limits, action_rate_l2. Sensor-dependent rewards
  (feet_air_time / feet_clearance / feet_swing_height / feet_slip /
  soft_landing / angular_momentum_penalty / self_collision_cost) are
  declared with `_requires_sensor` stubs that emit a one-shot warning on
  first call and return zeros so a config can keep mjlab's term list
  unmodified while sensor wiring is in progress.

- manipulation/mdp/rewards: staged_position_reward, bring_object_reward,
  multi_cube_*, joint_velocity_hinge_penalty. These read site/object
  world-frame positions via `physics.data.site_xpos / data.xpos`. Newton
  path receives a one-shot warning + zero return until the equivalent
  query is wired.

- tracking/mdp/rewards: motion_global_anchor_position_error_exp /
  motion_global_anchor_orientation_error_exp /
  motion_relative_body_position_error_exp /
  motion_relative_body_orientation_error_exp /
  motion_global_body_linear_velocity_error_exp /
  motion_global_body_angular_velocity_error_exp. All read from the
  `MotionCommand` manager registered on `env.command_managers[name]`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from ._math import (
    GRAVITY_W,
    base_ang_vel_b,
    base_lin_vel_b,
    body_ang_vel_w,
    body_quat_w_xyzw,
    projected_gravity_b,
    quat_apply_inverse_xyzw,
)
from .scene_entity import (
    SceneEntityCfg,
    entity_joint_pos,
    entity_joint_vel,
    resolve_joint_ids,
)

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# tolerance helpers (dm_control / mjlab `value_at_margin=0.1` default)
# ---------------------------------------------------------------------------

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    """Gaussian sigmoid tolerance: 1 at x=0, ``value_at_margin=0.1`` at ``|x|=margin``."""
    if margin == 0:
        return (x == 0).float()
    scaled = x / margin * _GAUSSIAN_SCALE
    return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    """Quadratic sigmoid tolerance: 1 at x=0, 0 at ``|x|>=margin``."""
    if margin == 0:
        return (x == 0).float()
    scaled = x / margin * _QUADRATIC_SCALE
    return torch.clamp(1 - scaled**2, min=0.0)


# ---------------------------------------------------------------------------
# cartpole — 1 term
# ---------------------------------------------------------------------------


def cartpole_smooth_reward(
    env,
    env_states,
    cart_cfg: SceneEntityCfg,
    hinge_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """dm_control smooth cartpole reward: ``upright * centered * small_control * small_velocity``.

    Bit-identical to mjlab's ``cartpole_smooth_reward`` modulo the
    raw-action accessor: mjlab reads ``env.action_manager.action``,
    we read ``env._action`` (stored by ``ManagerBasedRVEnv.step`` before
    ``_process_action`` runs).

    Returns shape ``(num_envs,)`` in ``[0, 1]``.
    """
    hinge_angle = entity_joint_pos(env, env_states, hinge_cfg).squeeze(-1)
    pole_cos = torch.cos(hinge_angle)
    upright = (pole_cos + 1) / 2

    cart_pos = entity_joint_pos(env, env_states, cart_cfg).squeeze(-1)
    centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

    if env._action is None:
        control = torch.zeros_like(cart_pos)
    else:
        control = env._action.squeeze(-1)
    small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

    hinge_vel = entity_joint_vel(env, env_states, hinge_cfg).squeeze(-1)
    small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

    return upright * centered * small_control * small_velocity


# ---------------------------------------------------------------------------
# velocity — 7 sensor-free rewards ported from mjlab/tasks/velocity/mdp/rewards.py
# ---------------------------------------------------------------------------


def _read_command(env, command_name: str | None) -> torch.Tensor | None:
    """Pull the current ``(N, 3)`` velocity command from env.command_managers."""
    if command_name is None:
        return None
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None:
        return None
    return mgr.current()


def track_linear_velocity(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target: tuple[float, float] = (1.0, 0.0),
    std: float = 0.5,
) -> torch.Tensor:
    """Track commanded base linear velocity (xy in body frame).

    Mirrors ``mdp.track_linear_velocity``. When ``command_name`` resolves to a
    registered ``VelocityCommandManager`` on env, uses its per-env target.
    Falls back to static ``target`` xy if no command manager is wired.

    Shape: ``(num_envs,)``.
    """
    actual_b = base_lin_vel_b(env, env_states, asset_cfg.name)  # (N, 3)
    command = _read_command(env, command_name)
    if command is None:
        cmd_xy = torch.tensor(target, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    else:
        cmd_xy = command[:, :2]
    xy_error = torch.sum((cmd_xy - actual_b[:, :2]) ** 2, dim=1)
    z_error = actual_b[:, 2] ** 2
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_yaw_rate: float = 0.0,
    std: float = 0.707,
) -> torch.Tensor:
    """Track commanded yaw rate. ``mdp.track_angular_velocity`` port.

    Shape: ``(num_envs,)``.
    """
    actual_b = base_ang_vel_b(env, env_states, asset_cfg.name)
    command = _read_command(env, command_name)
    if command is None:
        cmd_yaw = torch.full((env.num_envs,), target_yaw_rate, device=env.device, dtype=torch.float32)
    else:
        cmd_yaw = command[:, 2]
    z_error = (cmd_yaw - actual_b[:, 2]) ** 2
    xy_error = torch.sum(actual_b[:, :2] ** 2, dim=1)  # penalize non-yaw rotations
    return torch.exp(-(z_error + xy_error) / std**2)


def upright(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
    std: float = 0.447,
) -> torch.Tensor:
    """Reward for keeping the base upright (no terrain-normal variant).

    Penalizes the xy components of projected gravity expressed in the
    *configured body's* frame — zero when upright. ``mdp.upright`` port for
    the flat-ground case (terrain-normal variant deferred to sensor wiring).

    mjlab projects ``gravity_vec_w`` into ``body_link_quat_w[body_ids]`` — the
    orientation of the body named in ``asset_cfg.body_names`` (g1: ``torso_link``,
    go1: ``trunk``), NOT necessarily the free-joint root. We read that body's
    world quat so the frame matches mjlab even when the body ≠ root.

    Shape: ``(num_envs,)``.
    """
    body_names = getattr(asset_cfg, "body_names", ()) or ()
    if body_names:
        body_name = body_names[0] if isinstance(body_names, (tuple, list)) else body_names
        quat = body_quat_w_xyzw(env, env_states, asset_cfg.name, body_name)
        gravity_w = torch.tensor(GRAVITY_W, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
        pg = quat_apply_inverse_xyzw(quat, gravity_w)
    else:
        pg = projected_gravity_b(env, env_states, asset_cfg.name)
    xy_sq = pg[:, 0] ** 2 + pg[:, 1] ** 2
    return torch.exp(-xy_sq / std**2)


def _resolve_named_stds(
    spec: dict[str, float] | float | torch.Tensor | None,
    joint_names: tuple[str, ...] | None,
    default: float,
    device,
) -> torch.Tensor:
    """Resolve a {regex-or-name: std} spec into a per-joint std tensor.

    Mirrors mjlab's ``resolve_matching_names_values``. Falls back to a
    single scalar broadcast or the supplied default when no per-joint
    spec is provided.
    """
    if isinstance(spec, torch.Tensor):
        return spec.to(device=device, dtype=torch.float32)
    if isinstance(spec, (int, float)):
        if joint_names is None:
            return torch.tensor([float(spec)], device=device, dtype=torch.float32)
        return torch.full((len(joint_names),), float(spec), device=device, dtype=torch.float32)
    if spec is None or not spec:
        if joint_names is None:
            return torch.tensor([default], device=device, dtype=torch.float32)
        return torch.full((len(joint_names),), default, device=device, dtype=torch.float32)

    import re

    assert joint_names is not None, "Per-joint std spec requires joint_names list."
    out: list[float] = []
    for jname in joint_names:
        match_val: float | None = None
        for pat, val in spec.items():
            if pat == jname or re.fullmatch(pat, jname):
                match_val = float(val)
                break
        out.append(default if match_val is None else match_val)
    return torch.tensor(out, device=device, dtype=torch.float32)


def variable_posture(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
    std: float | None = None,
    default_pose: torch.Tensor | None = None,
    command_name: str | None = None,
    std_standing: dict[str, float] | float | None = None,
    std_walking: dict[str, float] | float | None = None,
    std_running: dict[str, float] | float | None = None,
    walking_threshold: float = 0.05,
    running_threshold: float = 1.5,
) -> torch.Tensor:
    """Speed-regime ``mdp.variable_posture`` port — 1:1 with mjlab.

    Two modes:

    1. **Single-std** (backward compat with phase-3 scaffold): pass ``std``
       only — every joint gets the same gaussian tolerance regardless of
       commanded velocity.
    2. **3-regime** (mjlab parity): pass ``std_standing`` / ``std_walking``
       / ``std_running`` as either scalars or ``{joint-name-or-regex: std}``
       maps plus ``command_name`` to read the velocity command. The std
       used per step is selected by the magnitude of
       ``cmd_lin_norm + cmd_yaw_abs``.

    Shape: ``(num_envs,)``.
    """
    joint_ids = resolve_joint_ids(env, asset_cfg)
    qpos = entity_joint_pos(env, env_states, asset_cfg)
    n_envs, n_joints = qpos.shape

    if default_pose is None:
        default = getattr(env, "default_dof_pos_original", None)
        if default is not None:
            default = default[:, joint_ids]
        else:
            default = torch.zeros_like(qpos)
    else:
        default = default_pose.to(qpos.device, qpos.dtype) if default_pose.device != qpos.device else default_pose
    err_sq = (qpos - default) ** 2

    if std_standing is None and std_walking is None and std_running is None:
        # Single-std mode.
        s = std if std is not None else 0.5
        return torch.exp(-err_sq.mean(dim=1) / (s * s))

    # 3-regime mode.
    joint_names = asset_cfg.joint_names if isinstance(asset_cfg.joint_names, tuple) else tuple(asset_cfg.joint_names)
    s_stand = _resolve_named_stds(std_standing, joint_names, default=0.1, device=qpos.device)
    s_walk = _resolve_named_stds(std_walking, joint_names, default=0.3, device=qpos.device)
    s_run = _resolve_named_stds(std_running, joint_names, default=0.6, device=qpos.device)

    if command_name is None or command_name not in getattr(env, "command_managers", {}):
        # Without a registered command, assume "walking" tolerance.
        std_per_joint = s_walk
        std_b = std_per_joint.unsqueeze(0).expand(n_envs, -1)
    else:
        cmd = env.command_managers[command_name].current()  # (N, 3)
        speed = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()  # (N,)
        standing_mask = (speed < walking_threshold).float().unsqueeze(1)
        walking_mask = ((speed >= walking_threshold) & (speed < running_threshold)).float().unsqueeze(1)
        running_mask = (speed >= running_threshold).float().unsqueeze(1)
        std_b = (
            s_stand.unsqueeze(0) * standing_mask
            + s_walk.unsqueeze(0) * walking_mask
            + s_run.unsqueeze(0) * running_mask
        )  # (N, J)

    return torch.exp(-torch.mean(err_sq / (std_b * std_b), dim=1))


def body_angular_velocity_penalty(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize body xy angular velocity (don't penalize yaw — that's commanded).

    ``mdp.body_angular_velocity_penalty`` port. mjlab reads the *world-frame*
    angular velocity of the configured body (``body_link_ang_vel_w[body_ids]``)
    and squares its xy components — NOT the body-frame root ang vel. We read the
    named body's world-frame ang vel so the frame + body both match mjlab
    (g1: ``torso_link``); falls back to base-frame root ang vel only if no body
    is configured.

    Shape: ``(num_envs,)`` — typically combined with a negative weight.
    """
    body_names = getattr(asset_cfg, "body_names", ()) or ()
    if body_names:
        body_name = body_names[0] if isinstance(body_names, (tuple, list)) else body_names
        ang = body_ang_vel_w(env, env_states, asset_cfg.name, body_name)
    else:
        ang = base_ang_vel_b(env, env_states, asset_cfg.name)
    return ang[:, 0] ** 2 + ang[:, 1] ** 2


def joint_pos_limits(env, env_states, *, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for joint positions outside the URDF's soft-limit range.

    ``mdp.joint_pos_limits`` port. Reads joint position vs the model's
    ``jnt_range`` (already encoded in MJCF / URDF). Soft limit = 0.9 of
    hard range, matching IsaacLab convention.

    Shape: ``(num_envs,)`` — combined with a negative weight.
    """
    import mujoco

    joint_ids = resolve_joint_ids(env, asset_cfg).cpu().numpy()
    qpos_t = entity_joint_pos(env, env_states, asset_cfg)
    # Sim-aware joint range lookup:
    #   mujoco: read mp.jnt_range directly by joint name
    #   newton: pull from RobotCfg.joint_limits in scenario.robots[0]
    if asset_cfg._joint_ids is not None and isinstance(asset_cfg.joint_names, (tuple, list)):
        ranges: list[tuple[float, float]] = []
        if hasattr(env.handler, "physics"):
            model = env.handler.physics.model
            mp = model.ptr if hasattr(model, "ptr") else model
            for jname in asset_cfg.joint_names:
                jid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, jname)
                ranges.append((float(mp.jnt_range[jid, 0]), float(mp.jnt_range[jid, 1])))
        else:
            robot_cfg = next((r for r in env.scenario.robots if r.name == asset_cfg.name), None)
            jl = getattr(robot_cfg, "joint_limits", {}) if robot_cfg else {}
            for jname in asset_cfg.joint_names:
                lo, hi = jl.get(jname, (-3.14, 3.14))
                ranges.append((float(lo), float(hi)))
        ranges_t = torch.tensor(ranges, device=env.device, dtype=torch.float32)
        soft = 0.9
        mid = (ranges_t[:, 0] + ranges_t[:, 1]) / 2.0
        half = (ranges_t[:, 1] - ranges_t[:, 0]) / 2.0 * soft
        low = mid - half
        high = mid + half
        below = torch.clamp(low.unsqueeze(0) - qpos_t, min=0.0)
        above = torch.clamp(qpos_t - high.unsqueeze(0), min=0.0)
        return torch.sum(below + above, dim=1)
    return torch.zeros(env.num_envs, device=env.device)


def action_rate_l2(env, env_states) -> torch.Tensor:
    """Penalty for large action changes between steps. ``mdp.action_rate_l2`` port.

    Reads ``env._action`` (current raw policy output) vs ``env._prev_action``
    (stored by ``ManagerBasedRVEnv.step``).

    Shape: ``(num_envs,)`` — combined with a negative weight.
    """
    if env._action is None or env._prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    diff = env._action - env._prev_action
    return torch.sum(diff * diff, dim=1)


# ---------------------------------------------------------------------------
# manipulation rewards — mjlab/tasks/manipulation/mdp/rewards.py port
#
# The mjlab versions read ``robot.data.site_pos_w`` / ``obj.data.root_link_pos_w``;
# we read the equivalent MuJoCo arrays here. Newton path emits a one-shot
# warning + zero return until the equivalent query is wired.
# ---------------------------------------------------------------------------


def _mujoco_site_pos_w(env, site_name: str) -> np.ndarray | None:
    """World-frame site position via ``physics.data.site_xpos``."""
    handler = env.handler
    if not hasattr(handler, "physics"):
        return None
    import mujoco

    m = handler.physics.model
    mp = m.ptr if hasattr(m, "ptr") else m
    sid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        return None
    return np.asarray(handler.physics.data.site_xpos[sid], dtype=np.float32)


def _mujoco_body_pos_w(env, body_name: str) -> np.ndarray | None:
    """World-frame body com position via ``physics.data.xpos``."""
    handler = env.handler
    if not hasattr(handler, "physics"):
        return None
    import mujoco

    m = handler.physics.model
    mp = m.ptr if hasattr(m, "ptr") else m
    bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    return np.asarray(handler.physics.data.xpos[bid], dtype=np.float32)


def _read_command_tensor(env, command_name: str) -> torch.Tensor | None:
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None:
        return None
    cur = mgr.current()
    return cur if isinstance(cur, torch.Tensor) else None


# ---------------------------------------------------------------------------
# Sim-agnostic world-position readers (Newton path uses handler.get_states).
# mjlab reads ``robot.data.site_pos_w`` / ``obj.data.root_link_pos_w``; on
# Newton there is no MuJoCo ``physics`` object, so we read the equivalent
# batched tensors from ``handler.get_states(mode="tensor")``. Both return a
# ``(num_envs, 3)`` tensor so reward bodies are per-env on Newton.
# ---------------------------------------------------------------------------


def _newton_object_pos_w(env, name: str) -> torch.Tensor | None:
    """``(N,3)`` world position of a scene *object* on the Newton path."""
    st = env.handler.get_states(mode="tensor")
    objs = getattr(st, "objects", None) or {}
    if name in objs:
        return objs[name].root_state[:, :3]
    return None


def _newton_robot_ee_pos_w(env, body_suffix: str | None = None) -> torch.Tensor | None:
    """``(N,3)`` world position of the robot end-effector body on Newton.

    If ``body_suffix`` is given, match the body whose name ends with it;
    otherwise pick the deepest (most ``/``-segments) non-camera body, which
    for a serial manipulator is the gripper/tcp link — the closest analogue
    to mjlab's ``tcp_site``.
    """
    st = env.handler.get_states(mode="tensor")
    robots = getattr(st, "robots", None) or {}
    if not robots:
        return None
    rname = env.scenario.robots[0].name if env.scenario.robots else next(iter(robots))
    rs = robots.get(rname)
    if rs is None or rs.body_state is None:
        return None
    names = rs.body_names
    idx = None
    if body_suffix is not None:
        idx = next(
            (i for i, n in enumerate(names) if n.split("/")[-1] == body_suffix or n.endswith(body_suffix)),
            None,
        )
    if idx is None:
        cands = [(n.count("/"), i) for i, n in enumerate(names) if "camera" not in n.lower()]
        if not cands:
            return None
        idx = max(cands)[1]
    return rs.body_state[:, idx, :3]


def staged_position_reward(
    env,
    env_states,
    *,
    command_name: str,
    object_name: str,
    reaching_std: float = 0.2,
    bringing_std: float = 0.3,
    asset_cfg: SceneEntityCfg | None = None,
    site_name: str | None = None,
) -> torch.Tensor:
    """Mjlab ``manipulation.staged_position_reward`` port.

    Returns ``reaching * (1 + bringing)`` where:
      reaching = exp(-||ee - obj||^2 / reaching_std^2)
      bringing = exp(-||target - obj||^2 / bringing_std^2)

    The mjlab version reads ``robot.data.site_pos_w[:, site_ids]``; we
    accept either an explicit ``site_name`` (preferred for scene-MJCF
    tasks like lift_cube_yam) or fall back to the body-name of the asset
    if no site is given.

    Shape: ``(num_envs,)``.
    """
    if site_name is None and asset_cfg is None:
        raise ValueError("staged_position_reward needs either site_name or asset_cfg")

    if not hasattr(env.handler, "physics"):
        # Newton path: read ee + object positions per-env from get_states.
        ee = _newton_robot_ee_pos_w(env, site_name if site_name and site_name != "tcp_site" else None)
        obj = _newton_object_pos_w(env, object_name)
        if ee is None or obj is None:
            return torch.zeros(env.num_envs, device=env.device)
        cmd = _read_command_tensor(env, command_name)
        target = cmd if (cmd is not None and cmd.shape[-1] >= 3) else obj
        target = target.to(obj.device)
        reach = torch.exp(-((ee - obj) ** 2).sum(-1) / (reaching_std**2))
        bring = torch.exp(-((target - obj) ** 2).sum(-1) / (bringing_std**2))
        return reach * (1.0 + bring)

    ee_p = (
        _mujoco_site_pos_w(env, site_name)
        if site_name is not None
        else _mujoco_body_pos_w(env, asset_cfg.body_names[0] if asset_cfg.body_names else asset_cfg.name)
    )
    obj_p = _mujoco_body_pos_w(env, object_name)
    if ee_p is None or obj_p is None:
        return torch.zeros(env.num_envs, device=env.device)

    cmd = _read_command_tensor(env, command_name)
    target_p_np = cmd[0].detach().cpu().numpy().astype(np.float32) if cmd is not None and cmd.numel() >= 3 else obj_p

    reach_err = float(np.sum((ee_p - obj_p) ** 2))
    reach = float(np.exp(-reach_err / (reaching_std**2)))

    bring_err = float(np.sum((target_p_np - obj_p) ** 2))
    bring = float(np.exp(-bring_err / (bringing_std**2)))

    return torch.full((env.num_envs,), reach * (1.0 + bring), device=env.device)


def bring_object_reward(
    env,
    env_states,
    *,
    command_name: str,
    object_name: str,
    std: float = 0.05,
) -> torch.Tensor:
    """Mjlab ``manipulation.bring_object_reward`` port.

    Returns ``exp(-||target - obj||^2 / std^2)``. Reads the
    ``LiftingCommand``'s target via ``env.command_managers[command_name].current()``.

    Shape: ``(num_envs,)``.
    """
    if not hasattr(env.handler, "physics"):
        # Newton path: object + target both per-env tensors.
        obj = _newton_object_pos_w(env, object_name)
        cmd = _read_command_tensor(env, command_name)
        if obj is None or cmd is None or cmd.shape[-1] < 3:
            return torch.zeros(env.num_envs, device=env.device)
        target = cmd.to(obj.device)
        return torch.exp(-((target - obj) ** 2).sum(-1) / (std**2))

    obj_p = _mujoco_body_pos_w(env, object_name)
    if obj_p is None:
        return torch.zeros(env.num_envs, device=env.device)

    cmd = _read_command_tensor(env, command_name)
    if cmd is None or cmd.numel() < 3:
        return torch.zeros(env.num_envs, device=env.device)
    target_p_np = cmd[0].detach().cpu().numpy().astype(np.float32)
    err = float(np.sum((target_p_np - obj_p) ** 2))
    return torch.full((env.num_envs,), float(np.exp(-err / (std**2))), device=env.device)


def multi_cube_staged_position_reward(
    env,
    env_states,
    *,
    command_name: str,
    reaching_std: float = 0.2,
    bringing_std: float = 0.3,
    asset_cfg: SceneEntityCfg | None = None,
    site_name: str | None = None,
) -> torch.Tensor:
    """Mjlab ``manipulation.multi_cube_staged_position_reward`` port.

    Delegates target-cube identification to the ``MultiCubeLiftingCommand``
    on ``env.command_managers[command_name]``; the command must expose
    ``target_object_pos()`` and ``current()`` returning ``(N, 3)`` target.
    """
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None or not hasattr(mgr, "target_object_pos"):
        _warn_once(
            "multi_cube_staged_position_reward.no_command",
            "multi_cube_staged_position_reward requires a MultiCubeLiftingCommand; returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env.handler, "physics") or site_name is None:
        _warn_once(
            "multi_cube_staged_position_reward.unsupported",
            "multi_cube_staged_position_reward needs site_name + MuJoCo handler; returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)

    ee_p = _mujoco_site_pos_w(env, site_name)
    obj_pos_w = mgr.target_object_pos().detach().cpu().numpy()[0].astype(np.float32)
    target_p_np = mgr.current()[0].detach().cpu().numpy().astype(np.float32)
    if ee_p is None:
        return torch.zeros(env.num_envs, device=env.device)
    reach_err = float(np.sum((ee_p - obj_pos_w) ** 2))
    reach = float(np.exp(-reach_err / (reaching_std**2)))
    bring_err = float(np.sum((target_p_np - obj_pos_w) ** 2))
    bring = float(np.exp(-bring_err / (bringing_std**2)))
    return torch.full((env.num_envs,), reach * (1.0 + bring), device=env.device)


def multi_cube_bring_object_reward(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 0.05,
) -> torch.Tensor:
    """Mjlab ``manipulation.multi_cube_bring_object_reward`` port."""
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None or not hasattr(mgr, "target_object_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    obj_pos_w = mgr.target_object_pos().detach().cpu().numpy()[0].astype(np.float32)
    target_p_np = mgr.current()[0].detach().cpu().numpy().astype(np.float32)
    err = float(np.sum((target_p_np - obj_pos_w) ** 2))
    return torch.full((env.num_envs,), float(np.exp(-err / (std**2))), device=env.device)


def joint_velocity_hinge_penalty(
    env,
    env_states,
    *,
    asset_cfg: SceneEntityCfg,
    max_vel: float = 0.5,
) -> torch.Tensor:
    """Mjlab ``manipulation.joint_velocity_hinge_penalty`` port.

    Returns ``sum(max(|qvel| - max_vel, 0)^2)``. Penalty term — use with
    negative weight in the cfg.

    Shape: ``(num_envs,)``.
    """
    qvel = entity_joint_vel(env, env_states, asset_cfg)
    excess = (qvel.abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)


# ---------------------------------------------------------------------------
# tracking rewards — mjlab/tasks/tracking/mdp/rewards.py port
#
# Each reads a MotionCommand registered as
# ``env.command_managers[command_name]``. The command must expose:
#   .anchor_pos_w, .robot_anchor_pos_w               # (N, 3)
#   .anchor_quat_w, .robot_anchor_quat_w             # (N, 4) wxyz
#   .body_pos_relative_w, .robot_body_pos_w          # (N, B, 3)
#   .body_quat_relative_w, .robot_body_quat_w        # (N, B, 4)
#   .body_lin_vel_w, .robot_body_lin_vel_w           # (N, B, 3)
#   .body_ang_vel_w, .robot_body_ang_vel_w           # (N, B, 3)
#   .cfg.body_names: tuple[str, ...]
# If absent → emit one-shot warning + return zeros (training continues
# but tracking is no-op until command is registered).
# ---------------------------------------------------------------------------


def _quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angle between two unit quaternions (wxyz). Shape preserves leading dims.

    Native parity: mjlab ``quat_error_magnitude`` uses the box-minus log map
    ``||log(q1 * q2^-1)||`` (math.py:685-697), NOT ``2*acos(|dot|)``. The two
    agree near zero error but diverge for large angles, so we port the exact
    box-minus formula for float-eps reward parity.
    """
    from ._math import quat_error_magnitude_wxyz

    return quat_error_magnitude_wxyz(q1, q2)


def _get_motion_command(env, command_name: str):
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None or not hasattr(mgr, "anchor_pos_w"):
        return None
    return mgr


def _body_index_subset(mgr, body_names: tuple[str, ...] | None) -> list[int]:
    cfg_names = getattr(mgr.cfg, "body_names", None) if hasattr(mgr, "cfg") else None
    if cfg_names is None:
        return []
    if body_names is None:
        return list(range(len(cfg_names)))
    return [i for i, name in enumerate(cfg_names) if name in body_names]


def motion_global_anchor_position_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 0.3,
) -> torch.Tensor:
    """Return the exponential reward for global anchor position tracking error."""
    mgr = _get_motion_command(env, command_name)
    if mgr is None:
        _warn_once(
            "motion_global_anchor_position_error_exp.no_cmd",
            f"MotionCommand '{command_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    err = torch.sum((mgr.anchor_pos_w - mgr.robot_anchor_pos_w) ** 2, dim=-1)
    return torch.exp(-err / (std**2))


def motion_global_anchor_orientation_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 0.4,
) -> torch.Tensor:
    """Return the exponential reward for global anchor orientation tracking error."""
    mgr = _get_motion_command(env, command_name)
    if mgr is None:
        _warn_once(
            "motion_global_anchor_orientation_error_exp.no_cmd",
            f"MotionCommand '{command_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    err = _quat_error_magnitude(mgr.anchor_quat_w, mgr.robot_anchor_quat_w) ** 2
    return torch.exp(-err / (std**2))


def _motion_body_subset_error(
    env,
    command_name: str,
    std: float,
    body_names,
    attr_target: str,
    attr_robot: str,
    is_orient: bool = False,
) -> torch.Tensor:
    mgr = _get_motion_command(env, command_name)
    if mgr is None:
        _warn_once(
            f"motion.{attr_target}.no_cmd",
            f"MotionCommand '{command_name}' not registered for {attr_target} — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    body_idx = _body_index_subset(mgr, body_names)
    if not body_idx:
        return torch.zeros(env.num_envs, device=env.device)
    t = getattr(mgr, attr_target)[:, body_idx]
    r = getattr(mgr, attr_robot)[:, body_idx]
    if is_orient:
        err = _quat_error_magnitude(t, r) ** 2
        err_mean = err.mean(dim=-1)
    else:
        err = torch.sum((t - r) ** 2, dim=-1)  # (N, |body_idx|)
        err_mean = err.mean(dim=-1)
    return torch.exp(-err_mean / (std**2))


def motion_relative_body_position_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 0.3,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Return the exponential reward for relative body position tracking error."""
    return _motion_body_subset_error(env, command_name, std, body_names, "body_pos_relative_w", "robot_body_pos_w")


def motion_relative_body_orientation_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 0.4,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Return the exponential reward for relative body orientation tracking error."""
    return _motion_body_subset_error(
        env,
        command_name,
        std,
        body_names,
        "body_quat_relative_w",
        "robot_body_quat_w",
        is_orient=True,
    )


def motion_global_body_linear_velocity_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 1.0,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Return the exponential reward for global body linear-velocity tracking error."""
    return _motion_body_subset_error(env, command_name, std, body_names, "body_lin_vel_w", "robot_body_lin_vel_w")


def motion_global_body_angular_velocity_error_exp(
    env,
    env_states,
    *,
    command_name: str,
    std: float = 1.0,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Return the exponential reward for global body angular-velocity tracking error."""
    return _motion_body_subset_error(env, command_name, std, body_names, "body_ang_vel_w", "robot_body_ang_vel_w")


# ---------------------------------------------------------------------------
# sensor-dependent rewards — read from ``env._mjlab_sensors[<name>]``.
#
# Each function falls back to zeros (with one-shot warning) when the
# expected sensor isn't registered, so config-level mjlab parity is
# preserved without forcing every task to wire sensors.
# ---------------------------------------------------------------------------


def _get_sensor(env, sensor_name: str):
    return getattr(env, "_mjlab_sensors", {}).get(sensor_name)


def feet_air_time(
    env,
    env_states,
    *,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.feet_air_time`` — full port.

    Reward for feet spending time in air. Reads
    ``env._mjlab_sensors[sensor_name].data.current_air_time``.

    Shape: ``(num_envs,)``.
    """
    sensor = _get_sensor(env, sensor_name)
    if sensor is None:
        _warn_once(
            f"feet_air_time.no_sensor.{sensor_name}",
            f"feet_air_time: sensor '{sensor_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    data = sensor.data
    current_air = data.current_air_time
    if current_air is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_range = (current_air > threshold_min) & (current_air < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    if command_name is not None:
        cmd = _read_command(env, command_name)
        if cmd is not None:
            total_cmd = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
            reward = reward * (total_cmd > command_threshold).float()
    return reward


def feet_clearance(
    env,
    env_states,
    *,
    target_height: float = 0.1,
    height_sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg | None = None,
    site_names: tuple[str, ...] | None = None,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.feet_clearance`` — full port.

    Penalize deviation from ``target_height`` weighted by foot velocity.
    Foot height from the TerrainHeightSensor; foot velocity is computed
    from ``physics.data.site_xvel`` (or ``cvel`` for body fallback).

    Shape: ``(num_envs,)``.
    """
    h_sensor = _get_sensor(env, height_sensor_name)
    if h_sensor is None:
        _warn_once(
            f"feet_clearance.no_sensor.{height_sensor_name}",
            f"feet_clearance: height sensor '{height_sensor_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    heights = h_sensor.data.heights  # (N, P)
    delta = (heights - target_height).abs()

    # Foot velocity — prefer sites if the sensor was set up that way; else
    # fall back to body-level velocity from physics.data.
    site_names = h_sensor.cfg.primary_sites
    if site_names:
        foot_vel = _foot_xy_speed_via_sites(env, site_names)
    else:
        foot_vel = _foot_xy_speed_via_bodies(env, h_sensor.cfg.primary_bodies)
    cost = (delta * foot_vel).sum(dim=1)
    if command_name is not None:
        cmd = _read_command(env, command_name)
        if cmd is not None:
            total = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
            cost = cost * (total > command_threshold).float()
    return cost


def _foot_xy_speed_via_sites(env, site_names: tuple[str, ...]) -> torch.Tensor:
    """Per-site xy-speed (norm of linear xy velocity) read from physics.data.

    Returns ``(num_envs, len(site_names))``. Uses ``mj_objectVelocity`` for
    world-frame linear vel at each site, then takes the xy norm.
    """
    if not site_names or not hasattr(env.handler, "physics"):
        return torch.zeros((env.num_envs, len(site_names)), device=env.device)
    import mujoco

    physics = env.handler.physics
    mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    P = len(site_names)
    out = np.zeros((env.num_envs, P), dtype=np.float32)
    for j, sname in enumerate(site_names):
        sid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid < 0:
            continue
        v = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(mp, physics.data.ptr, mujoco.mjtObj.mjOBJ_SITE, sid, v, 0)
        out[0, j] = float(np.linalg.norm(v[3:5]))
    return torch.from_numpy(out).to(env.device)


def feet_swing_height(
    env,
    env_states,
    *,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float = 0.05,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.feet_swing_height`` — full port with peak-height tracking.

    Per-foot peak height is maintained over each swing phase; at landing
    (first_contact) the squared deviation from ``target_height`` is the
    cost. State is keyed off ``env`` via ``env._reward_state["feet_swing_height"]``
    so the function is stateless from the caller's view.
    """
    contact = _get_sensor(env, sensor_name)
    h_sensor = _get_sensor(env, height_sensor_name)
    if contact is None or h_sensor is None:
        _warn_once(
            f"feet_swing_height.no_sensor.{sensor_name}",
            "feet_swing_height: missing sensor — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    cmd = _read_command(env, command_name)
    if cmd is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_heights = h_sensor.data.heights  # (N, P)
    if not hasattr(env, "_reward_state"):
        env._reward_state = {}
    state_key = f"feet_swing_height.{height_sensor_name}"
    if state_key not in env._reward_state:
        env._reward_state[state_key] = torch.zeros_like(foot_heights)
    peak = env._reward_state[state_key]
    if peak.shape != foot_heights.shape:
        peak = torch.zeros_like(foot_heights)

    found = contact.data.found
    if found is None:
        found = torch.zeros_like(foot_heights)
    in_air = found == 0
    peak = torch.where(in_air, torch.maximum(peak, foot_heights), peak)

    first_contact = contact.compute_first_contact(env.step_dt)
    total = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
    active = (total > command_threshold).float()
    error = peak / max(target_height, 1e-6) - 1.0
    cost = torch.sum(error**2 * first_contact.float(), dim=1) * active

    # Reset peak on landing.
    peak = torch.where(first_contact, torch.zeros_like(peak), peak)
    env._reward_state[state_key] = peak
    return cost


def feet_slip(
    env,
    env_states,
    *,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg | None = None,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.feet_slip`` — full port.

    Penalize foot sliding (xy velocity while in contact).
    """
    sensor = _get_sensor(env, sensor_name)
    if sensor is None:
        _warn_once(
            f"feet_slip.no_sensor.{sensor_name}",
            f"feet_slip: sensor '{sensor_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    cmd = _read_command(env, command_name)
    if cmd is None:
        return torch.zeros(env.num_envs, device=env.device)
    total = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
    active = (total > command_threshold).float()
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_contact = (found > 0).float()
    # Match the sensor's primary list to grab xy velocities.
    foot_speed = _foot_xy_speed_via_sites(env, getattr(asset_cfg, "site_names", ()) or ())
    if foot_speed.shape[1] == 0:
        # Fallback: use primary bodies of the contact sensor.
        foot_speed = _foot_xy_speed_via_bodies(env, sensor.cfg.primary_bodies)
    vel_sq = foot_speed**2
    cost = torch.sum(vel_sq * in_contact, dim=1) * active
    return cost


def _foot_xy_speed_via_bodies(env, body_names: tuple[str, ...]) -> torch.Tensor:
    """Per-body xy-speed (linear-vel norm in xy)."""
    if not body_names:
        return torch.zeros((env.num_envs, len(body_names)), device=env.device)
    if not hasattr(env.handler, "physics"):
        # Newton path: foot xy linear speed from get_states body velocities.
        st = env.handler.get_states(mode="tensor")
        robots = getattr(st, "robots", None) or {}
        rname = env.scenario.robots[0].name if env.scenario.robots else next(iter(robots), None)
        rs = robots.get(rname) if rname is not None else None
        P = len(body_names)
        if rs is None or rs.body_state is None:
            return torch.zeros((env.num_envs, P), device=env.device)
        names = rs.body_names
        out = torch.zeros((env.num_envs, P), device=env.device)
        for j, pat in enumerate(body_names):
            col = next((k for k, n in enumerate(names) if n == pat or n.endswith("_" + pat) or n.endswith(pat)), -1)
            if 0 <= col < rs.body_state.shape[1]:
                out[:, j] = torch.norm(rs.body_state[:, col, 7:9], dim=-1)
        return out
    import mujoco

    physics = env.handler.physics
    mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    P = len(body_names)
    out = np.zeros((env.num_envs, P), dtype=np.float32)
    for j, bname in enumerate(body_names):
        bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        v = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(mp, physics.data.ptr, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        out[0, j] = float(np.linalg.norm(v[3:5]))
    return torch.from_numpy(out).to(env.device)


def soft_landing(
    env,
    env_states,
    *,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.soft_landing`` — full port.

    Penalize impact forces at landing (first_contact frame).
    """
    sensor = _get_sensor(env, sensor_name)
    if sensor is None:
        _warn_once(
            f"soft_landing.no_sensor.{sensor_name}",
            f"soft_landing: sensor '{sensor_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    force = sensor.data.force  # (N, P, 3)
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = torch.norm(force, dim=-1)  # (N, P)
    first_contact = sensor.compute_first_contact(env.step_dt)
    landing = force_mag * first_contact.float()
    cost = torch.sum(landing, dim=1)
    if command_name is not None:
        cmd = _read_command(env, command_name)
        if cmd is not None:
            total = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
            cost = cost * (total > command_threshold).float()
    return cost


def angular_momentum_penalty(
    env,
    env_states,
    *,
    sensor_name: str = "robot/root_angmom",
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity.angular_momentum_penalty`` — full port.

    Reads ``env._mjlab_sensors[sensor_name].data`` (a 3-vector per env)
    and returns its squared magnitude. Sensor is typically a
    :class:`BuiltinSensor` over ``data.subtree_angmom[robot_body]``.
    """
    sensor = _get_sensor(env, sensor_name)
    if sensor is None:
        _warn_once(
            f"angular_momentum_penalty.no_sensor.{sensor_name}",
            f"angular_momentum_penalty: sensor '{sensor_name}' not registered — returning zeros.",
        )
        return torch.zeros(env.num_envs, device=env.device)
    angmom = sensor.data  # (N, 3)
    if not isinstance(angmom, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(angmom**2, dim=-1)


def self_collision_cost(
    env,
    env_states,
    *,
    sensor_name: str = "self_contact",
    force_threshold: float = 10.0,
    **_unused,
) -> torch.Tensor:
    """Mjlab ``velocity/tracking.self_collision_cost`` — full port.

    Reads ``env._mjlab_sensors[sensor_name].data.force_history`` (rolling
    buffer) and counts substeps where any contact force magnitude exceeds
    ``force_threshold``.
    """
    sensor = _get_sensor(env, sensor_name)
    if sensor is None:
        # Fallback to ContactForces query for backward-compat with older tasks.
        query = getattr(env.handler, "contact_forces_query", None) or getattr(env, "_contact_forces_query", None)
        if query is None:
            _warn_once(
                f"self_collision_cost.no_sensor.{sensor_name}",
                f"self_collision_cost: sensor '{sensor_name}' not registered — returning zeros.",
            )
            return torch.zeros(env.num_envs, device=env.device)
        try:
            forces = query.contact_forces
            mag = torch.norm(forces, dim=-1)
            return (mag > force_threshold).any(dim=-1).float()
        except Exception:
            return torch.zeros(env.num_envs, device=env.device)

    data = sensor.data
    history = getattr(data, "force_history", None)
    if history is not None:
        # history shape: (N, P, H, 3)
        mag = torch.norm(history, dim=-1)  # (N, P, H)
        hit_any = (mag > force_threshold).any(dim=1)  # (N, H)
        return hit_any.sum(dim=-1).float()
    found = data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return found.sum(dim=-1).float()
