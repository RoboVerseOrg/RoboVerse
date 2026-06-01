"""Event term functions ported from mjlab.

Signature for reset-mode events:
    ``func(env, env_ids: torch.Tensor, **params) -> None``  — mutates state in place

Reset events run after ``ManagerBasedRVEnv._reset_idx`` has restored
initial states; they're the standard mechanism for per-episode noise
injection (joint angle/velocity randomization, base velocity push, etc.).
"""

from __future__ import annotations

import torch

from .scene_entity import SceneEntityCfg, resolve_joint_ids


def reset_joints_by_offset(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Add uniform noise to joint positions and velocities for the reset envs.

    Reads the current state (which the parent ``_reset_idx`` just restored
    to initial), perturbs the selected joint subset, and writes back via
    ``handler.set_states``. This matches mjlab's
    ``mdp.reset_joints_by_offset`` exactly.

    Args:
        env: The env.
        env_ids: Indices of envs to perturb. Shape ``(K,)``.
        position_range: ``(low, high)`` uniform offset for joint positions.
        velocity_range: ``(low, high)`` uniform offset for joint velocities.
        asset_cfg: Selects the entity + joint subset.
    """
    if env_ids.numel() == 0:
        return

    states = env.handler.get_states(mode="tensor")
    state = states.robots[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    n_envs_reset = env_ids.numel()
    n_joints_sel = joint_ids.numel()

    pos_lo, pos_hi = position_range
    vel_lo, vel_hi = velocity_range
    pos_offsets = torch.empty((n_envs_reset, n_joints_sel), device=env.device).uniform_(pos_lo, pos_hi)
    vel_offsets = torch.empty((n_envs_reset, n_joints_sel), device=env.device).uniform_(vel_lo, vel_hi)

    # In-place edit of the cached TensorState — handler will copy on set_states.
    state.joint_pos[env_ids[:, None], joint_ids[None, :]] += pos_offsets
    state.joint_vel[env_ids[:, None], joint_ids[None, :]] += vel_offsets

    env.handler.set_states(states, env_ids=env_ids.tolist())


def reset_robot_to_default_newton(
    env,
    env_ids: torch.Tensor,
    *,
    asset_cfg: SceneEntityCfg,
    default_pose: torch.Tensor,
    base_height: float,
    pose_range: dict[str, tuple[float, float]],
    base_velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
    joint_velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Newton-path per-env reset matching mjlab ``reset_base`` + ``reset_robot_joints``.

    Writes, per reset env, via ``handler.set_states``:
      - base root pose: pos ``(x, y, base_height + z)`` where x/y/z are sampled
        from ``pose_range`` (mjlab ``reset_root_state_uniform`` pose offset on top
        of the default root z), and an orientation = identity quat rotated by a
        uniform ``yaw`` (and optional roll/pitch) from ``pose_range`` — this yaw
        spread is the load-bearing per-env exploration driver mjlab relies on.
      - base root velocity: lin/ang vel from ``base_velocity_range`` (mjlab uses
        an empty range = zero for the velocity task; kept configurable).
      - joints: ``default_pose`` (mapped to the handler's joint column order via
        ``resolve_joint_ids``) plus uniform ``joint_position_range`` offset; joint
        velocity = uniform ``joint_velocity_range``.

    ``default_pose`` is a 1-D tensor in ``asset_cfg.joint_names`` order; this fn
    scatters it onto the resolved (possibly differently-ordered) handler columns
    so the stance lands on the correct joints regardless of Newton's sort order.
    """
    if env_ids.numel() == 0:
        return

    states = env.handler.get_states(mode="tensor")
    state = states.robots[asset_cfg.name]
    device = env.device
    joint_ids = resolve_joint_ids(env, asset_cfg)
    n = env_ids.numel()

    # --- base root state (pos + quat_wxyz + lin/ang vel) ---
    def _u(key: str) -> torch.Tensor:
        rng = pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(rng[0], rng[1])

    dx, dy, dz = _u("x"), _u("y"), _u("z")
    roll, pitch, yaw = _u("roll"), _u("pitch"), _u("yaw")

    # Identity base orientation rotated by (roll, pitch, yaw). quat_wxyz.
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    root = state.root_state  # (N, 13): pos(3) quat_wxyz(4) lin(3) ang(3)
    root[env_ids, 0] = dx
    root[env_ids, 1] = dy
    root[env_ids, 2] = base_height + dz
    root[env_ids, 3] = qw
    root[env_ids, 4] = qx
    root[env_ids, 5] = qy
    root[env_ids, 6] = qz

    bvr = base_velocity_range or {}
    vel_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    for i, key in enumerate(vel_keys):
        if key in bvr:
            lo, hi = bvr[key]
            root[env_ids, 7 + i] = torch.empty(n, device=device).uniform_(lo, hi)
        else:
            root[env_ids, 7 + i] = 0.0

    # --- joints: default pose (reordered to handler columns) + uniform noise ---
    default_pose = default_pose.to(device=device, dtype=state.joint_pos.dtype)
    # default_pose[k] is the value for asset_cfg.joint_names[k]; joint_ids[k] is
    # the handler column for that same joint. Scatter so each default lands right.
    pos = default_pose.unsqueeze(0).expand(n, -1).clone()
    pos += torch.empty((n, joint_ids.numel()), device=device).uniform_(*joint_position_range)
    vel = torch.empty((n, joint_ids.numel()), device=device).uniform_(*joint_velocity_range)

    state.joint_pos[env_ids[:, None], joint_ids[None, :]] = pos
    state.joint_vel[env_ids[:, None], joint_ids[None, :]] = vel

    env.handler.set_states(states, env_ids=env_ids.tolist())
