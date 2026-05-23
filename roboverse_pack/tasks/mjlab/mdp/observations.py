"""Observation term functions ported from mjlab.

Signature: ``func(env: ManagerBasedRVEnv, env_states: TensorState, **params) -> torch.Tensor``
Returns shape ``(num_envs, D)`` where ``D`` depends on the term. The
``ManagerBasedRVEnv._observation_group`` loop concatenates all terms in
a group along the last dim, so each term must produce 2-D output.
"""

from __future__ import annotations

import torch

from ._math import base_ang_vel_b, base_lin_vel_b, projected_gravity_b
from .rewards import (
    _mujoco_body_pos_w,
    _mujoco_site_pos_w,
    _newton_object_pos_w,
    _newton_robot_ee_pos_w,
    _read_command_tensor,
)
from .scene_entity import (
    SceneEntityCfg,
    entity_joint_pos,
    entity_joint_vel,
    resolve_joint_ids,
)

# ---------------------------------------------------------------------------
# Base proprioception (mjlab velocity_env_cfg actor obs: base_lin_vel /
# base_ang_vel / projected_gravity) + command (generated_commands). These are
# the obs terms mjlab feeds the locomotion policy in addition to joint state;
# without them the policy cannot see its own base velocity or the command,
# so it can only learn to stand. Ported 1:1 from mjlab's obs functions
# (base velocities are body-frame; projected_gravity uses the normalized
# gravity unit vector — see ._math.GRAVITY_W).
# ---------------------------------------------------------------------------


def base_lin_vel(env, env_states, asset_cfg: SceneEntityCfg, imu_offset=None) -> torch.Tensor:
    """Body-frame base linear velocity. mjlab ``builtin_sensor(imu_lin_vel)``.

    mjlab's ``base_lin_vel`` obs term reads the ``imu_lin_vel`` velocimeter,
    which measures the linear velocity at the IMU **site** -- offset
    ``imu_offset`` (in body frame) from the base body origin -- not at the
    origin itself. A velocimeter on a rigid body therefore reports
    ``v_base_b + omega_b x r_imu``. When ``imu_offset`` is given we add that
    ``omega x r`` term so the observation matches the sensor 1:1; when it is
    ``None`` we return the plain body-origin velocity (back-compat for tasks
    with no offset IMU). NOTE: the velocity-tracking *reward* uses the true
    body velocity (mjlab ``root_link_lin_vel_b``), so this offset must stay in
    the observation path only and never leak into the reward.
    """
    v_b = base_lin_vel_b(env, env_states, asset_cfg.name)
    if imu_offset is None:
        return v_b
    w_b = base_ang_vel_b(env, env_states, asset_cfg.name)
    r = torch.tensor(imu_offset, device=v_b.device, dtype=v_b.dtype).expand_as(v_b)
    return v_b + torch.linalg.cross(w_b, r, dim=-1)


def base_ang_vel(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body-frame base angular velocity. mjlab ``builtin_sensor(imu_ang_vel)``."""
    return base_ang_vel_b(env, env_states, asset_cfg.name)


def projected_gravity(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity unit vector in body frame. mjlab ``mdp.projected_gravity``."""
    return projected_gravity_b(env, env_states, asset_cfg.name)


def generated_commands(env, env_states, *, command_name: str) -> torch.Tensor:
    """Current command vector. mjlab ``mdp.generated_commands``.

    Reads ``env.command_managers[command_name].current()`` → ``(N, D)``.
    Returns zeros if the command manager isn't wired yet (keeps obs dim stable).
    """
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    cur = mgr.current()
    if not isinstance(cur, torch.Tensor):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return cur


def ee_to_object_distance(env, env_states, *, object_name: str, site_name: str = "tcp_site") -> torch.Tensor:
    """Distance vector ``object - end_effector``. mjlab ``ee_to_object_distance``.

    mjlab returns the vector in the robot *base* frame
    (``quat_apply(quat_inv(base_quat), obj_w - ee_w)``). The yam lift task is a
    **fixed-base** arm mounted at identity orientation, so the base frame equals
    the world frame and we return the world-frame vector directly (1:1). Reuses
    the same world-position readers as the lift rewards, so mujoco and Newton
    paths stay consistent. Shape ``(num_envs, 3)``.
    """
    if not hasattr(env.handler, "physics"):  # Newton path (per-env tensors)
        ee = _newton_robot_ee_pos_w(env, site_name if site_name and site_name != "tcp_site" else None)
        obj = _newton_object_pos_w(env, object_name)
        if ee is None or obj is None:
            return torch.zeros(env.num_envs, 3, device=env.device)
        return (obj - ee).to(env.device, torch.float32)
    ee = _mujoco_site_pos_w(env, site_name)
    obj = _mujoco_body_pos_w(env, object_name)
    if ee is None or obj is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    vec = torch.as_tensor(obj - ee, device=env.device, dtype=torch.float32)
    return vec.unsqueeze(0).expand(env.num_envs, -1)


def object_to_goal_distance(env, env_states, *, object_name: str, command_name: str) -> torch.Tensor:
    """Distance vector ``goal - object``. mjlab ``object_to_goal_distance``.

    Goal is the ``LiftingCommand`` target read via
    ``command_managers[command_name].current()[:, :3]``. World frame == base
    frame for the fixed-base yam (see :func:`ee_to_object_distance`). Shape
    ``(num_envs, 3)``.
    """
    cmd = _read_command_tensor(env, command_name)
    if cmd is None or cmd.shape[-1] < 3:
        return torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env.handler, "physics"):  # Newton path
        obj = _newton_object_pos_w(env, object_name)
        if obj is None:
            return torch.zeros(env.num_envs, 3, device=env.device)
        target = cmd[:, :3].to(obj.device, torch.float32)
        return (target - obj).to(env.device, torch.float32)
    obj = _mujoco_body_pos_w(env, object_name)
    if obj is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    obj_t = torch.as_tensor(obj, device=env.device, dtype=torch.float32)
    target = cmd[0, :3].to(env.device, torch.float32)
    return (target - obj_t).unsqueeze(0).expand(env.num_envs, -1)


def height_scan(env, env_states, *, sensor_name: str = "terrain_scan") -> torch.Tensor:
    """Terrain height scan. mjlab velocity-rough ``height_scan``.

    Reads a :class:`~..sensors.TerrainGridScanSensor` registered on
    ``env._mjlab_sensors[sensor_name]`` → ``(N, num_rays)`` (frame_z - hit_z per
    grid ray). Returns zeros if the sensor isn't wired (keeps obs dim stable).
    """
    sensors = getattr(env, "_mjlab_sensors", {})
    sensor = sensors.get(sensor_name)
    if sensor is None or sensor.data.heights is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return sensor.data.heights


def joint_pos_rel(env, env_states, asset_cfg: SceneEntityCfg, default=None) -> torch.Tensor:
    """Joint positions, optionally relative to a default pose (mjlab 1:1).

    Mjlab's ``joint_pos_rel`` returns ``data.joint_pos - data.default_joint_pos``.
    For free-base assets with zero default (cartpole) this equals raw joint_pos,
    so ``default`` is omitted there. For legged robots (go1 thigh=0.9/calf=-1.8,
    g1 stance) pass ``default`` = per-joint default pose (in ``asset_cfg`` joint
    order) so the obs is centered the same way mjlab feeds its policy.

    Shape: ``(num_envs, len(asset_cfg.joint_ids))``.
    """
    pos = entity_joint_pos(env, env_states, asset_cfg)
    if default is not None:
        d = torch.as_tensor(default, dtype=pos.dtype, device=pos.device).reshape(1, -1)
        pos = pos - d
    return pos


def joint_pos_rel_default(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions relative to the robot's default joint positions.

    Requires the env to have ``default_dof_pos_original`` populated (set
    by ``LeggedRobotTask``-style envs); falls back to raw joint_pos if
    not present. Mjlab's velocity / tracking tasks use this everywhere.
    """
    pos = entity_joint_pos(env, env_states, asset_cfg)
    default = getattr(env, "default_dof_pos_original", None)
    if default is not None:
        joint_ids = resolve_joint_ids(env, asset_cfg)
        pos = pos - default[:, joint_ids]
    return pos


def joint_vel_rel(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities. Shape: ``(num_envs, len(asset_cfg.joint_ids))``."""
    return entity_joint_vel(env, env_states, asset_cfg)


def pole_angle_cos_sin(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """``[cos(angle), sin(angle)]`` for the selected hinge joint.

    Same ordering as mjlab's cartpole observation; the manager-loop
    concatenation produces the 5-D obs vector
    ``[cart_pos, cos(pole), sin(pole), cart_vel, pole_vel]`` when wired
    in mjlab's PolicyCfg order.

    Shape: ``(num_envs, 2)``.
    """
    angle = entity_joint_pos(env, env_states, asset_cfg)
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def last_action(env, env_states) -> torch.Tensor:
    """Previous policy action (pre-processing). Shape: ``(num_envs, num_actions)``."""
    if env._action is None:
        return torch.zeros((env.num_envs, env.num_actions), device=env.device)
    return env._action


# ---------------------------------------------------------------------------
# camera observations — port of mjlab manipulation obs (depth / rgb / seg)
# ---------------------------------------------------------------------------


def _get_camera(env, camera_name: str):
    cameras = getattr(env_states := env.handler.get_states(mode="tensor"), "cameras", {})
    if cameras is None or camera_name not in cameras:
        return None
    return cameras[camera_name]


def camera_rgb(env, env_states, *, camera_name: str, scale: float = 1.0 / 255.0) -> torch.Tensor:
    """Flattened RGB obs from a named camera. Mjlab parity for rgb variants.

    Reads ``env_states.cameras[camera_name].rgb`` (shape ``(N, H, W, 3)``,
    uint8), scales to float, returns shape ``(N, H*W*3)``. If the camera
    is missing or rgb is None, returns zeros (one-shot warning).

    For mjlab's ``lift_cube_yam_rgb`` the camera was a wrist-mounted
    256x256 RGB cam => obs dim = 256*256*3 = 196608. The exact resolution
    is configured by the camera cfg in the task scenario.
    """
    cameras = getattr(env_states, "cameras", {}) or {}
    cam = cameras.get(camera_name)
    if cam is None or getattr(cam, "rgb", None) is None:
        N = env.num_envs
        return torch.zeros((N, 0), device=env.device)
    rgb = cam.rgb
    if rgb.dtype != torch.float32:
        rgb = rgb.to(torch.float32) * scale
    N = rgb.shape[0]
    return rgb.reshape(N, -1)


def camera_depth(env, env_states, *, camera_name: str, scale: float = 1.0) -> torch.Tensor:
    """Flattened depth obs from a named camera. ``(N, H, W) → (N, H*W)``.

    Reads ``env_states.cameras[camera_name].depth``. Missing cam → zeros.
    """
    cameras = getattr(env_states, "cameras", {}) or {}
    cam = cameras.get(camera_name)
    if cam is None or getattr(cam, "depth", None) is None:
        return torch.zeros((env.num_envs, 0), device=env.device)
    depth = cam.depth
    if depth.dtype != torch.float32:
        depth = depth.to(torch.float32)
    if scale != 1.0:
        depth = depth * scale
    return depth.reshape(depth.shape[0], -1)


def camera_instance_seg(env, env_states, *, camera_name: str) -> torch.Tensor:
    """Flattened instance-segmentation obs from a named camera.

    Reads ``env_states.cameras[camera_name].instance_seg``. Returns int32
    cast to float (so the manager-loop concatenate is type-consistent).
    Missing cam → zeros.
    """
    cameras = getattr(env_states, "cameras", {}) or {}
    cam = cameras.get(camera_name)
    seg = getattr(cam, "instance_seg", None) if cam is not None else None
    if seg is None:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return seg.reshape(seg.shape[0], -1).to(torch.float32)
