"""Observation term functions ported from mjlab.

Signature: ``func(env: ManagerBasedRVEnv, env_states: TensorState, **params) -> torch.Tensor``
Returns shape ``(num_envs, D)`` where ``D`` depends on the term. The
``ManagerBasedRVEnv._observation_group`` loop concatenates all terms in
a group along the last dim, so each term must produce 2-D output.
"""

from __future__ import annotations

import torch

from ._math import base_ang_vel_imu_b, base_lin_vel_imu_b, projected_gravity_b, quat_apply_inverse_xyzw
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


def base_lin_vel(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base linear velocity obs. mjlab ``builtin_sensor(imu_lin_vel)`` (IMU site)."""
    return base_lin_vel_imu_b(env, env_states, asset_cfg.name)


def base_ang_vel(env, env_states, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base angular velocity obs. mjlab ``builtin_sensor(imu_ang_vel)`` (IMU gyro)."""
    return base_ang_vel_imu_b(env, env_states, asset_cfg.name)


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


def height_scan(
    env, env_states, *, sensor_name: str, offset: float = 0.0, scale: float = 1.0, num_rays: int = 187
) -> torch.Tensor:
    """Terrain height-scan obs from a grid raycast sensor (mjlab 1:1).

    Port of mjlab ``envs/mdp/observations.py:height_scan`` + the rough velocity
    cfg's ``scale=1/terrain_scan.max_distance`` (``velocity_env_cfg.py:204``).
    Reads the :class:`HeightScanSensor` registered on ``env._mjlab_sensors``: the
    per-ray height is ``frame_z - hit_z`` (misses already filled with the sensor's
    ``max_distance``), then multiplied by ``scale`` and with ``offset`` subtracted.

    During env construction the first ``reset()`` computes obs before the task
    registers its sensors; if the sensor is missing we emit a fixed-dim zero
    vector (``num_rays``) so the obs layout stays stable. By the time the policy
    runs (or the parity harness recomputes), the sensor is present.

    Shape: ``(num_envs, num_rays)`` (187 for the velocity grid).
    """
    sensor = getattr(env, "_mjlab_sensors", {}).get(sensor_name)
    if sensor is None:
        return torch.zeros(env.num_envs, num_rays, device=env.device)
    sensor.update()
    heights = sensor.data.heights  # (N, R)
    return (heights - offset) * scale


# ---------------------------------------------------------------------------
# motion-tracking obs — port of mjlab tasks/tracking/mdp/observations.py
#
# These read the MotionCommandManager's anchor frames (all quats wxyz, the same
# convention the manager reads from ``data.xquat`` and the motion npz) and
# express the desired anchor pose in the robot anchor's body frame, mirroring
# ``motion_anchor_pos_b`` / ``motion_anchor_ori_b`` exactly.
# ---------------------------------------------------------------------------


def motion_anchor_pos_b(env, env_states, *, command_name: str) -> torch.Tensor:
    """Desired anchor position in the robot anchor body frame. Shape ``(N, 3)``.

    Native (tracking/mdp/observations.py:18-28):
    ``subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w,
    anchor_pos_w, anchor_quat_w)`` and keep the position part.
    """
    from mjlab.utils.lab_api.math import subtract_frame_transforms

    mgr = env.command_managers[command_name]
    pos, _ = subtract_frame_transforms(
        mgr.robot_anchor_pos_w,
        mgr.robot_anchor_quat_w,
        mgr.anchor_pos_w,
        mgr.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env, env_states, *, command_name: str) -> torch.Tensor:
    """Desired anchor orientation in the robot anchor body frame. Shape ``(N, 6)``.

    Native (tracking/mdp/observations.py:31-41): take the rotation from
    ``subtract_frame_transforms``, convert to a rotation matrix and keep its
    first two columns (6-D continuous rotation representation).
    """
    from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

    mgr = env.command_managers[command_name]
    _, ori = subtract_frame_transforms(
        mgr.robot_anchor_pos_w,
        mgr.robot_anchor_quat_w,
        mgr.anchor_pos_w,
        mgr.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# manipulation obs — port of mjlab tasks/manipulation/mdp/observations.py
# ---------------------------------------------------------------------------


def _site_pos_w(env, *, site_name: str) -> torch.Tensor:
    """World-frame position of a named MuJoCo site. Shape ``(num_envs, 3)``.

    Scene-MJCF path (YAM lift_cube): the robot loads as the scene rather
    than a registered RobotCfg entity, so read the site directly from
    ``physics.data.site_xpos`` (mirrors ``robot.data.site_pos_w`` on the
    mjlab side, which for the YAM ``grasp_site`` is the same world point).
    """
    import mujoco
    import numpy as np

    physics = env.handler.physics
    mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    sid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = np.asarray(physics.data.site_xpos[sid], dtype=np.float32)
    return torch.as_tensor(pos, device=env.device).unsqueeze(0).expand(env.num_envs, -1)


def _body_pos_w(env, *, body_name: str) -> torch.Tensor:
    """World-frame position of a named MuJoCo body. Shape ``(num_envs, 3)``.

    Mirrors mjlab ``obj.data.root_link_pos_w`` for the free-floating cube.
    """
    import mujoco
    import numpy as np

    physics = env.handler.physics
    mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = np.asarray(physics.data.xpos[bid], dtype=np.float32)
    return torch.as_tensor(pos, device=env.device).unsqueeze(0).expand(env.num_envs, -1)


def _base_quat_xyzw(env, *, body_name: str) -> torch.Tensor:
    """World-frame orientation (xyzw) of the robot base body. Shape ``(num_envs, 4)``.

    mjlab rotates the ee/cube/goal deltas into the robot base frame via
    ``quat_inv(robot.data.root_link_quat_w)``. For the fixed-base YAM the
    base (``arm`` body) is at identity, but read it from ``data.xquat``
    (wxyz) and convert to xyzw so this stays correct for any base pose.
    """
    import mujoco
    import numpy as np

    physics = env.handler.physics
    mp = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, body_name)
    q_wxyz = np.asarray(physics.data.xquat[bid], dtype=np.float32)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    return torch.as_tensor(q_xyzw, device=env.device).unsqueeze(0).expand(env.num_envs, -1)


def ee_to_object_distance(
    env,
    env_states,
    *,
    object_name: str,
    site_name: str,
    base_body_name: str = "arm",
) -> torch.Tensor:
    """Distance vector from end effector to object in the robot base frame.

    Port of mjlab ``manipulation.mdp.ee_to_object_distance``: returns
    ``quat_apply(quat_inv(base_quat_w), obj_pos_w - ee_pos_w)`` where the EE
    is the ``site_name`` site and the object is ``object_name``'s body.
    Shape ``(num_envs, 3)``.
    """
    ee_pos_w = _site_pos_w(env, site_name=site_name)
    obj_pos_w = _body_pos_w(env, body_name=object_name)
    distance_vec_w = obj_pos_w - ee_pos_w
    base_quat = _base_quat_xyzw(env, body_name=base_body_name)
    return quat_apply_inverse_xyzw(base_quat, distance_vec_w)


def object_to_goal_distance(
    env,
    env_states,
    *,
    object_name: str,
    command_name: str,
    base_body_name: str = "arm",
) -> torch.Tensor:
    """Distance vector from object to goal in the robot base frame.

    Port of mjlab ``manipulation.mdp.object_to_goal_distance``: returns
    ``quat_apply(quat_inv(base_quat_w), goal_pos_w - obj_pos_w)`` where the
    goal is the ``command_name`` LiftingCommand target. Shape ``(num_envs, 3)``.
    """
    obj_pos_w = _body_pos_w(env, body_name=object_name)
    mgr = getattr(env, "command_managers", {}).get(command_name)
    if mgr is None:
        goal_pos_w = obj_pos_w
    else:
        goal_pos_w = mgr.target_pos
    distance_vec_w = goal_pos_w - obj_pos_w
    base_quat = _base_quat_xyzw(env, body_name=base_body_name)
    return quat_apply_inverse_xyzw(base_quat, distance_vec_w)


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
