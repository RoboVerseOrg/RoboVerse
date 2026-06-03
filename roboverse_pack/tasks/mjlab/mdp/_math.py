"""Quaternion / frame-transform helpers used by velocity MDP rewards.

Mjlab's reward functions expect body-frame quantities (e.g.
``asset.data.root_link_lin_vel_b``). RoboVerse's ``TensorState.root_state``
stores world-frame linear/angular velocity at ``[:, 7:10]`` / ``[:, 10:13]``
(quaternion at ``[:, 3:7]`` in xyzw order); these helpers do the
world-frame → body-frame transform that mjlab's Entity does internally.

For scene-MJCF tasks where the asset isn't a RoboVerse RobotCfg, fall
back to reading from ``physics.data.qpos`` / ``qvel`` directly (MuJoCo
free-joint convention: qpos[3:7] is wxyz, qvel[0:3] is world-frame
linear vel, qvel[3:6] is body-frame angular vel — note the body-frame
default for ang vel, unlike lin vel).
"""

from __future__ import annotations

import numpy as np
import torch

from metasim.utils.math import convert_quat, quat_apply, quat_conjugate

# mjlab uses the *normalized* gravity direction (entity.py: gravity_vec_w =
# [0, 0, -1.0]) for projected_gravity_b — NOT the 9.81 m/s² magnitude. The
# `upright` reward and projected_gravity obs both consume this, so matching
# mjlab's unit vector is required for 1:1 (a 9.81× scale silently saturates
# the upright reward to ~0 on any tilt).
GRAVITY_W: tuple[float, float, float] = (0.0, 0.0, -1.0)


def quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` from world frame into body frame using the inverse of ``q``.

    Thin **xyzw**-convention adapter over :func:`metasim.utils.math.quat_apply`
    (RoboVerse ``root_state`` stores the quaternion xyzw; core math is wxyz).
    Rotating by the inverse of a unit quaternion equals rotating by its
    conjugate. Bitwise-identical to the previous hand-rolled implementation.

    Args:
        q: ``(..., 4)`` quaternion in **xyzw** order.
        v: ``(..., 3)`` world-frame vector.

    Returns:
        ``(..., 3)`` body-frame vector.
    """
    return quat_apply(quat_conjugate(convert_quat(q, to="wxyz")), v)


def quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` from body frame into world frame using ``q`` (xyzw).

    Thin xyzw-convention adapter over :func:`metasim.utils.math.quat_apply`.
    """
    return quat_apply(convert_quat(q, to="wxyz"), v)


def get_base_state(env, env_states, asset_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(root_quat_xyzw, lin_vel_w, ang_vel_w)`` of the asset's base.

    Reads from ``env_states.robots[name].root_state`` (RoboVerse layout:
    pos(3)+quat_xyzw(4)+lin_vel(3)+ang_vel(3)) when the asset is a
    registered robot; falls back to ``physics.data.qpos[3:7] (wxyz)`` +
    ``qvel[:6]`` for scene-MJCF assets.

    Note: MuJoCo qvel for a free joint stores **body-frame** angular vel
    in qvel[3:6], not world. We rotate it to world here so caller
    consistently gets world-frame vels.
    """
    if asset_name in env_states.robots:
        rs = env_states.robots[asset_name].root_state
        # MetaSim TensorState.root_state stores the quaternion in **wxyz**
        # order (same as the MuJoCo free-joint qpos convention used in the
        # scene-MJCF fallback below). The body-frame helpers here expect
        # xyzw, so reorder. (Reading it as xyzw directly silently applies a
        # 180°-about-x rotation — identity wxyz [1,0,0,0] becomes xyzw
        # [1,0,0,0] = 180° about x — which flipped projected_gravity to
        # [0,0,+1] and corrupted base_lin/ang_vel on every Newton run.)
        q_wxyz = rs[:, 3:7]
        quat_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)
        lin_w = rs[:, 7:10]
        ang_w = rs[:, 10:13]
        return quat_xyzw, lin_w, ang_w
    # Scene-MJCF fallback (num_envs=1 for now).
    qpos = np.asarray(env.handler.physics.data.qpos, dtype=np.float32)
    qvel = np.asarray(env.handler.physics.data.qvel, dtype=np.float32)
    # MuJoCo free joint: qpos[3:7] = quat in wxyz; convert to xyzw.
    w, x, y, z = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
    quat_xyzw = torch.tensor([[x, y, z, w]], device=env.device, dtype=torch.float32)
    lin_w = torch.tensor([[float(qvel[0]), float(qvel[1]), float(qvel[2])]], device=env.device, dtype=torch.float32)
    # MuJoCo free joint: qvel[3:6] = body-frame angular vel; rotate to world.
    ang_b = torch.tensor([[float(qvel[3]), float(qvel[4]), float(qvel[5])]], device=env.device, dtype=torch.float32)
    ang_w = quat_apply_xyzw(quat_xyzw, ang_b)
    # Broadcast to num_envs (scene-MJCF is single env but keep generic).
    return (
        quat_xyzw.expand(env.num_envs, -1),
        lin_w.expand(env.num_envs, -1),
        ang_w.expand(env.num_envs, -1),
    )


def read_builtin_sensor(env, sensor_name: str) -> torch.Tensor | None:
    """Read a named MuJoCo sensor's value from ``physics.data.sensordata``.

    mjlab's velocity / tracking actor obs read ``base_lin_vel`` and
    ``base_ang_vel`` from the robot's IMU **velocimeter** (``imu_lin_vel``)
    and **gyro** (``imu_ang_vel``) site sensors, not from the free-joint
    ``qvel``. The velocimeter reports the linear velocity of the *IMU site*
    expressed in the site frame, which differs from the body-origin velocity
    by ``ω × r_site_offset``; the gyro reports the site-frame angular
    velocity. The scene-MJCF the RoboVerse tasks load retains these sensors,
    so reading them here reproduces mjlab's ``builtin_sensor`` term bitwise.

    Returns ``(num_envs, dim)`` or ``None`` if the sensor or the MuJoCo
    physics handle is unavailable (e.g. the Newton GPU path), in which case
    callers fall back to the qvel-based reconstruction.
    """
    handler = getattr(env, "handler", None)
    physics = getattr(handler, "physics", None)
    if physics is None:
        return None
    try:
        import mujoco

        model = physics.model
        mp = model.ptr if hasattr(model, "ptr") else model
        sid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None
        adr = int(mp.sensor_adr[sid])
        dim = int(mp.sensor_dim[sid])
        sd = np.asarray(physics.data.sensordata[adr : adr + dim], dtype=np.float32)
        return torch.tensor(sd, device=env.device, dtype=torch.float32).reshape(1, dim).expand(env.num_envs, -1)
    except Exception:
        return None


def base_lin_vel_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Body-origin linear velocity of the asset's base, body frame. ``(num_envs, 3)``.

    This is the **body-link** velocity (mjlab ``asset.data.root_link_lin_vel_b``),
    used by the locomotion *rewards* (``track_linear_velocity`` etc). The
    *observations* instead read the IMU velocimeter via ``base_lin_vel_imu_b`` —
    mjlab deliberately uses different velocity sources for obs vs reward (the IMU
    site velocity differs from the body-origin velocity by ``ω × r_imu_offset``).
    """
    quat, lin_w, _ = get_base_state(env, env_states, asset_name)
    return quat_apply_inverse_xyzw(quat, lin_w)


def base_ang_vel_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Body-origin angular velocity of the asset's base, body frame. ``(num_envs, 3)``.

    Body-link angular velocity (mjlab ``asset.data.root_link_ang_vel_b``), used by
    *rewards*. Observations use the IMU gyro via ``base_ang_vel_imu_b``.
    """
    quat, _, ang_w = get_base_state(env, env_states, asset_name)
    return quat_apply_inverse_xyzw(quat, ang_w)


_IMU_OFFSET_CACHE: dict[str, torch.Tensor] = {}


def _imu_lin_vel_site_offset(env, asset_name: str) -> torch.Tensor | None:
    """Position of the ``imu_lin_vel`` velocimeter site in its parent body frame.

    Reads the offset once from the robot's MJCF (the site referenced by the
    ``imu_lin_vel`` velocimeter) and caches it per ``mjcf_path``. Returns a
    ``(3,)`` tensor on ``env.device`` or ``None`` if it cannot be resolved.

    mjlab's velocimeter reports the velocity of the IMU *site* expressed in the
    site frame. Both go1/g1 IMU sites carry only a translation (no rotation),
    so site frame == parent-body frame and the site velocity in body frame is
    ``base_lin_vel_b + cross(base_ang_vel_b, r_imu_offset)`` — see
    ``base_lin_vel_imu_b`` (verified bit-for-bit against the MuJoCo sensor).
    """
    mjcf_path = None
    scenario = getattr(env, "scenario", None)
    robots = getattr(scenario, "robots", None) or []
    for r in robots:
        if getattr(r, "name", None) == asset_name:
            mjcf_path = getattr(r, "mjcf_path", None)
            break
    if mjcf_path is None and robots:
        mjcf_path = getattr(robots[0], "mjcf_path", None)
    if mjcf_path is None:
        return None
    if mjcf_path in _IMU_OFFSET_CACHE:
        off = _IMU_OFFSET_CACHE[mjcf_path]
        return None if off is None else off.to(env.device)
    off = None
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(mjcf_path)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_vel")
        if sid >= 0:
            site_id = int(m.sensor_objid[sid])
            pos = np.asarray(m.site_pos[site_id], dtype=np.float32)
            quat = np.asarray(m.site_quat[site_id], dtype=np.float64)  # wxyz
            # Site frame must equal body frame for the simple cross-product
            # formula; both go1/g1 IMU sites satisfy this (identity quat).
            if not np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-7):
                off = None
            else:
                off = torch.tensor(pos, dtype=torch.float32)
    except Exception:
        off = None
    _IMU_OFFSET_CACHE[mjcf_path] = off
    return None if off is None else off.to(env.device)


def base_lin_vel_imu_b(env, env_states, asset_name: str) -> torch.Tensor:
    """IMU-velocimeter linear velocity for *observations*. ``(num_envs, 3)``.

    Prefers the robot's ``imu_lin_vel`` velocimeter sensor (mjlab obs parity:
    velocity of the IMU site, in site frame). On the Newton path no MuJoCo
    sensor handle exists, so reconstruct the velocimeter reading from the
    body-origin velocity and the body-frame angular velocity:
    ``v_imu_b = base_lin_vel_b + cross(base_ang_vel_b, r_imu_offset)``,
    where ``r_imu_offset`` is the velocimeter site position in the parent body
    frame (read from the MJCF). This matches the MuJoCo velocimeter to ~1e-6.
    Falls back to the body-origin velocity only if the offset can't be read.
    """
    sensed = read_builtin_sensor(env, "imu_lin_vel")
    if sensed is not None:
        return sensed
    lin_b = base_lin_vel_b(env, env_states, asset_name)
    r = _imu_lin_vel_site_offset(env, asset_name)
    if r is None:
        return lin_b
    ang_b = base_ang_vel_b(env, env_states, asset_name)
    return lin_b + torch.cross(ang_b, r.expand_as(ang_b), dim=-1)


def base_ang_vel_imu_b(env, env_states, asset_name: str) -> torch.Tensor:
    """IMU-gyro angular velocity for *observations*. ``(num_envs, 3)``.

    Prefers the robot's ``imu_ang_vel`` gyro sensor (mjlab obs parity); falls
    back to ``base_ang_vel_b`` when no such sensor exists.
    """
    sensed = read_builtin_sensor(env, "imu_ang_vel")
    if sensed is not None:
        return sensed
    return base_ang_vel_b(env, env_states, asset_name)


def projected_gravity_b(env, env_states, asset_name: str) -> torch.Tensor:
    """Gravity vector in body frame. Shape ``(num_envs, 3)``."""
    quat, _, _ = get_base_state(env, env_states, asset_name)
    gravity_w = torch.tensor(GRAVITY_W, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    return quat_apply_inverse_xyzw(quat, gravity_w)


# ---------------------------------------------------------------------------
# Named-body world-frame readers.
#
# mjlab's ``upright`` and ``body_angular_velocity_penalty`` rewards do NOT read
# the free-joint root; they read a *specific* body (``body_link_quat_w`` /
# ``body_link_ang_vel_w`` indexed by ``asset_cfg.body_ids``). For go1 that body
# (``trunk``) coincides with the root, but for g1 it is ``torso_link`` — a
# different body than the ``pelvis`` root — so reading the root quat/angvel
# diverges from mjlab. These helpers read the requested body by name from
# MuJoCo (``data.xquat`` for orientation, ``mj_objectVelocity`` for the
# world-frame spatial velocity) with a Newton ``body_state`` fallback.
# ---------------------------------------------------------------------------


def body_quat_w_xyzw(env, env_states, asset_name: str, body_name: str) -> torch.Tensor:
    """World-frame orientation (xyzw) of a named body. Shape ``(num_envs, 4)``.

    Mirrors mjlab ``asset.data.body_link_quat_w[:, body_ids]``.
    """
    handler = getattr(env, "handler", None)
    physics = getattr(handler, "physics", None)
    if physics is not None:
        import mujoco

        m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            q = np.asarray(physics.data.xquat[bid], dtype=np.float32)  # wxyz
            quat_xyzw = torch.tensor([q[1], q[2], q[3], q[0]], device=env.device, dtype=torch.float32)
            return quat_xyzw.unsqueeze(0).expand(env.num_envs, -1)
    # Newton fallback: pull the body's quat from the batched body_state.
    rs = env_states.robots.get(asset_name) if asset_name in env_states.robots else None
    if rs is not None and getattr(rs, "body_state", None) is not None:
        names = rs.body_names
        col = next(
            (i for i, n in enumerate(names) if n == body_name or n.endswith("/" + body_name) or n.endswith(body_name)),
            None,
        )
        if col is not None:
            q_wxyz = rs.body_state[:, col, 3:7]
            return torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)
    # Last resort: root quat (correct when the body IS the root, e.g. go1 trunk).
    quat, _, _ = get_base_state(env, env_states, asset_name)
    return quat


def body_ang_vel_w(env, env_states, asset_name: str, body_name: str) -> torch.Tensor:
    """World-frame angular velocity of a named body. Shape ``(num_envs, 3)``.

    Mirrors mjlab ``asset.data.body_link_ang_vel_w[:, body_ids]``.
    """
    handler = getattr(env, "handler", None)
    physics = getattr(handler, "physics", None)
    if physics is not None:
        import mujoco

        m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            v = np.zeros(6, dtype=np.float64)
            # flg_local=0 -> world frame; mj_objectVelocity returns (ang(3), lin(3)).
            mujoco.mj_objectVelocity(m, physics.data.ptr, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
            ang_w = torch.tensor(v[:3], device=env.device, dtype=torch.float32)
            return ang_w.unsqueeze(0).expand(env.num_envs, -1)
    rs = env_states.robots.get(asset_name) if asset_name in env_states.robots else None
    if rs is not None and getattr(rs, "body_state", None) is not None:
        names = rs.body_names
        col = next(
            (i for i, n in enumerate(names) if n == body_name or n.endswith("/" + body_name) or n.endswith(body_name)),
            None,
        )
        if col is not None:
            return rs.body_state[:, col, 10:13]
    _, _, ang_w = get_base_state(env, env_states, asset_name)
    return ang_w


# ---------------------------------------------------------------------------
# wxyz quaternion utilities.
#
# These were previously hand-rolled here as verbatim ports of mjlab's math
# (which is itself IsaacLab-derived). ``metasim.utils.math`` carries the SAME
# IsaacLab-lineage implementations with identical operation order, so callers
# now import ``quat_mul / quat_inv / quat_apply / yaw_quat /
# quat_error_magnitude`` etc. directly from core (verified bitwise-equal:
# quat_mul/apply/yaw/axis_angle/error_magnitude diff 0.0, quat_inv ~1e-7 and
# == conjugate on unit quats). The only mjlab-specific helpers kept in this
# module are the xyzw-convention rotation adapters above and the obs/sensor
# readers below, which have no core equivalent.
# ---------------------------------------------------------------------------
