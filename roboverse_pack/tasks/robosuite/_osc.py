"""Native OSC_POSE controller — robosuite-free reimplementation.

A faithful port of robosuite v1.5.2's ``OperationalSpaceController`` (the BASIC
composite controller's arm part) plus its parallel-jaw gripper mapping. Reads
state straight from a ``mujoco.MjModel``/``MjData`` so it can drive MetaSim's
handler with NO ``import robosuite`` at runtime.

Only the small pure-math helpers (opspace matrices, nullspace torques,
orientation error, axis-angle↔matrix) are reproduced here — these are standard
operational-space-control / rotation math, not robosuite-specific machinery.

STATUS — VERIFIED end-to-end. The full native controller (arm OSC + parallel-jaw
gripper) driving MetaSim's handler over the official robomimic Lift benchmark
*actions* completes the task **20/20**, matching robosuite's own controller 20/20,
with NO robosuite controller in the loop — see
``tools/robosuite_integration/verify_native_controller.py``. (Isolated per-substep
arm torques agree with robosuite to ~3-7e-3 N·m / ≈2e-5 relative when measured on
a fresh env; that micro-benchmark is sensitive to env-step capture timing, so the
authoritative check is the end-to-end task-success one.)

This is the controller half of making robosuite deletable. What remains to drop
the robosuite *package* entirely is vendoring the static per-task MJCF + meshes
and porting each task's reward/obs (the Lift success check is a few lines —
``_native_lift_success`` in verify_native_controller.py).
"""

from __future__ import annotations

import mujoco
import numpy as np

# ---- pure math (ported from robosuite.utils.{control_utils,transform_utils}) ----


def axisangle2quat(vec: np.ndarray) -> np.ndarray:
    """Axis-angle (axis*angle) -> quaternion [x,y,z,w]."""
    angle = np.linalg.norm(vec)
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat2mat(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix (robosuite convention)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    rc1, rc2, rc3 = current[:, 0], current[:, 1], current[:, 2]
    rd1, rd2, rd3 = desired[:, 0], desired[:, 1], desired[:, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    m_inv = np.linalg.inv(mass_matrix)
    lambda_full = np.linalg.pinv(J_full @ m_inv @ J_full.T)
    lambda_pos = np.linalg.pinv(J_pos @ m_inv @ J_pos.T)
    lambda_ori = np.linalg.pinv(J_ori @ m_inv @ J_ori.T)
    Jbar = (m_inv @ J_full.T) @ lambda_full
    nullspace = np.eye(J_full.shape[-1]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace


def nullspace_torques(mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel, joint_kp=10.0):
    joint_kv = np.sqrt(joint_kp) * 2
    pose_torques = mass_matrix @ (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    return nullspace_matrix.T @ pose_torques


class NativeOSC:
    """OSC_POSE (position+orientation, decoupled) for a fixed-base arm + gripper."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        eef_site: str,
        arm_joint_qpos: list[int],
        arm_joint_qvel: list[int],
        arm_actuator_ids: list[int],
        gripper_actuator_ids: list[int],
        initial_joint: np.ndarray,
        base_pos: np.ndarray,
        base_ori: np.ndarray,
        kp: float = 150.0,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        gripper_sign: float = 1.0,
    ):
        self.m, self.d = model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        self.qpos_index = np.asarray(arm_joint_qpos, dtype=int)
        self.qvel_index = np.asarray(arm_joint_qvel, dtype=int)
        self.arm_act = np.asarray(arm_actuator_ids, dtype=int)
        self.grip_act = np.asarray(gripper_actuator_ids, dtype=int)
        self.initial_joint = np.asarray(initial_joint, dtype=float)
        self.origin_pos = np.asarray(base_pos, dtype=float)
        self.origin_ori = np.asarray(base_ori, dtype=float).reshape(3, 3)
        self.kp = np.full(6, kp)
        self.kd = 2.0 * np.sqrt(self.kp)
        self.output_max = np.asarray(output_max)
        self.output_min = -self.output_max
        self.input_max = np.ones(6)
        self.input_min = -np.ones(6)
        self.gripper_sign = gripper_sign
        self.goal_pos = None
        self.goal_ori = None

    # ---- helpers reading mujoco state ----
    def _site_pose(self):
        pos = self.d.site_xpos[self.site_id].copy()
        mat = self.d.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, mat

    def _site_jac(self):
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site_id)
        J_pos = jacp[:, self.qvel_index]
        J_ori = jacr[:, self.qvel_index]
        return J_pos, J_ori, np.vstack([J_pos, J_ori])

    def _site_vel(self, J_pos, J_ori):
        qvel = self.d.qvel[self.qvel_index]
        return J_pos @ qvel, J_ori @ qvel

    def _mass_matrix(self):
        full = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, full, self.d.qM)
        return full[np.ix_(self.qvel_index, self.qvel_index)]

    def _world_to_origin(self, pos):
        return self.origin_ori.T @ (pos - self.origin_pos)

    def scale_action(self, a):
        a = np.clip(a, self.input_min, self.input_max)
        scale = np.abs(self.output_max - self.output_min) / np.abs(self.input_max - self.input_min)
        out_t = (self.output_max + self.output_min) / 2.0
        in_t = (self.input_max + self.input_min) / 2.0
        return (a - in_t) * scale + out_t

    def set_goal(self, action):
        ref_pos, ref_mat = self._site_pose()
        scaled = self.scale_action(np.asarray(action[:6], dtype=float))
        # goal in base (origin) frame, "achieved" update mode
        self.goal_pos = self._world_to_origin(ref_pos) + scaled[:3]
        cur_ori_base = self.origin_ori.T @ ref_mat
        R_err = quat2mat(axisangle2quat(scaled[3:6]))
        self.goal_ori = R_err @ cur_ori_base

    def compute_arm_torques(self):
        ref_pos, ref_mat = self._site_pose()
        J_pos, J_ori, J_full = self._site_jac()
        ref_pos_vel, ref_ori_vel = self._site_vel(J_pos, J_ori)
        mass_matrix = self._mass_matrix()

        desired_world_pos = self.origin_pos + self.origin_ori @ self.goal_pos
        desired_world_ori = self.origin_ori @ self.goal_ori

        position_error = desired_world_pos - ref_pos
        vel_pos_error = -ref_pos_vel  # fixed base: base vel = 0
        desired_force = self.kp[:3] * position_error + self.kd[:3] * vel_pos_error

        ori_error = orientation_error(desired_world_ori, ref_mat)
        vel_ori_error = -ref_ori_vel
        desired_torque = self.kp[3:] * ori_error + self.kd[3:] * vel_ori_error

        _, lambda_pos, lambda_ori, nullspace = opspace_matrices(mass_matrix, J_full, J_pos, J_ori)
        decoupled = np.concatenate([lambda_pos @ desired_force, lambda_ori @ desired_torque])

        qfrc_bias = self.d.qfrc_bias[self.qvel_index]
        torques = J_full.T @ decoupled + qfrc_bias
        torques += nullspace_torques(
            mass_matrix, nullspace, self.initial_joint, self.d.qpos[self.qpos_index], self.d.qvel[self.qvel_index]
        )
        return torques

    _grip_target = None
    _GRIP_SPEED = 0.004  # position target ramp per control step (empirical, robosuite SimpleGripController)
    _GRIP_RANGE = (0.0, 0.04)  # finger1 ctrlrange; finger2 is the mirror

    def apply(self, action):
        """Set arm goal + advance gripper target one control step. Call per control step."""
        self.set_goal(action)
        if len(self.grip_act):
            if self._grip_target is None:
                self._grip_target = float(self.d.ctrl[self.grip_act[0]]) or 0.02
            g = float(action[6]) if len(action) > 6 else 0.0
            # close (+1) drives target -> 0; open (-1) drives target -> 0.04
            self._grip_target = float(np.clip(self._grip_target - self._GRIP_SPEED * g, *self._GRIP_RANGE))

    def write_ctrl(self):
        """Compute torques at the current state and write actuator ctrl (per substep)."""
        self.d.ctrl[self.arm_act] = self.compute_arm_torques()
        if len(self.grip_act) and self._grip_target is not None:
            self.d.ctrl[self.grip_act[0]] = self._grip_target
            if len(self.grip_act) > 1:
                self.d.ctrl[self.grip_act[1]] = -self._grip_target
