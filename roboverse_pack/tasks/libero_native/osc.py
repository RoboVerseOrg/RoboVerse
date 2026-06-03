"""Self-contained OSC_POSE controller — NO ``libero`` and NO ``robosuite`` import.

Same operational-space control law as ``scripts/osc/osc_pose_controller.py`` but
with the robosuite ``control_utils`` / ``transform_utils`` math **inlined verbatim**
(all pure numpy), so a MetaSim-native LIBERO task depends only on ``mujoco`` +
``numpy``. Reads the dynamics (mass matrix, site Jacobian, EE pose/vel, bias
forces) from any MuJoCo ``model``/``data`` pair.

The inlined functions are copied from robosuite (Apache-2.0) so the control law
stays bitwise-identical to native LIBERO; verified against the robosuite-importing
port (per-step joint-torque Δ = 0).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

_EPS = np.finfo(float).eps * 4.0


# --- inlined robosuite transform_utils ------------------------------------- #
def _axisangle2quat(vec):
    angle = np.linalg.norm(vec)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def _quat2mat(quaternion):
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
    ])


# --- inlined robosuite control_utils --------------------------------------- #
def _orientation_error(desired, current):
    rc1, rc2, rc3 = current[:, 0], current[:, 1], current[:, 2]
    rd1, rd2, rd3 = desired[:, 0], desired[:, 1], desired[:, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def _opspace_matrices(mass_matrix, j_full, j_pos, j_ori):
    m_inv = np.linalg.inv(mass_matrix)
    lambda_full = np.linalg.pinv(np.dot(np.dot(j_full, m_inv), j_full.T))
    lambda_pos = np.linalg.pinv(np.dot(np.dot(j_pos, m_inv), j_pos.T))
    lambda_ori = np.linalg.pinv(np.dot(np.dot(j_ori, m_inv), j_ori.T))
    jbar = np.dot(m_inv, j_full.T).dot(lambda_full)
    nullspace = np.eye(j_full.shape[-1]) - np.dot(jbar, j_full)
    return lambda_full, lambda_pos, lambda_ori, nullspace


def _nullspace_torques(mass_matrix, nullspace, initial_joint, joint_pos, joint_vel, joint_kp=10):
    kv = np.sqrt(joint_kp) * 2
    pose_torques = np.dot(mass_matrix, joint_kp * (initial_joint - joint_pos) - kv * joint_vel)
    return np.dot(nullspace.T, pose_torques)


def _set_goal_position(delta, current):
    return current + delta


def _set_goal_orientation(delta, current_mat):
    return np.dot(_quat2mat(_axisangle2quat(delta)), current_mat)


class NativeOSCPose:
    """OSC_POSE over a MuJoCo (model, data); LIBERO config (kp=150, deltas, uncouple)."""

    def __init__(
        self,
        model,
        data,
        eef_site,
        arm_qvel_index,
        arm_act_index,
        *,
        initial_joint,
        kp=150.0,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        torque_limits=None,
    ):
        self.model, self.data = model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        self.qvi = np.asarray(arm_qvel_index, int)
        self.act = np.asarray(arm_act_index, int)
        self.kp = np.full(6, kp)
        self.kd = 2 * np.sqrt(self.kp)
        self.out = np.asarray(output_max)
        self.tlim = torque_limits
        self.initial_joint = np.asarray(initial_joint)
        self._update()
        self.goal_pos, self.goal_ori = self.ee_pos.copy(), self.ee_ori.copy()

    def _update(self):
        mujoco.mj_forward(self.model, self.data)
        d, m = self.data, self.model
        self.ee_pos = np.array(d.site_xpos[self.site_id])
        self.ee_ori = np.array(d.site_xmat[self.site_id]).reshape(3, 3)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
        self.ee_pv, self.ee_ov = jacp @ d.qvel, jacr @ d.qvel
        self.jp, self.jo = jacp[:, self.qvi], jacr[:, self.qvi]
        self.jf = np.vstack([self.jp, self.jo])
        self.jpos, self.jvel = np.array(d.qpos[self.qvi]), np.array(d.qvel[self.qvi])
        mfull = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, mfull, d.qM)
        self.M = mfull[np.ix_(self.qvi, self.qvi)]
        self.bias = np.array(d.qfrc_bias[self.qvi])

    def set_goal(self, action):
        """Capture the EE goal from a 6-D delta action (sticky orientation goal)."""
        self._update()
        scaled = np.clip(action, -1.0, 1.0) * self.out  # input ±1 -> output ±out
        if sum(0.0 if math.isclose(e, 0.0) else 1.0 for e in scaled[3:]) > 0.0:
            self.goal_ori = _set_goal_orientation(scaled[3:], self.ee_ori)
        self.goal_pos = _set_goal_position(scaled[:3], self.ee_pos)

    def run(self):
        """Compute the OSC joint torques for the current state toward the goal."""
        self._update()
        df = (self.goal_pos - self.ee_pos) * self.kp[:3] + (-self.ee_pv) * self.kd[:3]
        dt = _orientation_error(self.goal_ori, self.ee_ori) * self.kp[3:6] + (-self.ee_ov) * self.kd[3:6]
        lf, lp, lo, ns = _opspace_matrices(self.M, self.jf, self.jp, self.jo)
        tau = self.jf.T @ np.concatenate([lp @ df, lo @ dt]) + self.bias
        tau = tau + _nullspace_torques(self.M, ns, self.initial_joint, self.jpos, self.jvel)
        if self.tlim is not None:
            tau = np.clip(tau, self.tlim[0], self.tlim[1])
        return tau
