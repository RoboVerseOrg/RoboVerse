"""A MuJoCo OSC_POSE controller that reads MetaSim's dm_control physics.

This is the missing piece that lets a LIBERO-style EE-delta policy run *inside*
MetaSim (not just the robosuite passthrough): Operational Space Control maps the
7-D action ``[Δx,Δy,Δz, Δrx,Δry,Δrz, gripper]`` to joint torques using the
robot's own dynamics, exactly as robosuite does.

Faithfulness strategy: the *math* (operational-space inertia, nullspace,
orientation error, action scaling, goal setting) is reused **verbatim** from
robosuite's own ``control_utils`` / ``transform_utils`` -- we only reimplement
the sim-data access (mass matrix, Jacobian, EE pose/vel, bias forces) on top of
MetaSim's dm_control ``Physics`` instead of robosuite's ``MjSim``. So if the
per-substep torques match, the port is faithful by construction.

Control loop to mirror robosuite (single_arm.control + environments.base.step):
    set_goal(action)                      # once per policy step (captures EE pose)
    for _ in range(substeps):             # control_timestep / model_timestep
        ctrl[arm] = run_controller()      # recompute torques each substep
        mj_step(model, data)

The controller is sim-agnostic in spirit: it only needs (mass matrix, site
Jacobian, site pose/vel, bias forces, joint pos/vel) -- the exact interface a
backend-agnostic MetaSim OSC term would expose.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Verbatim robosuite math -- identical numerics, no reimplementation risk.
from robosuite.utils.control_utils import (
    nullspace_torques,
    opspace_matrices,
    orientation_error,
    set_goal_orientation,
    set_goal_position,
)
from robosuite.utils.transform_utils import axisangle2quat, quat2mat  # noqa: F401  (used via control_utils)


class MujocoOSCPose:
    """OSC_POSE over a MetaSim dm_control ``Physics`` (or any handler.physics).

    Mirrors robosuite ``OperationalSpaceController`` with LIBERO's config:
    fixed impedance, control_delta=True, uncouple_pos_ori=True, no interpolation.
    """

    def __init__(
        self,
        physics,
        eef_site: str,
        arm_joint_qvel_index,
        arm_actuator_index,
        *,
        kp: float = 150.0,
        damping_ratio: float = 1.0,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        input_max=1.0,
        input_min=-1.0,
        torque_limits=None,
        initial_joint=None,
    ):
        self.physics = physics
        self.model = physics.model.ptr
        self.data = physics.data.ptr
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        self.qvel_index = np.asarray(arm_joint_qvel_index, dtype=int)
        self.qpos_index = self.qvel_index  # LIBERO arm: qpos and qvel indices coincide
        self.act_index = np.asarray(arm_actuator_index, dtype=int)

        self.kp = np.full(6, kp, dtype=float)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        self.output_max = np.asarray(output_max, dtype=float)
        self.output_min = np.asarray(output_min, dtype=float)
        self.input_max = np.full(6, input_max, dtype=float)
        self.input_min = np.full(6, input_min, dtype=float)
        self._scale = np.abs(self.output_max - self.output_min) / np.abs(self.input_max - self.input_min)
        self._out_t = (self.output_max + self.output_min) / 2.0
        self._in_t = (self.input_max + self.input_min) / 2.0
        self.torque_limits = torque_limits

        self._update()
        # robosuite initializes the orientation goal to the current EE orientation
        # (reset_goal); pos goal starts at current EE pos.
        self.goal_pos = self.ee_pos.copy()
        self.goal_ori = self.ee_ori_mat.copy()
        self.initial_joint = np.array(initial_joint) if initial_joint is not None else self.joint_pos.copy()

    # --- sim-data access (the only sim-specific part) -------------------------
    def _update(self):
        mujoco.mj_forward(self.model, self.data)
        d, m = self.data, self.model
        self.ee_pos = np.array(d.site_xpos[self.site_id])
        self.ee_ori_mat = np.array(d.site_xmat[self.site_id]).reshape(3, 3)

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
        self.ee_pos_vel = jacp @ d.qvel  # == robosuite get_site_xvelp
        self.ee_ori_vel = jacr @ d.qvel  # == robosuite get_site_xvelr
        self.J_pos = jacp[:, self.qvel_index]
        self.J_ori = jacr[:, self.qvel_index]
        self.J_full = np.vstack([self.J_pos, self.J_ori])

        self.joint_pos = np.array(d.qpos[self.qpos_index])
        self.joint_vel = np.array(d.qvel[self.qvel_index])

        M_full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M_full, d.qM)
        self.mass_matrix = M_full[np.ix_(self.qvel_index, self.qvel_index)]
        self.torque_compensation = np.array(d.qfrc_bias[self.qvel_index])

    def _scale_action(self, action):
        action = np.clip(action, self.input_min, self.input_max)
        return (action - self._in_t) * self._scale + self._out_t

    # --- robosuite-identical control logic ------------------------------------
    def set_goal(self, action):
        import math

        self._update()
        scaled = self._scale_action(np.asarray(action, dtype=float))
        # IMPORTANT (matches robosuite): only update the orientation goal when the
        # ori delta is non-zero, otherwise keep the previous (sticky) goal_ori.
        # Always update the position goal.
        if sum(0.0 if math.isclose(e, 0.0) else 1.0 for e in scaled[3:]) > 0.0:
            self.goal_ori = set_goal_orientation(scaled[3:], self.ee_ori_mat, set_ori=None)
        self.goal_pos = set_goal_position(scaled[:3], self.ee_pos, set_pos=None)

    def run_controller(self):
        self._update()
        desired_pos = np.array(self.goal_pos)
        ori_err = orientation_error(np.array(self.goal_ori), self.ee_ori_mat)

        pos_err = desired_pos - self.ee_pos
        desired_force = pos_err * self.kp[0:3] + (-self.ee_pos_vel) * self.kd[0:3]
        desired_torque = ori_err * self.kp[3:6] + (-self.ee_ori_vel) * self.kd[3:6]

        lam_full, lam_pos, lam_ori, nullspace = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
        # uncouple_pos_ori = True
        decoupled = np.concatenate([lam_pos @ desired_force, lam_ori @ desired_torque])
        torques = self.J_full.T @ decoupled + self.torque_compensation
        torques = torques + nullspace_torques(
            self.mass_matrix, nullspace, self.initial_joint, self.joint_pos, self.joint_vel
        )
        if self.torque_limits is not None:
            torques = np.clip(torques, self.torque_limits[0], self.torque_limits[1])
        return torques

    def apply(self, torques):
        self.data.ctrl[self.act_index] = torques
