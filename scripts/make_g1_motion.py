"""Synthesize a schema-valid G1 motion npz for tracking obs-parity testing.

mjlab's tracking ``MotionLoader`` (``tasks/tracking/mdp/commands.py``) loads an
``np.load``-able archive with these arrays (and nothing else is read):

  joint_pos      (T, 29)      float  — per-frame actuated joint positions
  joint_vel      (T, 29)      float  — per-frame actuated joint velocities
  body_pos_w     (T, 30, 3)   float  — per-frame world position of EVERY robot body
  body_quat_w    (T, 30, 4)   float  — per-frame world orientation (wxyz) of EVERY body
  body_lin_vel_w (T, 30, 3)   float  — per-frame world linear velocity of EVERY body
  body_ang_vel_w (T, 30, 3)   float  — per-frame world angular velocity of EVERY body

The body axis spans the robot's FULL body list (30 for G1) because the loader
slices it with ``body_indexes = robot.find_bodies(cfg.body_names)`` AFTER load
(indices into the full body list). There is no ``fps`` key — the loader plays one
frame per env control step.

For OBS PARITY the motion *content* is irrelevant; both sides only need to load
the SAME schema-valid file. We synthesize a short, smooth, deterministic clip:
the G1 default standing pose plus tiny per-joint sinusoidal motion, with body
kinematics derived by MuJoCo forward kinematics (mj_kinematics / mj_comVel) so
the per-body pos/quat/vel arrays are mutually consistent (this also makes the
reward terms well-behaved if anyone reuses the clip).

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 python scripts/make_g1_motion.py [out.npz]
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from roboverse_pack.tasks.mjlab._locator import mjlab_asset
from roboverse_pack.tasks.mjlab.velocity_g1_v2 import (
    _G1_ALL_JOINT_NAMES,
    _G1_DEFAULT_POSE_NP,
    _G1_XML,
)

OUT_DEFAULT = "/tmp/g1_synth_motion.npz"

T = 120  # number of frames (~2.4 s at 50 Hz control)
FPS = 50.0
DT = 1.0 / FPS
BASE_HEIGHT = 0.78
AMP = 0.05  # joint sinusoid amplitude (rad)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT

    model = mujoco.MjModel.from_xml_path(mjlab_asset(_G1_XML))
    data = mujoco.MjData(model)

    n_joints = len(_G1_ALL_JOINT_NAMES)
    nbody = model.nbody - 1  # drop the MuJoCo worldbody (index 0)
    assert n_joints == 29, n_joints

    # Map actuated-joint name -> qpos / qvel addresses (free joint occupies 0:7 / 0:6).
    jpos_adr = np.zeros(n_joints, dtype=int)
    jvel_adr = np.zeros(n_joints, dtype=int)
    for i, name in enumerate(_G1_ALL_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jpos_adr[i] = model.jnt_qposadr[jid]
        jvel_adr[i] = model.jnt_dofadr[jid]

    joint_pos = np.zeros((T, n_joints), dtype=np.float32)
    joint_vel = np.zeros((T, n_joints), dtype=np.float32)
    body_pos_w = np.zeros((T, nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((T, nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((T, nbody, 3), dtype=np.float32)

    # Smooth per-joint phase offsets (deterministic).
    phase = np.linspace(0.0, 2.0 * np.pi, n_joints, endpoint=False)

    for t in range(T):
        ang = 2.0 * np.pi * t / T
        q = _G1_DEFAULT_POSE_NP + AMP * np.sin(ang + phase)
        qd = AMP * (2.0 * np.pi / T / DT) * np.cos(ang + phase)

        data.qpos[:] = 0.0
        data.qpos[0:3] = (0.0, 0.0, BASE_HEIGHT)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
        data.qpos[jpos_adr] = q
        data.qvel[:] = 0.0
        data.qvel[jvel_adr] = qd

        mujoco.mj_forward(model, data)

        joint_pos[t] = q.astype(np.float32)
        joint_vel[t] = qd.astype(np.float32)

        # Drop worldbody (index 0): bodies 1..nbody map to robot bodies 0..nbody-1.
        body_pos_w[t] = data.xpos[1:].astype(np.float32)
        body_quat_w[t] = data.xquat[1:].astype(np.float32)  # wxyz

        # World-frame body velocities at the body origin via mj_objectVelocity.
        for b in range(1, model.nbody):
            v6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            # mj_objectVelocity (flg_local=0) returns [ang(3), lin(3)] in world frame.
            body_ang_vel_w[t, b - 1] = v6[:3].astype(np.float32)
            body_lin_vel_w[t, b - 1] = v6[3:].astype(np.float32)

    np.savez(
        out,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        fps=np.float32(FPS),  # harmless extra; loader ignores it.
    )
    print(f"wrote {out}: T={T}, n_joints={n_joints}, nbody={nbody}")
    print(
        "  arrays: joint_pos%s joint_vel%s body_pos_w%s body_quat_w%s "
        "body_lin_vel_w%s body_ang_vel_w%s"
        % (
            joint_pos.shape,
            joint_vel.shape,
            body_pos_w.shape,
            body_quat_w.shape,
            body_lin_vel_w.shape,
            body_ang_vel_w.shape,
        )
    )


if __name__ == "__main__":
    main()
