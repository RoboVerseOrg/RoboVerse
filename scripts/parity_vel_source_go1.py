"""Verify the obs/reward velocity-source split for go1 vs mjlab native.

mjlab uses DIFFERENT base-velocity sources for obs vs reward:
  - obs  ``base_lin_vel`` / ``base_ang_vel`` = IMU site sensors (velocimeter/gyro)
  - reward ``track_*_velocity``              = ``asset.data.root_link_{lin,ang}_vel_b`` (body origin)

The two differ by ``ω × r_imu_offset``. This harness injects K matched states
WITH nonzero angular velocity and checks both RoboVerse paths reproduce the
correct mjlab source:
  (A) RV obs velocity (base_lin_vel_imu_b)  ==  mjlab IMU sensor
  (B) RV reward velocity (base_lin_vel_b)   ==  mjlab root_link_lin_vel_b
  (C) |imu - body| is materially nonzero (the split actually matters)

Run: MJLAB_REPO=... PYTHONPATH=$(pwd) MUJOCO_GL=egl python scripts/parity_vel_source_go1.py
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.parity_obs_all import (  # reuse the verified injection helpers
    DEVICE,
    _go1_flat_builder,
    _sample_velocity,
    build_mjlab,
    build_rv,
    mjlab_inject,
    rv_inject,
)

K = 24
SEED = 0


def main() -> None:
    rng = np.random.default_rng(SEED)
    mj = build_mjlab(_go1_flat_builder)
    rv = build_rv("mjlab.velocity_flat_go1_v2")

    from roboverse_pack.tasks.mjlab.mdp import _math
    from roboverse_pack.tasks.mjlab.velocity_go1_v2 import _GO1_TRUNK

    ent = mj.scene["robot"]
    n_joints = 12

    a_obs = []  # RV imu obs vs mjlab imu sensor
    b_rew = []  # RV body reward-vel vs mjlab root_link_lin_vel_b
    c_diff = []  # imu vs body magnitude

    for _ in range(K):
        root, jpos, jvel, command, last_action = _sample_velocity(rng, n_joints)
        # force a non-trivial angular velocity so the imu offset term is active
        root = list(root)
        root[10:13] = [float(rng.uniform(-2, 2)) for _ in range(3)]  # world ang vel

        mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=command,
                     last_action=last_action, ent_name="robot")
        rv_inject(rv, root=root, jpos=jpos, jvel=jvel, command=command,
                  last_action=last_action, n_qpos_joint=n_joints, free_base=True)

        # --- mjlab sources ---
        import mjlab.envs.mdp as mmdp
        mj_imu_lin = mmdp.builtin_sensor(mj, "robot/imu_lin_vel")[0].detach().cpu().numpy()
        mj_imu_ang = mmdp.builtin_sensor(mj, "robot/imu_ang_vel")[0].detach().cpu().numpy()
        mj_body_lin = ent.data.root_link_lin_vel_b[0].detach().cpu().numpy()
        mj_body_ang = ent.data.root_link_ang_vel_b[0].detach().cpu().numpy()

        # --- RV sources ---
        st = rv.handler.get_states(mode="tensor")
        rv_imu_lin = _math.base_lin_vel_imu_b(rv, st, _GO1_TRUNK.name)[0].cpu().numpy()
        rv_imu_ang = _math.base_ang_vel_imu_b(rv, st, _GO1_TRUNK.name)[0].cpu().numpy()
        rv_body_lin = _math.base_lin_vel_b(rv, st, _GO1_TRUNK.name)[0].cpu().numpy()
        rv_body_ang = _math.base_ang_vel_b(rv, st, _GO1_TRUNK.name)[0].cpu().numpy()

        a_obs.append(max(np.abs(rv_imu_lin - mj_imu_lin).max(), np.abs(rv_imu_ang - mj_imu_ang).max()))
        b_rew.append(max(np.abs(rv_body_lin - mj_body_lin).max(), np.abs(rv_body_ang - mj_body_ang).max()))
        c_diff.append(max(np.abs(mj_imu_lin - mj_body_lin).max(), np.abs(mj_imu_ang - mj_body_ang).max()))

    a = float(np.max(a_obs)); b = float(np.max(b_rew)); c = float(np.max(c_diff))
    print(f"(A) max|RV imu_obs   - mjlab imu sensor      | = {a:.3e}   (obs path)")
    print(f"(B) max|RV body_rew  - mjlab root_link_vel_b  | = {b:.3e}   (reward path)")
    print(f"(C) max|mjlab imu    - mjlab body             | = {c:.3e}   (split magnitude)")
    ok = a < 1e-4 and b < 1e-4 and c > 1e-3
    print("VERDICT:", "PASS" if ok else "FAIL",
          "— obs uses IMU (==mjlab), reward uses body (==mjlab), and they differ" if ok
          else "— see deltas above")


if __name__ == "__main__":
    main()
