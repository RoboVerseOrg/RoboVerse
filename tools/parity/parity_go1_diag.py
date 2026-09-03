"""Go1 obs parity diagnostic: RoboVerse velocity_flat_go1_v2 vs mjlab native.

Step 1 (this script): dump joint orderings + base-proprioception term values
at a matched root state, so we can confirm alignment (joint order, base-frame
conventions) before asserting full obs 1:1.
"""

from __future__ import annotations

import numpy as np
import torch

DEVICE = "cuda:0"


def build_mjlab():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    cfg = unitree_go1_flat_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=DEVICE)
    env.reset()
    return env


def build_rv():
    import roboverse_pack  # noqa
    from metasim.task.registry import get_task_class

    env = get_task_class("mjlab.velocity_flat_go1_v2")()
    env.reset()
    return env


def main():
    mj = build_mjlab()
    rv = build_rv()

    ent = mj.scene["robot"]
    print("=== mjlab go1 joint names (entity order) ===")
    mj_joints = list(ent.joint_names)
    print(mj_joints)
    print("=== RoboVerse _GO1_JOINTS order ===")
    from roboverse_pack.tasks.mjlab.velocity_go1_v2 import _GO1_JOINTS

    print(list(_GO1_JOINTS.joint_names))
    print("joint order identical?:", mj_joints == list(_GO1_JOINTS.joint_names))

    # mjlab actor obs term order + dims
    print("\n=== mjlab actor obs terms ===")
    try:
        for name in mj.observation_manager._group_obs_term_names["actor"]:
            print("  ", name)
    except Exception as e:
        print("  (could not list:", e, ")")

    # Base-term comparison at a matched root state (upright, known velocity).
    # mjlab: write_root_state_to_sim([pos3, quat_wxyz4, linvel3, angvel3]) then forward.
    root = torch.tensor(
        [[0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.0, 0.0, 0.0, 0.3]], dtype=torch.float, device=DEVICE
    )
    try:
        ent.write_root_state_to_sim(root)
        mj.sim.forward() if hasattr(mj, "sim") and hasattr(mj.sim, "forward") else None
    except Exception as e:
        print("mjlab root write failed:", e)

    # mjlab base obs via its own funcs
    import mjlab.envs.mdp as mmdp

    from roboverse_pack.tasks.mjlab.mdp import observations as O
    from roboverse_pack.tasks.mjlab.velocity_go1_v2 import _GO1_TRUNK

    mj.observation_manager._obs_buffer = None
    print("\n=== base terms @ matched root (pos z=0.4, linvel=(0.7,0.2,0) world, yaw rate 0.3) ===")
    try:
        bl = mmdp.builtin_sensor(mj, "robot/imu_lin_vel")[0].detach().cpu().numpy()
        ba = mmdp.builtin_sensor(mj, "robot/imu_ang_vel")[0].detach().cpu().numpy()
        pg = mmdp.projected_gravity(mj)[0].detach().cpu().numpy()
        print("  mjlab base_lin_vel:", np.array2string(bl, precision=4))
        print("  mjlab base_ang_vel:", np.array2string(ba, precision=4))
        print("  mjlab proj_gravity:", np.array2string(pg, precision=4))
    except Exception as e:
        print("  mjlab base term read failed:", e)

    # RoboVerse: set physics root and forward, read its obs funcs
    ph = rv.handler.physics
    with ph.reset_context():
        ph.data.qpos[0:3] = (0.0, 0.0, 0.4)
        ph.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        ph.data.qvel[0:3] = (0.7, 0.2, 0.0)
        ph.data.qvel[3:6] = (0.0, 0.0, 0.3)
    st = rv.handler.get_states(mode="tensor")
    print("  rv    base_lin_vel:", np.array2string(O.base_lin_vel(rv, st, _GO1_TRUNK)[0].cpu().numpy(), precision=4))
    print("  rv    base_ang_vel:", np.array2string(O.base_ang_vel(rv, st, _GO1_TRUNK)[0].cpu().numpy(), precision=4))
    print(
        "  rv    proj_gravity:", np.array2string(O.projected_gravity(rv, st, _GO1_TRUNK)[0].cpu().numpy(), precision=4)
    )


if __name__ == "__main__":
    main()
