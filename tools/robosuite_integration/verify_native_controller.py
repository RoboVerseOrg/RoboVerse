"""End-to-end check: the NATIVE OSC controller completes the benchmark tasks.

Drives MetaSim's MujocoHandler with the native, robosuite-free controller
(:class:`roboverse_pack.tasks.robosuite._osc.NativeOSC`) over the official
robomimic Lift benchmark *actions*, and judges success with a native check
(cube lifted above the table) — no robosuite controller and no robosuite
``_check_success`` in the rollout loop.

robosuite is used only to (a) emit the compiled MJCF and the controller config
(initial rest pose, base frame) once per demo, and (b) cross-check success. Both
are static and would be vendored to drop the robosuite package entirely.

Usage:
    python -m tools.robosuite_integration.verify_native_controller --n-demos 20
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def _native_lift_success(model, data, cube_body: str = "cube_main", table_top_z: float = 0.8) -> bool:
    """robosuite Lift success, ported: cube center > table top + 0.04 m."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body)
    return bool(data.xpos[bid][2] > table_top_z + 0.04)


def run(n_demos: int) -> dict:
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.scenario.simulator_params import SimParamCfg

    from roboverse_pack.tasks.robosuite._osc import NativeOSC
    from roboverse_pack.tasks.robosuite.robosuite_env import _GroundlessMujocoHandler

    from .inventory import get
    from .robosuite_rollout import make_robosuite_env

    f = h5py.File("datasets/lift_ph_low_dim_v141.hdf5", "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[:n_demos]

    native_succ = 0
    robo_succ = 0
    for dm in demos:
        states = f["data"][dm]["states"][()]
        actions = f["data"][dm]["actions"][()]

        env = make_robosuite_env(get("Lift"))
        env.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        c = env.robots[0].composite_controller.part_controllers["right"]
        xml = env.model.get_xml()
        qpos0, qvel0 = env.sim.data._data.qpos.copy(), env.sim.data._data.qvel.copy()
        cfg = (list(c.qpos_index), list(c.qvel_index), np.array(c.initial_joint), np.array(c.origin_pos), np.array(c.origin_ori))
        # robosuite reference success (action-replay through robosuite)
        for a in actions:
            env.step(a)
        robo_succ += int(env._check_success())
        env.close()

        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        tmp.write(xml)
        tmp.flush()
        sc = ScenarioCfg(
            scene=SceneCfg(mjcf_path=tmp.name), robots=[], lights=[], ground=None,
            sim_params=SimParamCfg(dt=0.002), decimation=25, simulator="mujoco", num_envs=1, headless=True,
        )
        h = _GroundlessMujocoHandler(sc)
        h.launch()
        m, d = h.physics.model.ptr, h.physics.data.ptr
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        arm_act = [aid(f"robot0_torq_j{i}") for i in range(1, 8)]
        grip = [aid("gripper0_right_gripper_finger_joint1"), aid("gripper0_right_gripper_finger_joint2")]
        with h.physics.reset_context():
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
        osc = NativeOSC(
            m, d, eef_site="gripper0_right_grip_site", arm_joint_qpos=cfg[0], arm_joint_qvel=cfg[1],
            arm_actuator_ids=arm_act, gripper_actuator_ids=grip, initial_joint=cfg[2], base_pos=cfg[3], base_ori=cfg[4],
        )
        for a in actions:
            osc.apply(a)
            for _ in range(25):
                osc.write_ctrl()
                mujoco.mj_step(m, d)
        native_succ += int(_native_lift_success(m, d))
    f.close()

    out = {
        "n_demos": len(demos),
        "native_controller_success": native_succ,
        "robosuite_controller_success": robo_succ,
        "native_success_rate": native_succ / len(demos),
        "robosuite_success_rate": robo_succ / len(demos),
    }
    print(
        f"native OSC controller (robosuite-free) on {len(demos)} Lift benchmark demos: "
        f"{native_succ}/{len(demos)} success ({out['native_success_rate']:.0%}) | "
        f"robosuite controller: {robo_succ}/{len(demos)} ({out['robosuite_success_rate']:.0%})"
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=20)
    run(ap.parse_args().n_demos)
