"""Action-replay determinism + ManiSkill demo replay in the shipped native tasks.

* ``test_action_replay_grasp_lift`` — replaying an identical grasp+lift action sequence in the shipped
  task vs native ManiSkill keeps the cube within ~1e-4 m through the contact-rich grasp+lift and both
  end grasped (so any successful action sequence replays deterministically; success threshold is
  0.025 m, ~250x the divergence).
* ``test_demo_replay_success`` — replaying official ManiSkill ``pd_joint_delta_pos`` demos through the
  shipped ``maniskill.pick_cube_native`` reproduces the demonstrated success on most episodes (the
  demos are GPU-generated, so open-loop replay on the CPU handler tracks native ``physx_cpu``, which
  itself reproduces ~24/25). Skips if the demo file isn't downloaded.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")

_DEMO = os.path.expanduser("~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5")


def test_action_replay_grasp_lift():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401
    from metasim.task.registry import get_task_class

    T = 40
    acts = np.zeros((T, 8), dtype=np.float32)
    acts[:, 7] = -1.0  # close gripper
    acts[12:, 1] = -0.25
    acts[12:, 3] = 0.25  # lift arm

    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu"
    )
    u = env.unwrapped
    env.reset(seed=0)
    tcp = u.agent.tcp_pose.p.cpu().numpy().ravel()
    sd = u.get_state_dict()
    cs = sd["actors"]["cube"].clone()
    cs[0, :3] = torch.tensor([float(x) for x in tcp])
    cs[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cs[0, 7:] = 0
    sd["actors"]["cube"] = cs
    u.set_state_dict(sd)
    q0 = u.agent.robot.get_qpos().cpu().numpy().ravel().astype(np.float32)
    qv0 = u.agent.robot.get_qvel().cpu().numpy().ravel().astype(np.float32)
    nat_cube, nat_grasp = [], []
    for a in acts:
        env.step(torch.tensor(a).unsqueeze(0))
        nat_cube.append(u.cube.pose.p.cpu().numpy().ravel().copy())
        nat_grasp.append(bool(u.agent.is_grasping(u.cube).item()))
    env.close()
    nat_cube = np.asarray(nat_cube)

    cls = get_task_class("maniskill.pick_cube_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset(seed=0)
    task.handler.object_ids["panda"].set_qpos(q0)
    task.handler.object_ids["panda"].set_qvel(qv0)
    task.handler.object_ids["cube"].set_pose(sapien.Pose([float(x) for x in tcp], [1.0, 0.0, 0.0, 0.0]))
    our_cube, our_grasp = [], []
    for a in acts:
        task.step(torch.tensor(a).unsqueeze(0))
        our_cube.append(task.obj_pos("cube").copy())
        our_grasp.append(task.is_grasped("cube"))
    task.close()

    delta = np.abs(nat_cube - np.asarray(our_cube)).max()
    assert delta < 1e-3, f"cube replay delta vs native through grasp+lift too large: {delta:.3e}"
    assert sum(int(a == b) for a, b in zip(nat_grasp, our_grasp)) >= T - 2
    assert nat_cube[-1][2] > 0.2 and np.asarray(our_cube)[-1][2] > 0.2  # both actually lifted


@pytest.mark.skipif(not os.path.exists(_DEMO), reason="ManiSkill PickCube demo not downloaded")
def test_demo_replay_success():
    from tools.maniskill_integration.replay_demo import replay

    r = replay("pick_cube", episodes=25)
    assert r["demo_success"] >= 24, r
    # Open-loop GPU demos on the CPU handler — native physx_cpu itself reproduces ~24/25; require most.
    assert r["replay_success"] >= 18, r
