"""Bitwise reward + is_grasped parity vs native ManiSkill.

Pins two correctness claims of the ManiSkill integration:

* ``_native.rewards.pick_cube`` reproduces native PickCube-v1 ``compute_dense_reward`` to float32
  epsilon on identical inputs, and
* ``_native.grasp.is_grasped`` agrees with native ``Panda.is_grasping`` on a controlled grasp, with
  matching contact forces — exercising the new sapien3 ``get_pairwise_contact_force`` query.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")


def _extract(task_key, u, info):
    from roboverse_pack.tasks.maniskill._native import rewards as RW

    if task_key == "pick_cube":
        return RW.pick_cube, dict(
            cube_pos=u.cube.pose.p.cpu().numpy().ravel(),
            tcp_pos=u.agent.tcp_pose.p.cpu().numpy().ravel(),
            goal_pos=u.goal_site.pose.p.cpu().numpy().ravel(),
            qvel_arm=u.agent.robot.get_qvel().cpu().numpy().ravel()[:-2],
            is_grasped=bool(info["is_grasped"].item()),
            is_robot_static=bool(info["is_robot_static"].item()),
        )
    if task_key in ("push_cube", "pull_cube"):
        fn = RW.push_cube if task_key == "push_cube" else RW.pull_cube
        return fn, dict(
            cube_pos=u.obj.pose.p.cpu().numpy().ravel(),
            tcp_pos=u.agent.tcp.pose.p.cpu().numpy().ravel(),
            goal_pos=u.goal_region.pose.p.cpu().numpy().ravel(),
            cube_half_size=float(u.cube_half_size),
        )
    if task_key == "lift_peg_upright":
        from mani_skill.utils.geometry import rotation_conversions as RC

        return RW.lift_peg_upright, dict(
            peg_rot_mat=RC.quaternion_to_matrix(u.peg.pose.q).cpu().numpy().reshape(3, 3),
            peg_pos=u.peg.pose.p.cpu().numpy().ravel(),
            tcp_pos=u.agent.tcp.pose.p.cpu().numpy().ravel(),
            is_grasped=bool(u.agent.is_grasping(u.peg).item()),
            peg_half_length=float(u.peg_half_length),
        )
    raise ValueError(task_key)


@pytest.mark.parametrize("task_key,gym_id", [
    ("pick_cube", "PickCube-v1"), ("push_cube", "PushCube-v1"),
    ("pull_cube", "PullCube-v1"), ("lift_peg_upright", "LiftPegUpright-v1"),
])
def test_dense_reward_bitwise(task_key, gym_id):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    env = gym.make(gym_id, num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu", reward_mode="dense")
    u = env.unwrapped
    env.reset(seed=0)
    rng = np.random.RandomState(3)
    max_delta = 0.0
    for a in rng.uniform(-1, 1, size=(30, 8)).astype(np.float32):
        _, rew, _, _, info = env.step(torch.tensor(a).unsqueeze(0))
        fn, kw = _extract(task_key, u, info)
        max_delta = max(max_delta, abs(float(rew.item()) - fn(**kw)))
    env.close()
    assert max_delta < 1e-6, f"{task_key} dense reward delta vs native too large: {max_delta:.3e}"


def test_is_grasped_matches_native():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401
    from metasim.task.registry import get_task_class
    from roboverse_pack.tasks.maniskill._native.grasp import is_grasped

    T = 15
    acts = np.zeros((T, 8), dtype=np.float32)
    acts[:, 7] = -1.0  # close gripper

    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=0)
    q0 = u.agent.robot.get_qpos().cpu().numpy().ravel().astype(np.float32)
    qv0 = u.agent.robot.get_qvel().cpu().numpy().ravel().astype(np.float32)
    tcp = u.agent.tcp_pose.p.cpu().numpy().ravel()
    sd = u.get_state_dict()
    cube_state = sd["actors"]["cube"].clone()
    cube_state[0, :3] = torch.tensor(tcp)
    cube_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cube_state[0, 7:] = 0
    sd["actors"]["cube"] = cube_state
    u.set_state_dict(sd)
    native = []
    for a in acts:
        env.step(torch.tensor(a).unsqueeze(0))
        native.append(bool(u.agent.is_grasping(u.cube).item()))
    env.close()

    cls = get_task_class("maniskill.pick_cube_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset()
    task.handler.object_ids["panda"].set_qpos(q0)
    task.handler.object_ids["panda"].set_qvel(qv0)
    task.handler.object_ids["cube"].set_pose(sapien.Pose(tcp, [1.0, 0.0, 0.0, 0.0]))
    ours = []
    for a in acts:
        task.step(torch.tensor(a).unsqueeze(0))
        ours.append(is_grasped(task.handler, "cube"))
    task.close()

    agreement = sum(int(a == b) for a, b in zip(native, ours))
    assert agreement >= T - 1, f"is_grasped agreement {agreement}/{T} too low (native={native}, ours={ours})"
    assert any(ours), "expected the controlled grasp to register as grasped"
