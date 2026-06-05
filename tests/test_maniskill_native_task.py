"""End-to-end: the shipped ``maniskill.pick_cube_native`` task reproduces native ManiSkill.

Drives the task through the standard registry + ``BaseTaskEnv`` + sapien3 handler path with the
native ManiSkill ``pd_joint_delta_pos`` action contract, starting from the native env's reset state,
and asserts the cube pose tracks native ``PickCube-v1`` (physx_cpu) to PhysX float32 roundoff.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")


def test_pick_cube_native_tracks_maniskill():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers tasks
    from metasim.task.registry import get_task_class

    SEED, T = 0, 20
    rng = np.random.RandomState(1)

    # --- native reference cube trajectory ---
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=SEED)
    robot = u.agent.robot
    init_qpos = robot.get_qpos().cpu().numpy().ravel().astype(np.float32)
    init_qvel = robot.get_qvel().cpu().numpy().ravel().astype(np.float32)
    cube_raw = u.cube.pose.raw_pose.cpu().numpy().ravel()
    actions = rng.uniform(-1, 1, size=(T, 8)).astype(np.float32)
    ref = []
    for a in actions:
        env.step(torch.tensor(a).unsqueeze(0))
        ref.append(u.cube.pose.raw_pose.cpu().numpy().ravel().copy())
    env.close()
    ref = np.asarray(ref)

    # --- shipped native task through the handler ---
    cls = get_task_class("maniskill.pick_cube_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset()
    # Match the native reset state exactly.
    task.handler.object_ids["panda"].set_qpos(init_qpos)
    task.handler.object_ids["panda"].set_qvel(init_qvel)
    task.handler.object_ids["cube"].set_pose(sapien.Pose(cube_raw[:3], cube_raw[3:]))

    rep = []
    for a in actions:
        task.step(torch.tensor(a).unsqueeze(0))
        p = task.handler.object_ids["cube"].get_pose()
        rep.append(np.concatenate([p.p, p.q]))
    rep = np.asarray(rep)
    task.close()

    delta = np.abs(ref - rep).max()
    # PhysX float32 roundoff (residual is body-creation order through the handler); far below 1e-3.
    assert delta < 1e-3, f"cube pose delta vs native ManiSkill too large: {delta:.3e}"
    assert np.isfinite(delta)
