"""End-to-end: every shipped ``maniskill.*_native`` task reproduces native ManiSkill.

For each task, drive the shipped task through the standard registry + ``BaseTaskEnv`` + sapien3
handler path with ManiSkill's native ``pd_joint_delta_pos`` action contract, starting from the native
env's reset state, and assert every object's pose tracks native (physx_cpu) to PhysX float32 roundoff.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")

from roboverse_pack.tasks.maniskill._native.specs import TASK_SPECS

_T = 15


def _native_reference(gym_id, names, seed):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    env = gym.make(gym_id, num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=seed)
    qpos = u.agent.robot.get_qpos().cpu().numpy().ravel().astype(np.float32)
    qvel = u.agent.robot.get_qvel().cpu().numpy().ravel().astype(np.float32)
    init = {n: u.scene.actors[n].pose.raw_pose.cpu().numpy().ravel().copy() for n in names}
    rng = np.random.RandomState(seed + 1)
    adim = env.action_space.shape[-1]  # 8 (panda) or 7 (panda_stick)
    actions = rng.uniform(-1, 1, size=(_T, adim)).astype(np.float32)
    traj = {n: [] for n in names}
    for a in actions:
        env.step(torch.tensor(a).unsqueeze(0))
        for n in names:
            traj[n].append(u.scene.actors[n].pose.raw_pose.cpu().numpy().ravel().copy())
    env.close()
    return qpos, qvel, init, actions, {n: np.asarray(v) for n, v in traj.items()}


@pytest.mark.parametrize("task_key", sorted(TASK_SPECS))
def test_native_task_tracks_maniskill(task_key):
    import sapien
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers tasks
    from metasim.task.registry import get_task_class

    spec = TASK_SPECS[task_key]
    names = [o[0] for o in spec["objects"]]
    qpos, qvel, init, actions, ref = _native_reference(spec["gym_id"], names, seed=0)

    cls = get_task_class(f"maniskill.{task_key}_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset()
    task.handler.object_ids["panda"].set_qpos(qpos)
    task.handler.object_ids["panda"].set_qvel(qvel)
    for n in names:
        task.handler.object_ids[n].set_pose(sapien.Pose(init[n][:3], init[n][3:]))

    rep = {n: [] for n in names}
    for a in actions:
        task.step(torch.tensor(a).unsqueeze(0))
        for n in names:
            p = task.handler.object_ids[n].get_pose()
            rep[n].append(np.concatenate([p.p, p.q]))
    task.close()

    delta = max(np.abs(ref[n] - np.asarray(rep[n])).max() for n in names)
    # PhysX float32 roundoff through the handler — measured ≤ 5e-6 across the suite.
    assert delta < 1e-3, f"{task_key}: object pose delta vs native ManiSkill too large: {delta:.3e}"
    assert np.isfinite(delta)
