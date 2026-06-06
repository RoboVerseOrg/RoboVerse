"""Bitwise success-predicate parity vs native ManiSkill ``evaluate()``.

Two checks per the verification done by hand:

* **Negative agreement** — over a random rollout, the ported ``_native.success`` predicate agrees with
  native ``info["success"]`` at every step (no false positives), exercising the geometric thresholds.
* **Positive firing** — forcing a success state (object teleported onto the goal, robot static) makes
  both native and the port return True.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")

from roboverse_pack.tasks.maniskill._native import success as SU


def _native_success(info):
    return bool(info["success"].item())


@pytest.mark.parametrize(
    "gym_id,fn",
    [
        (
            "PushCube-v1",
            lambda u, i: SU.push_cube(
                cube_pos=u.obj.pose.p.cpu().numpy().ravel(),
                goal_pos=u.goal_region.pose.p.cpu().numpy().ravel(),
                cube_half_size=float(u.cube_half_size),
            ),
        ),
        (
            "PullCube-v1",
            lambda u, i: SU.pull_cube(
                cube_pos=u.obj.pose.p.cpu().numpy().ravel(), goal_pos=u.goal_region.pose.p.cpu().numpy().ravel()
            ),
        ),
        (
            "RollBall-v1",
            lambda u, i: SU.roll_ball(
                ball_pos=u.ball.pose.p.cpu().numpy().ravel(), goal_pos=u.goal_region.pose.p.cpu().numpy().ravel()
            ),
        ),
        (
            "StackCube-v1",
            lambda u, i: SU.stack_cube(
                cubeA_pos=u.cubeA.pose.p.cpu().numpy().ravel(),
                cubeB_pos=u.cubeB.pose.p.cpu().numpy().ravel(),
                cube_half_size=u.cube_half_size.cpu().numpy().ravel(),
                is_cubeA_static=bool(u.cubeA.is_static(1e-2, 0.5).item()),
                is_cubeA_grasped=bool(i["is_cubeA_grasped"].item()),
            ),
        ),
    ],
)
def test_success_negative_agreement(gym_id, fn):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    env = gym.make(gym_id, num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=0)
    rng = np.random.RandomState(2)
    for a in rng.uniform(-1, 1, size=(60, 8)).astype(np.float32):
        _, _, _, _, info = env.step(torch.tensor(a).unsqueeze(0))
        assert fn(u, info) == _native_success(info)
    env.close()


def test_success_positive_firing():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    env = gym.make(
        "PushCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu"
    )
    u = env.unwrapped
    env.reset(seed=0)
    goal = u.goal_region.pose.p.cpu().numpy().ravel()
    sd = u.get_state_dict()
    cs = sd["actors"]["cube"].clone()
    cs[0, :3] = torch.tensor([float(goal[0]), float(goal[1]), 0.02])
    cs[0, 7:] = 0
    sd["actors"]["cube"] = cs
    u.set_state_dict(sd)
    info = u.evaluate()
    ours = SU.push_cube(
        cube_pos=u.obj.pose.p.cpu().numpy().ravel(), goal_pos=goal, cube_half_size=float(u.cube_half_size)
    )
    env.close()
    assert _native_success(info) is True and ours is True
