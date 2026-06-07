"""Action-level 1:1 parity across all ManiSkill robot action layouts.

Beyond the standard panda (7 arm + 1 mimic gripper), this pins the two other layouts:

* **arm-only** (``panda_stick``: PushT / DrawTriangle, 7-dim) via ``parity_native.run_task`` — the
  generalized controller auto-detects no gripper; object pose tracks native bit-for-bit / ~1e-5.
* **multi-agent** (``TwoRobotPickCube``: 2 x 8-dim) via ``parity_multi_agent.run`` — per-robot action
  slices replayed in one rebuilt scene; object pose Δ = 0.

Heavy (imports mani_skill + SAPIEN); skips cleanly when unavailable. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")


@pytest.mark.parametrize("gym_id,obj", [("PushT-v1", "Tee"), ("DrawTriangle-v1", "canvas")])
def test_arm_only_action_level(gym_id, obj):
    from tools.maniskill_integration.parity_native import run_task

    r = run_task(gym_id, steps=15, seed=0)
    assert r["action_dim"] == 7, r  # panda_stick: arm-only, no gripper
    assert obj in r["objects"], r
    assert r["objects"][obj]["delta_max"] < 1e-4, r


def test_multi_agent_action_level():
    from tools.maniskill_integration.parity_multi_agent import run

    r = run("TwoRobotPickCube-v1", steps=15, seed=0, obj_name="cube")
    assert r["n_robots"] == 2 and r["per_robot_action"] == 8, r
    assert r["obj_pose_delta_max"] < 1e-4, r


def test_two_robot_shipped_task():
    """The shipped multi-robot task runs the 16-dim action through the standard handler path."""
    import copy

    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401
    from metasim.task.registry import get_task_class

    cls = get_task_class("maniskill.two_robot_pick_cube_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset(seed=0)
    assert {"panda-0", "panda-1"}.issubset(task.handler.object_ids)
    out = task.step(torch.zeros((1, 16)))
    assert out[1].numel() == 1 and out[2].dtype == torch.bool
    task.close()
