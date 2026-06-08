"""Reset distribution: per-episode variety, seed reproducibility, native range match.

Regression for the bug where the goal/reset sampler re-seeded ``RandomState(0)`` every reset (all
episodes identical). Consecutive resets must differ; ``reset(seed=N)`` must be reproducible; and the
spawn distribution must match ManiSkill's ranges (cube XY in [-0.1, 0.1], goal Z in [0.02, 0.32]).

Heavy (SAPIEN); skips without it. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("sapien")


def _task():
    import roboverse_pack.tasks.maniskill  # noqa: F401
    from metasim.task.registry import get_task_class

    cls = get_task_class("maniskill.pick_cube_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    return cls(sc)


def test_reset_variety_and_reproducibility():
    t = _task()
    goals = []
    for _ in range(3):
        t.reset()
        goals.append(t.goal_pos.copy())
    # consecutive resets differ (no frozen episode)
    assert not np.allclose(goals[0], goals[1]) and not np.allclose(goals[1], goals[2])
    # explicit seed is reproducible
    t.reset(seed=7)
    g1 = t.goal_pos.copy()
    t.reset(seed=7)
    assert np.allclose(g1, t.goal_pos)
    t.close()


def test_reset_distribution_matches_native_ranges():
    t = _task()
    cube, goal = [], []
    for i in range(120):
        t.reset(seed=i)
        cube.append(t.obj_pos("cube").copy())
        goal.append(t.goal_pos.copy())
    t.close()
    cube, goal = np.asarray(cube), np.asarray(goal)
    # ManiSkill PickCube: cube XY in U(+-0.1); goal Z = U(0,0.3) + cube_z(0.02)
    assert cube[:, 0].min() >= -0.1 - 1e-3 and cube[:, 0].max() <= 0.1 + 1e-3
    assert cube[:, 1].min() >= -0.1 - 1e-3 and cube[:, 1].max() <= 0.1 + 1e-3
    assert 0.02 - 1e-3 <= goal[:, 2].min() and goal[:, 2].max() <= 0.32 + 1e-3
    # actually spans most of the range (not degenerate)
    assert goal[:, 2].max() - goal[:, 2].min() > 0.2
