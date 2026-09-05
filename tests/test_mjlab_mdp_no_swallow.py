"""The mjlab command manager and curriculum let a failing state / command read propagate.

Both used to catch ``Exception`` and return ``None``, which left the previous heading command in place
(a stale yaw-rate target fed the tracking rewards) or silently disabled the terrain curriculum.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.utils.math import euler_xyz_from_quat


def test_current_heading_is_the_yaw_and_a_failing_read_propagates():
    from roboverse_pack.tasks.mjlab.mdp.commands import VelocityCommandManager

    root = torch.tensor([[0.0, 0.0, 0.5, 0.7071068, 0.0, 0.0, 0.7071068] + [0.0] * 6])  # 90 deg about z
    states = SimpleNamespace(robots={"g1": SimpleNamespace(root_state=root)})
    mgr = VelocityCommandManager.__new__(VelocityCommandManager)
    mgr.env = SimpleNamespace(handler=SimpleNamespace(get_states=lambda mode="tensor": states))
    assert mgr._current_heading()[0] == pytest.approx(euler_xyz_from_quat(root[:, 3:7])[2][0])
    mgr.env = SimpleNamespace(
        handler=SimpleNamespace(get_states=lambda mode="tensor": (_ for _ in ()).throw(RuntimeError("worker died")))
    )
    with pytest.raises(RuntimeError, match="worker died"):
        mgr._current_heading()
    mgr.env = SimpleNamespace(handler=SimpleNamespace(get_states=lambda mode="tensor": SimpleNamespace(robots={})))
    assert mgr._current_heading() is None, "no robot in the scene is the one None left"


def test_terrain_curriculum_propagates_a_failing_command_read():
    from roboverse_pack.tasks.mjlab.mdp.curriculums import terrain_levels_vel

    class _Cmd:
        def current(self):
            raise RuntimeError("command manager broke")

    env = SimpleNamespace(terrain_manager=object(), command_managers={"base_velocity": _Cmd()}, handler=None)
    with pytest.raises(RuntimeError, match="command manager broke"):
        terrain_levels_vel(env, torch.tensor([0]), command_name="base_velocity")
