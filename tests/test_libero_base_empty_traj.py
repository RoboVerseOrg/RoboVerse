"""LiberoBaseTask must not crash on empty / missing trajectory data (H9).

The previous _get_initial_states unconditionally indexed and divided by
``len(initial_states)`` — empty trajectories produced ZeroDivisionError
(from ``num_envs // 0``) or IndexError (from ``initial_states[:0]``).
The reset() ordering was also flipped relative to checker init.

These tests stub out get_traj and BaseTaskEnv to test the
LiberoBaseTask methods in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from roboverse_pack.tasks.libero.libero_base import LiberoBaseTask


def _instance_without_init(num_envs: int, traj_filepath: str | None = "/tmp/x.pkl.gz"):
    """Build a LiberoBaseTask bound to minimum needed attrs without
    running its __init__ (which hits the HF downloader)."""
    obj = LiberoBaseTask.__new__(LiberoBaseTask)
    obj.scenario = SimpleNamespace(robots=[SimpleNamespace(name="franka")])
    obj.handler = MagicMock()
    obj.num_envs = num_envs
    obj.traj_filepath = traj_filepath
    obj.checker = MagicMock()
    return obj


@pytest.mark.general
def test_get_initial_states_returns_none_on_empty_trajectory(monkeypatch):
    monkeypatch.setattr(
        "roboverse_pack.tasks.libero.libero_base.get_traj",
        lambda *a, **k: ([], None, None),
    )
    task = _instance_without_init(num_envs=4)
    # Previously raised ZeroDivisionError on `num_envs // 0`.
    result = task._get_initial_states()
    assert result is None


@pytest.mark.general
def test_get_initial_states_returns_none_when_traj_missing(monkeypatch):
    def _raise_filenotfound(*a, **k):
        raise FileNotFoundError("traj.pkl.gz")

    monkeypatch.setattr("roboverse_pack.tasks.libero.libero_base.get_traj", _raise_filenotfound)
    task = _instance_without_init(num_envs=4)
    assert task._get_initial_states() is None


@pytest.mark.general
def test_get_initial_states_returns_none_when_traj_filepath_is_none(monkeypatch):
    monkeypatch.setattr(
        "roboverse_pack.tasks.libero.libero_base.get_traj",
        lambda *a, **k: ([], None, None),
    )
    task = _instance_without_init(num_envs=4, traj_filepath=None)
    assert task._get_initial_states() is None


@pytest.mark.general
def test_get_initial_states_replicates_to_match_num_envs(monkeypatch):
    """Fewer states than envs → replicate cleanly (no off-by-one)."""
    monkeypatch.setattr(
        "roboverse_pack.tasks.libero.libero_base.get_traj",
        lambda *a, **k: ([{"id": 0}, {"id": 1}, {"id": 2}], None, None),
    )
    task = _instance_without_init(num_envs=7)
    result = task._get_initial_states()
    assert result is not None
    assert len(result) == 7
    # 3 states × 2 + 1 remainder → ids should be 0,1,2,0,1,2,0
    assert [s["id"] for s in result] == [0, 1, 2, 0, 1, 2, 0]


@pytest.mark.general
def test_get_initial_states_truncates_when_more_states_than_envs(monkeypatch):
    monkeypatch.setattr(
        "roboverse_pack.tasks.libero.libero_base.get_traj",
        lambda *a, **k: ([{"id": i} for i in range(10)], None, None),
    )
    task = _instance_without_init(num_envs=3)
    result = task._get_initial_states()
    assert result is not None
    assert len(result) == 3


@pytest.mark.general
def test_reset_calls_checker_reset_before_super(monkeypatch):
    """checker.reset must run before super().reset so the latter's
    callbacks see fresh checker state."""
    task = _instance_without_init(num_envs=1)
    order = []

    def _checker_reset(*a, **k):
        order.append("checker")

    task.checker = MagicMock()
    task.checker.reset = _checker_reset

    # Patch super().reset to record ordering.
    from metasim.task.base import BaseTaskEnv

    def _super_reset(self, states=None, env_ids=None):
        order.append("super")
        return "sentinel"

    monkeypatch.setattr(BaseTaskEnv, "reset", _super_reset)

    result = task.reset(env_ids=[0])
    assert result == "sentinel"
    assert order == ["checker", "super"]


@pytest.mark.general
def test_reset_tolerates_missing_checker(monkeypatch):
    """If the subclass leaves ``checker = None`` (it does on the base
    class), reset() should still succeed."""
    task = _instance_without_init(num_envs=1)
    task.checker = None

    from metasim.task.base import BaseTaskEnv

    monkeypatch.setattr(BaseTaskEnv, "reset", lambda self, states=None, env_ids=None: "ok")

    assert task.reset(env_ids=[0]) == "ok"
