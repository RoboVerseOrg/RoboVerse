"""LiberoBaseTask must not crash on empty / missing trajectory data (H9).

The previous _get_initial_states unconditionally indexed and divided by
``len(initial_states)`` — empty trajectories produced ZeroDivisionError
(from ``num_envs // 0``) or IndexError (from ``initial_states[:0]``).
The reset() ordering was also flipped relative to checker init.

These tests stub out get_traj and BaseTaskEnv to test the
LiberoBaseTask methods in isolation. Because the BaseTaskEnv.reset stubs
below must mirror core's real signature, ``test_reset_signature_matches_base_task_env``
pins it — a stub that drifts out of sync stops checking anything.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from metasim.task.base import BaseTaskEnv
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
def test_reset_signature_matches_base_task_env():
    """``LiberoBaseTask.reset`` must keep accepting exactly what ``BaseTaskEnv.reset`` does.

    The reset-ordering tests below monkeypatch ``BaseTaskEnv.reset`` with a stub. When core
    grew a ``seed`` parameter, ``LiberoBaseTask.reset`` was updated to forward it but the
    stubs were not, so they silently went stale and the ordering guarantee stopped being
    checked. Pin the signature at the seam so that divergence fails loudly and points here.
    """
    base_params = list(inspect.signature(BaseTaskEnv.reset).parameters)
    assert base_params == ["self", "states", "env_ids", "seed"]
    assert list(inspect.signature(LiberoBaseTask.reset).parameters) == base_params


@pytest.mark.general
def test_reset_resets_the_checker_after_the_new_states_are_written():
    """The bbox detector places its region from the base object's pose at reset, so the checker
    must reset after ``set_states`` (it read the previous episode's pose when it ran first), and
    ``reset`` must still take the base signature (``states``, ``env_ids``, ``seed``)."""
    import torch

    task = _instance_without_init(num_envs=1)
    order = []
    task.handler.set_states.side_effect = lambda **k: order.append(("set_states", k["env_ids"]))
    task.handler.set_states_refreshes = True
    task.handler.set_seed.side_effect = lambda seed: order.append(("seed", seed))
    task.checker.reset.side_effect = lambda handler, env_ids: order.append(("checker", env_ids))
    task._episode_steps = torch.zeros(1, dtype=torch.int32)
    task._initial_states = "initial"
    task._prepare_callbacks()

    task.reset(env_ids=[0], seed=7)
    assert order == [("seed", 7), ("set_states", [0]), ("checker", [0])]


@pytest.mark.general
def test_reset_tolerates_missing_checker(monkeypatch):
    """If the subclass leaves ``checker = None`` (it does on the base
    class), reset() should still succeed."""
    task = _instance_without_init(num_envs=1)
    task.checker = None
    task._on_reset([0])  # the hook both reset paths call; a None checker is skipped
