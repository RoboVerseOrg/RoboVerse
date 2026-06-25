"""Regression tests (backend-free) for the libero task fixes.

Covers two independent hardening fixes:

* ``Libero90BaseTask._get_initial_states`` must degrade to None (handler
  defaults) when the trajectory is missing/empty or the scenario is robotless,
  matching the already-hardened ``LiberoBaseTask``. Previously it indexed
  ``scenario.robots[0]`` and called get_traj with no guard, so those cases
  crashed.
* Every ``libero_pick_*`` task must declare ``robots=["franka"]``. 9 of 10
  omitted it, so they instantiated with an empty robot list — a robotless pick
  task can never succeed and eval scores a silent 0%.
"""

from __future__ import annotations

import pathlib

import pytest

_LIBERO = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "libero"


@pytest.mark.general
def test_libero_90_get_initial_states_degrades_gracefully(monkeypatch):
    """Libero90BaseTask._get_initial_states must degrade to None (handler
    defaults) when the traj is missing/empty or the scenario is robotless,
    matching the hardened LiberoBaseTask. Previously it indexed
    ``scenario.robots[0]`` and called get_traj with no guard → crashed.
    """
    from types import SimpleNamespace

    import roboverse_pack.tasks.libero_90.libero_90_base as mod
    from roboverse_pack.tasks.libero_90.libero_90_base import Libero90BaseTask

    t = Libero90BaseTask.__new__(Libero90BaseTask)
    t.scenario = SimpleNamespace(robots=[])  # robotless
    t.num_envs = 1
    t.handler = None

    # 1. No traj_filepath → None (new guard; previously len(None) / robots[0] crash)
    t.traj_filepath = None
    assert t._get_initial_states() is None

    # 2. get_traj raises (missing file / bad key) → None (try/except)
    t.traj_filepath = "does/not/exist.pkl.gz"

    def _raise(*a, **k):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(mod, "get_traj", _raise)
    assert t._get_initial_states() is None

    # 3. get_traj returns empty → None (empty guard before len())
    monkeypatch.setattr(mod, "get_traj", lambda *a, **k: ([], None, None))
    assert t._get_initial_states() is None


@pytest.mark.general
def test_all_libero_pick_tasks_have_a_robot():
    """Every ``libero_pick_*`` task is a "pick up X and place in basket" task and
    therefore MUST declare a robot arm. Previously 9 of 10 omitted
    ``robots=["franka"]`` (only butter had it), so the scenario instantiated with
    an empty robot list — a robotless pick task can never succeed and eval scores
    a silent 0%. This pins every sibling to declare a robot.
    """
    import importlib

    pick_files = sorted(_LIBERO.glob("libero_pick_*.py"))
    assert pick_files, "no libero_pick_* task files found"
    missing = []
    for f in pick_files:
        mod = importlib.import_module(f"roboverse_pack.tasks.libero.{f.stem}")
        # the task class defines a class-level ScenarioCfg
        for obj in vars(mod).values():
            scen = getattr(obj, "scenario", None)
            if scen is not None and hasattr(scen, "robots"):
                if not scen.robots:
                    missing.append(f.stem)
                break
    assert not missing, f"libero_pick tasks with no robot (robotless pick can never succeed): {missing}"
