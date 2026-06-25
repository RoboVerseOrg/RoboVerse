"""Regression tests for two LIBERO content bugs (backend-free).

- scene3 "turn on the stove" termination did ``(stove_joint > thr).all(dim=-1)``
  on a ``(N,)`` tensor, collapsing the per-env result to a scalar (reducing over
  the batch). The sibling scene9 does it correctly.
- two pick tasks had registration-id typos (a misspelled alias and a primary id
  missing the ``pick_`` prefix that every sibling uses).
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from metasim.types import ObjectState, TensorState

_LIBERO = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "libero"


@pytest.mark.general
def test_stove_terminated_is_per_env():
    from roboverse_pack.tasks.libero_90.libero_kitchen_scene3_turn_on_the_stove import (
        LiberoKitchenScene3TurnOnStoveTask,
    )

    states = TensorState(
        objects={
            "flat_stove": ObjectState(
                root_state=torch.zeros(2, 13),
                joint_pos=torch.tensor([[0.6], [0.1]]),  # env0 on, env1 off
                joint_vel=torch.zeros(2, 1),
            )
        },
        robots={},
        cameras={},
    )
    task = LiberoKitchenScene3TurnOnStoveTask.__new__(LiberoKitchenScene3TurnOnStoveTask)
    out = task._terminated(states)
    assert out.shape == (2,), f"_terminated must be per-env (2,), got {tuple(out.shape)}"
    assert out.tolist() == [True, False]


@pytest.mark.general
def test_libero_pick_registration_ids_are_correct():
    soup = (_LIBERO / "libero_pick_alphabet_soup.py").read_text()
    assert "pick_alphnabet_soup" not in soup, "misspelled alias 'pick_alphnabet_soup'"
    assert '"libero.pick_alphabet_soup", "pick_alphabet_soup"' in soup

    juice = (_LIBERO / "libero_pick_orange_juice.py").read_text()
    assert '"libero.orange_juice"' not in juice, "primary id is missing the 'pick_' prefix"
    assert '"libero.pick_orange_juice", "pick_orange_juice"' in juice


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
