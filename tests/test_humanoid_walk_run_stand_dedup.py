"""``humanoid.{walk,run,stand}`` must be registered exactly once (M9).

Before the fix:

- ``metasim.example.example_pack.tasks.humanoid_env`` registered
  ``humanoid.walk`` / ``humanoid.run`` / ``humanoid.stand``.
- ``roboverse_pack.tasks.humanoid_bench.humanoid_env`` also defined
  ``WalkEnv`` / ``RunEnv`` / ``StandEnv`` with the ``@register_task``
  lines *commented out*. The bodies had drifted from the metasim copy —
  notably ``RunEnv.reward_functions`` was assigned as a bare callable
  rather than wrapped in a list, which mismatches the reward-loop
  contract.

If a user ever uncommented the second copy (or another module
re-imported the file with import-order surprises), ``register_task``
would raise ``ValueError("Task name 'humanoid.walk' is already
registered to WalkEnv")`` at import time, breaking the whole task
discovery pass.

After the fix:

- ``roboverse_pack.tasks.humanoid_bench.humanoid_env`` no longer defines
  its own ``WalkEnv``/``RunEnv``/``StandEnv``. It re-exports them from
  the metasim canonical module so any consumer of either path gets the
  same class.
- ``BaseLocomotionEnv`` / ``BaseLocomotionReward`` still live in the
  roboverse module because ``crawl``/``slide``/``stair``/``sit`` extend
  them.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.general
def test_walk_run_stand_come_from_single_module():
    """Both import paths resolve to the same class — no drift possible."""
    rv = importlib.import_module("roboverse_pack.tasks.humanoid_bench.humanoid_env")
    ms = importlib.import_module("metasim.example.example_pack.tasks.humanoid_env")

    assert rv.WalkEnv is ms.WalkEnv, "WalkEnv must be re-exported from metasim, not redefined"
    assert rv.RunEnv is ms.RunEnv, "RunEnv must be re-exported from metasim, not redefined"
    assert rv.StandEnv is ms.StandEnv, "StandEnv must be re-exported from metasim, not redefined"


@pytest.mark.general
def test_no_duplicate_humanoid_registrations():
    """Importing the humanoid_bench package must not raise on
    duplicate registration. Pre-fix, removing a single ``#`` from the
    roboverse copy would have made ``register_task`` raise — we want
    that footgun gone permanently, so we also assert via the registry
    that each name maps to exactly one class.
    """
    # Force the metasim example pack to load so its decorators run.
    importlib.import_module("metasim.example.example_pack.tasks.humanoid_env")
    # Now load the roboverse_pack humanoid_bench module which used to
    # be the second copy.
    importlib.import_module("roboverse_pack.tasks.humanoid_bench.humanoid_env")

    from metasim.task.registry import TASK_REGISTRY

    for name in ("humanoid.walk", "humanoid.run", "humanoid.stand"):
        assert name in TASK_REGISTRY, f"{name} must be registered"
    # The classes referenced under those names are the metasim ones.
    from metasim.example.example_pack.tasks.humanoid_env import RunEnv, StandEnv, WalkEnv

    assert TASK_REGISTRY["humanoid.walk"] is WalkEnv
    assert TASK_REGISTRY["humanoid.run"] is RunEnv
    assert TASK_REGISTRY["humanoid.stand"] is StandEnv


@pytest.mark.general
def test_base_locomotion_classes_stay_in_roboverse_pack():
    """``BaseLocomotionEnv``/``BaseLocomotionReward`` live in the
    roboverse module because crawl/slide/stair/sit extend them. The
    M9 fix must not remove these."""
    rv = importlib.import_module("roboverse_pack.tasks.humanoid_bench.humanoid_env")
    assert hasattr(rv, "BaseLocomotionEnv")
    assert hasattr(rv, "BaseLocomotionReward")


@pytest.mark.general
def test_run_env_reward_functions_is_a_list():
    """The dropped copy had ``self.reward_functions = BaseLocomotionReward(...)``
    (bare callable). Confirm the canonical implementation wraps it in a list.

    AST inspection rather than instantiation — this works without a
    physics backend.
    """
    import ast
    import inspect

    from metasim.example.example_pack.tasks.humanoid_env import RunEnv

    src = inspect.getsource(RunEnv)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "reward_functions":
                    assert isinstance(node.value, ast.List), (
                        "RunEnv.reward_functions must be a list of callables, not a bare callable"
                    )
                    return
    pytest.fail("RunEnv does not assign self.reward_functions in its __init__")
