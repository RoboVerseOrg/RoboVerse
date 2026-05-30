"""Backward-compat guard: gym adapter forwarding ``seed=N`` must not
break task subclasses that override ``reset()`` without a seed parameter.

Background: the 2026-05-26 reproducibility-wiring commit added
``seed=None`` to ``TaskBase.reset`` / ``RLTaskEnv.reset`` so
``env.reset(seed=42)`` could propagate to ``handler.set_seed``.
Downstream RoboVerse tasks (maniskill / libero / calvin / mjlab / etc.)
override ``reset(self, states=None, env_ids=None)`` — *without* the
``seed`` kwarg. Forwarding ``seed=`` to those tasks would TypeError on
unexpected keyword.

The gym adapter introspects each task's ``reset`` signature and only
passes ``seed=`` when the target accepts it. This test pins the
behaviour against:

  (a) tasks whose reset accepts ``seed`` — seed is forwarded
  (b) tasks whose reset has no seed — seed is silently dropped (only
      gym's base RNG is seeded, but the user gets no crash)
  (c) tasks whose reset accepts ``**kwargs`` — seed flows through

Pure introspection-level test — no GPU, no sim env.
"""

from __future__ import annotations

import pytest


class _TaskAcceptsSeed:
    def reset(self, states=None, env_ids=None, seed: int | None = None):
        self.last_seed = seed
        return "obs", {"info_with_seed": True}


class _TaskNoSeed:
    def reset(self, states=None, env_ids=None):
        self.last_seed = "unset"
        return "obs", {"info_no_seed": True}


class _TaskKwargs:
    def reset(self, states=None, env_ids=None, **kwargs):
        self.last_seed = kwargs.get("seed", "absent")
        return "obs", {"info_kwargs": True}


@pytest.mark.general
def test_accepts_seed_detection_positive():
    from metasim.task.gym_registration import _task_reset_accepts_seed

    assert _task_reset_accepts_seed(_TaskAcceptsSeed()) is True


@pytest.mark.general
def test_accepts_seed_detection_negative():
    from metasim.task.gym_registration import _task_reset_accepts_seed

    assert _task_reset_accepts_seed(_TaskNoSeed()) is False


@pytest.mark.general
def test_accepts_seed_detection_kwargs():
    """``**kwargs`` counts as accepting seed — the task is free to ignore it
    but won't crash on the forwarded keyword."""
    from metasim.task.gym_registration import _task_reset_accepts_seed

    assert _task_reset_accepts_seed(_TaskKwargs()) is True


@pytest.mark.general
def test_seed_forward_does_not_break_legacy_subclass():
    """Smoke check the exact scenario that motivated the fix: calling
    ``reset(seed=N)`` through a wrapper that mimics ``GymEnvAdapter``
    against a task that doesn't accept ``seed`` must not raise."""
    from metasim.task.gym_registration import _task_reset_accepts_seed

    task = _TaskNoSeed()
    # Mimic the gym adapter's branching logic.
    seed = 42
    if _task_reset_accepts_seed(task):
        task.reset(seed=seed)
    else:
        task.reset()  # backward-compat fallback
    assert task.last_seed == "unset"


@pytest.mark.general
def test_accepts_seed_detection_is_cached():
    """Introspection is cached per instance so reset() isn't paying the
    inspect.signature cost every call."""
    from metasim.task.gym_registration import _task_reset_accepts_seed

    task = _TaskAcceptsSeed()
    _task_reset_accepts_seed(task)
    assert getattr(task, "_reset_accepts_seed_cached", None) is True
    # Second call must return the same answer without re-inspecting.
    assert _task_reset_accepts_seed(task) is True
