"""``ScenarioCfg.update(**kwargs)` surfaces unknown field names.

``update`` is the public patch boundary: ``gym.make(id, **kwargs)`` and
``gym.make_vec`` forward their caller's kwargs straight into
``scenario.update(...)`` (see ``metasim.task.gym_registration``). Pre-fix,
``update`` did a raw ``setattr`` per key, so a typo silently created a dead
attribute and left the real field at its default — e.g. ``update(num_env=4)``
ran 1 env, ``update(headles=True)`` left the sim non-headless — with no error.

The fix **warns and skips** unknown keys (instead of setting dead attributes).
A warning, not a hard error, is deliberate: several shipped scripts pass a
long-dead ``renderer=`` kwarg alongside their real ``simulator=``; raising
would break them, while warning surfaces the dead kwarg without breakage —
consistent with the existing ``_warn_duplicate_names`` boundary behaviour.

These tests lock in: valid fields still patch (and post-init reruns), unknown
fields warn + are skipped (no dead attribute), and the real field is intact.
"""

from __future__ import annotations

import pytest
from loguru import logger as _loguru_logger

from metasim.scenario.scenario import ScenarioCfg


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING records (the repo logs via loguru, not stdlib
    ``logging``, so ``caplog`` would miss these). Mirrors the fixture in
    ``test_scenario_duplicate_names_general.py``."""
    records: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        yield records
    finally:
        _loguru_logger.remove(sink_id)


@pytest.mark.general
def test_update_patches_known_fields():
    """A normal update of real fields still works and reruns post-init."""
    scenario = ScenarioCfg(num_envs=1)
    ret = scenario.update(num_envs=4, headless=True, simulator="mujoco")
    assert ret is scenario  # update returns self for chaining
    assert scenario.num_envs == 4
    assert scenario.headless is True
    assert scenario.simulator == "mujoco"


@pytest.mark.general
def test_update_reruns_post_init_on_valid_update():
    """A valid update must rerun ``__post_init__`` — proven via a side effect
    it produces (string robot name → resolved RobotCfg instance). Guards
    against a future change that drops the ``self.__post_init__()`` call."""
    scenario = ScenarioCfg(num_envs=1)
    scenario.update(robots=["franka"])
    # __post_init__ resolves the "franka" string into a RobotCfg object.
    assert len(scenario.robots) == 1
    resolved = scenario.robots[0]
    assert not isinstance(resolved, str), "post-init did not resolve the robot name string"
    assert getattr(resolved, "name", None) == "franka"


@pytest.mark.general
@pytest.mark.parametrize(
    "bad_key",
    ["num_env", "headles", "totally_bogus", "renderer"],
)
def test_update_warns_and_skips_unknown_fields(bad_key, loguru_warnings):
    """An unknown/misspelled field is warned about and skipped — NOT set as a
    dead attribute, and NOT a hard error (shipped ``renderer=`` callers keep
    working)."""
    scenario = ScenarioCfg(num_envs=1)
    ret = scenario.update(**{bad_key: "whatever"})
    assert ret is scenario  # does not raise
    # The dead attribute must NOT have been created.
    assert not hasattr(scenario, bad_key)
    # A warning naming the offending key must have been emitted.
    out = "\n".join(loguru_warnings)
    assert bad_key in out, f"no warning mentioning {bad_key!r}"


@pytest.mark.general
def test_update_applies_valid_keys_even_when_an_unknown_key_is_present():
    """Mixing a valid + an unknown key: the valid one still applies, the
    unknown one is dropped with a warning (no all-or-nothing surprise)."""
    scenario = ScenarioCfg(num_envs=1)
    scenario.update(num_envs=8, renderer="mujoco")  # renderer is the dead shipped kwarg
    assert scenario.num_envs == 8
    assert not hasattr(scenario, "renderer")
