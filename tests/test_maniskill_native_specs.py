"""The native ManiSkill specs agree with the installed ``mani_skill`` env registry.

The ``maniskill.*_native`` tier only earns its name if its episode horizon and control/sim rates are
ManiSkill's, not defaults that happen to sit nearby: a truncation limit below upstream's silently
caps the achievable success rate, so any number reported from the tier stops being comparable to
ManiSkill's published ones (``plug_charger`` truncated at 50 against an upstream limit of 200 —
below the length of a median demonstration — is unsolvable by construction).

Every expectation is **derived from the installed package** (``REGISTERED_ENVS`` +
``_default_sim_config``) rather than a hardcoded table, so the test tracks upstream instead of
rotting alongside the code it guards. Cheap: reads the registry, builds no scene.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mani_skill")

from roboverse_pack.tasks.maniskill._native.recipe import DECIMATION, maniskill_sim_params
from roboverse_pack.tasks.maniskill._native.specs import TASK_SPECS


def _registered_env(gym_id: str):
    """The upstream ``EnvSpec`` for ``gym_id`` (importing ``mani_skill.envs`` registers them all)."""
    import mani_skill.envs  # noqa: F401 — import registers every ManiSkill env
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert gym_id in REGISTERED_ENVS, f"{gym_id} is not a registered ManiSkill env"
    return REGISTERED_ENVS[gym_id]


@pytest.mark.parametrize("task_key", sorted(TASK_SPECS))
def test_max_steps_matches_maniskill(task_key):
    """``max_steps`` equals the ManiSkill env's registered ``max_episode_steps``."""
    spec = TASK_SPECS[task_key]
    expected = _registered_env(spec["gym_id"]).max_episode_steps
    assert spec["max_steps"] == expected, (
        f"{task_key}: max_steps={spec['max_steps']} but ManiSkill {spec['gym_id']} truncates at"
        f" {expected} — the native tier is not comparable to ManiSkill's numbers at a different horizon"
    )


@pytest.mark.parametrize("task_key", sorted(TASK_SPECS))
def test_registered_task_carries_the_native_horizon(task_key):
    """The registered ``maniskill.<name>_native`` task class exposes that horizon."""
    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers the native tasks
    from metasim.task.registry import get_task_class

    cls = get_task_class(f"maniskill.{task_key}_native")
    assert cls.max_episode_steps == _registered_env(TASK_SPECS[task_key]["gym_id"]).max_episode_steps


@pytest.mark.parametrize("task_key", sorted(TASK_SPECS))
def test_sim_and_control_freq_match_maniskill(task_key):
    """The shipped PhysX recipe steps at ManiSkill's ``sim_freq`` with its ``control_freq`` decimation."""
    env_cls = _registered_env(TASK_SPECS[task_key]["gym_id"]).cls
    sim_config = env_cls._default_sim_config.fget(env_cls)  # property, no instance needed

    params = maniskill_sim_params()
    assert params.dt == pytest.approx(1.0 / sim_config.sim_freq)
    assert DECIMATION == sim_config.sim_freq // sim_config.control_freq
