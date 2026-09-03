"""The native ManiSkill specs agree with the installed ``mani_skill`` env registry.

The ``maniskill.*_native`` tier only earns its name if its episode horizon and control/sim rates are
ManiSkill's, not defaults that happen to sit nearby: a truncation limit below upstream's silently
caps the achievable success rate, so any number reported from the tier stops being comparable to
ManiSkill's published ones (``plug_charger`` truncated at 50 against an upstream limit of 200 —
below the length of a median demonstration — is unsolvable by construction).

Two layers: a pinned table of upstream horizons (``mani_skill==3.0.1``) that always runs, and
registry-derived checks (``REGISTERED_ENVS`` + ``_default_sim_config``) that run when ``mani_skill``
is installed so the pin tracks upstream instead of rotting. Cheap: reads the registry, builds no scene.
"""

from __future__ import annotations

import pytest

from roboverse_pack.tasks.maniskill._native.recipe import DECIMATION, maniskill_sim_params
from roboverse_pack.tasks.maniskill._native.specs import TASK_SPECS

# ``max_episode_steps`` registered by mani_skill==3.0.1 (``mani_skill/envs/tasks/tabletop/*.py``).
UPSTREAM_MAX_EPISODE_STEPS = {
    "DrawTriangle-v1": 300,
    "LiftPegUpright-v1": 50,
    "PegInsertionSide-v1": 100,
    "PickCube-v1": 50,
    "PlaceSphere-v1": 50,
    "PlugCharger-v1": 200,
    "PokeCube-v1": 50,
    "PullCube-v1": 50,
    "PullCubeTool-v1": 100,
    "PushCube-v1": 50,
    "PushT-v1": 100,
    "RollBall-v1": 80,
    "StackCube-v1": 50,
    "StackPyramid-v1": 250,
}
# ManiSkill's default tabletop ``SimConfig``: sim_freq=100, control_freq=20.
UPSTREAM_SIM_FREQ, UPSTREAM_CONTROL_FREQ = 100, 20


@pytest.mark.parametrize("task_key", sorted(TASK_SPECS))
def test_max_steps_matches_pinned_upstream_horizon(task_key):
    """Always runs: ``max_steps`` equals the horizon pinned from the upstream source."""
    spec = TASK_SPECS[task_key]
    assert spec["gym_id"] in UPSTREAM_MAX_EPISODE_STEPS, f"{task_key}: add {spec['gym_id']} to the pinned table"
    assert spec["max_steps"] == UPSTREAM_MAX_EPISODE_STEPS[spec["gym_id"]]


def test_recipe_matches_pinned_upstream_rates():
    params = maniskill_sim_params()
    assert params.dt == pytest.approx(1.0 / UPSTREAM_SIM_FREQ)
    assert DECIMATION == UPSTREAM_SIM_FREQ // UPSTREAM_CONTROL_FREQ


def _registered_env(gym_id: str):
    """The upstream ``EnvSpec`` for ``gym_id`` (importing ``mani_skill.envs`` registers them all)."""
    pytest.importorskip("mani_skill")
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
