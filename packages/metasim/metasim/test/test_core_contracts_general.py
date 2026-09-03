"""Contracts that used to hold only by accident (architecture review, 2026-09-03).

* Content packs win over MetaSim's bundled example pack when both define a config name, and a
  shadowed name is reported.
* ``gym.make``-style construction never mutates the task class's shared ``scenario``.
* ``get_states(env_ids=...)`` returns exactly those envs even when the backend ignores ``env_ids``.
* The task-index cache lives in a per-user directory, not the shared temp dir.
"""

from __future__ import annotations

import os
import textwrap

import pytest
import torch

from metasim.task import _static_index
from metasim.utils import package_discovery, setup_util


@pytest.mark.general
def test_defaults_are_searched_last(monkeypatch):
    monkeypatch.setattr(package_discovery, "_entry_point_packages", lambda role: ("content_pack.robots",))
    cands = package_discovery.get_package_candidates("robots", defaults=["metasim.example.example_pack.robots"])
    assert cands.index("content_pack.robots") < cands.index("metasim.example.example_pack.robots")


@pytest.mark.general
def test_shadowed_config_name_is_reported_and_first_wins(tmp_path, monkeypatch):
    for pkg in ("packa", "packb"):
        d = tmp_path / pkg
        d.mkdir()
        (d / "__init__.py").write_text(
            textwrap.dedent(f'''
            class DemoCfg:
                origin = "{pkg}"
            '''),
            encoding="utf-8",
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    setup_util._SHADOW_WARNED.clear()
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        cfg = setup_util._lookup_cfg("DemoCfg", ["packa", "packb"], "Robot")
    finally:
        logger.remove(sink_id)
    assert cfg.origin == "packa"
    assert any("defined in several packages" in m and "packa" in m for m in messages)


@pytest.mark.general
def test_scenario_replace_leaves_original_untouched():
    from metasim.scenario.scenario import ScenarioCfg

    base = ScenarioCfg(num_envs=1)
    other = base.replace(num_envs=8)
    assert base.num_envs == 1 and other.num_envs == 8 and other is not base


@pytest.mark.general
def test_gym_wrapper_does_not_mutate_class_scenario(monkeypatch):
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.task import gym_registration

    class _Env:
        scenario = ScenarioCfg(num_envs=4)

        def __init__(self, scenario, device=None):
            self.scenario = scenario
            self.device = "cpu"
            self.action_space = self.observation_space = None
            self.num_envs = scenario.num_envs

    monkeypatch.setattr(gym_registration, "get_task_class", lambda name: _Env)
    wrapper = gym_registration.GymEnvWrapper.__new__(gym_registration.GymEnvWrapper)
    gym_registration.GymEnvWrapper.__init__(wrapper, "demo")
    assert wrapper.scenario.num_envs == 1
    assert _Env.scenario.num_envs == 4, "gym.make rewrote the task's class-level scenario"


@pytest.mark.general
def test_get_states_slices_when_backend_ignores_env_ids():
    from metasim.sim.base import BaseSimHandler
    from metasim.types import ObjectState, TensorState

    full = TensorState(
        objects={"cube": ObjectState(root_state=torch.arange(4 * 13, dtype=torch.float32).view(4, 13))},
        robots={},
        cameras={},
    )

    class _Handler:
        num_envs = 4
        _get_states = staticmethod(lambda env_ids=None: full)  # ignores env_ids, like six real backends

    sub = BaseSimHandler._enforce_env_subset(_Handler(), full, [1, 3])
    assert sub.objects["cube"].root_state.shape[0] == 2
    assert torch.equal(sub.objects["cube"].root_state, full.objects["cube"].root_state[[1, 3]])

    # A batch that is neither the requested subset nor the full env set is a backend bug, not a slice.
    class _Broken(_Handler):
        num_envs = 5

    with pytest.raises(RuntimeError):
        BaseSimHandler._enforce_env_subset(_Broken(), full, [0, 1])


@pytest.mark.general
def test_index_cache_defaults_to_user_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("METASIM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _static_index.default_cache_path() == os.path.join(str(tmp_path), "metasim", "task_index.json")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert _static_index.default_cache_path().startswith(os.path.join(os.path.expanduser("~"), ".cache", "metasim"))
