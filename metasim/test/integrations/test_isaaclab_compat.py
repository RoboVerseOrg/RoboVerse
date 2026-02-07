from __future__ import annotations

import pytest

from metasim.integrations.isaaclab.cfg_convert import scenario_from_isaaclab_cfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.factory import make_task_env
from metasim.task.registry import register_task


@pytest.mark.general
def test_scenario_from_isaaclab_cfg_runtime_knobs():
    class _Scene:
        num_envs = 123
        env_spacing = 2.5

    class _Sim:
        dt = 0.005

    class _Cfg:
        scene = _Scene()
        decimation = 4
        sim = _Sim()

    scenario = scenario_from_isaaclab_cfg(_Cfg(), simulator="isaacsim", headless=True)
    assert scenario.simulator == "isaacsim"
    assert scenario.headless is True
    assert scenario.num_envs == 123
    assert scenario.env_spacing == 2.5
    assert scenario.decimation == 4
    assert scenario.sim_params.dt == 0.005


@register_task("test.metasim.factory_dummy")
class _DummyMetaSimTask(BaseTaskEnv):
    scenario = ScenarioCfg()

    def __init__(self, scenario: ScenarioCfg, args=None, device=None):
        # Intentionally do not call BaseTaskEnv.__init__ (no simulator in unit tests).
        self.scenario = scenario
        self.args = args
        self.device = device


@register_task("test.isaaclab.factory_dummy")
class _DummyIsaacLabTask:
    scenario = ScenarioCfg()

    def __init__(self, scenario: ScenarioCfg, args=None, device=None):
        self.scenario = scenario
        self.args = args
        self.device = device


@pytest.mark.general
def test_make_task_env_routes_metasim_task_without_launching_sim():
    scenario = ScenarioCfg(simulator="pybullet", num_envs=1, headless=True)
    args = object()
    env = make_task_env("test.metasim.factory_dummy", scenario=scenario, args=args, device="cpu")
    assert isinstance(env, _DummyMetaSimTask)
    assert env.args is args


@pytest.mark.general
def test_make_task_env_routes_isaaclab_style_task_without_runtime():
    scenario = ScenarioCfg(simulator="isaacsim", num_envs=1, headless=True)
    args = object()
    env = make_task_env("test.isaaclab.factory_dummy", scenario=scenario, args=args, device="cpu")
    assert isinstance(env, _DummyIsaacLabTask)
    assert env.args is args
