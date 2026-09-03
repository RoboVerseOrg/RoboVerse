"""``BaseTaskEnv.supported_simulators``: a declared task refuses a backend it does not list, at construction."""

from __future__ import annotations

import pytest

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv


class _Declared(BaseTaskEnv):
    supported_simulators = ("mujoco", "newton")


class _Undeclared(BaseTaskEnv):
    pass


@pytest.mark.general
def test_declared_task_rejects_other_backend_before_touching_a_handler():
    with pytest.raises(ValueError, match=r"does not support simulator 'sapien3'.*\('mujoco', 'newton'\)"):
        _Declared._check_supported_simulator(ScenarioCfg(simulator="sapien3"))


@pytest.mark.general
def test_declared_task_accepts_listed_backend_and_undeclared_task_is_unchecked():
    _Declared._check_supported_simulator(ScenarioCfg(simulator="newton"))
    _Undeclared._check_supported_simulator(ScenarioCfg(simulator="sapien3"))


@pytest.mark.general
def test_check_runs_inside_init(monkeypatch):
    calls = []
    monkeypatch.setattr(BaseTaskEnv, "_instantiate_env", lambda self, scenario: calls.append("instantiated"))
    with pytest.raises(ValueError):
        _Declared(ScenarioCfg(simulator="pybullet"))
    assert calls == [], "the handler must not be built for an unsupported backend"
