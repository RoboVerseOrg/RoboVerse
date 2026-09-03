"""General test: get_handler forwards optional_queries to the handler (no backend)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import metasim.utils.setup_util as setup_util
from metasim.utils.setup_util import get_handler


@pytest.mark.general
def test_get_handler_second_param_is_optional_queries():
    """Regression: the 2nd param was named 'device' but forwarded into the
    handler's optional_queries slot (dead when None, crash if a device passed)."""
    params = list(inspect.signature(get_handler).parameters)
    assert params[1] == "optional_queries"


@pytest.mark.general
def test_get_handler_forwards_optional_queries(monkeypatch):
    received = {}

    class _StubHandler:
        def __init__(self, scenario, optional_queries=None):
            received["optional_queries"] = optional_queries

        def launch(self):
            received["launched"] = True

    monkeypatch.setattr(setup_util, "get_sim_handler_class", lambda sim: _StubHandler)

    scenario = SimpleNamespace(simulator="mujoco")
    queries = {"imu_pos": object()}
    handler = get_handler(scenario, optional_queries=queries)

    assert isinstance(handler, _StubHandler)
    assert received["optional_queries"] is queries
    assert received["launched"] is True
