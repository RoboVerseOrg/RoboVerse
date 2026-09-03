"""General regression tests for two small core defects (no simulator backend)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.general
def test_default_buffer_size_multiplier_is_int():
    """Regression: the field is annotated int but defaulted to float 2.0."""
    from metasim.scenario.simulator_params import SimParamCfg

    value = SimParamCfg().default_buffer_size_multiplier
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value == 2


@pytest.mark.general
def test_object_registry_register_overwrites():
    """Regression: register() had a dead if/else (both branches identical)."""
    from metasim.randomization.core.object_registry import ObjectRegistry

    reg = ObjectRegistry.__new__(ObjectRegistry)  # bypass __init__
    reg._registry = {}
    first = SimpleNamespace(name="box")
    second = SimpleNamespace(name="box")
    reg.register(first)
    reg.register(second)
    assert reg._registry["box"] is second  # last write wins
    assert len(reg._registry) == 1
