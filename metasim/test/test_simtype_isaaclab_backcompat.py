"""Backward-compat tests for the ``SimType.ISAACLAB`` deprecated alias.

The standalone IsaacLab handler was removed in commit b752bb2 (C3) — the
IsaacSim handler now uses the ``isaaclab`` Python package internally.
Downstream code in the RoboVerse repo still imports ``SimType.ISAACLAB``
(``scripts/conversion/convert_traj_v1_to_v2.py``,
``scripts/advanced/train_ppo.py``), so the enum value is kept as a
deprecated alias that the dispatcher rewrites to ``ISAACSIM``.
"""

from __future__ import annotations

import warnings

import pytest


@pytest.mark.general
def test_simtype_isaaclab_enum_value_still_present():
    """``SimType.ISAACLAB`` must remain importable for existing callers."""
    from metasim.constants import SimType

    assert hasattr(SimType, "ISAACLAB")
    assert SimType.ISAACLAB.value == "isaaclab"


@pytest.mark.general
def test_isaaclab_dispatch_routes_to_isaacsim_with_deprecation():
    """``get_sim_handler_class(SimType.ISAACLAB)`` must rewrite to ISAACSIM
    and emit a DeprecationWarning, instead of raising or returning a stale
    class."""
    from metasim.constants import SimType
    from metasim.utils.setup_util import get_sim_handler_class

    try:
        from metasim.sim.isaacsim import IsaacsimHandler
    except ImportError:
        pytest.skip("IsaacSim runtime not installed in this environment")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler_cls = get_sim_handler_class(SimType.ISAACLAB)

    assert handler_cls is IsaacsimHandler
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning when dispatching ISAACLAB"
    assert any("ISAACLAB" in str(w.message) and "ISAACSIM" in str(w.message) for w in deprecations)


@pytest.mark.general
def test_isaaclab_not_in_scenario_simulator_literal():
    """The deprecated alias must NOT appear in ``ScenarioCfg.simulator``'s
    Literal — new code should be steered toward ``isaacsim``."""
    import typing

    from metasim.scenario.scenario import ScenarioCfg

    hint = typing.get_type_hints(ScenarioCfg).get("simulator")
    assert hint is not None, "ScenarioCfg.simulator type hint missing"
    literal_values: set[str] = set()
    for arg in typing.get_args(hint):
        literal_values.update(str(v) for v in typing.get_args(arg))
    assert "isaaclab" not in literal_values, (
        "isaaclab should not be in the Literal; the ISAACLAB enum is only a "
        "deprecated alias for existing call sites."
    )
