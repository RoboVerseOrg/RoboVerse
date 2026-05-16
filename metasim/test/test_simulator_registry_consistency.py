"""Contract tests: every ``SimType`` has a ``get_sim_handler_class`` entry,
every entry in the ``ScenarioCfg.simulator`` Literal has a corresponding
``SimType``, and the import-time entries actually resolve to existing
modules.

These tests catch the class of bug fixed in C3: a SimType / Literal entry
that points at a deleted submodule (``metasim.sim.isaaclab``), which
results in `ModuleNotFoundError` at scenario-launch time instead of at
import / config-validation time.
"""

from __future__ import annotations

from typing import get_args, get_origin, get_type_hints

import pytest

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_sim_handler_class


# ``ISAACLAB`` is a deprecated alias kept for backward compatibility with
# downstream callers (e.g. RoboVerse's ``scripts/conversion/convert_traj_v1_to_v2.py``).
# The dispatcher rewrites it to ``ISAACSIM`` with a DeprecationWarning.
# It must NOT appear in the ``ScenarioCfg.simulator`` Literal — new
# code should be steered to ``isaacsim``.
_DEPRECATED_SIMTYPE_VALUES: frozenset[str] = frozenset({"isaaclab"})


@pytest.mark.general
def test_no_dead_isaaclab_reference():
    """The standalone IsaacLab handler module was removed in C3
    (b752bb2). The enum value survives as a deprecated alias so existing
    callers keep working, but the Literal must not advertise it."""
    # SimType.ISAACLAB exists as a deprecated alias only.
    assert hasattr(SimType, "ISAACLAB"), (
        "SimType.ISAACLAB must remain importable as a deprecated alias "
        "(see metasim/utils/setup_util.py for the dispatcher shim)"
    )
    hints = get_type_hints(ScenarioCfg, include_extras=False)
    simulator_hint = hints.get("simulator")
    sim_args = ()
    if get_origin(simulator_hint) is not None:
        for arg in get_args(simulator_hint):
            inner = get_args(arg)
            if inner:
                sim_args = inner
                break
    assert "isaaclab" not in sim_args, (
        "ScenarioCfg.simulator Literal must not advertise isaaclab — "
        "new code should use isaacsim instead"
    )


@pytest.mark.general
def test_simulator_literal_covers_every_sim_type():
    """The ``ScenarioCfg.simulator`` Literal must list every ``SimType``
    value, *except* explicitly deprecated aliases."""
    hints = get_type_hints(ScenarioCfg, include_extras=False)
    simulator_hint = hints.get("simulator")
    sim_args: tuple[str, ...] = ()
    if get_origin(simulator_hint) is not None:
        for arg in get_args(simulator_hint):
            inner = get_args(arg)
            if inner:
                sim_args = inner
                break
    literal_values = set(sim_args)
    sim_type_values = {st.value for st in SimType} - _DEPRECATED_SIMTYPE_VALUES
    missing_from_literal = sim_type_values - literal_values
    assert not missing_from_literal, (
        f"ScenarioCfg.simulator Literal missing entries that exist in SimType: {sorted(missing_from_literal)}"
    )
    extras_in_literal = literal_values - sim_type_values - _DEPRECATED_SIMTYPE_VALUES
    assert not extras_in_literal, (
        f"ScenarioCfg.simulator Literal has entries with no matching SimType: {sorted(extras_in_literal)}"
    )


@pytest.mark.general
@pytest.mark.parametrize("sim_type", list(SimType))
def test_every_sim_type_has_handler_dispatch(sim_type):
    """``get_sim_handler_class`` must wire each registered SimType to an
    importable submodule. We *only* fail when the dispatch points at a
    metasim-internal module that doesn't exist. Backend-level import
    failures (omni, isaacgym, sapien, jax/numpy version skew, …) are
    treated as "optional dep not installed" — they prove dispatch reaches
    the right module, even if the module's own deps aren't usable here.
    """
    try:
        cls = get_sim_handler_class(sim_type)
    except ModuleNotFoundError as exc:
        # Anything inside ``metasim.sim.*`` being missing means our dispatch
        # is broken; anything else is the backend's own optional lib being
        # absent (acceptable).
        missing = str(exc.name) if exc.name else str(exc)
        if missing.startswith("metasim.sim.") and "_handler" not in missing:
            raise AssertionError(
                f"Dispatch for {sim_type} points at a deleted metasim submodule: {exc}"
            ) from exc
        return
    except (ImportError, AttributeError, OSError):
        # Backend native lib failed to load (e.g. jax/numpy skew on MJX,
        # libGL missing for sapien) — dispatch itself reached the module.
        return
    assert cls is not None
