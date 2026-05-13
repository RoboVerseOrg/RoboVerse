"""``ScenarioCfg.simulator`` Literal must match ``SimType`` (L1).

Pre-fix the Literal listed 7 of the 12 registered backends (``isaaclab``,
``isaacgym``, ``sapien2``, ``sapien3``, ``genesis``, ``pybullet``,
``mujoco``). Five were missing: ``isaacsim``, ``mjx``, ``newton``,
``blender``, ``pyrep``. Type checkers and downstream Literal-based
dispatchers therefore couldn't validate those names. Setting
``scenario.simulator = "isaacsim"`` silently violated the type
annotation while the constant ``SimType.ISAACSIM`` happily existed.

After the fix the Literal mirrors ``SimType`` exactly. This test locks
that contract: any new backend added to ``SimType`` is forced to also
appear in the Literal (or vice versa).
"""

from __future__ import annotations

from typing import get_args, get_type_hints

import pytest

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg


@pytest.mark.general
def test_scenario_simulator_literal_matches_simtype():
    """The set of strings accepted by ``ScenarioCfg.simulator`` must be
    exactly the values of ``SimType``."""
    hints = get_type_hints(ScenarioCfg)
    sim_field_type = hints["simulator"]

    # The annotation is ``Literal[...] | None`` — flatten and drop None.
    args = get_args(sim_field_type)
    literal_values = []
    for arg in args:
        # ``Literal[...]`` itself has its members as ``__args__``.
        if hasattr(arg, "__args__"):
            literal_values.extend(arg.__args__)

    literal_set = set(literal_values)
    simtype_set = {sim.value for sim in SimType}

    missing_from_literal = simtype_set - literal_set
    extra_in_literal = literal_set - simtype_set

    assert not missing_from_literal, (
        f"ScenarioCfg.simulator is missing these SimType values: {sorted(missing_from_literal)}"
    )
    assert not extra_in_literal, (
        f"ScenarioCfg.simulator carries values not in SimType: {sorted(extra_in_literal)}"
    )


@pytest.mark.general
def test_each_simtype_value_is_a_string():
    """Sanity check used by the field comparison above."""
    for sim in SimType:
        assert isinstance(sim.value, str)
