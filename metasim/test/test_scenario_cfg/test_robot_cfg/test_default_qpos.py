"""Integration tests for default joint position configuration."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.mujoco
def test_default_qpos(handler):
    """Test that default joint positions are correctly applied."""
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    }
                }
            }
        ]
        * handler.scenario.num_envs
    )

    # Check initial state matches the default positions from scenario
    states_default = handler.get_states(mode="dict")
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint1"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint2"], -0.785398 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint3"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint4"], -2.356194 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint5"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint6"], 1.570796 + 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint7"], 0.785398 + 0.1, atol=1e-3)

    # Simulate and check state converges to targets
    for _ in range(20):
        handler.simulate()

    states_after = handler.get_states(mode="dict")
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint1"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint2"], -0.785398, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint3"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint4"], -2.356194, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint5"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint6"], 1.570796, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint7"], 0.785398, atol=1e-3)

    log.info(f"Default qpos test passed for {handler.scenario.simulator}")
