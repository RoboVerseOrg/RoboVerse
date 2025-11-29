"""Integration tests for default joint position configuration."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx", "sapien2", "sapien3")
def test_default_pos(handler):
    """Test that default robot positions are correctly applied."""
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

    default_pos = handler.scenario.robots[0].default_position
    default_rot = handler.scenario.robots[0].default_orientation
    assert_close(states_default[0]["robots"]["franka"]["pos"], default_pos, atol=1e-3, message="franka pos error")
    assert_close(states_default[0]["robots"]["franka"]["rot"], default_rot, atol=1e-3, message="franka rot error")

    log.info(f"Default pos test passed for {handler.scenario.simulator}")
