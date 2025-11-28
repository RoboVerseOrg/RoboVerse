"""Integration tests for 1-robot scenario configuration."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx")
def test_1_robot(handler):
    """Test that scenario correctly reports 1 robot."""
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
    handler.simulate()

    robot_keys = list(handler.get_states(mode="tensor").robots.keys())
    assert len(robot_keys) == 1 and "franka" in robot_keys
    assert robot_keys == ["franka"]

    robot_keys_dict = list(handler.get_states(mode="dict")[0]["robots"].keys())
    assert robot_keys_dict == ["franka"]

    log.info(f"1-robot test passed for {handler.scenario.simulator}")
