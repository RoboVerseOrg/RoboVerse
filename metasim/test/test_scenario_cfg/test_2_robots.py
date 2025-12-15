"""Integration tests for 2-robots scenario configuration."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)


@pytest.mark.mujoco
def test_2_robots(handler):
    """Test that scenario correctly reports 2 robots."""
    handler.simulate()

    robot_keys = list(handler.get_states(mode="tensor").robots.keys())
    assert len(robot_keys) == 2 and "franka1" in robot_keys and "franka2" in robot_keys
    assert robot_keys == ["franka1", "franka2"]

    robot_keys_dict = list(handler.get_states(mode="dict")[0]["robots"].keys())
    assert robot_keys_dict == ["franka1", "franka2"]

    log.info(f"2-robots test passed for {handler.scenario.simulator}")
