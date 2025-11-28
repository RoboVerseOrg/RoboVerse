"""Integration tests for joint position limits."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_pack.robots.franka_cfg import FrankaCfg


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx")
def test_qpos_limit(handler):
    """Test that joint limits are respected during simulation."""
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": -3.0,  # Exceeds lower limit
                        "panda_joint2": +2.0,  # Exceeds upper limit
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

    # Simulate to let joints reach their targets (clamped by limits)
    for _ in range(10):
        handler.simulate()

    states_after = handler.get_states(mode="dict")

    # Get joint limits from FrankaCfg
    franka_cfg = FrankaCfg()

    # Check that joints are within limits
    assert (
        states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint1"]
        >= franka_cfg.joint_limits["panda_joint1"][0] - 1e-3
    )
    assert (
        states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint2"]
        <= franka_cfg.joint_limits["panda_joint2"][1] + 1e-3
    )

    log.info(f"Qpos limit test passed for {handler.scenario.simulator}")
