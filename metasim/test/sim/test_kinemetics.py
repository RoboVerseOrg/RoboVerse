"""Integration tests for kinematics (forward kinematics)."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sapien3
def test_kinematics(handler):
    """Test that forward kinematics are computed correctly."""
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

    # Need to step once to update the kinematics
    handler.simulate()
    state = handler.get_states(mode="dict")

    # Check hand position and orientation if body data is available
    if "body" in state[0]["robots"]["franka"].keys():
        hand_pos = state[0]["robots"]["franka"]["body"]["panda_hand"]["pos"]
        hand_rot = state[0]["robots"]["franka"]["body"]["panda_hand"]["rot"]

        expected_hand_pos = torch.Tensor([3.0689e-01, 0.0, 5.9028e-01])
        expected_hand_rot = torch.Tensor([0.0, 1.0000e00, 0.0, 0.0])

        assert_close(hand_pos, expected_hand_pos, atol=1e-3, message="hand pos")
        assert_close(hand_rot, expected_hand_rot, atol=1e-3, message="hand rot")

    log.info(f"Kinematics test passed for {handler.scenario.simulator}")
