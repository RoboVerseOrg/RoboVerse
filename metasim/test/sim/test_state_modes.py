"""Integration tests for state mode conversion between tensor and dict formats.

Tests the get_states(mode="tensor") and get_states(mode="dict") functionality
and conversion between the two formats.
"""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_get_states_dict_mode(handler):
    """Test that get_states with mode='dict' returns list of dicts."""
    dict_state = handler.get_states(mode="dict")

    assert isinstance(dict_state, list)
    assert len(dict_state) == handler.scenario.num_envs

    # Verify structure of first environment
    env0_state = dict_state[0]
    assert "robots" in env0_state
    assert "objects" in env0_state
    assert "franka" in env0_state["robots"]
    assert "cube" in env0_state["objects"]

    # Verify robot dict structure
    robot_state = env0_state["robots"]["franka"]
    assert "pos" in robot_state
    assert "rot" in robot_state
    assert "dof_pos" in robot_state
    assert "dof_vel" in robot_state

    # Verify types
    assert isinstance(robot_state["pos"], torch.Tensor)
    assert robot_state["pos"].shape == (3,)
    assert isinstance(robot_state["rot"], torch.Tensor)
    assert robot_state["rot"].shape == (4,)
    assert isinstance(robot_state["dof_pos"], dict)

    log.info(f"Get states dict mode test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_state_mode_conversion_consistency(handler):
    """Test that tensor and dict modes represent the same state."""
    # Set a known state
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.3,
                        "panda_joint2": -0.6,
                        "panda_joint3": 0.2,
                        "panda_joint4": -2.2,
                        "panda_joint5": 0.1,
                        "panda_joint6": 1.6,
                        "panda_joint7": 0.9,
                        "panda_finger_joint1": 0.03,
                        "panda_finger_joint2": 0.03,
                    }
                }
            }
        ]
        * handler.scenario.num_envs
    )

    for _ in range(10):
        handler.simulate()

    # Get both formats
    tensor_state = handler.get_states(mode="tensor")
    dict_state = handler.get_states(mode="dict")

    # Compare positions for first environment
    tensor_robot_pos = tensor_state.robots["franka"].root_state[0, 0:3]
    dict_robot_pos = dict_state[0]["robots"]["franka"]["pos"]
    assert_close(tensor_robot_pos.cpu(), dict_robot_pos, atol=1e-6, message="robot root pos")

    # Compare orientations
    tensor_robot_quat = tensor_state.robots["franka"].root_state[0, 3:7]
    dict_robot_rot = dict_state[0]["robots"]["franka"]["rot"]
    assert_close(tensor_robot_quat.cpu(), dict_robot_rot, atol=1e-6, message="robot root quat")

    # Compare joint positions
    sorted_joint_names = handler.get_joint_names("franka", sort=True)
    tensor_joint_pos = tensor_state.robots["franka"].joint_pos[0]

    for idx, joint_name in enumerate(sorted_joint_names):
        dict_joint_pos = dict_state[0]["robots"]["franka"]["dof_pos"][joint_name]
        assert_close(
            tensor_joint_pos[idx].item(),
            dict_joint_pos,
            atol=1e-6,
            message=f"joint {joint_name} pos",
        )

    log.info(f"State mode conversion consistency test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_state_mode_multiple_calls(handler):
    """Test that calling get_states with different modes multiple times works correctly."""
    # Call with tensor mode
    tensor_state_1 = handler.get_states(mode="tensor")

    # Call with dict mode
    dict_state = handler.get_states(mode="dict")

    # Call with tensor mode again
    tensor_state_2 = handler.get_states(mode="tensor")

    # Both tensor calls should give same results (cache working correctly)
    assert_close(
        tensor_state_1.robots["franka"].root_state[:, 0:3],
        tensor_state_2.robots["franka"].root_state[:, 0:3],
        atol=1e-9,
        message="tensor state consistency across calls",
    )

    # Simulate and verify cache is invalidated
    handler.simulate()

    tensor_state_3 = handler.get_states(mode="tensor")

    # After simulate, positions may have changed slightly
    # Just verify we can still get the state
    assert tensor_state_3.robots["franka"].root_state is not None

    log.info(f"State mode multiple calls test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_dict_state_all_objects(handler):
    """Test that dict state includes all objects and robots."""
    dict_state = handler.get_states(mode="dict")

    env0_state = dict_state[0]

    # Verify all objects are present
    assert "cube" in env0_state["objects"]
    assert "sphere" in env0_state["objects"]
    assert "bbq_sauce" in env0_state["objects"]
    assert "box_base" in env0_state["objects"]

    # Verify robot is present
    assert "franka" in env0_state["robots"]

    # Verify articulation object has dof_pos
    assert "dof_pos" in env0_state["objects"]["box_base"]
    assert "box_joint" in env0_state["objects"]["box_base"]["dof_pos"]

    log.info(f"Dict state all objects test passed for {handler.scenario.simulator}")
