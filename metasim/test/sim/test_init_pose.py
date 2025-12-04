"""Integration tests for initial pose consistency."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sapien3
def test_init_pose(handler):
    """Test that initial poses are correctly applied."""
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

    state = handler.get_states(mode="dict")

    # Check robot DoF positions and velocities
    dof_pos = state[0]["robots"]["franka"]["dof_pos"]
    dof_vel = state[0]["robots"]["franka"]["dof_vel"]
    dof_pos_tensor = torch.Tensor([
        dof_pos[joint_name] for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    dof_vel_tensor = torch.Tensor([
        dof_vel[joint_name] for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    init_dof_pos_tensor = torch.Tensor([
        handler.scenario.robots[0].default_joint_positions[joint_name]
        for joint_name in sorted(handler.get_joint_names("franka", True))
    ])
    assert_close(dof_pos_tensor, init_dof_pos_tensor, atol=1e-3, message="DoF pos")
    assert_close(dof_vel_tensor, torch.zeros(9), atol=1e-3, message="DoF vel")

    # Check robot position and orientation
    pos = state[0]["robots"]["franka"]["pos"]
    rot = state[0]["robots"]["franka"]["rot"]
    assert_close(pos, torch.Tensor(handler.scenario.robots[0].default_position), atol=1e-3, message="franka pos")
    assert_close(rot, torch.Tensor(handler.scenario.robots[0].default_orientation), atol=1e-3, message="franka rot")

    # Check object positions and orientations
    pos = state[0]["objects"]["cube"]["pos"]
    rot = state[0]["objects"]["cube"]["rot"]
    assert_close(pos, torch.Tensor(handler.scenario.objects[0].default_position), atol=1e-3, message="cube pos")
    assert_close(rot, torch.Tensor(handler.scenario.objects[0].default_orientation), atol=1e-3, message="cube rot")

    pos = state[0]["objects"]["sphere"]["pos"]
    rot = state[0]["objects"]["sphere"]["rot"]
    assert_close(pos, torch.Tensor(handler.scenario.objects[1].default_position), atol=1e-3, message="sphere pos")
    assert_close(rot, torch.Tensor(handler.scenario.objects[1].default_orientation), atol=1e-3, message="sphere rot")

    pos = state[0]["objects"]["bbq_sauce"]["pos"]
    rot = state[0]["objects"]["bbq_sauce"]["rot"]
    assert_close(pos, torch.Tensor(handler.scenario.objects[2].default_position), atol=1e-3, message="bbq_sauce pos")
    assert_close(rot, torch.Tensor(handler.scenario.objects[2].default_orientation), atol=1e-3, message="bbq_sauce rot")

    pos = state[0]["objects"]["box_base"]["pos"]
    rot = state[0]["objects"]["box_base"]["rot"]
    assert_close(pos, torch.Tensor(handler.scenario.objects[3].default_position), atol=1e-3, message="box_base pos")
    assert_close(rot, torch.Tensor(handler.scenario.objects[3].default_orientation), atol=1e-3, message="box_base rot")

    # Check tensor state
    tensor_state = handler.get_states(mode="tensor")
    assert_close(tensor_state.robots["franka"].joint_pos, init_dof_pos_tensor, atol=1e-3, message="tensor DoF pos")
    assert_close(tensor_state.robots["franka"].joint_vel, torch.zeros(9), atol=1e-3, message="tensor DoF vel")

    log.info(f"Init pose test passed for {handler.scenario.simulator}")
