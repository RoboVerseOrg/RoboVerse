"""Integration tests for joint position limits."""

from __future__ import annotations

import math

import pytest
import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from metasim.sim.base import BaseSimHandler


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx", "sapien2", "sapien3")
def test_self_collision(handler: BaseSimHandler):
    """Test that joint limits are respected during simulation."""
    dof_targets = [
        {
            "franka1": {
                "dof_pos_target": {
                    "panda_joint1": 0.0,
                    "panda_joint2": 1.3,  # Self collision
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.0,  # Self collision
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
            },
            "franka2": {
                "dof_pos_target": {
                    "panda_joint1": 0.0,
                    "panda_joint2": 1.3,  # Self collision
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.0,  # Self collision
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
            },
        }
    ] * handler.scenario.num_envs
    handler.set_dof_targets(dof_targets)

    # Simulate to let joints reach their targets (clamped by limits)
    for _ in range(10):
        handler.simulate()

    states_after = handler.get_states(mode="dict")

    assert math.isclose(states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint2"], 1.3, abs_tol=1e-3), (
        "franka1 (without self collisions) joint 2 should be close to 1.3"
    )
    assert not math.isclose(states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint2"], 1.3, abs_tol=1e-3), (
        "franka2 (with self collisions) joint 2 should not be close to 1.3"
    )


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx", "sapien2", "sapien3")
def test_mutual_collision(handler: BaseSimHandler):
    """Test that joint limits are respected during simulation."""

    dof_resets = [
        {
            "robots": {
                "franka1": {
                    "dof_pos_target": {
                        "panda_joint1": 0.0 - 0.1,
                        "panda_joint2": -0.785398 - 0.1,
                        "panda_joint3": 0.0 - 0.1,
                        "panda_joint4": -2.356194 - 0.1,
                        "panda_joint5": 0.0 - 0.1,
                        "panda_joint6": 1.570796 + 0.1,
                        "panda_joint7": 0.785398 + 0.1,
                        "panda_finger_joint1": 0.0,
                        "panda_finger_joint2": 0.0,
                    }
                },
                "franka2": {
                    "dof_pos_target": {
                        "panda_joint1": 0.0 - 0.1,
                        "panda_joint2": -0.785398 - 0.1,
                        "panda_joint3": 0.0 - 0.1,
                        "panda_joint4": -2.356194 - 0.1,
                        "panda_joint5": 0.0 - 0.1,
                        "panda_joint6": 1.570796 + 0.1,
                        "panda_joint7": 0.785398 + 0.1,
                        "panda_finger_joint1": 0.0,
                        "panda_finger_joint2": 0.0,
                    }
                },
            },
            "objects": {},
        }
    ] * handler.scenario.num_envs

    dof_targets = [
        {
            "franka1": {
                "dof_pos_target": {
                    "panda_joint1": 0.65,
                    "panda_joint2": 0.564,
                    "panda_joint3": 0.25,
                    "panda_joint4": -1.27,
                    "panda_joint5": -0.08,
                    "panda_joint6": 2.13,
                    "panda_joint7": 0.785,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
            },
            "franka2": {
                "dof_pos_target": {
                    "panda_joint1": -0.82,
                    "panda_joint2": 0.05,
                    "panda_joint3": 0.01,
                    "panda_joint4": -1.71,
                    "panda_joint5": 0.04,
                    "panda_joint6": 2.98,
                    "panda_joint7": 0.785,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
            },
        }
    ] * handler.scenario.num_envs

    # Reset the simulation to the default positions
    handler.set_states(dof_resets)
    handler.set_dof_targets(dof_targets)

    # Simulate to let joints reach their targets (clamped by limits)
    for _ in range(10):
        handler.simulate()

    states_after = handler.get_states(mode="dict")

    franka1_dof_state_list = [
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint1"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint2"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint3"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint4"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint5"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint6"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_joint7"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_finger_joint1"],
        states_after[0]["robots"]["franka1"]["dof_pos"]["panda_finger_joint2"],
    ]
    franka2_dof_state_list = [
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint1"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint2"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint3"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint4"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint5"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint6"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_joint7"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_finger_joint1"],
        states_after[0]["robots"]["franka2"]["dof_pos"]["panda_finger_joint2"],
    ]
    franka1_dof_state_tensor = torch.tensor(franka1_dof_state_list)
    franka2_dof_state_tensor = torch.tensor(franka2_dof_state_list)
    concated_states = torch.cat([franka1_dof_state_tensor, franka2_dof_state_tensor], dim=0)
    franka1_dof_target_list = [
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint1"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint2"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint3"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint4"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint5"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint6"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_joint7"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_finger_joint1"],
        dof_targets[0]["franka1"]["dof_pos_target"]["panda_finger_joint2"],
    ]
    franka2_dof_target_list = [
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint1"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint2"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint3"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint4"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint5"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint6"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_joint7"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_finger_joint1"],
        dof_targets[0]["franka2"]["dof_pos_target"]["panda_finger_joint2"],
    ]
    franka1_dof_target_tensor = torch.tensor(franka1_dof_target_list)
    franka2_dof_target_tensor = torch.tensor(franka2_dof_target_list)
    concated_targets = torch.cat([franka1_dof_target_tensor, franka2_dof_target_tensor], dim=0)
    assert not torch.allclose(concated_states, concated_targets, atol=1e-3), (
        "franka1 and franka2 should be in collision, thus the states should not be close to the targets"
    )
