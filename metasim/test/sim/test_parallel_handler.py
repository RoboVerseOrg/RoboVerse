"""Integration tests for parallel simulation with multiple environments.

Based on get_started/3_parallel_envs.py - Tests ParallelSimWrapper functionality.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sim("isaacsim", "isaacgym")
def test_parallel_envs_state_independence(handler):
    """Test that parallel environments maintain independent states."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    # Set different states for each environment
    states = []
    for i in range(handler.scenario.num_envs):
        states.append({
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3 + i * 0.1, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6 + i * 0.1, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14 + i * 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": i * 0.1,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        })

    handler.set_states(states)
    retrieved_states = handler.get_states(mode="dict")

    # Verify each environment has its unique state
    for i in range(handler.scenario.num_envs):
        assert_close(
            retrieved_states[i]["objects"]["cube"]["pos"],
            torch.tensor([0.3 + i * 0.1, -0.2, 0.05]),
            atol=1e-3,
            message=f"env {i} cube pos",
        )
        assert_close(
            retrieved_states[i]["robots"]["franka"]["dof_pos"]["panda_joint1"],
            i * 0.1,
            atol=1e-3,
            message=f"env {i} joint1 pos",
        )

    log.info(f"Parallel envs state independence test passed for {handler.scenario.simulator}")


@pytest.mark.sim("isaacsim", "isaacgym")
def test_parallel_envs_actions(handler):
    """Test that different actions can be applied to each parallel environment."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    # Apply different actions to each environment
    actions = []
    for i in range(handler.scenario.num_envs):
        actions.append({
            "franka": {
                "dof_pos_target": {
                    "panda_joint1": float(i) * 0.2,
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
        })

    handler.set_dof_targets(actions)

    # Simulate multiple steps to let actions take effect
    for _ in range(20):
        handler.simulate()

    states = handler.get_states(mode="dict")

    # Verify that each environment moved towards its target
    for i in range(handler.scenario.num_envs):
        joint1_pos = states[i]["robots"]["franka"]["dof_pos"]["panda_joint1"]
        # Joint should be moving towards target (not exact due to PD control)
        assert abs(joint1_pos - float(i) * 0.2) < 0.1, f"Env {i} didn't move towards target"

    log.info(f"Parallel envs actions test passed for {handler.scenario.simulator}")


@pytest.mark.sim("isaacsim", "isaacgym")
def test_parallel_envs_simulate_consistency(handler):
    """Test that simulate() updates all environments consistently."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    # Set identical initial states
    identical_state = {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce": {
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([0.5, 0.2, 0.1]),
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        },
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
    handler.set_states([identical_state] * handler.scenario.num_envs)

    # Apply identical actions
    identical_action = {
        "franka": {
            "dof_pos_target": {
                "panda_joint1": 0.5,
                "panda_joint2": -0.5,
                "panda_joint3": 0.3,
                "panda_joint4": -2.0,
                "panda_joint5": 0.1,
                "panda_joint6": 1.5,
                "panda_joint7": 0.8,
                "panda_finger_joint1": 0.02,
                "panda_finger_joint2": 0.02,
            }
        }
    }
    handler.set_dof_targets([identical_action] * handler.scenario.num_envs)

    # Simulate
    for _ in range(10):
        handler.simulate()

    states = handler.get_states(mode="dict")

    # All environments should have identical states (within tolerance)
    reference_state = states[0]
    for i in range(1, handler.scenario.num_envs):
        for joint_name in reference_state["robots"]["franka"]["dof_pos"].keys():
            assert_close(
                states[i]["robots"]["franka"]["dof_pos"][joint_name],
                reference_state["robots"]["franka"]["dof_pos"][joint_name],
                atol=1e-5,
                message=f"env {i} vs env 0 joint {joint_name}",
            )

    log.info(f"Parallel envs simulate consistency test passed for {handler.scenario.simulator}")


@pytest.mark.sim("isaacsim", "isaacgym")
def test_parallel_tensor_state_batching(handler):
    """Test that tensor state correctly batches data across environments."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    tensor_state = handler.get_states(mode="tensor")

    # Verify tensor state has correct batch dimensions
    assert tensor_state.robots["franka"].root_state.shape[0] == handler.scenario.num_envs
    assert tensor_state.robots["franka"].joint_pos.shape[0] == handler.scenario.num_envs
    assert tensor_state.objects["cube"].root_state.shape[0] == handler.scenario.num_envs

    # Verify data types
    assert isinstance(tensor_state.robots["franka"].root_state, torch.Tensor)
    assert isinstance(tensor_state.robots["franka"].joint_pos, torch.Tensor)

    log.info(f"Parallel tensor state batching test passed for {handler.scenario.simulator}")


@pytest.mark.sim("isaacsim", "isaacgym")
def test_parallel_envs_partial_env_ids(handler):
    """Test setting states for specific environment IDs."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    ##FIXME: IsaacSim needs a simulate or scene update after set_states to flush changes, otherwise would cause issues in this test.
    if handler.scenario.simulator == "isaacsim":
        handler.scene.update(dt=handler.physics_dt)

    # Get initial states
    initial_states = handler.get_states(mode="dict")

    # Modify only first environment
    modified_env0 = deepcopy(initial_states[0])
    modified_env0["robots"]["franka"]["pos"] = torch.tensor([0.1, 0.2, 0.0])
    modified_env0["robots"]["franka"]["rot"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    modified_env0["robots"]["franka"]["dof_pos"] = {
        "panda_joint1": 0.5,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    modified_state = [modified_env0]

    # Set state only for env 0
    handler.set_states(modified_state, env_ids=[0])

    new_states = handler.get_states(mode="dict")

    # Env 0 should have changed
    assert_close(
        new_states[0]["robots"]["franka"]["pos"],
        torch.tensor([0.1, 0.2, 0.0]),
        atol=1e-3,
        message="env 0 pos after modification",
    )

    # Other envs should remain unchanged (approximately)
    for i in range(1, handler.scenario.num_envs):
        assert_close(
            new_states[i]["robots"]["franka"]["pos"],
            initial_states[i]["robots"]["franka"]["pos"],
            atol=1e-2,
            message=f"env {i} pos should not change",
        )

    log.info(f"Parallel envs partial env_ids test passed for {handler.scenario.simulator}")
