"""Integration tests for DOF (degree of freedom) control via set_dof_targets."""

from __future__ import annotations

from copy import deepcopy

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)


@pytest.fixture(autouse=True)
def reset_robot_to_default(handler, request):
    """Reset robot to default joint positions before each test.

    This ensures test isolation when using session-scoped handler fixtures.
    """
    # Skip for general tests that don't need a handler
    if isinstance(handler, dict) and handler.get("general"):
        return

    # Get current state to preserve object positions
    current_states = handler.get_states(mode="dict")

    # Get the robot's default joint positions from the scenario config
    robot = handler.scenario.robots[0]
    default_positions = robot.default_joint_positions

    # Update only the robot DOF positions, keeping everything else the same
    reset_states = []
    for state in current_states:
        reset_state = {"objects": state.get("objects", {}), "robots": {robot.name: {"dof_pos": default_positions}}}
        reset_states.append(reset_state)

    # Reset the robot state
    handler.set_states(reset_states)

    # Simulate a few steps to stabilize
    for _ in range(10):
        handler.simulate()


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_set_dof_targets_basic(handler):
    """Test basic set_dof_targets functionality."""
    target_positions = {
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

    actions = [{"franka": {"dof_pos_target": target_positions}}] * handler.scenario.num_envs

    handler.set_dof_targets(actions)

    # Simulate to let PD controller apply targets
    for _ in range(50):
        handler.simulate()

    states = handler.get_states(mode="dict")

    # Verify joints moved towards targets (within tolerance due to PD control)
    for joint_name, target_value in target_positions.items():
        actual_value = states[0]["robots"]["franka"]["dof_pos"][joint_name]
        assert abs(actual_value - target_value) < 0.05, (
            f"Joint {joint_name}: expected ~{target_value}, got {actual_value}"
        )

    log.info(f"Set dof targets basic test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_set_dof_targets_within_limits(handler):
    """Test that dof targets respect joint limits."""
    robot = handler.scenario.robots[0]

    # Get joint limits
    joint_limits = robot.joint_limits

    # Generate random targets within limits
    random_targets = {}
    for joint_name, (lower, upper) in joint_limits.items():
        random_value = torch.rand(1).item() * (upper - lower) + lower
        random_targets[joint_name] = random_value

    actions = [{"franka": {"dof_pos_target": random_targets}}] * handler.scenario.num_envs

    handler.set_dof_targets(actions)

    # Simulate
    for _ in range(50):
        handler.simulate()

    states = handler.get_states(mode="dict")

    # Verify all joints are within limits
    for joint_name, (lower, upper) in joint_limits.items():
        actual_value = states[0]["robots"]["franka"]["dof_pos"][joint_name]
        assert lower - 0.01 <= actual_value <= upper + 0.01, (
            f"Joint {joint_name} out of bounds: {actual_value} not in [{lower}, {upper}]"
        )

    log.info(f"Set dof targets within limits test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_set_dof_targets_sequential_changes(handler):
    """Test applying sequential dof targets."""
    # First target
    target1 = {
        "panda_joint1": 0.2,
        "panda_joint2": -0.6,
        "panda_joint3": 0.1,
        "panda_joint4": -2.2,
        "panda_joint5": 0.0,
        "panda_joint6": 1.6,
        "panda_joint7": 0.7,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    handler.set_dof_targets([{"franka": {"dof_pos_target": target1}}] * handler.scenario.num_envs)

    for _ in range(30):
        handler.simulate()

    state1 = handler.get_states(mode="dict")
    joint1_value1 = state1[0]["robots"]["franka"]["dof_pos"]["panda_joint1"]

    # Second target (different from first)
    target2 = {
        "panda_joint1": 0.6,
        "panda_joint2": -0.4,
        "panda_joint3": 0.3,
        "panda_joint4": -2.0,
        "panda_joint5": 0.2,
        "panda_joint6": 1.4,
        "panda_joint7": 0.9,
        "panda_finger_joint1": 0.02,
        "panda_finger_joint2": 0.02,
    }

    handler.set_dof_targets([{"franka": {"dof_pos_target": target2}}] * handler.scenario.num_envs)

    for _ in range(30):
        handler.simulate()

    state2 = handler.get_states(mode="dict")
    joint1_value2 = state2[0]["robots"]["franka"]["dof_pos"]["panda_joint1"]

    # Verify joint moved from first to second target
    assert abs(joint1_value1 - 0.2) < abs(joint1_value1 - 0.6), "Joint should be closer to first target initially"
    assert abs(joint1_value2 - 0.6) < abs(joint1_value2 - 0.2), "Joint should be closer to second target after change"

    log.info(f"Set dof targets sequential changes test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_set_dof_targets_per_env(handler):
    """Test setting different dof targets for each environment."""
    if handler.scenario.num_envs < 2:
        pytest.skip("Test requires at least 2 parallel environments")

    # Create different targets for each environment
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

    # Simulate
    for _ in range(50):
        handler.simulate()

    states = handler.get_states(mode="dict")

    # Verify each environment has different joint1 position
    for i in range(handler.scenario.num_envs):
        joint1_value = states[i]["robots"]["franka"]["dof_pos"]["panda_joint1"]
        expected_target = float(i) * 0.2
        assert abs(joint1_value - expected_target) < 0.1, (
            f"Env {i}: joint1 should be near {expected_target}, got {joint1_value}"
        )

    log.info(f"Set dof targets per env test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_dof_targets_gripper_control(handler):
    """Test gripper control via finger joints."""
    # Open gripper
    open_target = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
        "panda_finger_joint1": 0.04,  # Open
        "panda_finger_joint2": 0.04,  # Open
    }

    handler.set_dof_targets([{"franka": {"dof_pos_target": open_target}}] * handler.scenario.num_envs)

    for _ in range(30):
        handler.simulate()

    state_open = handler.get_states(mode="dict")
    finger1_open = deepcopy(state_open[0]["robots"]["franka"]["dof_pos"]["panda_finger_joint1"])

    # Close gripper
    close_target = open_target.copy()
    close_target["panda_finger_joint1"] = 0.01  # Close
    close_target["panda_finger_joint2"] = 0.01  # Close

    handler.set_dof_targets([{"franka": {"dof_pos_target": close_target}}] * handler.scenario.num_envs)

    for _ in range(30):
        handler.simulate()

    state_close = handler.get_states(mode="dict")
    finger1_close = state_close[0]["robots"]["franka"]["dof_pos"]["panda_finger_joint1"]

    # Verify gripper opened and closed
    assert finger1_open > finger1_close, f"Gripper should be more open initially: {finger1_open} > {finger1_close}"

    log.info(f"Dof targets gripper control test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym")
def test_dof_convergence_to_target(handler):
    """Test that DOF positions converge to targets over time."""
    target_positions = {
        "panda_joint1": 0.4,
        "panda_joint2": -0.5,
        "panda_joint3": 0.2,
        "panda_joint4": -2.1,
        "panda_joint5": 0.15,
        "panda_joint6": 1.55,
        "panda_joint7": 0.82,
        "panda_finger_joint1": 0.03,
        "panda_finger_joint2": 0.03,
    }

    handler.set_dof_targets([{"franka": {"dof_pos_target": target_positions}}] * handler.scenario.num_envs)

    # Track error over time
    errors_at_step_2 = []
    errors_at_step_9 = []

    for step in range(10):
        handler.simulate()

        if step == 2:
            state = handler.get_states(mode="dict")
            for joint_name, target in target_positions.items():
                actual = state[0]["robots"]["franka"]["dof_pos"][joint_name]
                errors_at_step_2.append(abs(actual - target))

        if step == 9:
            state = handler.get_states(mode="dict")
            for joint_name, target in target_positions.items():
                actual = state[0]["robots"]["franka"]["dof_pos"][joint_name]
                errors_at_step_9.append(abs(actual - target))

    # Error should decrease over time
    avg_error_2 = sum(errors_at_step_2) / len(errors_at_step_2)
    avg_error_9 = sum(errors_at_step_9) / len(errors_at_step_9)

    assert avg_error_9 < avg_error_2 or avg_error_2 < 1e-3, (
        f"Error should decrease: step 2 error {avg_error_2} > step 9 error {avg_error_9}"
    )
    log.info(f"Dof convergence to target test passed for {handler.scenario.simulator}")
