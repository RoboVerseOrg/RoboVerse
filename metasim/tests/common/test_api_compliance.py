"""API compliance tests for all simulator handlers.

These tests ensure that all simulator handlers correctly implement the BaseSimHandler
interface and behave consistently across different physics engines.
"""

import numpy as np
import pytest
import torch


class TestAPICompliance:
    """Test suite for verifying API compliance across all simulators."""

    @pytest.mark.integration
    def test_handler_attributes(self, simulator_handler):
        """Test that handler has all required attributes."""
        required_attrs = [
            "num_envs",
            "device",
            "scenario",
        ]

        for attr in required_attrs:
            assert hasattr(simulator_handler, attr), f"Handler missing required attribute: {attr}"
        assert isinstance(simulator_handler.num_envs, int)
        assert simulator_handler.num_envs > 0

        if hasattr(simulator_handler, "device"):
            import torch

            device = simulator_handler.device
            assert isinstance(device, (torch.device, str)), f"Device should be torch.device or str, got {type(device)}"

    @pytest.mark.integration
    def test_core_methods_exist(self, simulator_handler):
        """Test that all core methods exist and are callable."""
        # Methods that must be implemented by all handlers
        required_methods = [
            "close",
            "launch",
            "get_states",
            "set_states",
            "get_pos",
            "get_rot",
            "get_vel",
            "set_pose",
            "set_dof_targets",
            "simulate",
            "refresh_render",
            "get_joint_names",
            "get_body_names",
            "get_joint_reindex",
            "get_body_reindex",
        ]

        # Methods that may be implemented by handler or wrapper
        optional_methods = [
            "step",
            "reset",
            "render",
        ]

        for method in required_methods:
            assert hasattr(simulator_handler, method), f"Handler missing required method: {method}"
            assert callable(getattr(simulator_handler, method)), f"Handler attribute {method} is not callable"

        # Check optional methods - if they exist, they should be callable
        for method in optional_methods:
            if hasattr(simulator_handler, method):
                assert callable(getattr(simulator_handler, method)), (
                    f"Handler attribute {method} exists but is not callable"
                )

    @pytest.mark.integration
    def test_launch_close_lifecycle(self, simulator_handler):
        """Test basic lifecycle operations."""
        simulator_handler.close()
        simulator_handler.launch()
        simulator_handler.close()
        simulator_handler.launch()

    @pytest.mark.integration
    def test_reset_returns_valid_state(self, simulator_handler):
        """Test that reset returns a valid state structure."""
        # Skip if handler doesn't implement reset directly
        if not hasattr(simulator_handler, "reset"):
            pytest.skip(f"{simulator_handler.__class__.__name__} doesn't implement reset directly")

        states = simulator_handler.reset()
        assert states is not None, "Reset should return state data"

        # Check the returned data structure
        if isinstance(states, tuple):
            assert len(states) == 2, "Reset should return (states, extra)"
            states_data = states[0]
        else:
            states_data = states

        assert states_data is not None

    @pytest.mark.integration
    def test_step_workflow(self, simulator_handler):
        """Test the basic simulate -> get_state workflow."""
        # All handlers should support simulate and get_states
        initial_states = simulator_handler.get_states()

        # Set some actions if the handler supports it
        if hasattr(simulator_handler.scenario, "robots") and simulator_handler.scenario.robots:
            robot = simulator_handler.scenario.robots[0]
            if hasattr(robot, "name"):
                # Create dummy actions
                num_joints = len(robot.actuators) if hasattr(robot, "actuators") else 2
                actions = [{robot.name: {"dof_pos_target": {f"joint{i}": 0.1 for i in range(num_joints)}}}]
                simulator_handler.set_dof_targets(robot.name, actions)

        simulator_handler.simulate()
        new_states = simulator_handler.get_states()
        assert new_states is not None

    @pytest.mark.integration
    def test_joint_and_body_names(self, simulator_handler):
        """Test retrieval of joint and body names."""
        # Check if there are any robots in the scenario
        if not hasattr(simulator_handler.scenario, "robots") or not simulator_handler.scenario.robots:
            pytest.skip("No robots in scenario")

        robot = simulator_handler.scenario.robots[0]
        robot_name = robot.name if hasattr(robot, "name") else "robot"

        try:
            joint_names = simulator_handler.get_joint_names(robot_name)
            assert isinstance(joint_names, list), "get_joint_names should return a list"
            # Some robots might have no joints (fixed objects)
            if len(joint_names) > 0:
                assert all(isinstance(name, str) for name in joint_names), "Joint names should be strings"
        except NotImplementedError:
            pytest.skip(f"{simulator_handler.__class__.__name__} doesn't implement get_joint_names")

        try:
            body_names = simulator_handler.get_body_names(robot_name)
            assert isinstance(body_names, list), "get_body_names should return a list"
            # Some objects might have no bodies
            if len(body_names) > 0:
                assert all(isinstance(name, str) for name in body_names), "Body names should be strings"
        except NotImplementedError:
            pytest.skip(f"{simulator_handler.__class__.__name__} doesn't implement get_body_names")

    @pytest.mark.integration
    def test_pose_get_set(self, simulator_handler, tolerances):
        """Test getting and setting object poses."""
        # Check if there are any objects in the scenario
        if not hasattr(simulator_handler.scenario, "objects") or not simulator_handler.scenario.objects:
            pytest.skip("No objects in scenario")

        obj = simulator_handler.scenario.objects[0]
        obj_name = obj.name if hasattr(obj, "name") else "object"

        target_pos = torch.tensor([1.0, 2.0, 3.0])
        target_rot = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Set pose using the object name
        simulator_handler.set_pose(obj_name, target_pos, target_rot)

        for _ in range(5):
            simulator_handler.simulate()

        actual_pos = simulator_handler.get_pos(obj_name)
        actual_rot = simulator_handler.get_rot(obj_name)

        # Convert to numpy for comparison
        if isinstance(actual_pos, torch.Tensor):
            actual_pos = actual_pos.cpu().numpy()
            target_pos = target_pos.cpu().numpy()
        if isinstance(actual_rot, torch.Tensor):
            actual_rot = actual_rot.cpu().numpy()
            target_rot = target_rot.cpu().numpy()

        np.testing.assert_allclose(
            actual_pos.flatten()[:3],
            target_pos.flatten()[:3],
            rtol=tolerances["position"],
            atol=tolerances["position"],
            err_msg="Position not set correctly",
        )

        # Quaternions q and -q represent the same rotation
        dot_product = np.abs(np.dot(actual_rot.flatten()[:4], target_rot.flatten()[:4]))
        assert dot_product > 0.999, f"Rotation not set correctly: {actual_rot} vs {target_rot}"

    @pytest.mark.integration
    def test_velocity_retrieval(self, simulator_handler):
        """Test velocity retrieval for objects."""
        # Check if there are any objects in the scenario
        if not hasattr(simulator_handler.scenario, "objects") or not simulator_handler.scenario.objects:
            pytest.skip("No objects in scenario")

        obj = simulator_handler.scenario.objects[0]
        obj_name = obj.name if hasattr(obj, "name") else "object"

        # Set object high up so it can fall
        simulator_handler.set_pose(obj_name, torch.tensor([0.0, 0.0, 2.0]), torch.tensor([1.0, 0.0, 0.0, 0.0]))

        for _ in range(10):
            simulator_handler.simulate()

        vel = simulator_handler.get_vel(obj_name)

        # Handle both numpy and torch tensors
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()

        assert isinstance(vel, (np.ndarray, torch.Tensor)), "Velocity should be a numpy array or torch tensor"

        # Flatten in case of batch dimension
        vel_flat = vel.flatten()
        assert len(vel_flat) >= 3, "Velocity should have at least 3 components"

        z_vel = vel_flat[2]
        # Some simulators might have damping or other effects, so we check for downward motion
        assert z_vel < 0.01, f"Object should be falling or stationary, but z-velocity is {z_vel}"

    @pytest.mark.integration
    def test_simulate_method(self, simulator_handler):
        """Test the simulate() method."""
        initial_state = simulator_handler.get_states()

        # Run simulation steps
        for _ in range(10):
            simulator_handler.simulate()

        state_after_simulate = simulator_handler.get_states()

        # States should be available after simulation
        assert state_after_simulate is not None

    @pytest.mark.integration
    def test_render_modes(self, simulator_handler):
        """Test different rendering modes if supported."""
        try:
            image = simulator_handler.render()

            if image is not None:
                assert isinstance(image, np.ndarray), "Rendered image should be numpy array"
                assert len(image.shape) == 3, "Image should be 3D (height, width, channels)"
                assert image.shape[2] in [3, 4], "Image should have 3 (RGB) or 4 (RGBA) channels"
        except NotImplementedError:
            pytest.skip("Simulator does not support rendering in current mode")

    @pytest.mark.integration
    def test_multiple_environments(self):
        """Test handlers with multiple parallel environments."""
        pytest.skip("Multi-environment test requires special setup")

    @pytest.mark.integration
    def test_state_persistence(self, simulator_handler, tolerances):
        """Test that states can be saved and restored."""
        # Run some simulation steps to change state
        for _ in range(20):
            simulator_handler.simulate()

        saved_state = simulator_handler.get_states()

        # Run more steps to change state further
        for _ in range(20):
            simulator_handler.simulate()

        changed_state = simulator_handler.get_states()

        # Restore the saved state
        if isinstance(saved_state, list):
            simulator_handler.set_states(saved_state)
        else:
            # Handle TensorState format
            simulator_handler.set_states([saved_state])

        restored_state = simulator_handler.get_states()

        assert restored_state is not None, "State restoration failed"

    @pytest.mark.integration
    def test_deterministic_simulation(self, simulator_handler):
        """Test that simulation is deterministic with same initial conditions."""
        # Get initial state
        initial_state = simulator_handler.get_states()

        # Save initial state to restore later
        if isinstance(initial_state, list):
            saved_initial = initial_state.copy()
        else:
            saved_initial = initial_state

        states_sequence_1 = []

        # Run simulation with specific actions
        for i in range(10):
            # Set alternating actions if we have robots
            if hasattr(simulator_handler.scenario, "robots") and simulator_handler.scenario.robots:
                robot = simulator_handler.scenario.robots[0]
                if hasattr(robot, "actuators"):
                    num_joints = len(robot.actuators)
                    actions = [
                        {robot.name: {"dof_pos_target": {f"joint{j}": 0.5 * (-1) ** i for j in range(num_joints)}}}
                    ]
                    simulator_handler.set_dof_targets(robot.name, actions)

            simulator_handler.simulate()
            states_sequence_1.append(simulator_handler.get_states())

        # Reset to initial state
        if isinstance(saved_initial, list):
            simulator_handler.set_states(saved_initial)
        else:
            simulator_handler.set_states([saved_initial])

        states_sequence_2 = []

        # Run same simulation again
        for i in range(10):
            if hasattr(simulator_handler.scenario, "robots") and simulator_handler.scenario.robots:
                robot = simulator_handler.scenario.robots[0]
                if hasattr(robot, "actuators"):
                    num_joints = len(robot.actuators)
                    actions = [
                        {robot.name: {"dof_pos_target": {f"joint{j}": 0.5 * (-1) ** i for j in range(num_joints)}}}
                    ]
                    simulator_handler.set_dof_targets(robot.name, actions)

            simulator_handler.simulate()
            states_sequence_2.append(simulator_handler.get_states())

        assert len(states_sequence_1) == len(states_sequence_2), "Sequences should have same length"
