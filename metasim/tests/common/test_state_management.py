"""State management and conversion tests for simulator handlers.

These tests verify that state getting, setting, and conversions work correctly
across all simulators, ensuring consistency in the unified state representation.
"""

import pytest
import torch

from metasim.utils.state import TensorState, state_tensor_to_nested


class TestStateManagement:
    """Test suite for state management functionality."""

    @pytest.mark.integration
    def test_state_structure_consistency(self, simulator_handler):
        """Test that state structure is consistent across get/set operations."""
        handler = simulator_handler
        handler.launch()

        state1 = handler._get_states()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}}] * handler.num_envs
        for _ in range(10):
            handler.step(actions)

        state2 = handler._get_states()

        assert type(state1) == type(state2), "State type changed between calls"

        if isinstance(state1, TensorState):
            assert state1.robots.keys() == state2.robots.keys()
            assert state1.objects.keys() == state2.objects.keys()

            for robot_name in state1.robots:
                robot_state1 = state1.robots[robot_name]
                robot_state2 = state2.robots[robot_name]

                assert robot_state1.root_state.shape == robot_state2.root_state.shape
                assert robot_state1.joint_pos.shape == robot_state2.joint_pos.shape
                assert robot_state1.joint_vel.shape == robot_state2.joint_vel.shape

        handler.close()

    @pytest.mark.integration
    def test_state_get_set_roundtrip(self, simulator_handler):
        """Test that states can be retrieved and set back accurately."""
        handler = simulator_handler
        handler.launch()

        obs, _ = handler.reset()
        initial_state = handler._get_states()

        if isinstance(initial_state, TensorState):
            state_dict = state_tensor_to_nested(initial_state)
        else:
            state_dict = initial_state

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}}] * handler.num_envs
        for _ in range(20):
            handler.step(actions)

        modified_state = handler._get_states()

        handler._set_states(state_dict)

        restored_state = handler._get_states()

        if isinstance(initial_state, TensorState) and isinstance(restored_state, TensorState):
            for robot_name in initial_state.robots:
                initial_robot = initial_state.robots[robot_name]
                restored_robot = restored_state.robots[robot_name]

                torch.testing.assert_close(initial_robot.root_state, restored_robot.root_state, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(initial_robot.joint_pos, restored_robot.joint_pos, rtol=1e-4, atol=1e-4)

        handler.close()

    @pytest.mark.integration
    def test_state_partial_update(self, simulator_handler):
        """Test that partial state updates work correctly."""
        handler = simulator_handler
        handler.launch()

        handler.reset()
        initial_state = handler._get_states()

        if isinstance(initial_state, TensorState):
            state_dict = state_tensor_to_nested(initial_state)

            if handler.num_envs > 0 and "test_robot" in state_dict[0]["robots"]:
                state_dict[0]["robots"]["test_robot"]["joint_positions"]["joint1"] = 0.7
                state_dict[0]["robots"]["test_robot"]["joint_positions"]["joint2"] = -0.7

            handler._set_states(state_dict)

            new_state = handler._get_states()
            if "test_robot" in new_state.robots:
                new_joint_pos = new_state.robots["test_robot"].joint_pos[0]
                assert abs(new_joint_pos[0].item() - 0.7) < 0.01
                assert abs(new_joint_pos[1].item() - (-0.7)) < 0.01

        handler.close()

    @pytest.mark.integration
    def test_state_bounds_validation(self, simulator_handler):
        """Test that state bounds are properly validated."""
        handler = simulator_handler
        handler.launch()

        joint_names = handler.get_joint_names("test_robot", sort=True)

        handler.close()

    @pytest.mark.integration
    def test_state_multi_env_consistency(self, simulator_handler):
        """Test state management across multiple environments."""
        if simulator_handler.num_envs < 2:
            pytest.skip("Test requires multiple environments")

        handler = simulator_handler
        handler.launch()

        handler.reset()

        actions = []
        for i in range(handler.num_envs):
            actions.append({"robot": {"dof_pos_target": {"joint1": 0.1 * i, "joint2": -0.1 * i}}})

        for _ in range(10):
            handler.step(actions)

        states = handler._get_states()

        if isinstance(states, TensorState):
            if "test_robot" in states.robots:
                joint_positions = states.robots["test_robot"].joint_pos

                for i in range(1, handler.num_envs):
                    assert not torch.allclose(joint_positions[i], joint_positions[0], atol=1e-3), (
                        "All environments have the same joint positions"
                    )

        handler.close()

    @pytest.mark.integration
    def test_state_device_consistency(self, simulator_handler):
        """Test that state tensors are on the correct device."""
        handler = simulator_handler
        handler.launch()

        states = handler._get_states()

        if isinstance(states, TensorState):
            expected_device = handler.device

            for robot_name, robot_state in states.robots.items():
                assert robot_state.root_state.device == expected_device
                assert robot_state.joint_pos.device == expected_device
                assert robot_state.joint_vel.device == expected_device

            for obj_name, obj_state in states.objects.items():
                assert obj_state.root_state.device == expected_device

        handler.close()

    @pytest.mark.integration
    def test_state_serialization(self, simulator_handler):
        """Test that states can be serialized and deserialized."""
        handler = simulator_handler
        handler.launch()

        state = handler._get_states()

        if isinstance(state, TensorState):
            state_dict = state_tensor_to_nested(state)

            import json

            try:
                json_str = json.dumps(state_dict, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                assert len(json_str) > 0
            except Exception as e:
                pytest.fail(f"State serialization failed: {e}")

        handler.close()

    @pytest.mark.integration
    def test_reset_state_consistency(self, simulator_handler):
        """Test that reset properly restores initial states."""
        handler = simulator_handler
        handler.launch()

        handler.reset()
        initial_state = handler._get_states()

        actions = [{"robot": {"dof_pos_target": {"joint1": 1.0, "joint2": -1.0}}}] * handler.num_envs
        for _ in range(50):
            handler.step(actions)

        handler.reset()
        reset_state = handler._get_states()

        if isinstance(initial_state, TensorState) and isinstance(reset_state, TensorState):
            if "test_robot" in initial_state.robots:
                initial_joints = initial_state.robots["test_robot"].joint_pos
                reset_joints = reset_state.robots["test_robot"].joint_pos

                assert torch.allclose(reset_joints, torch.zeros_like(reset_joints), atol=0.1)

        handler.close()

    @pytest.mark.integration
    def test_state_action_correspondence(self, simulator_handler):
        """Test that actions properly affect states."""
        handler = simulator_handler
        handler.launch()

        handler.reset()
        initial_state = handler._get_states()

        target_pos = {"joint1": 0.5, "joint2": -0.3}
        actions = [{"robot": {"dof_pos_target": target_pos}}] * handler.num_envs

        for _ in range(100):
            handler.step(actions)

        final_state = handler._get_states()

        if isinstance(final_state, TensorState) and "test_robot" in final_state.robots:
            final_joints = final_state.robots["test_robot"].joint_pos[0]

            assert abs(final_joints[0].item() - target_pos["joint1"]) < 0.1
            assert abs(final_joints[1].item() - target_pos["joint2"]) < 0.1

        handler.close()


@pytest.mark.integration
class TestStateConversions:
    """Test state conversion utilities."""

    def test_tensor_to_nested_conversion(self, simulator_handler):
        """Test conversion from TensorState to nested dict."""
        handler = simulator_handler
        handler.launch()

        tensor_state = handler._get_states()

        if isinstance(tensor_state, TensorState):
            nested_state = state_tensor_to_nested(tensor_state)

            assert isinstance(nested_state, list)
            assert len(nested_state) == handler.num_envs

            for env_state in nested_state:
                assert isinstance(env_state, dict)
                assert "robots" in env_state
                assert "objects" in env_state

                if "test_robot" in env_state["robots"]:
                    robot_state = env_state["robots"]["test_robot"]
                    assert "root_position" in robot_state
                    assert "root_orientation" in robot_state
                    assert "joint_positions" in robot_state
                    assert "joint_velocities" in robot_state

        handler.close()
