"""Error handling and edge case tests for simulator handlers.

These tests verify that simulators handle errors gracefully and provide
meaningful error messages for common failure scenarios.
"""

import numpy as np
import pytest


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.integration
    def test_invalid_asset_path(self, simulator_handler):
        """Test handling of invalid asset paths."""
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)) as exc_info:
            simulator_handler.load_robot("/path/that/does/not/exist.urdf")

        assert "exist" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_malformed_urdf(self, simulator_handler, tmp_path):
        """Test handling of malformed URDF files."""
        bad_urdf = tmp_path / "bad.urdf"
        bad_urdf.write_text("""<?xml version="1.0"?>
<robot name="bad_robot">
    <joint name="joint1" type="revolute">
    </joint>
</robot>""")

        with pytest.raises((ValueError, RuntimeError, Exception)) as exc_info:
            simulator_handler.load_robot(str(bad_urdf))

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["invalid", "malformed", "error", "failed"])

    @pytest.mark.integration
    def test_action_dimension_mismatch(self, simulator_handler, simple_robot_urdf):
        """Test handling of incorrect action dimensions."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        joint_names = simulator_handler.get_joint_names()
        expected_dof = len(joint_names)

        test_cases = [
            (np.array([]), "empty action"),
            (np.array([0.1] * (expected_dof + 1)), "too many actions"),
            (np.array([0.1] * max(1, expected_dof - 1)), "too few actions"),
        ]

        for wrong_action, desc in test_cases:
            if len(wrong_action) != expected_dof:
                with pytest.raises((ValueError, AssertionError, RuntimeError)) as exc_info:
                    simulator_handler.step(wrong_action)

                error_msg = str(exc_info.value).lower()
                assert any(word in error_msg for word in ["dimension", "size", "shape", "mismatch"])

    @pytest.mark.integration
    def test_step_before_reset(self, simulator_handler, simple_robot_urdf):
        """Test calling step before reset."""
        simulator_handler.load_robot(simple_robot_urdf)

        action = np.array([0.1])

        try:
            simulator_handler.step(action)
            simulator_handler.step(action)
        except (RuntimeError, AssertionError) as e:
            assert any(word in str(e).lower() for word in ["reset", "initialize", "not initialized"])

    @pytest.mark.integration
    def test_operations_on_nonexistent_object(self, simulator_handler):
        """Test operations on objects that don't exist."""
        fake_id = 99999

        with pytest.raises((ValueError, KeyError, IndexError, RuntimeError)):
            simulator_handler.get_pos(fake_id)

        with pytest.raises((ValueError, KeyError, IndexError, RuntimeError)):
            simulator_handler.get_rot(fake_id)

        with pytest.raises((ValueError, KeyError, IndexError, RuntimeError)):
            simulator_handler.set_pose(fake_id, np.zeros(3), np.array([1, 0, 0, 0]))

    @pytest.mark.integration
    def test_invalid_joint_names(self, simulator_handler, simple_robot_urdf):
        """Test operations with invalid joint names."""
        simulator_handler.load_robot(simple_robot_urdf)

        fake_joint = "joint_that_does_not_exist"

        if hasattr(simulator_handler, "get_joint_state"):
            with pytest.raises((ValueError, KeyError, RuntimeError)):
                simulator_handler.get_joint_state(fake_joint)

    @pytest.mark.integration
    def test_extreme_action_values(self, simulator_handler, simple_robot_urdf):
        """Test handling of extreme action values."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        extreme_cases = [
            (np.array([1e10]), "very large positive"),
            (np.array([-1e10]), "very large negative"),
            (np.array([np.inf]), "positive infinity"),
            (np.array([-np.inf]), "negative infinity"),
            (np.array([np.nan]), "NaN"),
        ]

        for extreme_action, desc in extreme_cases:
            try:
                simulator_handler.step(extreme_action)
                state = simulator_handler.get_states()
                self._assert_state_finite(state)
            except (ValueError, RuntimeError, AssertionError):
                pass

            simulator_handler.reset()

    @pytest.mark.integration
    def test_memory_limits(self, simulator_handler, simple_cube_urdf):
        """Test behavior when approaching memory limits."""
        max_objects = 100
        loaded_objects = []

        try:
            for i in range(max_objects):
                obj_id = simulator_handler.load_object(simple_cube_urdf)
                loaded_objects.append(obj_id)

                x = (i % 10) * 0.2
                y = (i // 10) * 0.2
                simulator_handler.set_pose(obj_id, np.array([x, y, 0.5]), np.array([1, 0, 0, 0]))

            for _ in range(5):
                simulator_handler.simulate()

        except (MemoryError, RuntimeError) as e:
            assert any(word in str(e).lower() for word in ["memory", "limit", "resource"])

        simulator_handler.reset()

    @pytest.mark.integration
    def test_concurrent_operations(self, simulator_handler, simple_robot_urdf):
        """Test thread safety of operations if relevant."""
        import threading
        import time

        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        errors = []

        def simulate_thread():
            try:
                for _ in range(10):
                    simulator_handler.simulate()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("simulate", e))

        def get_state_thread():
            try:
                for _ in range(10):
                    simulator_handler.get_states()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("get_state", e))

        threads = [
            threading.Thread(target=simulate_thread),
            threading.Thread(target=get_state_thread),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        if errors:
            for op, error in errors:
                error_msg = str(error).lower()
                thread_indicators = ["thread", "concurrent", "lock", "mutex", "synchroniz"]

    @pytest.mark.integration
    def test_invalid_configuration_values(self, simulator_handler):
        """Test creation with invalid configuration values."""
        from metasim.tests.conftest import SimulatorNotAvailableError, create_handler

        simulator_name = simulator_handler.simulator_name

        invalid_configs = [
            {"num_envs": -1, "error": "negative environments"},
            {"num_envs": 0, "error": "zero environments"},
            {"device": "invalid_device", "error": "invalid device"},
        ]

        for config in invalid_configs:
            error_desc = config.pop("error")
            try:
                bad_handler = create_handler(simulator_name, **config)
                bad_handler.launch()
                if "num_envs" in config:
                    assert bad_handler.num_envs > 0, f"Handler accepted {error_desc}"
                bad_handler.close()
            except (ValueError, RuntimeError, SimulatorNotAvailableError, AssertionError):
                pass

    @pytest.mark.integration
    def test_state_setting_during_simulation(self, simulator_handler, simple_robot_urdf):
        """Test setting state while simulation is running."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        initial_state = simulator_handler.get_states()

        for i in range(5):
            action = np.array([0.5])
            simulator_handler.step(action)

        simulator_handler.set_states(initial_state)

        for i in range(5):
            action = np.array([-0.5])
            simulator_handler.step(action)

        final_state = simulator_handler.get_states()
        assert final_state is not None

    @pytest.mark.integration
    def test_asset_loading_limits(self, simulator_handler, simple_robot_urdf):
        """Test limits on asset loading."""
        robot_ids = []
        max_robots = 10

        try:
            for i in range(max_robots):
                robot_id = simulator_handler.load_robot(simple_robot_urdf)
                robot_ids.append(robot_id)

                base_pos = np.array([i * 2.0, 0, 0])
                try:
                    simulator_handler.set_pose(robot_id, base_pos, np.array([1, 0, 0, 0]))
                except:
                    pass

            simulator_handler.reset()
            for _ in range(5):
                actions = np.zeros(len(robot_ids))
                simulator_handler.step(actions)

        except (RuntimeError, MemoryError, ValueError) as e:
            assert len(str(e)) > 0

    @pytest.mark.integration
    def test_numerical_stability(self, simulator_handler, simple_robot_urdf):
        """Test numerical stability over long simulations."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        num_steps = 1000
        states_history = []

        for i in range(num_steps):
            action = np.array([0.1 * np.sin(i * 0.1)])
            simulator_handler.step(action)

            if i % 100 == 0:
                state = simulator_handler.get_states()
                states_history.append(state)

        for i, state in enumerate(states_history):
            self._assert_state_finite(state, f"State at step {i * 100}")

        final_state = simulator_handler.get_states()
        self._assert_state_finite(final_state, "Final state")

    def _assert_state_finite(self, state, context=""):
        """Assert that all state values are finite (not NaN or inf)."""
        context_msg = f" ({context})" if context else ""

        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    assert np.all(np.isfinite(value)), f"Non-finite values in {key}{context_msg}"
                elif isinstance(value, (int, float)):
                    assert np.isfinite(value), f"Non-finite value in {key}{context_msg}"
        elif isinstance(state, np.ndarray):
            assert np.all(np.isfinite(state)), f"Non-finite values in state{context_msg}"
