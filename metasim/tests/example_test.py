"""Example test file to demonstrate test structure and best practices.

This file shows how to write tests for the Metasim module.
It's not meant to be run, but serves as a template for new tests.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from metasim.cfg.robots import BaseRobotCfg
from metasim.utils.state import TensorState


class TestExampleUnit:
    """Example unit tests for individual components."""

    def test_simple_assertion(self):
        """Test basic assertions."""
        assert 2 + 2 == 4
        assert "metasim" in "metasim is great"
        assert [1, 2, 3] == [1, 2, 3]

    def test_with_fixture(self, tmp_path):
        """Test using pytest fixtures."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

    @pytest.mark.parametrize(
        "input,expected",
        [
            (1, 2),
            (2, 4),
            (3, 6),
            (4, 8),
        ],
    )
    def test_parameterized(self, input, expected):
        """Test with multiple input/output pairs."""
        result = input * 2
        assert result == expected

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])

        np.testing.assert_array_equal(arr1, arr2)
        np.testing.assert_allclose(arr1 * 0.1, [0.1, 0.2, 0.3])

    def test_torch_tensors(self):
        """Test with PyTorch tensors."""
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.0, 2.0, 3.0])

        assert torch.allclose(tensor1, tensor2)
        assert torch.equal(tensor1, tensor2)

    def test_with_mock(self):
        """Test using mocks."""
        mock_robot = Mock(spec=BaseRobotCfg)
        mock_robot.name = "test_robot"
        mock_robot.get_joint_names.return_value = ["joint1", "joint2"]

        assert mock_robot.name == "test_robot"
        assert len(mock_robot.get_joint_names()) == 2
        mock_robot.get_joint_names.assert_called_once()

    def test_exception_handling(self):
        """Test that exceptions are raised correctly."""
        with pytest.raises(ValueError, match="Invalid value"):
            raise ValueError("Invalid value: -1")

        with pytest.raises(TypeError):
            "string" + 123


class TestExampleIntegration:
    """Example integration tests for multiple components."""

    @pytest.mark.integration
    def test_component_interaction(self):
        """Test interaction between multiple components."""
        state = TensorState()
        state.joint_pos = torch.zeros(7)
        state.joint_vel = torch.zeros(7)

        assert hasattr(state, "joint_pos")
        assert state.joint_pos.shape == (7,)

    @pytest.mark.slow
    def test_long_running_operation(self):
        """Test that takes significant time."""
        import time

        start = time.time()
        time.sleep(0.1)
        duration = time.time() - start
        assert duration >= 0.1

    @pytest.mark.gpu
    def test_gpu_required(self):
        """Test that requires GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        assert tensor.is_cuda


class TestExampleFixtures:
    """Example of custom fixtures."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data."""
        return {
            "robot_name": "franka",
            "num_joints": 7,
            "joint_positions": [0.0] * 7,
        }

    @pytest.fixture
    def mock_simulator(self):
        """Fixture providing a mock simulator."""
        sim = Mock()
        sim.num_envs = 4
        sim.reset.return_value = Mock(joint_pos=torch.zeros(4, 7))
        return sim

    def test_using_fixtures(self, sample_data, mock_simulator):
        """Test using custom fixtures."""
        assert sample_data["robot_name"] == "franka"
        assert sample_data["num_joints"] == 7

        obs = mock_simulator.reset()
        assert obs.joint_pos.shape == (4, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExampleConditional:
    """Example of conditional test execution."""

    def test_cuda_operations(self):
        """Test CUDA operations."""
        device = torch.device("cuda")
        tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(tensor, tensor.T)
        assert result.shape == (100, 100)
        assert result.device.type == "cuda"


def test_module_level():
    """Example of module-level test function."""
    assert True


@pytest.fixture(scope="module")
def module_fixture():
    """Fixture with module scope (shared across tests in module)."""
    print("Setting up module fixture")
    yield {"shared": "data"}
    print("Tearing down module fixture")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
