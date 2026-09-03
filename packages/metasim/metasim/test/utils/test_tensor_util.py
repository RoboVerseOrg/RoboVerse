"""Tests for metasim.utils.tensor_util module.

These are pure unit tests for tensor utility functions.
All tests are marked @pytest.mark.general.
"""

import numpy as np
import pytest
import torch

from metasim.utils.tensor_util import array_to_tensor, tensor_to_cpu, tensor_to_str


@pytest.mark.general
def test_tensor_to_str_1d():
    """Test tensor to string conversion for 1D tensors."""
    arr = torch.tensor([1.234, 2.567, 3.891])
    result = tensor_to_str(arr)

    assert result == "[1.23, 2.57, 3.89]"


@pytest.mark.general
def test_tensor_to_str_list():
    """Test tensor to string conversion for Python lists."""
    arr = [1.234, 2.567, 3.891]
    result = tensor_to_str(arr)

    assert result == "[1.23, 2.57, 3.89]"


@pytest.mark.general
def test_tensor_to_str_2d():
    """Test tensor to string conversion for 2D tensors."""
    arr = torch.tensor([[1.1, 2.2], [3.3, 4.4]])
    result = tensor_to_str(arr)

    # Should have nested structure
    assert "[" in result
    assert "1.10" in result or "1.1" in result
    assert "4.40" in result or "4.4" in result


@pytest.mark.general
def test_tensor_to_cpu_dict():
    """Test moving tensors in dictionary to CPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = {
        "a": torch.tensor([1.0, 2.0]).cuda(),
        "b": torch.tensor([3.0, 4.0]).cuda(),
        "nested": {
            "c": torch.tensor([5.0]).cuda(),
        },
    }

    result = tensor_to_cpu(data)

    assert result["a"].device.type == "cpu"
    assert result["b"].device.type == "cpu"
    assert result["nested"]["c"].device.type == "cpu"

    # Values should be preserved
    assert torch.allclose(result["a"], torch.tensor([1.0, 2.0]))


@pytest.mark.general
def test_tensor_to_cpu_list():
    """Test moving tensors in list to CPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = [
        torch.tensor([1.0, 2.0]).cuda(),
        torch.tensor([3.0, 4.0]).cuda(),
    ]

    result = tensor_to_cpu(data)

    assert result[0].device.type == "cpu"
    assert result[1].device.type == "cpu"

    # Values should be preserved
    assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))


@pytest.mark.general
def test_array_to_tensor_from_list():
    """Test converting list to tensor."""
    data = [1.0, 2.0, 3.0]
    result = array_to_tensor(data)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.general
def test_array_to_tensor_from_numpy():
    """Test converting numpy array to tensor."""
    data = np.array([1.0, 2.0, 3.0])
    result = array_to_tensor(data)

    assert isinstance(result, torch.Tensor)
    # Compare values, handling potential dtype differences
    assert torch.allclose(result.float(), torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.general
def test_array_to_tensor_from_tensor():
    """Test that tensor input is returned as tensor."""
    data = torch.tensor([1.0, 2.0, 3.0])
    result = array_to_tensor(data)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, data)


@pytest.mark.general
def test_array_to_tensor_with_device():
    """Test converting with device specification."""
    data = [1.0, 2.0, 3.0]
    result = array_to_tensor(data, device="cpu")

    assert result.device.type == "cpu"
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.general
def test_array_to_tensor_invalid_type():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="Input data must be"):
        array_to_tensor("invalid")

    with pytest.raises(ValueError, match="Input data must be"):
        array_to_tensor(123)


@pytest.mark.general
def test_array_to_tensor_2d_list():
    """Test converting 2D list to tensor."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = array_to_tensor(data)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.general
def test_array_to_tensor_2d_numpy():
    """Test converting 2D numpy array to tensor."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = array_to_tensor(data)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    # Compare values, handling potential dtype differences
    assert torch.allclose(result.float(), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
