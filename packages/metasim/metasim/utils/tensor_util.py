"""Tensor utilities."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_str(arr: torch.Tensor):
    """Convert a list or 1D array of float to a string with 2 decimal places."""
    if isinstance(arr, list) or (isinstance(arr, torch.Tensor) and len(arr.shape) == 1):
        return "[" + ", ".join([f"{e:.2f}" for e in arr]) + "]"
    elif isinstance(arr, torch.Tensor) and len(arr.shape) == 2:
        return "[\n  " + "\n  ".join([tensor_to_str(e) for e in arr]) + "\n]"
    else:
        return str(arr)


def tensor_to_cpu(data: dict | list) -> dict | list:
    """Convert all tensors in a nested dictionary or list to CPU tensors."""
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.cpu() if isinstance(v, torch.Tensor) else tensor_to_cpu(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            data[i] = v.cpu() if isinstance(v, torch.Tensor) else tensor_to_cpu(v)
    return data


def array_to_tensor(data, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert a NumPy array or list to a PyTorch tensor on the specified device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, list):
        tensor = torch.tensor(data, dtype=torch.float32)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        raise ValueError("Input data must be a list, NumPy array, or PyTorch tensor.")
    if device is not None:
        tensor = tensor.to(device)
    return tensor
