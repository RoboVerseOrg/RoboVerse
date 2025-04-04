"""Tensor utilities."""

import torch


def tensor_to_str(arr: torch.Tensor):
    """Convert a list or 1D array of float to a string with 2 decimal places."""
    if isinstance(arr, list) or (isinstance(arr, torch.Tensor) and len(arr.shape) == 1):
        return "[" + ", ".join([f"{e:.2f}" for e in arr]) + "]"
    elif isinstance(arr, torch.Tensor) and len(arr.shape) == 2:
        return "[\n  " + "\n  ".join([tensor_to_str(e) for e in arr]) + "\n]"
    else:
        return str(arr)
