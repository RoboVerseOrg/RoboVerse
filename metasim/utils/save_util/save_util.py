"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import torch

from .save_util_v1 import is_v1_demo, save_demo_v1
from .save_util_v2 import is_v2_demo, save_demo_v2


def save_demo(save_dir: str, demo: list[dict[str, torch.Tensor]]):
    """Save a demo to a directory.

    Args:
        save_dir: The directory to save the demo.
        demo: The demo to save.
    """
    if is_v1_demo(demo):
        save_demo_v1(save_dir, demo)
    elif is_v2_demo(demo):
        save_demo_v2(save_dir, demo)
    else:
        raise ValueError("Unknown demo format")
