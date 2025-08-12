from __future__ import annotations

from typing import Literal

import torch


def position_shift_success(
    current_pos: torch.Tensor,
    initial_pos: torch.Tensor,
    axis: Literal["x", "y", "z"],
    distance: float,
) -> torch.BoolTensor:
    """Pure function: success if object's position shift along axis reaches threshold.

    Args:
        current_pos: (N, 3) current positions.
        initial_pos: (N, 3) baseline positions captured at reset.
        axis: one of "x", "y", "z".
        distance: threshold (m). Positive => +axis, Negative => -axis.
    Returns:
        (N,) bool tensor mask.
    """
    dim = {"x": 0, "y": 1, "z": 2}[axis]
    delta = current_pos[:, dim] - initial_pos[:, dim]
    if distance >= 0:
        return delta >= distance
    else:
        return delta <= distance
