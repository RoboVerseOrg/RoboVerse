"""``select_envs`` and ``state_to_device`` keep a ``TensorState`` consistent when rows or devices change.

Both are used by the hybrid handler and the env-subset path: a wrong row selection or a tensor left on
the wrong device surfaces later as an unhelpful backend error, so the contract is pinned here.
"""

from __future__ import annotations

import pytest
import torch

from metasim.utils.state import TensorState, select_envs, state_to_device

pytestmark = pytest.mark.general


def _state(n: int = 3) -> TensorState:
    from metasim.utils.state import ObjectState, RobotState

    root = torch.arange(n * 13, dtype=torch.float32).reshape(n, 13)
    return TensorState(
        objects={"cube": ObjectState(root_state=root.clone())},
        robots={
            "franka": RobotState(
                root_state=root.clone(),
                body_names=["a", "b"],
                joint_pos=torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
            )
        },
        cameras={},
        extras={"step": torch.arange(n)},
    )


def test_select_envs_takes_rows_and_shares_names():
    s = _state(3)
    sub = select_envs(s, [2, 0])
    assert torch.equal(sub.objects["cube"].root_state, s.objects["cube"].root_state[[2, 0]])
    assert torch.equal(sub.robots["franka"].joint_pos, s.robots["franka"].joint_pos[[2, 0]])
    assert sub.robots["franka"].body_names is s.robots["franka"].body_names
    assert sub.robots["franka"].root_state.shape[0] == 2
    # the source is untouched
    assert s.objects["cube"].root_state.shape[0] == 3


def test_state_to_device_moves_every_tensor_and_keeps_values():
    s = _state(2)
    target = "cuda" if torch.cuda.is_available() else "cpu"
    moved = state_to_device(s, target)
    for holder in (moved.objects["cube"], moved.robots["franka"]):
        assert holder.root_state.device.type == target
    assert moved.robots["franka"].joint_pos.device.type == target
    assert moved.extras["step"].device.type == target
    assert torch.equal(moved.robots["franka"].joint_pos.cpu(), s.robots["franka"].joint_pos)
    assert moved.robots["franka"].body_names == ["a", "b"]
    # non-tensor fields and None tensors pass through
    assert moved.objects["cube"].joint_pos is None


def test_state_to_device_is_noop_on_same_device():
    s = _state(2)
    same = state_to_device(s, s.objects["cube"].root_state.device)
    assert same.objects["cube"].root_state is s.objects["cube"].root_state
