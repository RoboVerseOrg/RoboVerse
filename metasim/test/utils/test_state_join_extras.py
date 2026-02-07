"""Tests for joining TensorState extras in parallel mode."""

import pytest
import torch

from metasim.queries.contact_force import ContactForcesData
from metasim.types import RobotState, TensorState
from metasim.utils.state import join_tensor_states


def _make_robot_state() -> RobotState:
    root = torch.zeros((1, 13), dtype=torch.float32)
    joint = torch.zeros((1, 0), dtype=torch.float32)
    return RobotState(
        root_state=root,
        body_names=[],
        body_state=None,
        joint_pos=joint,
        joint_vel=joint,
        joint_pos_target=None,
        joint_vel_target=None,
        joint_effort_target=None,
    )


@pytest.mark.general
def test_join_tensor_states_merges_dataclass_extras():
    extra0 = ContactForcesData(
        contact_forces_history=torch.zeros((1, 3, 2, 3), dtype=torch.float32),
        contact_forces=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    extra1 = ContactForcesData(
        contact_forces_history=torch.ones((1, 3, 2, 3), dtype=torch.float32),
        contact_forces=torch.ones((1, 2, 3), dtype=torch.float32),
    )

    s0 = TensorState(
        objects={}, robots={"robot": _make_robot_state()}, cameras={}, extras={"contact_forces": {"robot": extra0}}
    )
    s1 = TensorState(
        objects={}, robots={"robot": _make_robot_state()}, cameras={}, extras={"contact_forces": {"robot": extra1}}
    )

    joined = join_tensor_states([s0, s1])

    payload = joined.extras["contact_forces"]["robot"]
    assert isinstance(payload, ContactForcesData)
    assert payload.contact_forces_history.shape == (2, 3, 2, 3)
    assert payload.contact_forces.shape == (2, 2, 3)
    assert torch.allclose(payload.contact_forces_history[0], torch.zeros((3, 2, 3)))
    assert torch.allclose(payload.contact_forces_history[1], torch.ones((3, 2, 3)))
