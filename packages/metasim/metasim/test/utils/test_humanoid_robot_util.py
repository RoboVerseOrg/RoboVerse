"""General tests for metasim.utils.humanoid_robot_util accessor guards (no backend)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.utils.humanoid_robot_util import (
    actuator_knee_pos_tensor,
    dof_pos_tensor,
    dof_vel_tensor,
)


@pytest.mark.general
def test_dof_pos_vel_tensor_none_does_not_crash():
    """Regression: the None fallback did torch.zeros_like(None) -> TypeError."""
    es = SimpleNamespace(robots={"r": SimpleNamespace(joint_pos=None, joint_vel=None)})
    assert dof_pos_tensor(es, "r") is None
    assert dof_vel_tensor(es, "r") is None

    es2 = SimpleNamespace(robots={"r": SimpleNamespace(joint_pos=torch.zeros(2, 3), joint_vel=torch.ones(2, 3))})
    assert torch.equal(dof_pos_tensor(es2, "r"), torch.zeros(2, 3))
    assert torch.equal(dof_vel_tensor(es2, "r"), torch.ones(2, 3))


@pytest.mark.general
def test_actuator_knee_pos_tensor_checks_none_before_slicing():
    """Regression: the None check ran AFTER slicing, so a None knee_states crashed
    with TypeError instead of the intended ValueError."""
    es_none = SimpleNamespace(robots={"r": SimpleNamespace(extra={"knee_states": None})})
    with pytest.raises(ValueError):
        actuator_knee_pos_tensor(es_none, "r")

    es_ok = SimpleNamespace(robots={"r": SimpleNamespace(extra={"knee_states": torch.zeros(2, 2, 3)})})
    out = actuator_knee_pos_tensor(es_ok, "r")
    assert out.shape == (2, 2, 2)  # last dim sliced to :2
