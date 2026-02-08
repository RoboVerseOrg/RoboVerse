from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from metasim.utils.state import adapt_actions_to_dict


@dataclass
class _DummyRobot:
    name: str
    control_type: dict[str, str] | None = None


class _DummyHandler:
    def __init__(self, robot: _DummyRobot, joint_names_sorted: list[str]):
        self.robots = [robot]
        self._joint_names = list(joint_names_sorted)

    def get_joint_names(self, obj_name: str, sort: bool = True):
        assert obj_name == self.robots[0].name
        names = list(self._joint_names)
        if sort:
            names.sort()
        return names


@pytest.mark.general
def test_adapt_actions_tensor_maps_to_effort_for_effort_robots():
    robot = _DummyRobot(name="r1", control_type={"j1": "effort", "j2": "effort"})
    handler = _DummyHandler(robot, joint_names_sorted=["j1", "j2"])

    out = adapt_actions_to_dict(handler, torch.tensor([1.0, 2.0]))
    assert "r1" in out
    assert "dof_effort_target" in out["r1"]
    assert out["r1"]["dof_effort_target"]["j1"] == 1.0
    assert out["r1"]["dof_effort_target"]["j2"] == 2.0


@pytest.mark.general
def test_adapt_actions_tensor_maps_to_pos_for_position_robots():
    robot = _DummyRobot(name="r1", control_type={"j1": "position", "j2": "position"})
    handler = _DummyHandler(robot, joint_names_sorted=["j1", "j2"])

    out = adapt_actions_to_dict(handler, torch.tensor([1.0, 2.0]))
    assert "r1" in out
    assert "dof_pos_target" in out["r1"]
    assert out["r1"]["dof_pos_target"]["j1"] == 1.0
    assert out["r1"]["dof_pos_target"]["j2"] == 2.0
