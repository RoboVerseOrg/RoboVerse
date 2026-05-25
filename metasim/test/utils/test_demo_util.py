"""Tests for metasim.utils.demo_util trajectory loading.

Pure unit tests (no simulator) covering the v2->v3 single-agent path and the
multi-agent (bimanual) merge added for keyed-by-robot-name trajectory files.
All tests are marked @pytest.mark.general.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.utils.demo_util import get_traj
from metasim.utils.demo_util.loader import save_traj_file


def _robot(name):
    """Minimal stand-in: the loader only reads ``robot.name``."""
    return SimpleNamespace(name=name)


def _agent_demo(robot_name, joint_value, with_states):
    """One demo for a single agent, sharing a ``cube`` object."""
    init_state = {
        robot_name: {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "dof_pos": {"j0": joint_value}},
        "cube": {"pos": [0.4, 0.0, 0.05], "rot": [1.0, 0.0, 0.0, 0.0]},
    }
    actions = [{"dof_pos_target": {"j0": joint_value + t}} for t in range(3)]
    states = None
    if with_states:
        states = [
            {
                robot_name: {"pos": [0.0, 0.0, float(t)], "rot": [1.0, 0.0, 0.0, 0.0]},
                "cube": {"pos": [0.4, 0.0, 0.05], "rot": [1.0, 0.0, 0.0, 0.0]},
            }
            for t in range(3)
        ]
    return {"init_state": init_state, "actions": actions, "states": states}


def _write_dataset(tmp_path, agent_names, with_states=False):
    dataset = {name: [_agent_demo(name, i, with_states)] for i, name in enumerate(agent_names)}
    dataset["metadata"] = {"num_agents": len(agent_names), "agents": list(agent_names)}
    path = str(tmp_path / "bimanual_handover_v2.pkl")
    save_traj_file(dataset, path)
    return path


@pytest.mark.general
def test_single_agent_backward_compatible(tmp_path):
    """A single RobotCfg still returns only that agent's v3 slice."""
    path = _write_dataset(tmp_path, ["franka_left", "franka_right"])
    init_states, all_actions, _ = get_traj(path, _robot("franka_left"))

    assert list(init_states[0]["robots"]) == ["franka_left"]
    assert "cube" in init_states[0]["objects"]
    assert list(all_actions[0][0]) == ["franka_left"]


@pytest.mark.general
def test_multiagent_merges_robots_and_shares_objects(tmp_path):
    """A list of robots unions ``robots`` per step and shares ``objects`` once."""
    names = ["franka_left", "franka_right"]
    path = _write_dataset(tmp_path, names)
    init_states, all_actions, _ = get_traj(path, [_robot(n) for n in names])

    # init: both arms present, single shared object.
    assert set(init_states[0]["robots"]) == set(names)
    assert list(init_states[0]["objects"]) == ["cube"]

    # actions: every step carries a namespaced entry per agent.
    assert len(all_actions[0]) == 3
    for step in all_actions[0]:
        assert set(step) == set(names)
        assert step["franka_left"]["dof_pos_target"]["j0"] != step["franka_right"]["dof_pos_target"]["j0"]


@pytest.mark.general
def test_multiagent_merges_states(tmp_path):
    """Per-step states union each agent's ``robots`` and share ``objects``."""
    names = ["franka_left", "franka_right"]
    path = _write_dataset(tmp_path, names, with_states=True)
    _, _, all_states = get_traj(path, [_robot(n) for n in names])

    assert all_states is not None
    for step in all_states[0]:
        assert set(step["robots"]) == set(names)
        assert list(step["objects"]) == ["cube"]


@pytest.mark.general
def test_multiagent_init_tensorized(tmp_path):
    """Merged init poses are tensors, matching the single-agent v3 contract."""
    names = ["franka_left", "franka_right"]
    path = _write_dataset(tmp_path, names)
    init_states, _, _ = get_traj(path, [_robot(n) for n in names])

    assert isinstance(init_states[0]["robots"]["franka_left"]["pos"], torch.Tensor)
    assert isinstance(init_states[0]["objects"]["cube"]["rot"], torch.Tensor)


@pytest.mark.general
def test_multiagent_requires_v3(tmp_path):
    """Multi-agent namespacing needs the v3 format; v2_as_v3=False is rejected."""
    names = ["franka_left", "franka_right"]
    path = _write_dataset(tmp_path, names)
    with pytest.raises(ValueError, match="v3"):
        get_traj(path, [_robot(n) for n in names], v2_as_v3=False)


@pytest.mark.general
def test_multiagent_empty_robot_list(tmp_path):
    """An empty robot list is an explicit error, not a silent no-op."""
    path = _write_dataset(tmp_path, ["franka_left"])
    with pytest.raises(ValueError, match="empty robot list"):
        get_traj(path, [])
