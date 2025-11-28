"""Integration tests for state consistency across simulators."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close
from metasim.utils.state import state_tensor_to_nested


def _create_init_states(num_envs: int) -> list[dict]:
    """Create initial states for testing."""
    return [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        }
    ] * num_envs


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym")
def test_state_consistency(handler):
    """Test that set_states and get_states are consistent."""
    num_envs = handler.scenario.num_envs
    init_states = _create_init_states(num_envs)

    handler.set_states(init_states)
    states = state_tensor_to_nested(handler, handler.get_states())

    for i in range(num_envs):
        assert_close(states[i]["objects"]["cube"]["pos"], init_states[i]["objects"]["cube"]["pos"])
        assert_close(states[i]["objects"]["sphere"]["pos"], init_states[i]["objects"]["sphere"]["pos"])
        assert_close(states[i]["objects"]["bbq_sauce"]["pos"], init_states[i]["objects"]["bbq_sauce"]["pos"])
        assert_close(states[i]["objects"]["box_base"]["pos"], init_states[i]["objects"]["box_base"]["pos"])
        assert_close(states[i]["objects"]["box_base"]["rot"], init_states[i]["objects"]["box_base"]["rot"])
        assert_close(states[i]["robots"]["franka"]["pos"], init_states[i]["robots"]["franka"]["pos"])
        assert_close(states[i]["robots"]["franka"]["rot"], init_states[i]["robots"]["franka"]["rot"])
        assert_close(
            states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
            init_states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
        )
        for k in states[i]["robots"]["franka"]["dof_pos"].keys():
            assert_close(
                states[i]["robots"]["franka"]["dof_pos"][k],
                init_states[i]["robots"]["franka"]["dof_pos"][k],
            )

    log.info(f"State consistency test passed for {handler.scenario.simulator}")
