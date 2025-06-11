"""Sub-module containing utilities for loading and saving trajectories in v3 format. v3 doesn't define a new trajectory format, but a new state format."""

from __future__ import annotations

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.types import Action, RobotAction


def convert_state_v2_to_v3(state: dict, robot: BaseRobotCfg):
    """Convert v2 state format to v3 state format.

    Args:
        state: The v2 state.
        robot: The robot cfg instance.

    Returns:
        The converted v3 state.
    """
    state_v3 = {"objects": {}, "robots": {}}
    for obj_name in state:
        if obj_name == robot.name:
            state_v3["robots"][obj_name] = state[obj_name]
        else:
            state_v3["objects"][obj_name] = state[obj_name]
    return state_v3


def convert_actions_v2_to_v3(actions_v2: list[RobotAction], robot: BaseRobotCfg) -> list[Action]:
    """Convert v2 action format to v3 action format.

    Args:
        actions_v2: The v2 actions.
        robot: The robot cfg instance.

    Returns:
        The converted v3 action.
    """
    return [[{robot.name: a} for a in action] for action in actions_v2]


def convert_traj_v2_to_v3(
    init_states: list[dict] | None,
    all_actions: list[list[dict]],
    all_states: list[list[dict]] | None,
    robot: BaseRobotCfg,
):
    """Convert v2 trajectory data to v3 trajectory data.

    Args:
        init_states: The v2 initial states.
        all_actions: The v2 actions.
        all_states: The v2 states.
        robot: The robot cfg instance.

    Returns:
        The converted v3 trajectory data.
    """
    init_states_v3 = [convert_state_v2_to_v3(init_state, robot) for init_state in init_states]
    if all_states is not None:
        all_states_v3 = [[convert_state_v2_to_v3(state, robot) for state in states] for states in all_states]
    else:
        all_states_v3 = None
    if all_actions is not None:
        all_actions_v3 = convert_actions_v2_to_v3(all_actions, robot)
    else:
        all_actions_v3 = None
    return init_states_v3, all_actions_v3, all_states_v3
