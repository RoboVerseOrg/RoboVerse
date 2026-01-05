"""Sub-module containing utilities for loading and saving trajectories in v3 format. v3 doesn't define a new trajectory format, but a new state format."""

from __future__ import annotations

from metasim.scenario.robot import RobotCfg
from metasim.types import Action


def convert_state_v2_to_v3(state: dict, robot: RobotCfg):
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


def convert_actions_v2_to_v3(actions_v2, robot: RobotCfg):
    """Convert v2 action format to v3 action format.

    This repo has multiple "v2" producer styles:
    - **episode-major**: all_actions is a list of episodes; each episode is a list of step actions.
      Each step action is either:
        - payload dict: {"dof_pos_target": {...}, ...}
        - already-wrapped v3 dict: {robot.name: {"dof_pos_target": {...}, ...}}
    - **legacy**: step action could be a 1-element list containing the payload dict.

    v3 format expected by handlers is:
      - all_actions_v3: list[episode]
      - episode: list[step_action]
      - step_action: {robot.name: payload_dict}
    """

    def _wrap_step(step) -> Action:
        # already v3
        if isinstance(step, dict) and robot.name in step and isinstance(step.get(robot.name), dict):
            return step  # type: ignore[return-value]

        # common v2: payload dict
        if isinstance(step, dict):
            return {robot.name: step}

        # legacy: [payload_dict]
        if isinstance(step, (list, tuple)) and len(step) == 1 and isinstance(step[0], dict):
            return {robot.name: step[0]}

        raise ValueError(f"Unsupported v2 action step format for v3 conversion: {type(step)}")

    # actions_v2 is list of episodes
    return [[_wrap_step(step) for step in episode] for episode in actions_v2]


def convert_traj_v2_to_v3(
    init_states: list[dict] | None,
    all_actions: list[list[dict]],
    all_states: list[list[dict]] | None,
    robot: RobotCfg,
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
