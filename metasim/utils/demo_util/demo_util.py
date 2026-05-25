"""Sub-module containing utilities for loading and saving trajectories."""

from __future__ import annotations

import os

from loguru import logger as log

from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler

from .demo_util_v2 import get_traj_v2
from .demo_util_v3 import convert_traj_v2_to_v3


def get_traj(
    traj_filepath,
    robot: RobotCfg | list[RobotCfg],
    handler: BaseSimHandler | None = None,
    v2_as_v3: bool = True,
):
    """Get the trajectory data.

    Args:
        traj_filepath: Traj data path.
        robot: The robot cfg instance, or a list of robot cfgs for a multi-agent
            (e.g. bimanual) trajectory. A multi-agent trajectory file is just a
            single-agent file with one ``{robot_name: [demos]}`` entry per agent,
            so a single-agent file is the one-robot special case.
        handler: The handler instance. Only used for v1 data format.
        v2_as_v3: Whether to convert v2 data format to v3 data format.

    Returns:
        ``(init_states, all_actions, all_states)``. For a multi-agent request the
        returned shape is identical to the single-agent v3 format, with every
        agent merged into each per-step dict: init/state entries union their
        ``robots`` (and share ``objects``), and each action step is the union of
        the agents' ``{robot_name: action}`` dicts. Existing single-robot callers
        are unaffected.
    """
    if isinstance(robot, (list, tuple)):
        return _get_traj_multiagent(traj_filepath, list(robot), handler=handler, v2_as_v3=v2_as_v3)

    if traj_filepath.find("v2") != -1:
        log.info("Reading trajectory using v2 data format")
        if os.path.exists(traj_filepath):
            if v2_as_v3:
                return convert_traj_v2_to_v3(*get_traj_v2(traj_filepath, robot), robot)
            else:
                return get_traj_v2(traj_filepath, robot)
        else:
            raise FileNotFoundError(
                "The trajectory file does not exist, please check the path or convert the trajectory file to v2 format"
            )
    else:
        log.warning("Reading trajectory using v1 data format, which is deprecated")


def _get_traj_multiagent(
    traj_filepath,
    robots: list[RobotCfg],
    handler: BaseSimHandler | None = None,
    v2_as_v3: bool = True,
):
    """Load and merge a multi-agent trajectory keyed by robot name.

    Each agent's slice is read through the single-robot path (the on-disk file is
    keyed by robot name, so ``data[robot.name]`` already isolates one agent), then
    the per-agent v3 streams are merged in lock-step into a single namespaced
    trajectory. Multi-agent trajectories require the v3 namespaced format, because
    that is what keeps each agent's observations and actions indexed by name.
    """
    if not robots:
        raise ValueError("get_traj received an empty robot list")
    if not v2_as_v3:
        raise ValueError("Multi-agent trajectories require the v3 namespaced format; call get_traj with v2_as_v3=True")

    log.info(f"Reading multi-agent trajectory for {len(robots)} agents: {[r.name for r in robots]}")
    per_agent = [get_traj(traj_filepath, robot, handler=handler, v2_as_v3=True) for robot in robots]
    init_list, action_list, state_list = zip(*per_agent)
    return (
        _merge_agent_init_states(init_list),
        _merge_agent_actions(action_list),
        _merge_agent_states(state_list),
    )


def _merge_agent_init_states(init_list):
    """Union each agent's ``robots`` (and shared ``objects``) per demo."""
    num_demos = min(len(init) for init in init_list)
    merged = []
    for demo_idx in range(num_demos):
        objects, robots = {}, {}
        for init in init_list:
            objects.update(init[demo_idx]["objects"])
            robots.update(init[demo_idx]["robots"])
        merged.append({"objects": objects, "robots": robots})
    return merged


def _merge_agent_actions(action_list):
    """Merge each step's ``{robot_name: action}`` dict across agents, per demo."""
    if any(actions is None for actions in action_list):
        return None
    num_demos = min(len(actions) for actions in action_list)
    merged = []
    for demo_idx in range(num_demos):
        horizon = min(len(actions[demo_idx]) for actions in action_list)
        demo = []
        for step_idx in range(horizon):
            step = {}
            for actions in action_list:
                step.update(actions[demo_idx][step_idx])
            demo.append(step)
        merged.append(demo)
    return merged


def _merge_agent_states(state_list):
    """Union each step's ``robots``/``objects`` across agents, per demo."""
    if any(states is None for states in state_list):
        return None
    num_demos = min(len(states) for states in state_list)
    merged = []
    for demo_idx in range(num_demos):
        horizon = min(len(states[demo_idx]) for states in state_list)
        demo = []
        for step_idx in range(horizon):
            objects, robots = {}, {}
            for states in state_list:
                objects.update(states[demo_idx][step_idx]["objects"])
                robots.update(states[demo_idx][step_idx]["robots"])
            demo.append({"objects": objects, "robots": robots})
        merged.append(demo)
    return merged
