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
        # The v1 format is no longer supported. Previously this branch only
        # logged a warning and fell off the end, returning None — every caller
        # unpacks a 3-tuple (`init_states, _, _ = get_traj(...)`), so they hit
        # an opaque `TypeError: cannot unpack non-iterable NoneType` that is NOT
        # in the (FileNotFoundError, KeyError, ValueError) set task base classes
        # catch, masking the real cause (an unsupported/misnamed traj file).
        raise ValueError(
            f"Unsupported v1 trajectory format for '{traj_filepath}'. "
            "Convert it to v2 (the filename must contain 'v2', e.g. '<robot>_v2.pkl.gz')."
        )


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

    Agents are aligned to the shortest common length: if they carry a differing
    number of demos (or differing per-demo horizons), the merge keeps the overlap
    and warns. Robot names must be unique, since each agent's slice is addressed
    by ``robot.name``; duplicate names raise rather than silently collapse.
    """
    if not robots:
        raise ValueError("get_traj received an empty robot list")
    if not v2_as_v3:
        raise ValueError("Multi-agent trajectories require the v3 namespaced format; call get_traj with v2_as_v3=True")

    names = [robot.name for robot in robots]
    if len(set(names)) != len(names):
        raise ValueError(
            f"Multi-agent robots must have unique names so each agent's slice is addressable by name, "
            f"but got duplicates in {names}. Give each agent a distinct name, e.g. robot.replace(name=...)."
        )

    log.info(f"Reading multi-agent trajectory for {len(robots)} agents: {names}")
    per_agent = [get_traj(traj_filepath, robot, handler=handler, v2_as_v3=True) for robot in robots]
    init_list, action_list, state_list = zip(*per_agent)

    demo_counts = [len(init) for init in init_list]
    if len(set(demo_counts)) > 1:
        log.warning(
            f"Multi-agent agents have differing demo counts {dict(zip(names, demo_counts))}; "
            f"merging the {min(demo_counts)} demo(s) common to all agents."
        )
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
