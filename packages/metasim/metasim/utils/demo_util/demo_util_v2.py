"""Sub-module containing utilities for loading and saving trajectories in v2 format."""

from __future__ import annotations

import os

import torch
from loguru import logger as log

from metasim.scenario.robot import RobotCfg

from .loader import load_traj_file


def get_traj_v2(traj_filepath, robot: RobotCfg, data=None):
    """Get the trajectory data.

    Args:
        traj_filepath: The task cfg instance.
        robot: The robot cfg instance.

        data: The already-loaded file content (``get_traj`` passes it so the file is read once).

    Returns:
        The trajectory data.
    """
    ## Load the file (``data`` is the content when ``get_traj`` already loaded it to recognise the format)
    from metasim.utils.demo_util.demo_util import _resolve_traj_file

    path = _resolve_traj_file(traj_filepath, robot.name)
    if path != traj_filepath:
        log.info(f"Loading trajectory from {path}")
    if data is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The trajectory file does not exist: {path}")
        data = load_traj_file(path)
    loaded = data
    if not isinstance(loaded, dict) or robot.name not in loaded:
        raise KeyError(
            f"{path} has no entry for robot {robot.name!r}; robots in the file: "
            f"{[k for k in loaded] if isinstance(loaded, dict) else type(loaded).__name__}"
        )
    data = loaded[robot.name]

    ## Parse initial states
    # Guard before indexing data[0]: an empty trajectory (filtered/failed
    # collection) would otherwise raise IndexError, which escapes the
    # (FileNotFoundError, KeyError, ValueError) set callers catch.
    if not data:
        raise ValueError(f"Trajectory for robot {robot.name!r} in '{traj_filepath}' is empty")
    if "init_state" in data[0]:
        init_states = [traj["init_state"] for traj in data]
    else:
        raise ValueError("No init_state found in the trajectory data")
    for demo_idx, init_state in enumerate(init_states):
        for obj_name in init_state:
            # import ipdb; ipdb.set_trace()
            init_states[demo_idx][obj_name]["pos"] = torch.tensor(init_states[demo_idx][obj_name]["pos"])
            init_states[demo_idx][obj_name]["rot"] = torch.tensor(init_states[demo_idx][obj_name]["rot"])

    ## Parse actions
    if "actions" in data[0]:
        all_actions = [traj["actions"] for traj in data]
    else:
        log.error("No actions found in the trajectory data")
        all_actions = None

    ## Parse states
    if "states" in data[0] and data[0]["states"] is not None:
        all_states = [traj["states"] for traj in data]
        for demo_idx, states in enumerate(all_states):
            for step_idx, state in enumerate(states):
                for obj_name in state:
                    all_states[demo_idx][step_idx][obj_name]["pos"] = torch.tensor(state[obj_name]["pos"])
                    all_states[demo_idx][step_idx][obj_name]["rot"] = torch.tensor(state[obj_name]["rot"])
    else:
        log.error("No states found in the trajectory data")
        all_states = None

    return init_states, all_actions, all_states
