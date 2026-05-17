"""Catalog of ManiSkill tabletop tasks we target for MetaSim parity.

Each entry maps a ManiSkill gym ID to the MetaSim task we register (or plan
to register) and to a one-line description of the success / reward
contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManiskillTask:
    """One ManiSkill task we want parity with."""

    gym_id: str  # e.g. "PickCube-v1"
    metasim_name: str  # e.g. "maniskill.pick_cube"
    robot: str  # ManiSkill robot key (Panda / etc.)
    control_mode: str  # default control mode for parity tests
    obs_mode: str  # default obs mode
    max_episode_steps: int
    description: str = ""


ALL_TASKS = [
    ManiskillTask(
        gym_id="PickCube-v1",
        metasim_name="maniskill.pick_cube",
        robot="panda",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        max_episode_steps=50,
        description=(
            "Lift a 4 cm red cube and place it inside a 2.5 cm goal sphere "
            "with the robot static. Default Panda with finger friction 2.0."
        ),
    ),
    ManiskillTask(
        gym_id="PushCube-v1",
        metasim_name="maniskill.push_cube",
        robot="panda",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        max_episode_steps=50,
        description="Push a 2 cm cube to a target on the table.",
    ),
    ManiskillTask(
        gym_id="StackCube-v1",
        metasim_name="maniskill.stack_cube",
        robot="panda",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        max_episode_steps=50,
        description="Pick the red cube and stack it on top of a green base cube.",
    ),
    ManiskillTask(
        gym_id="PullCube-v1",
        metasim_name="maniskill.pull_cube",
        robot="panda",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        max_episode_steps=50,
        description="Pull a 2 cm cube on the table backward to a target.",
    ),
]


def task_by_gym_id(gym_id: str) -> ManiskillTask:
    for t in ALL_TASKS:
        if t.gym_id == gym_id:
            return t
    raise KeyError(gym_id)


def task_by_metasim_name(name: str) -> ManiskillTask:
    for t in ALL_TASKS:
        if t.metasim_name == name:
            return t
    raise KeyError(name)
