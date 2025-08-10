from __future__ import annotations

import torch

from metasim.utils.state import TensorState

from .registry import register_task
from .rl_task import RLTaskWrapper


def negative_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    """Calculate negative distance from end effector to origin as reward."""
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    distances = torch.norm(ee_pos, dim=1)
    return -distances  # Negative distance as reward


def x_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    """Calculate x-distance from end effector as reward."""
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    return ee_pos[:, 0]  # Return x-coordinate


class ReachingWrapper(RLTaskWrapper):
    """Base wrapper for reaching tasks.

    This class provides common functionality for all reaching tasks.
    """

    def _load_task_config(self, scenario) -> None:
        """Configure common reaching task parameters."""
        super()._load_task_config(scenario)

        # Common configuration for reaching tasks
        self.objects = []
        self.max_episode_steps = 100

        return scenario

    def _get_initial_states(self) -> list[dict]:
        """Get the initial states of the environment."""
        return [
            {
                "objects": {},
                "robots": {
                    "franka": {
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": 0.0,
                            "panda_joint3": 0.0,
                            "panda_joint4": 0.0,
                            "panda_joint5": 0.0,
                            "panda_joint6": 0.0,
                            "panda_joint7": 0.0,
                            "panda_finger_joint1": 0.0,
                            "panda_finger_joint2": 0.0,
                        },
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    }
                },
            }
        ]

    def _observation(self, states: TensorState) -> torch.Tensor:
        """Get the observation for the current state."""
        joint_pos = states.robots["franka"].joint_pos
        panda_hand_index = states.robots["franka"].body_names.index("panda_hand")
        ee_pos = states.robots["franka"].body_state[:, panda_hand_index, :3]

        return torch.cat([joint_pos, ee_pos], dim=1)


@register_task("reach.origin", "reach_origin", "franka.reach_origin")
class ReachOriginWrapper(ReachingWrapper):
    """Wrapper for reaching origin task.

    This task encourages the robot's end effector to move towards the origin (0, 0, 0).
    The reward is based on the negative distance to the origin.
    """

    def _load_task_config(self, scenario) -> None:
        """Configure for reaching origin task."""
        super()._load_task_config(scenario)

        # Reward function: negative distance to origin
        self.reward_functions = [negative_distance]
        self.reward_weights = [1.0]

        return scenario


@register_task("reach.far", "reach_far", "franka.reach_far")
class ReachFarAwayWrapper(ReachingWrapper):
    """Wrapper for reaching far away task.

    This task encourages the robot's end effector to move as far as possible
    in the positive x direction.
    """

    def _load_task_config(self, scenario) -> None:
        """Configure for reaching far away task."""
        super()._load_task_config(scenario)

        # Reward function: x-distance (encourages moving away in x direction)
        self.reward_functions = [x_distance]
        self.reward_weights = [1.0]

        return scenario
