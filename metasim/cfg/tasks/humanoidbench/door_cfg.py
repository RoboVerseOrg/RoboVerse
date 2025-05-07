"""Exit Door task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _DoorChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward


class DoorReward(HumanoidBaseReward):
    """Reward function for the door task."""

    def __init__(self, robot_name="h1"):
        """Initialize the door reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the door reward."""
        results = []
        for state in states:
            # Get IMU position (for passage reward)
            imu_pos = state[f"metasim_site_{self.robot_name}/imu"]["pos"]

            # Get hinge angle (for opendoor reward)
            door_hinge_angle = state["door"]["dof_pos"]["door_hinge"]

            # Get door handle/lock angle (for openhatch reward)
            door_hatch_angle = state["door"]["dof_pos"]["door_hatch_hinge"]

            # Get left and right hand positions (for proximity reward)
            left_hand_pos = humanoid_robot_util.left_hand_position(state, self.robot_name)
            right_hand_pos = humanoid_robot_util.right_hand_position(state, self.robot_name)

            # Get door handle position
            door_handle_pos = state["metasim_body_door/door_hatch"]["pos"]

            # Calculate the distance between the hand and the doorknob.
            left_dist = torch.norm(left_hand_pos - door_handle_pos)
            right_dist = torch.norm(right_hand_pos - door_handle_pos)
            min_hand_dist = torch.min(left_dist, right_dist)

            # Calculate stable reward
            stable = StableReward(robot_name=self.robot_name)(states)

            # Calculate each sub-reward
            # opendoor = min(1, θdoor**2)
            opendoor = min(1.0, door_hinge_angle**2)

            # openhatch = tol(θhatch, (0.75, 2), 0.75)
            openhatch = humanoid_reward_util.tolerance(door_hatch_angle, bounds=(0.75, 2.0), margin=0.75)

            # proximitydoor = tol(min(d(handieft, door), d(handright, door)), (0, 0.25), 1)
            proximitydoor = humanoid_reward_util.tolerance(min_hand_dist, bounds=(0, 0.25), margin=1.0)

            # passage = tol(XIMU, (1.2, +∞), 1)
            passage = humanoid_reward_util.tolerance(imu_pos[0], bounds=(1.2, float("inf")), margin=1.0)

            # Total reward R = 0.1 * stable + 0.45 * opendoor + 0.05 * openhatch + 0.05 * proximitydoor + 0.35 * passage
            reward = 0.1 * stable + 0.45 * opendoor + 0.05 * openhatch + 0.05 * proximitydoor + 0.35 * passage
            results.append(reward)

        return torch.tensor(results)


@configclass
class DoorCfg(HumanoidTaskCfg):
    """Door task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="door",
            mjcf_path="roboverse_data/assets/humanoidbench/door/door/mjcf/door.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/door/v2/initial_state_v2.json"
    checker = _DoorChecker()
    reward_weights = [1.0]
    reward_functions = [DoorReward]
