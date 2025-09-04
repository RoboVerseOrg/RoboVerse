from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.rl_task import RLTaskEnv
from metasim.types import TensorState
from metasim.utils import humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    neck_height_tensor,
    robot_local_velocity_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)
from metasim.utils.state import list_state_to_tensor

# thresholds
H1_STAND_NECK_HEIGHT = 1.41
G1_STAND_NECK_HEIGHT = 1.00


class StableReward:
    """Standing * upright * small-control."""

    def __init__(self, robot_name: str = "h1"):
        self.robot_name = robot_name

    def __call__(self, states: TensorState) -> torch.FloatTensor:
        """Calculate the stability reward based on neck height, upright position, and small control forces."""
        if self.robot_name.startswith("h1"):
            neck_target = H1_STAND_NECK_HEIGHT
        else:
            neck_target = G1_STAND_NECK_HEIGHT

        neck_h = neck_height_tensor(states, self.robot_name)  # (N,)
        standing = humanoid_reward_util.tolerance_tensor(
            neck_h, bounds=(neck_target, float("inf")), margin=neck_target / 4
        )

        upright_score = torso_upright_tensor(states, self.robot_name)  # (N,)
        upright = humanoid_reward_util.tolerance_tensor(
            upright_score, bounds=(0.9, float("inf")), margin=1.9, value_at_margin=0.0, sigmoid="linear"
        )

        forces = actuator_forces_tensor(states, self.robot_name)  # (N, dof)
        small_control = humanoid_reward_util.tolerance_tensor(
            forces, margin=10.0, value_at_margin=0.0, sigmoid="quadratic"
        ).mean(dim=-1)
        small_control = (4.0 + small_control) / 5.0

        return (standing * upright * small_control).float()


class BaseLocomotionReward:
    """Stable (move or don't-move)."""

    _move_speed = 0.0

    def __init__(self, robot_name: str = "h1", move_speed: float | None = None):
        self.robot_name = robot_name
        if move_speed is not None:
            self._move_speed = float(move_speed)

    def __call__(self, states: TensorState) -> torch.FloatTensor:
        """Calculate the locomotion reward based on stability and movement speed."""
        stable = StableReward(self.robot_name)(states)

        if self._move_speed == 0:
            hv = robot_velocity_tensor(states, self.robot_name)[:, [0, 1]]  # (N,2)
            dont_move = humanoid_reward_util.tolerance_tensor(hv, margin=2.0).mean(dim=-1)
            move_term = dont_move
        else:
            vx_local = robot_local_velocity_tensor(states, self.robot_name)[:, 0]  # (N,)
            move = humanoid_reward_util.tolerance_tensor(
                vx_local,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            )
            move_term = (5.0 * move + 1.0) / 6.0

        return (stable * move_term).float()


# ---------- task env ----------
class BaseLocomotionEnv(RLTaskEnv):
    """locomotion reward with _move_speed = 0."""

    scenario = ScenarioCfg(
        objects=[],
        robots=["h1"],
    )

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        self.robot_name = (
            self.scenario.robots[0] if isinstance(self.scenario.robots[0], str) else self.scenario.robots[0].name
        )
        self.max_episode_steps = 800
        self.scenario.sim_params = SimParamCfg(
            dt=0.002,
            contact_offset=0.01,
            num_position_iterations=8,
            num_velocity_iterations=0,
            bounce_threshold_velocity=0.5,
            replace_cylinder_with_capsule=True,
        )
        self.scenario.decimation = 10
        super().__init__(scenario, device)

    def _observation(self, states: TensorState) -> torch.Tensor:
        results_state = []
        for _, object_state in sorted(states.objects.items()):
            results_state.append(object_state.root_state)
        for _, robot_state in sorted(states.robots.items()):
            results_state.append(robot_state.root_state)
            results_state.append(robot_state.joint_pos)
            results_state.append(robot_state.joint_vel)
        return torch.cat(results_state, dim=1)

    def _get_initial_states(self) -> list[dict] | None:
        init = [
            {
                "objects": {},
                "robots": {
                    "h1": {
                        "dof_pos": {
                            "left_hip_yaw": 0.0,
                            "left_hip_roll": 0.0,
                            "left_hip_pitch": -0.4,
                            "left_knee": 0.8,
                            "left_ankle": -0.4,
                            "right_hip_yaw": 0.0,
                            "right_hip_roll": 0.0,
                            "right_hip_pitch": -0.4,
                            "right_knee": 0.8,
                            "right_ankle": -0.4,
                            "torso": 0.0,
                            "left_shoulder_pitch": 0.0,
                            "left_shoulder_roll": 0.0,
                            "left_shoulder_yaw": 0.0,
                            "left_elbow": 0.0,
                            "right_shoulder_pitch": 0.0,
                            "right_shoulder_roll": 0.0,
                            "right_shoulder_yaw": 0.0,
                            "right_elbow": 0.0,
                        },
                        "pos": torch.tensor([0.0, 0.0, 0.98]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    }
                },
            }
            for _ in range(self.num_envs)
        ]

        return list_state_to_tensor(init)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        robot_position_tensor = states.robots[self.robot_name].root_state[:, 0:3]
        terminated = robot_position_tensor[:, 2] < 0.2
        return terminated


# @register_task("humanoid.walk", "walk", "h1.walk")
class WalkEnv(BaseLocomotionEnv):
    """Walking task for humanoid robots."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        self.max_episode_steps = 1000
        self.reward_functions = [BaseLocomotionReward(self.robot_name, move_speed=1.0)]
        self.reward_weights = [1.0]


# @register_task("humanoid.run", "run", "h1.run")
class RunEnv(BaseLocomotionEnv):
    """Run task for humanoid robots."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        self.max_episode_steps = 1000
        self.reward_functions = BaseLocomotionReward(self.robot_name, move_speed=5.0)
        self.reward_weights = [1.0]


# @register_task("humanoid.stand", "stand", "h1.stand")
class StandEnv(BaseLocomotionEnv):
    """Stand task for humanoid robots."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        self.max_episode_steps = 1000
        self.reward_functions = [BaseLocomotionReward(self.robot_name, move_speed=0.0)]
        self.reward_weights = [1.0]
