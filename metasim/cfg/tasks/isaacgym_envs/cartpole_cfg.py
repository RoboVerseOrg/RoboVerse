from __future__ import annotations

import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.cartpole_cfg import CartpoleCfg as CartpoleRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg
from .isaacgym_task_base import IsaacGymTaskBase


@configclass
class CartpoleCfg(BaseTaskCfg, IsaacGymTaskBase):
    episode_length = 500
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION  # Using closest available type

    # Task parameters
    reset_dist = 3.0
    max_push_effort = 400.0

    # Robot configuration
    robot: CartpoleRobotCfg = CartpoleRobotCfg()

    # No additional objects needed
    objects: list[RigidObjCfg] = []

    # Control configuration
    control: ControlCfg = ControlCfg(
        action_scale=400.0,  # Max push effort from IsaacGymEnvs
        action_offset=False,
        torque_limit_scale=1.0,
    )

    checker = EmptyChecker()

    # Observation space: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    observation_space = {"shape": [4]}
    action_space = {"shape": [1]}  # Single action for cart force

    randomize = {
        "robot": {
            "cartpole": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.1,  # 0.2 * 0.5 = 0.1
                    "high": 0.1,
                },
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.25,  # 0.5 * 0.5 = 0.25
                    "high": 0.25,
                },
            }
        }
    }

    def __post_init__(self):
        super().__post_init__()
        self._reset_buf = None
        self._progress_buf = None

    def get_observation(self, states):
        observations = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for env_state in states:
            robot_state = env_state["robots"]["cartpole"]

            # Get joint positions and velocities
            if "joint_qpos" in robot_state:
                joint_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32, device=device)
            elif "dof_pos" in robot_state:
                # Convert dict to tensor in correct order
                joint_pos = torch.tensor(
                    [robot_state["dof_pos"]["slider_to_cart"], robot_state["dof_pos"]["cart_to_pole"]],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                joint_pos = torch.zeros(2, dtype=torch.float32, device=device)

            if "joint_qvel" in robot_state:
                joint_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32, device=device)
            elif "dof_vel" in robot_state:
                # Convert dict to tensor in correct order
                joint_vel = torch.tensor(
                    [robot_state["dof_vel"]["slider_to_cart"], robot_state["dof_vel"]["cart_to_pole"]],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                joint_vel = torch.zeros(2, dtype=torch.float32, device=device)

            # Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
            obs = torch.tensor(
                [
                    joint_pos[0],  # cart position
                    joint_vel[0],  # cart velocity
                    joint_pos[1],  # pole angle
                    joint_vel[1],  # pole angular velocity
                ],
                dtype=torch.float32,
                device=device,
            )

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 4))

    def reward_fn(self, states, actions):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            # Handle TensorState from IsaacGym
            robot_state = states.robots["cartpole"]

            # Get joint positions and velocities
            cart_pos = robot_state.joint_pos[:, 0]
            cart_vel = robot_state.joint_vel[:, 0]
            pole_angle = robot_state.joint_pos[:, 1]
            pole_vel = robot_state.joint_vel[:, 1]

            # Compute reward similar to IsaacGymEnvs
            # reward = 1.0 - pole_angle^2 - 0.01 * |cart_vel| - 0.005 * |pole_vel|
            reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

            # Penalties for exceeding bounds
            reward = torch.where(torch.abs(cart_pos) > self.reset_dist, torch.ones_like(reward) * -2.0, reward)
            reward = torch.where(torch.abs(pole_angle) > 3.14159 / 2, torch.ones_like(reward) * -2.0, reward)

            return reward
        else:
            # Handle dict states
            rewards = []
            for env_state in states:
                robot_state = env_state["robots"]["cartpole"]

                if "joint_qpos" in robot_state:
                    cart_pos = robot_state["joint_qpos"][0]
                    pole_angle = robot_state["joint_qpos"][1]
                elif "dof_pos" in robot_state:
                    cart_pos = robot_state["dof_pos"]["slider_to_cart"]
                    pole_angle = robot_state["dof_pos"]["cart_to_pole"]
                else:
                    cart_pos = 0.0
                    pole_angle = 0.0

                if "joint_qvel" in robot_state:
                    cart_vel = robot_state["joint_qvel"][0]
                    pole_vel = robot_state["joint_qvel"][1]
                elif "dof_vel" in robot_state:
                    cart_vel = robot_state["dof_vel"]["slider_to_cart"]
                    pole_vel = robot_state["dof_vel"]["cart_to_pole"]
                else:
                    cart_vel = 0.0
                    pole_vel = 0.0

                # Compute reward
                reward = 1.0 - pole_angle**2 - 0.01 * abs(cart_vel) - 0.005 * abs(pole_vel)

                # Penalties
                if abs(cart_pos) > self.reset_dist:
                    reward = -2.0
                if abs(pole_angle) > 3.14159 / 2:
                    reward = -2.0

                rewards.append(reward)

            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            # Handle TensorState from IsaacGym
            robot_state = states.robots["cartpole"]

            cart_pos = robot_state.joint_pos[:, 0]
            pole_angle = robot_state.joint_pos[:, 1]

            # Terminate if cart exceeds bounds or pole falls too far
            terminations = (torch.abs(cart_pos) > self.reset_dist) | (torch.abs(pole_angle) > 3.14159 / 2)

            return terminations
        else:
            # Handle dict states
            terminations = []
            for env_state in states:
                robot_state = env_state["robots"]["cartpole"]

                if "joint_qpos" in robot_state:
                    cart_pos = robot_state["joint_qpos"][0]
                    pole_angle = robot_state["joint_qpos"][1]
                elif "dof_pos" in robot_state:
                    cart_pos = robot_state["dof_pos"]["slider_to_cart"]
                    pole_angle = robot_state["dof_pos"]["cart_to_pole"]
                else:
                    cart_pos = 0.0
                    pole_angle = 0.0

                # Check termination conditions
                terminated = abs(cart_pos) > self.reset_dist or abs(pole_angle) > 3.14159 / 2
                terminations.append(terminated)

            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        # Initialize buffers if needed
        self._reset_buf = None
        self._progress_buf = None

    def reset(self, env_ids=None):
        # Reset is handled by the simulator
        pass

    def post_reset(self):
        # Post reset operations if needed
        pass

    def initialize_buffers(self, num_envs, device):
        # Initialize any task-specific buffers if needed
        self._reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._progress_buf = torch.zeros(num_envs, dtype=torch.int32, device=device)
