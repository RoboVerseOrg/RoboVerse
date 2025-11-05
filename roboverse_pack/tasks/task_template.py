"""Minimal Task Template for RoboVerse.

Key components:
1. Register task with @register_task decorator
2. Define default scenario as class variable
3. Set initial states in _get_initial_states()
4. Implement termination condition in _terminated()
5. (Optional) Override step() and reset() for custom control logic
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task


@register_task("template.minimal")
class MinimalTask(BaseTaskEnv):
    """Minimal task example with a robot and a target object."""

    # ========================================
    # 1. Define default scenario (class variable)
    # ========================================
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="target",
                mass=0.01,
                size=(0.02, 0.02, 0.02),
                physics=PhysicStateType.XFORM,
                color=(0.2, 0.8, 0.2),
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator="mujoco",
        num_envs=1,
        headless=False,
    )

    max_episode_steps = 200

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        # Custom task variables (will be reset in reset()) e.g.
        # self.step_count = None

    # ========================================
    # 2. Set initial states
    # ========================================
    def _get_initial_states(self) -> list[dict] | None:
        """Return initial states for all envs.

        Format: list[dict] of length num_envs
        Each dict contains: "objects", "robots", "cameras", "extras"
        """
        return [
            {
                "objects": {
                    "target": {
                        "pos": torch.tensor([0.5, 0.0, 0.3]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),  # quaternion (w,x,y,z)
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": -0.5,
                            "panda_joint3": 0.0,
                            "panda_joint4": -2.0,
                            "panda_joint5": 0.0,
                            "panda_joint6": 1.5,
                            "panda_joint7": 0.785,
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                        },
                    }
                },
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]

    # ========================================
    # 3. Implement termination condition
    # ========================================
    def _terminated(self, states) -> torch.Tensor:
        """Check if task is completed.

        Args:
            states: Current environment states (TensorState)

        Returns:
            Boolean tensor of shape [num_envs], True if task succeeded
        """
        # Example: task succeeds when end-effector reaches target
        # ee_pos = states.sites["panda_hand"]["pos"]
        # target_pos = states.objects["target"]["pos"]
        # distance = torch.norm(ee_pos - target_pos, dim=-1)
        # return distance < 0.05

        # Default: never terminate
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ========================================
    # 4. (Optional) Override step and reset
    # ========================================
    def step(self, actions: torch.Tensor):
        """Execute one simulation step.

        Optional customizations:
        - Action normalization/unnormalization
        - End-effector control mapping
        - Custom control logic
        - Other logic
        """
        # Example: unnormalize actions from [-1,1] to joint limits
        # actions = (actions + 1.0) / 2.0 * (action_high - action_low) + action_low

        return super().step(actions)

    def reset(self, states=None, env_ids=None):
        """Reset environment.

        Use this to reset custom member variables.
        """
        states = super().reset(states, env_ids)

        # Reset custom variables， e.g.
        # if env_ids is None:
        #     self.step_count = torch.zeros(self.num_envs, device=self.device)
        # else:
        #     self.step_count[env_ids] = 0

        return states


# ========================================
# Working with TensorState
# ========================================
"""
TensorState is the core data structure for accessing simulation states.
It contains: objects, robots, cameras, sites, and extras.

Structure:
    states.objects[name]: ObjectState
        - root_state: [num_envs, 13] = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        - joint_pos: [num_envs, num_joints] (for articulated objects)
        - joint_vel: [num_envs, num_joints] (for articulated objects)
        - body_state: [num_envs, num_bodies, 13]
        - body_names: list[str]

    states.robots[name]: RobotState
        - root_state: [num_envs, 13] = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        - joint_pos: [num_envs, num_joints]
        - joint_vel: [num_envs, num_joints]
        - joint_pos_target: [num_envs, num_joints]
        - joint_vel_target: [num_envs, num_joints]
        - joint_effort_target: [num_envs, num_joints]
        - body_state: [num_envs, num_bodies, 13]
        - body_names: list[str]

    states.cameras[name]: CameraState
        - rgb: [num_envs, H, W, 3]
        - depth: [num_envs, H, W]
        - pos: [num_envs, 3]
        - quat_world: [num_envs, 4]
        - intrinsics: [num_envs, 3, 3]
        - instance_id_seg: [num_envs, H, W]

    states.extras: dict for custom data

Common access patterns:
    # Object state
    object_pos = states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
    object_quat = states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]
    object_lin_vel = states.objects["target"].root_state[:, 7:10]  # [num_envs, 3]
    object_ang_vel = states.objects["target"].root_state[:, 10:13]  # [num_envs, 3]

    # Robot state
    robot_joint_pos = states.robots["franka"].joint_pos  # [num_envs, 9]
    robot_joint_vel = states.robots["franka"].joint_vel  # [num_envs, 9]
    robot_root_pos = states.robots["franka"].root_state[:, 0:3]  # [num_envs, 3]

"""


def flatten_observation(states) -> torch.Tensor:
    """Example: Flatten TensorState into 1D observation tensor for RL algorithms.

    This function demonstrates how to extract and concatenate various state
    components into a single flat observation vector.

    Args:
        states: TensorState object from env.reset() or env.step()

    Returns:
        Flattened observation tensor of shape [num_envs, obs_dim]
    """
    # Extract object states
    target_pos = states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
    target_quat = states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

    # Extract robot states
    robot_joint_pos = states.robots["franka"].joint_pos  # [num_envs, num_joints=9]
    robot_joint_vel = states.robots["franka"].joint_vel  # [num_envs, num_joints=9]

    # Concatenate all components into flat observation
    # Total dim: 3 + 4 + 9 + 9 = 25
    obs = torch.cat(
        [
            target_pos,  # 3
            target_quat,  # 4
            robot_joint_pos,  # 9
            robot_joint_vel,  # 9
        ],
        dim=-1,
    )

    return obs  # shape: [num_envs, 28]


# ========================================
# Usage Examples
# ========================================

if __name__ == "__main__":
    """Example usage of this task template."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from metasim.task.registry import get_task_class

    task_cls = get_task_class("template.minimal")
    scenario = task_cls.scenario.update(num_envs=2, headless=True)
    env = task_cls(scenario=scenario, device=device)

    # Reset environment and get TensorState
    states, info = env.reset()

    # Access TensorState components directly
    target_pos = states.objects["target"].root_state[:, 0:3]  # [2, 3]
    robot_qpos = states.robots["franka"].joint_pos  # [2, 9]

    # Flatten observation for algorithms, get prue tensor obs
    obs_flat = flatten_observation(states)  # [2, 25]

    # Run a few steps
    for i in range(10):
        # Random actions for demonstration
        actions = torch.randn(2, 9, device=device)
        states, _, terminated, truncated, info = env.step(actions)

        # Extract flattened observation each step
        obs_flat = flatten_observation(states)

    env.close()
