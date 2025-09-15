from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    dof_pos_tensor,
    dof_vel_tensor,
)
from metasim.utils.state import TensorState
from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.rl.unitree_rl.envs.base_legged import LeggedRobot


# Training Config
@configclass
class LeggedWalkingCfgPPO(LeggedRobotCfgPPO):
    policy = LeggedRobotCfgPPO.Policy(
        init_noise_std=1.0, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu"
    )
    algorithm = LeggedRobotCfgPPO.Algorithm(entropy_coef=0.01)
    runner = LeggedRobotCfgPPO.Runner(experiment_name="legged_walking")


# Config
@configclass
class LeggedWalkingCfg(BaseLeggedTaskCfg):
    """Configuration for the walking task."""

    task_name = "legged_walking"

    ppo_cfg = LeggedWalkingCfgPPO()

    frame_stack = 1
    c_frame_stack = 1

    control: BaseLeggedTaskCfg.ControlCfg = BaseLeggedTaskCfg.ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)

    reward_cfg = BaseLeggedTaskCfg.RewardCfg(soft_dof_pos_limit=0.9, base_height_target=0.25)

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        # "orientation": -0.0,
        "orientation_sq": -0.0,
        "dof_vel": -0.0,
        "dof_acc": -2.5e-7,
        # "base_height": 0.0,
        "base_height_sq": -0.0,  # -10.0,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "action_rate": -0.01,
        "stand_still": -0.0,
        "dof_pos_limits": -10.0,
        "torques": -0.0002,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9 + 3 * self.num_actions  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 0
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


# Environment
class LeggedWalkingTask(LeggedRobot):
    """
    Wrapper for walking task for legged robots.
    """

    def __init__(self, task_cfg: LeggedWalkingCfg, scenario: ScenarioCfg):
        # Set decimation from scenario for consistent dt
        self.decimation = scenario.decimation
        # Bootstrap configuration into the env prior to BaseTaskEnv init
        self._init_from_cfg(task_cfg)
        super().__init__(scenario)

    def _init_from_cfg(self, task_cfg: LeggedWalkingCfg):
        self.cfg = task_cfg
        # Dimensions
        self.num_obs = self.cfg.num_observations
        self.num_actions = self.cfg.num_actions
        self.num_privileged_obs = self.cfg.num_privileged_obs
        # Timing
        self.dt = self.decimation * self.cfg.sim_params.dt
        import math
        self.max_episode_length = math.ceil(self.cfg.max_episode_length_s / self.dt)
        # PPO train cfg for rsl_rl
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(self.cfg.ppo_cfg)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _get_noise_scale_vec(self, cfg: BaseLeggedTaskCfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        noise_vec[0:3] = 0.0  # commands
        noise_vec[3:6] = noise_scales.lin_vel * noise_level * cfg.normalization.obs_scales.lin_vel
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * cfg.normalization.obs_scales.ang_vel
        noise_vec[9:12] = noise_scales.gravity * noise_level
        noise_vec[12 : 12 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * cfg.normalization.obs_scales.dof_pos
        )
        noise_vec[12 + self.num_actions : 12 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[12 + 2 * self.num_actions :] = 0.0  # previous actions

        return noise_vec

    def compute_observations(self, envstate: TensorState):
        """compute observations and priviledged observation"""
        q = (
            dof_pos_tensor(envstate, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstate, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel

        self.obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                q,
                dq,
                self.actions,
            ),
            dim=-1,
        )

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
