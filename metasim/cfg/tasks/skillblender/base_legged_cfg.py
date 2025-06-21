from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""
import inspect
from typing import Callable

import torch

from metasim.cfg.checkers.base_checker import BaseChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_action_rate,
    reward_ang_vel_xy,
    reward_base_height,
    reward_collision,
    reward_dof_acc,
    reward_dof_pos_limits,
    reward_dof_vel,
    reward_dof_vel_limits,
    reward_feet_air_time,
    reward_feet_contact_forces,
    reward_lin_vel_z,
    reward_orientation,
    reward_stand_still,
    reward_stumble,
    reward_termination,
    reward_torque_limits,
    reward_torques,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
)
from metasim.sim import BaseSimHandler
from metasim.utils import configclass


@configclass
class RewardCfg:
    """Base class for reward computation.

    Attributes:
    scales: for reward weights shaping.
    base_height_target: 0.89
    min_dist: TODO: fill it
    max_dist: TODO: fill it
    target_joint_pos_scale: 0.17 TODO: fill it
    target_feet_height: 0.06 TODO: fill it
    cycle_time: 0.64  # sec
    only_positiverewards: whether only use positive rewards
    tracking_sigma: tracking reward = exp(error*sigma)
    max_contact_force: 700  # forces above this value are penalized
    soft_torque_limit: 0.001
    """

    base_height_target: float = 0.89
    min_dist: float = 0.2
    max_dist: float = 0.5
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale: float = 0.17  # rad
    target_feet_height: float = 0.06  # m
    cycle_time: float = 0.64  # sec
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positiverewards: bool = True
    # tracking reward = exp(error*sigma)
    tracking_sigma: float = 5.0
    max_contact_force: float = 700.0  # forces above this value are penalized
    soft_torque_limit: float = 0.001


@configclass
class CommandRanges:
    """Command Ranges for random command sampling when training."""

    lin_vel_x: list[float] = [-1.0, 2.0]
    lin_vel_y: list[float] = [-1.0, 1.0]
    ang_vel_yaw: list[float] = [-1.0, 1.0]
    heading: list[float] = [-3.14, 3.14]


@configclass
class CommandsConfig:
    """Configuration for command generation.

    Attributes:
        curriculum: whether to start curriculum training
        max_curriculum.
        num_commands: number of commands.
        resampling_time: time before command are changed[s].
        heading_command: whether to compute ang vel command from heading error.
        ranges: upperbound and lowerbound of sampling ranges.
    """

    curriculum: bool = False
    max_curriculum: float = 1.0
    num_commands: int = 4  # linear x, linear y, angular velocity, heading
    resampling_time: float = 10.0
    heading_command: bool = True


# FIXME align config in with @configclass
class BaseConfig:
    def __init__(self) -> None:
        """Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


# FIXME align config in with @configclass
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        wandb: False


# FIXME align config
class Normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 1.0
        dof_pos = 1.0
        dof_vel = 0.05
        quat = 1.0
        height_measurements = 5.0

    clip_observations = 18.0
    clip_actions = 18.0


@configclass
class BaseLeggedRobotChecker(BaseChecker):
    # FIXME I want to get epsiode length which is updated at env_wrapper, but the params handler.

    def check(self, handler: BaseSimHandler):
        from metasim.utils.humanoid_robot_util import contact_forces_tensor

        states = handler.get_states()
        contact_forces = contact_forces_tensor(states, handler.robot.name)
        reset_buf = torch.any(
            torch.norm(contact_forces[:, handler.task.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        return reset_buf


@configclass
class BaseLeggedTaskCfg(BaseRLTaskCfg):
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    decimation: int = 10
    num_obs: int = 124
    num_privileged_obs: int = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
    num_actions: int = 12
    env_spacing: float = 1.0  # not used with heightfields/trimeshes
    send_timeouts: bool = True  # send time out information to the algorithm
    episode_length_s: float = 20.0  # episode length in seconds

    # TODO given name, assign in envHandler
    feet_indices: torch.Tensor | None = None
    penalised_contact_indices: torch.Tensor | None = None
    termination_contact_indices: torch.Tensor | None = None
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        num_position_iterations=8,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.5,
        replace_cylinder_with_capsule=True,
    )
    dt = decimation * sim_params.dt  # simulation time step
    objects = []
    traj_filepath: str | None = None

    # TODO read form max_episode_length_s and divide s
    max_episode_length_s: int = 24
    episode_length: int = 2400
    max_episode_length: int = 2400
    control: ControlCfg = ControlCfg(action_scale=0.5, action_offset=True, torque_limit_scale=0.85)
    reward_cfg: RewardCfg = RewardCfg()
    commands = CommandsConfig()
    command_ranges: CommandRanges = CommandRanges()

    use_vision: bool = False

    # use Callable for Inheritance of reward functions
    # FIXME no correct import
    reward_functions: list[Callable] = [
        # copied from legged gym
        reward_termination,
        reward_tracking_lin_vel,
        reward_tracking_ang_vel,
        reward_lin_vel_z,
        reward_ang_vel_xy,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_base_height,
        reward_feet_air_time,
        reward_collision,
        reward_stumble,
        reward_action_rate,
        reward_stand_still,
        # copied from skillblender
        reward_feet_contact_forces,
        reward_dof_pos_limits,
        reward_dof_vel_limits,
        reward_torque_limits,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.0,
        "torques": -0.00001,
        "dof_vel": -0.0,
        "dof_acc": -2.5e-7,
        "base_height": -0.0,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "action_rate": -0.0,
        "stand_still": -0.0,
    }

    ppo_cfg: LeggedRobotCfgPPO = LeggedRobotCfgPPO()
    checker: BaseLeggedRobotChecker = BaseLeggedRobotChecker()
    normalization = Normalization()
