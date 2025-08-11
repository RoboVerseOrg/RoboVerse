from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

from dataclasses import MISSING
from typing import Callable

import torch
from loguru import logger as log

from metasim.cfg.checkers.base_checker import BaseChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.randomization import FrictionRandomCfg, MassRandomCfg, RandomizationCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.sim import BaseSimHandler
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import contact_forces_tensor


@configclass
class LeggedRobotCfgPPO:
    """Configuration for PPO."""

    seed = 1
    runner_class_name = "OnPolicyRunner"

    @configclass
    class Policy:
        """Network config class for PPO."""

        init_noise_std = 1.0
        """Initial noise std for actor network."""
        actor_hidden_dims = [512, 256, 128]
        """Hidden dimensions for actor network."""
        critic_hidden_dims = [768, 256, 128]
        """Hidden dimensions for critic network."""

    @configclass
    class Algorithm:
        """Training config class for PPO."""

        value_loss_coef = 1.0
        """Value loss coefficient."""
        use_clipped_value_loss = True
        """Use clipped value loss."""
        clip_param = 0.2
        """Clipping parameter for PPO."""
        entropy_coef = 0.001
        """Entropy coefficient."""
        num_learning_epochs = 5
        """Number of learning epochs."""
        num_mini_batches = 4
        """mini batch size = num_envs*n_steps / num_mini_batches"""
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    @configclass
    class Runner:
        """Runner config class for PPO."""

        policy_class_name = "ActorCritic"
        """Policy class name."""
        algorithm_class_name = "PPO"
        """Algorithm class name."""
        num_steps_per_env = 24
        """per iteration"""
        max_iterations = 1500
        """max number of iterations"""

        # logging
        save_interval = 1000
        """save interval for checkpoints"""
        experiment_name = "test"
        """experiment name"""
        run_name = ""
        resume = False
        """resume from checkpoint"""
        load_run = -1
        """load run number"""
        checkpoint = -1
        """checkpoint name"""
        resume_path = None
        """resume path"""
        wandb = False
        """Whether to use wandb."""

    policy: Policy = Policy()
    algorithm: Algorithm = Algorithm()
    runner: Runner = Runner()


@configclass
class LeggedRobotDomainRandCfg(RandomizationCfg):
    """Randomization config for legged robots."""

    def sample_uniform_buckets(params_dict: dict) -> torch.Tensor:
        """Sample friction coefficients uniformly via discrete buckets."""

        try:
            num_buckets = params_dict["num_buckets"]
            range = params_dict["range"]
            num_envs = params_dict["num_envs"]
            device = params_dict["device"]
        except KeyError as e:
            log.error("num_buckets, range and device must be specified for uniform sampling")
            raise e

        shape = (num_buckets, 1)
        bucket_ids = torch.randint(0, num_buckets, (num_envs, 1))
        friction_buckets = (range[1] - range[0]) * torch.rand(*shape, device=device) + range[0]
        friction_coeffs = friction_buckets[bucket_ids]
        return friction_coeffs

    def sample_uniform(params_dict: dict) -> torch.Tensor:
        """Sample friction coefficients uniformly."""

        try:
            range = params_dict["range"]
            num_envs = params_dict["num_envs"]
            device = params_dict["device"]
        except KeyError as e:
            log.error("range and device must be specified for uniform sampling")
            raise e

        shape = (num_envs, 1)
        friction_coeffs = (range[1] - range[0]) * torch.rand(*shape, device=device) + range[0]
        return friction_coeffs

    @configclass
    class PushRandomCfg:
        """Configuration for random push forces."""

        enabled: bool = False
        """Whether to enable random push forces."""
        max_push_vel_xy: float = 0.2
        """Maximum push velocity in xy plane."""
        max_push_ang_vel: float = 0.4
        """Maximum push angular velocity."""
        push_interval: int = 4
        """Interval in steps for applying random push forces and torques."""

    push = PushRandomCfg(enabled=True)

    def __post_init__(self):
        super().__post_init__()
        self.friction = FrictionRandomCfg(
            enabled=True, range=[0.1, 2.0], dist_fn=self.sample_uniform_buckets, num_buckets=256
        )
        self.mass = MassRandomCfg(enabled=True, range=[-1.0, 1.0], dist_fn=self.sample_uniform)


@configclass
class BaseLeggedTaskCfg(BaseRLTaskCfg):
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    @configclass
    class RewardCfg:
        """Constants for reward computation."""

        base_height_target: float = 0.89
        """target height of the base"""
        min_dist: float = 0.2
        """minimum distance between feet"""
        max_dist: float = 0.5
        """maximum distance between feet"""

        target_joint_pos_scale: float = 0.17
        """target joint position scale"""
        target_feet_height: float = 0.06
        """target feet height"""
        cycle_time: float = 0.64
        """cycle time"""

        only_positive_rewards: bool = True
        """whether to use only positive rewards"""
        tracking_sigma: float = 5.0
        """tracking reward = exp(error*sigma)"""
        max_contact_force: float = 700.0
        """maximum contact force"""
        soft_torque_limit: float = 0.001
        """soft torque limit"""

    reward_cfg: RewardCfg = RewardCfg()

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
        """whether to start curriculum training"""
        max_curriculum: float = 1.0
        """maximum value of curriculum"""
        num_commands: int = 4
        """number of commands. linear x, linear y, angular velocity, heading"""
        resampling_time: float = 10.0
        """time before command are changed[s]."""
        heading_command: bool = True
        """whether to compute ang vel command from heading error."""

    @configclass
    class BaseLeggedRobotChecker(BaseChecker):
        def check(self, handler: BaseSimHandler):
            """Check if the contact forces are above threshold threshold."""

            states = handler.get_states()
            contact_forces = contact_forces_tensor(states, handler.robot.name)
            reset_buf = torch.any(
                torch.norm(contact_forces[:, handler.task.termination_contact_indices, :], dim=-1) > 1.0, dim=1
            )
            return reset_buf

    @configclass
    class Normalization:
        """Normalization constants for observations and actions."""

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
    class CommandRanges:
        """Command Ranges for random command sampling when training."""

        lin_vel_x: list[float] = [-1.0, 2.0]
        lin_vel_y: list[float] = [-1.0, 1.0]
        ang_vel_yaw: list[float] = [-1.0, 1.0]
        heading: list[float] = [-3.14, 3.14]

    reward_functions: list[Callable] = MISSING
    reward_weights: dict[str, float] = MISSING

    robots: list[BaseRobotCfg] | None = None
    """List of robots in the environment."""
    command_ranges: CommandRanges = CommandRanges()
    """Command Ranges for random command sampling when training."""
    commands = CommandsConfig()
    """Configuration for command generation."""

    use_vision: bool = False
    """Whether to use vision observations."""
    ppo_cfg: LeggedRobotCfgPPO = LeggedRobotCfgPPO()
    """PPO config."""
    checker: BaseLeggedRobotChecker = BaseLeggedRobotChecker()
    """Checker for resetting the environment."""
    normalization = Normalization()
    """Normalization config."""
    decimation: int = 10
    """Decimation pd control loop."""
    num_obs: int = 124
    """Number of observations."""
    num_privileged_obs: int = None
    """Number of privileged observations. If not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned """
    num_actions: int = 12
    """Number of actions."""
    env_spacing: float = 1.0
    """Environment spacing."""
    send_timeouts: bool = True
    """Whether to send time out information to the algorithm"""
    episode_length_s: float = 20.0
    """episode length in seconds"""
    feet_indices: torch.Tensor = MISSING
    """feet indices"""
    penalised_contact_indices: torch.Tensor = MISSING
    """penalised contact indices for reward computation"""
    termination_contact_indices: torch.Tensor = MISSING
    """termination contact indices for reward computation"""
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.5,
        replace_cylinder_with_capsule=True,
        friction_offset_threshold=0.04,
        num_threads=10,
    )
    """Simulation parameters with physics engine settings."""
    dt = decimation * sim_params.dt
    """simulation time step in s"""
    objects = []
    """objects in the environment"""
    traj_filepath = None
    """path to the trajectory file"""
    # TODO read form max_episode_length_s and divide s
    max_episode_length_s: int = 24
    """maximum episode length in seconds"""
    episode_length: int = 2400
    """episode length in steps"""
    max_episode_length: int = 2400
    """episode length in steps"""
    control: ControlCfg = ControlCfg(action_scale=0.5, action_offset=True, torque_limit_scale=0.85)
    """Control config."""
    random: LeggedRobotDomainRandCfg = LeggedRobotDomainRandCfg()
    """Randomization config."""
