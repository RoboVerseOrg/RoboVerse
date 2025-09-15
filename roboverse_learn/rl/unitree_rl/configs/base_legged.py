from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

import pathlib
from dataclasses import MISSING
from typing import Callable, Literal

import torch
from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg

from metasim.sim import BaseSimHandler
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import contact_forces_tensor, get_euler_xyz_tensor, robot_rotation_tensor
from roboverse_learn.rl.unitree_rl.helper.utils import torch_rand_float
from metasim.queries.base import BaseQueryType
from roboverse_learn.rl.unitree_rl.configs.optional_queries import NetContactForce
from roboverse_learn.rl.unitree_rl.configs.base_terrain import TerrainConfig


# Training Config
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
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = None
        rnn_hidden_size = None
        rnn_num_layers = None

    @configclass
    class Algorithm:
        """Training config class for PPO."""

        value_loss_coef = 1.0
        """Value loss coefficient."""
        use_clipped_value_loss = True
        """Use clipped value loss."""
        clip_param = 0.2
        """Clipping parameter for PPO."""
        entropy_coef = 0.01
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
        max_iterations = 150001
        """max number of iterations"""

        # logging
        save_interval = 50
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


# Randomization
@configclass
class LeggedRobotDomainRandCfg:
    """Randomization config for legged robots."""

    camera: bool = False
    """Randomize camera pose"""
    light: bool = False
    """Randomize light direction, temperature, intensity"""
    ground: bool = False
    """Randomize ground"""
    reflection: bool = False
    """Randomize reflection (roughness, metallic, reflectance), attach random diffuse color to surfaces that have no material"""
    table: bool = False
    """Randomize table albedo"""
    wall: bool = False
    """Add wall and roof, randomize wall"""
    scene: bool = False
    """Randomize scene"""
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level"""

    # New-style randomization configs (require robot name via obj_name)
    friction = None
    """Friction randomization config (new API). Must set obj_name later."""
    mass = None
    """Mass randomization config (new API). Must set obj_name later."""

    def sample_uniform_buckets(params_dict: dict) -> torch.Tensor:
        """Sample friction coefficients uniformly via discrete buckets."""

        try:
            num_buckets = params_dict["num_buckets"]
            range = params_dict["range"]
            num_envs = params_dict["num_envs"]
            device = str(params_dict["device"])
        except KeyError as e:
            print("num_buckets, range and device must be specified for uniform sampling")
            raise e

        bucket_ids = torch.randint(0, num_buckets, (num_envs, 1))
        friction_buckets = torch_rand_float(range[0], range[1], (num_buckets, 1), device=device)
        friction_coeffs = friction_buckets[bucket_ids]
        return friction_coeffs

    def sample_uniform(params_dict: dict) -> torch.Tensor:
        """Sample friction coefficients uniformly."""

        try:
            range = params_dict["range"]
            num_envs = params_dict["num_envs"]
            device = str(params_dict["device"])
        except KeyError as e:
            print("range and device must be specified for uniform sampling")
            raise e

        shape = (num_envs, 1)
        friction_coeffs = torch_rand_float(range[0], range[1], shape, device=device)
        return friction_coeffs

    @configclass
    class PushRandomCfg:
        """Configuration for random push forces."""

        enabled: bool = False
        """Whether to enable random push forces."""
        max_push_vel_xy: float = 1.5
        """Maximum push velocity in xy plane."""
        max_push_ang_vel: float = 0.5
        """Maximum push angular velocity."""
        push_interval: int = 500
        """Interval in steps for applying random push forces and torques."""

    push = PushRandomCfg(enabled=True)

    terrain_cfg: TerrainConfig = TerrainConfig.from_yaml(pathlib.Path(__file__).parent / "terrain.yaml")
    """Terrain randomization configuration"""

    def __post_init__(self):
        """Post-initialization configuration."""
        assert self.level in [0, 1, 2, 3]
        if self.level >= 0:
            pass
        if self.level >= 1:
            self.table = True
            self.ground = True
            self.wall = True
        if self.level >= 2:
            self.camera = True
        if self.level >= 3:
            self.light = True
            self.reflection = True
        # Friction and mass configs are populated in the TaskCfg __post_init__
        # once the robot name is available.
        if self.ground:
            assert self.terrain_cfg is not None, (
                "Terrain randomization config must be provided if ground randomization is enabled."
            )


# Config
@configclass
class BaseLeggedTaskCfg:
    """Base class for legged-gym style humanoid tasks. Deault values are designed for Unitree Go2.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    @configclass
    class RewardCfg:
        """Constants for reward computation."""

        base_height_target: float = 1.0
        """target height of the base"""
        min_dist: float = 0.2
        """minimum distance between feet"""
        max_dist: float = 0.5
        """maximum distance between feet"""

        target_joint_pos_scale: float = 0.17
        """target joint position scale"""
        target_feet_height: float = 0.06
        """target feet height"""
        feet_cycle_time: float = 0.64
        """cycle time"""
        feet_full_contact_time: float = 0.1
        """the phase that all feet contact with the ground"""
        feet_contact_threshold: float = 1.0
        """the contact force threshold"""

        only_positive_rewards: bool = True
        """whether to use only positive rewards"""
        tracking_sigma: float = 1 / 0.25
        """tracking reward = exp(error*sigma)"""
        max_contact_force: float = 100.0
        """maximum contact force"""
        soft_dof_pos_limit: float = 1.0
        """# percentage of urdf limits, values above this limit are penalized"""
        soft_dof_vel_limit: float = 1.0
        """soft dof velocity limit"""
        soft_torque_limit: float = 1.0
        """soft torque limit"""

    reward_cfg: RewardCfg = RewardCfg()
    reward_functions: list[Callable] | str = "roboverse_learn.rl.unitree_rl.configs.reward_funcs"
    reward_weights: dict[str, float] = MISSING

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

        class Ranges:
            """Command Ranges for random command sampling when training."""

            lin_vel_x: list[float] = [-1.0, 1.0]
            lin_vel_y: list[float] = [-1.0, 1.0]
            ang_vel_yaw: list[float] = [-1.0, 1.0]
            heading: list[float] = [-3.14, 3.14]

        curriculum: bool = False
        """whether to start curriculum training"""
        max_curriculum: float = 1.0
        """maximum value of curriculum"""
        num_commands: int = 4
        """number of commands. linear x, linear y, angular velocity, heading"""
        commands_dim: int = 3
        "the used commands dimension, e.g., 3 for linear x, linear y, angular velocity"
        resampling_time: float = 10.0
        """time before command are changed[s]."""
        heading_command: bool = True
        """whether to compute ang vel command from heading error."""
        ranges: Ranges = Ranges()
        """upperbound and lowerbound of commands."""

    commands = CommandsConfig()
    """Configuration for command generation with command ranges."""


    @configclass
    class Normalization:
        """Normalization constants for observations and actions."""

        class ObsScales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.0

        clip_observations = 100.0
        clip_actions = 100.0
        obs_scales = ObsScales()

    normalization = Normalization()
    """Normalization config."""

    @configclass
    class Noise:
        class NoiseScales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

        add_noise = True
        noise_level = 1.0  # scales other values
        noise_scales = NoiseScales()

    noise = Noise()
    """Noise config."""

    @configclass
    class ControlCfg:
        action_scale: float = 1.0  # to scale the actions
        torque_limit_scale: float = 1.0  # scale it down can ensure safety
        action_offset: bool = False  # set true if target position = action * action_scale + default position

    control: ControlCfg = ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)
    """Control config."""

    robots: list[RobotCfg] | None = None
    """List of robots in the environment."""
    objects = []
    """objects in the environment"""
    use_vision: bool = False
    """Whether to use vision observations."""
    ppo_cfg: LeggedRobotCfgPPO = LeggedRobotCfgPPO()
    """PPO config."""
    num_observations: int = None
    """Number of observations."""
    num_privileged_obs: int = None
    """Number of privileged observations. If not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned """
    num_actions: int = None
    """Number of actions."""
    env_spacing: float = 3.0
    """Environment spacing."""
    send_timeouts: bool = True
    """Whether to send time out information to the algorithm"""
    # episode_length_s: float = 20.0
    # """episode length in seconds"""
    feet_indices: torch.Tensor = MISSING
    """feet indices"""
    penalised_contact_indices: torch.Tensor = MISSING
    """penalised contact indices for reward computation"""
    termination_contact_indices: torch.Tensor = MISSING
    """termination contact indices for reward computation"""
    sim_params = SimParamCfg(
        dt=0.005,
        substeps=1,
        num_threads=10,
        solver_type=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        contact_offset=0.01,
        rest_offset=0.0,
        bounce_threshold_velocity=0.5,
        max_depenetration_velocity=1.0,
        default_buffer_size_multiplier=5,
        replace_cylinder_with_capsule=True,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )
    traj_filepath = None
    """path to the trajectory file"""
    # TODO read form max_episode_length_s and divide s
    max_episode_length_s: float = 20.0
    """maximum episode length in seconds"""
    # episode_length: int = 2400
    # """episode length in steps"""
    # max_episode_length: int = 2400
    # """episode length in steps"""
    random: LeggedRobotDomainRandCfg = LeggedRobotDomainRandCfg(level=0)
    """Randomization config."""
    init_states = [
        {
            "objects": {},
            "robots": {
                "go2": {
                    "pos": torch.tensor([0.0, 0.0, 0.42]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "FL_hip_joint": 0.1,
                        "RL_hip_joint": 0.1,
                        "FR_hip_joint": -0.1,
                        "RR_hip_joint": -0.1,
                        "FL_thigh_joint": 0.8,
                        "RL_thigh_joint": 1.0,
                        "FR_thigh_joint": 0.8,
                        "RR_thigh_joint": 1.0,
                        "FL_calf_joint": -1.5,
                        "RL_calf_joint": -1.5,
                        "FR_calf_joint": -1.5,
                        "RR_calf_joint": -1.5,
                    },
                },
                "g1_dof12": {
                    "pos": torch.tensor([0.0, 0.0, 0.8]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # Hips & legs
                        "left_hip_yaw_joint": 0.0,
                        "left_hip_roll_joint": 0.0,
                        "left_hip_pitch_joint": -0.1,
                        "left_knee_joint": 0.3,
                        "left_ankle_pitch_joint": -0.2,
                        "left_ankle_roll_joint": 0.0,
                        "right_hip_yaw_joint": 0.0,
                        "right_hip_roll_joint": 0.0,
                        "right_hip_pitch_joint": -0.1,
                        "right_knee_joint": 0.3,
                        "right_ankle_pitch_joint": -0.2,
                        "right_ankle_roll_joint": 0.0,
                    },
                },
            },
        }
    ]
    """Initial states for the environment. Only used for legged robots, e.g., go2-12dof, g1-12dof, h1-12dof."""
    extra_spec: dict[str, BaseQueryType] = {"contact_forces": NetContactForce()}


    def __post_init__(self):
        # super().__post_init__()
        """simulation time step in s"""
        # self.dt = self.decimation * self.sim_params.dt
        from math import ceil

        # self.max_episode_length = ceil(self.max_episode_length_s / self.dt)
        # self.episode_length = ceil(self.max_episode_length_s / self.dt)
        """maximum episode length in steps"""
        # set the number of actions based on the robot's joints if available
        if self.robots is not None:
            self.num_actions = self.robots[0].num_joints
        assert self.num_actions is not None, "num_actions must be set in the task config"

        # TODO: Populate domain randomization configs that require robot name
