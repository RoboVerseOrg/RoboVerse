import math

from metasim.utils import configclass

from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.configs.algorithm import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
import roboverse_learn.rl.unitree_rl.helper.curriculum_utils as curr_funs
from roboverse_learn.rl.unitree_rl.configs.cfg_queries import ContactForces
from roboverse_learn.rl.unitree_rl.configs.cfg_randomizers import (
    MaterialRandomizer,
    MassRandomizer,
)

from roboverse_learn.rl.unitree_rl.configs.callback_funcs import (
    termination_funcs,
    reset_funcs,
    step_funcs,
    reward_funcs,
)


@configclass
class WalkG1Dof29EnvCfg(BaseEnvCfg):
    """
    Environment configuration for humanoid walking task.
    """

    obs_len_history = 5
    priv_obs_len_history = 5
    episode_length_s = 20.0

    control = BaseEnvCfg.Control(action_scale=0.25, soft_joint_pos_limit_factor=0.9)

    @configclass
    class RewardsScales:
        track_lin_vel_xy = (1.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (0.5, {"std": math.sqrt(0.25)})
        is_alive = 0.15
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        joint_vel = -0.001
        joint_acc = -2.5e-7
        action_rate = -0.05
        joint_pos_limits = -5.0
        energy = -2e-5
        joint_deviation_arms = (
            -0.1,
            {"joint_names": (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*")},
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_waists = (
            -1.0,
            {"joint_names": "waist.*"},
            reward_funcs.joint_deviation_l1,
        )
        joint_deviation_legs = (
            -1.0,
            {"joint_names": (".*_hip_roll_joint", ".*_hip_yaw_joint")},
            reward_funcs.joint_deviation_l1,
        )
        flat_orientation = -5.0
        base_height = (-10.0, {"target_height": 0.78})
        feet_gait = (
            0.5,
            {
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "body_names": (".*ankle_roll.*"),
            },
        )
        feet_slide = (-0.2, {"body_names": (".*ankle_roll.*")})
        feet_clearance = (
            1.0,
            {
                "std": math.sqrt(0.05),
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "body_names": (".*ankle_roll.*"),
            },
        )
        undesired_contacts = (-1.0, {"threshold": 1, "body_names": ("(?!.*ankle.*).*")})

    rewards = BaseEnvCfg.Rewards(
        only_positive_rewards=False,
        scales=RewardsScales(),
    )

    commands = BaseEnvCfg.Commands(
        value=None,
        resample=step_funcs.resample_commands,
        heading_command=False,
        rel_standing_envs=0.02,
        ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_yaw=(-0.1, 0.1)
        ),
        limit_ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_yaw=(-0.2, 0.2)
        ),
    )

    curriculum = BaseEnvCfg.Curriculum(
        enabled=True,
        funcs={
            "lin_vel_cmd_levels": curr_funs.lin_vel_cmd_levels,
            #  "terrain_levels": curr_funs.terrain_levels_vel
        },
    )

    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_dof29",
            static_friction_range=(0.3, 1.0),
            dynamic_friction_range=(0.3, 1.0),
            restitution_range=(0.0, 0.0),
            num_buckets=64,
        ),
        "mass_randomizer": MassRandomizer(
            obj_name="g1_dof29",
            body_names="torso_link",
            mass_distribution_params=(-1.0, 3.0),
            operation="add",
        ),
    }
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state,
            {
                "pose_range": [
                    [-0.5, -0.5, 0.0, 0, 0, -3.14],  # x,y,z roll,pitch,yaw
                    [0.5, 0.5, 0.0, 0, 0, 3.14],
                ],
                "velocity_range": [[0] * 6, [0] * 6],
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
        ),
    }
    callbacks_post_step = {
        "push_robot": (
            step_funcs.push_by_setting_velocity,
            {
                "interval_range_s": (5.0, 5.0),
                "velocity_range": [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
            },
        )
    }
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "base_height": (
            termination_funcs.root_height_below_minimum,
            {"minimum_height": 0.2},
        ),
        "bad_orientation": (termination_funcs.bad_orientation, {"limit_angle": 0.8}),
    }


@configclass
class WalkG1Dof29EnvRslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
