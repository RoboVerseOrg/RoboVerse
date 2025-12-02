import math
from metasim.utils import configclass
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.configs.rsl_rl.algorithm import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
)
from roboverse_learn.rl.unitree_rl.helper.curriculum_utils import lin_vel_cmd_levels
from metasim.queries import ContactForces
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
class WalkG1Dof12EnvCfg(BaseEnvCfg):
    episode_length_s = 20.0
    obs_len_history = 1
    priv_obs_len_history = 1

    control = BaseEnvCfg.Control(
        action_scale=0.25, action_clip=100, soft_joint_pos_limit_factor=0.9
    )

    @configclass
    class RewardsScales:
        track_lin_vel_xy = (1.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (0.5, {"std": math.sqrt(0.25)})
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        flat_orientation = -1.0
        base_height = (-10.0, {"target_height": 0.78})
        joint_acc = -2.5e-7
        joint_vel = -0.001
        action_rate = -0.01
        joint_pos_limits = -5.0
        is_alive = 0.15
        joint_deviation_legs = (
            -1.0,
            {"joint_names": (".*_hip_roll_joint", ".*_hip_yaw_joint")},
            reward_funcs.joint_deviation_l1,
        )
        feet_slide = (-0.2, {"body_names": (".*ankle_roll.*")})
        # feet_swing_height = -20.0
        feet_clearance = (
            1.0,
            {
                "std": math.sqrt(0.05),
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "body_names": (".*ankle_roll.*"),
            },
        )
        # contact = 0.18
        feet_gait = (
            0.18,
            {
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "body_names": (".*ankle_roll.*"),
            },
        )
        energy = -1e-5
        ########################

    rewards = BaseEnvCfg.Rewards(scales=RewardsScales(), only_positive_rewards=True)

    commands = BaseEnvCfg.Commands(
        value=None,
        resample=step_funcs.resample_commands,
        heading_command=True,
        resampling_time=10.0,
        rel_standing_envs=0.02,
        ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        limit_ranges=BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

    curriculum = BaseEnvCfg.Curriculum(
        enabled=False, funcs={"lin_vel_cmd_levels": lin_vel_cmd_levels}
    )

    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_dof12",
            static_friction_range=(0.1, 1.25),
            dynamic_friction_range=(0.1, 1.25),
            restitution_range=(0.0, 0.0),
            num_buckets=64,
        ),
        "mass_randomizer": MassRandomizer(
            obj_name="g1_dof12",
            body_names="pelvis",
            mass_distribution_params=(-1.0, 3.0),
            operation="add",
        ),
    }
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state_terrain_aware,
            {
                "pose_range": [
                    [0., 0., 0, 0, 0, 0],  # x, y, z_offset, roll, pitch, yaw
                    [0., 0., 0, 0, 0, 0],
                ],
                "velocity_range": [[-0.5] * 6, [0.5] * 6],
                # base_height_offset is None by default, uses robot's default z position (0.8m from cfg_base.py)
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (0.5, 1.5), "velocity_range": (1.0, 1.0)},
        ),
    }
    callbacks_post_step = {
        "push_robot": (
            step_funcs.push_by_setting_velocity,
            {
                "interval_range_s": (5.0, 5.0),
                "velocity_range": [[-1.5, -1.5, 0.0], [1.5, 1.5, 0.0]],
            },
        )
    }
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "undesired_contact": (
            termination_funcs.undesired_contact,
            {
                "contact_names": [
                    ".*_elbow_.*",
                    ".*_shoulder_.*",
                    ".*_wrist_.*",
                    "pelvis",
                    "torso_link",
                ],
                "limit_range": 1.0,
            },
        ),
        "bad_orientation": (termination_funcs.bad_orientation, {"limit_angle": 0.8}),
    }


@configclass
class WalkG1Dof12RslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate=1.0e-3,  # 5.e-4
        schedule="adaptive",  # could be adaptive, fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
