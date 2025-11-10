from typing import Callable
from metasim.utils import configclass
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.configs.algorithm.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CatchHumanoidTaskCfg(BaseEnvCfg):
    """
    Environment configuration for humanoid Catching task.
    """
    obs_len_history = 5
    priv_obs_len_history = 5
    control = BaseEnvCfg.Control(action_scale = 0.25)
    noise = BaseEnvCfg.Noise(add_noise=True)  # disable noise by default
    normalization = BaseEnvCfg.Normalization(
        obs_scales=BaseEnvCfg.Normalization.ObsScales(
            lin_vel = 1.0,
            ang_vel = 0.20,
            dof_pos = 1.0,
            dof_vel = 0.05,
            # height_measurements = 5.0
        )
    )
    class rewards:
        send_timeouts = True
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions = "roboverse_learn.rl.unitree_rl.configs.cfg_reward_funcs"
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
    class InitialStates:
        objects = {"ball": {"pos": [0.0, 0.0, 0.8]}}
        robots = {
            "g1_dof29_dex3": {"pos": [0.0, 0.0, 0.8]},
                }
    initial_states = InitialStates()

@configclass
class CatchHumanoidRslRlTrainCfg(RslRlOnPolicyRunnerCfg):
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
