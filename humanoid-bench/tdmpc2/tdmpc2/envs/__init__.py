import warnings
from copy import deepcopy

import gymnasium as gym

from tdmpc2.envs.wrappers.multitask import MultitaskWrapper
from tdmpc2.envs.wrappers.pixels import PixelWrapper
from tdmpc2.envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(f"Missing dependencies for task {task}; install dependencies to use this environment.")


from tdmpc2.envs.dmcontrol import make_env as make_dm_control_env
from tdmpc2.envs.humanoid import make_env as make_humanoid_env

try:
    from tdmpc2.envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from tdmpc2.envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from tdmpc2.envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """Make a multi-task environment for TD-MPC2 experiments."""
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """Make an environment for TD-MPC2 experiments."""
    gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        for fn in [
            make_humanoid_env,
            make_dm_control_env,
            make_maniskill_env,
            make_metaworld_env,
            make_myosuite_env,
        ]:
            try:
                env = fn(cfg)
            except ValueError:
                pass
        if env is None:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.'
            )
        env = TensorWrapper(env)
    if cfg.get("obs", "state") == "rgb":
        env = PixelWrapper(cfg, env)
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
