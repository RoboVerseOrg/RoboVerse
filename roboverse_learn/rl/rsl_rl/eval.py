from __future__ import annotations

import os
import random

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import rootutils
import torch
import tyro
import datetime
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_learn.rl.configs.rsl_rl.ppo import RslRlPPOConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.registry import get_task_class

def get_log_dir(robot_name: str, task_name: str, now=None) -> str:
    """Get the log directory."""
    if now is None:
        now = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    log_dir = f"./outputs/{robot_name}/{task_name}/{now}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log.info("Log directory: {}", log_dir)
    return log_dir


def get_load_path(load_root: str, checkpoint: int | str = None) -> str:
    """Get the path to load the model from."""
    if isinstance(checkpoint, int):
        if checkpoint == -1:
            models = [file for file in os.listdir(load_root) if "model" in file and file.endswith(".pt")]
            models.sort(key=lambda m: f"{m!s:0>15}")
            model = models[-1]
            load_path = f"{load_root}/{model}"
        else:
            load_path = f"{load_root}/model_{checkpoint}.pt"
    else:
        load_path = f"{load_root}/{checkpoint}.pt"
    log.info(f"Loading checkpoint {checkpoint} from {load_root}")
    return load_path

def make_roboverse_env(args: RslRlPPOConfig):
    """Create RoboVerse task environment"""
    task_cls = get_task_class(args.task)

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[]
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")

    env = task_cls(scenario=scenario, device=device)
    return env, task_cls


def evaluate(args: RslRlPPOConfig):
    """Evaluate a trained RSL-RL PPO policy"""
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create environment
    print(f"Creating environment: {args.task} with {args.num_envs} environments")
    env, task_cls = make_roboverse_env(args)

    # Load checkpoint
    if not args.resume:
        raise ValueError("Please provide --resume (timestamp/log dir) for evaluation.")

    # Convert resume string to full log directory path
    log_dir = (
        args.resume
        if os.path.isdir(args.resume)
        else get_log_dir(robot_name=args.robot, task_name=args.task, now=args.resume)
    )

    # Use get_load_path helper to handle checkpoint loading logic
    # If checkpoint is None, default to -1 (latest checkpoint)
    checkpoint_num = args.checkpoint if args.checkpoint is not None else -1
    checkpoint_path = get_load_path(load_root=log_dir, checkpoint=checkpoint_num)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Loaded training config from task: {task_cls.__name__}")

    # Create environment wrapper
    wrapped_env = RslRlEnvWrapper(env, train_cfg=args.train_cfg)

    # Get observations from environment (needed for resolve_obs_groups)
    from rsl_rl.utils import resolve_obs_groups
    from rsl_rl.modules import ActorCritic

    obs = wrapped_env.get_observations()

    # Resolve obs_groups (mimicking OnPolicyRunner.__init__)
    default_sets = ["critic"]
    args.obs_groups = resolve_obs_groups(obs, {}, default_sets)
    obs_groups = args.obs_groups

    # Extract policy config
    policy_cfg = args.policy

    # Create actor-critic model with obs and obs_groups
    actor_critic = ActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=env.num_actions,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std,
    ).to(device)

    # Load the model weights
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()

    # Create inference policy (just the actor part)
    policy = actor_critic.act_inference

    # Disable curriculum and command resampling for eval
    env.cfg.curriculum.enabled = False
    env.cfg.commands.resampling_time = 1e6  # effectively disable command changes

    # Reset environment
    env.reset()
    obs, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=device))
    obs = wrapped_env.get_observations()

    print(f"Starting evaluation for 1000000 steps...")
    for i in range(1000000):
        # set fixed command
        env.commands_manager.value[:, 0] = 0.5
        env.commands_manager.value[:, 1] = 0.0
        env.commands_manager.value[:, 2] = 0.0
        actions = policy(obs)
        obs, _, _, _ = wrapped_env.step(actions)

        if (i + 1) % 1000 == 0:
            print(f"Step {i + 1}/1000000")

    print("Evaluation complete!")


if __name__ == "__main__":
    args = tyro.cli(RslRlPPOConfig)
    evaluate(args)
