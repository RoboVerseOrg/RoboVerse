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
from rsl_rl.runners import OnPolicyRunner

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_learn.rl.configs.rsl_rl.ppo import RslRlPPOConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.registry import get_task_class


def make_roboverse_env(args: RslRlPPOConfig):
    """Create RoboVerse task environment"""
    task_cls = get_task_class(args.task)

    # Load environment configuration from task

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[]
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")

    # Pass env_cfg to task constructor
    env = task_cls(scenario=scenario, device=device)
    return env


def train(args: RslRlPPOConfig):
    """Train RSL-RL PPO"""
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.model_dir, exist_ok=True)

    # Initialize WandB
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.exp_name,
            save_code=True
        )

    # Create environment and wrapper
    print(f"Creating environment: {args.task} with {args.num_envs} environments")
    env = make_roboverse_env(args)

    # Use training config directly from args
    train_cfg = args.train_cfg

    # Create environment wrapper
    env_wrapper = RslRlEnvWrapper(env, train_cfg=train_cfg)


    runner = OnPolicyRunner(
        env=env_wrapper,
        train_cfg=train_cfg,
        log_dir=args.model_dir,
        device=device
    )

    # Train
    print(f"Training RSL-RL PPO on {args.task} with {args.num_envs} environments")
    print(f"Model directory: {args.model_dir}")
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True
    )

    # Export policy
    print("Exporting policy...")
    policy = runner.get_inference_policy()
    policy_path = os.path.join(args.model_dir, "policy.pt")
    torch.jit.script(policy).save(policy_path)
    print(f"Policy exported to {policy_path}")

    if args.use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    args = tyro.cli(RslRlPPOConfig)
    train(args)
