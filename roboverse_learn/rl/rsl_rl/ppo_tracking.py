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

from roboverse_learn.rl.configs.rsl_rl.ppo_tracking import RslRlPPOTrackingConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.registry import get_task_class


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def make_roboverse_env(args: RslRlPPOTrackingConfig):
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
    env = task_cls(scenario=scenario, args=args, device=device)
    return env


def _export_policy(runner: OnPolicyRunner, model_dir: str) -> str:
    policy_filename = "policy.pt"
    policy_path = os.path.join(model_dir, policy_filename)
    runner.export_policy_to_jit(path=model_dir, filename=policy_filename)
    print(f"Policy exported to {policy_path} (runner.export_policy_to_jit)")
    return policy_path


def train(args: RslRlPPOTrackingConfig):
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
        # use artifact for training
        if args.registry_name:
            wandb.run.use_artifact(args.registry_name)

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
    _export_policy(runner=runner, model_dir=args.model_dir)

    if args.use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    args = tyro.cli(RslRlPPOTrackingConfig)
    train(args)
