from __future__ import annotations

import os
import random

try:
    # IsaacGym requires importing its modules before PyTorch.
    # Importing only the top-level `isaacgym` package is not sufficient; we need to
    # import `gymapi` (which loads the native deps) before `torch` is imported.
    from isaacgym import gymapi  # noqa: F401
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
from metasim.task.factory import make_task_env
from metasim.task.registry import get_task_class


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def make_roboverse_env(args: RslRlPPOTrackingConfig):
    """Create RoboVerse task environment"""
    if args.task == "motion-tracking-isaaclab":
        # Ensure the task is registered (it is registered via module import side effects).
        import roboverse_pack.tasks.beyondmimic.isaaclab.envs.tracking_rl_env  # noqa: F401

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

    env = make_task_env(args.task, scenario=scenario, args=args, device=device)
    return env


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
    policy_path = os.path.join(args.model_dir, "policy.pt")
    actor_critic = runner.alg.policy

    class _ExportablePolicy(torch.nn.Module):
        def __init__(self, actor_critic_module: torch.nn.Module):
            super().__init__()
            self.actor = getattr(actor_critic_module, "actor")
            self.actor_obs_normalizer = getattr(actor_critic_module, "actor_obs_normalizer", torch.nn.Identity())

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            obs = self.actor_obs_normalizer(obs)
            return self.actor(obs)

    export_policy = _ExportablePolicy(actor_critic).eval().cpu()
    torch.jit.script(export_policy).save(policy_path)
    print(f"Policy exported to {policy_path}")

    if args.use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    args = tyro.cli(RslRlPPOTrackingConfig)
    train(args)
