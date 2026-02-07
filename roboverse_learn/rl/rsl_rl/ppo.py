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
from rsl_rl.runners import OnPolicyRunner

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_learn.rl.configs.rsl_rl.ppo import RslRlPPOConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.registry import get_task_class


def get_log_dir(exp_name: str, task_name: str, now=None) -> str:
    """Get the log directory (aligned with ppo.py saving logic)."""
    if now is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./outputs/{exp_name}/{task_name}/{now}"
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

    # Load environment configuration from task

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[]
    )
    if args.sim == "newton":
        if args.newton_use_mujoco_contacts is None:
            args.newton_use_mujoco_contacts = True
        scenario.sim_params.newton_use_mujoco_contacts = args.newton_use_mujoco_contacts
    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")

    # Pass env_cfg to task constructor
    env = task_cls(scenario=scenario, device=device)
    return env


def _export_policy(runner: OnPolicyRunner, model_dir: str) -> str:
    policy_filename = "policy.pt"
    policy_path = os.path.join(model_dir, policy_filename)
    runner.export_policy_to_jit(path=model_dir, filename=policy_filename)
    print(f"Policy exported to {policy_path} (runner.export_policy_to_jit)")
    return policy_path


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

    if args.resume:
        # Get the run directory
        exp_name = args.exp_name or args.experiment_name or args.task

        load_root = (
            args.resume
            if os.path.isdir(args.resume)
            else get_log_dir(exp_name=exp_name, task_name=args.task, now=args.resume)
        )

        checkpoint_num = args.checkpoint if args.checkpoint is not None else -1
        checkpoint_path = get_load_path(load_root=load_root, checkpoint=checkpoint_num)
        runner.load(checkpoint_path)

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
    args = tyro.cli(RslRlPPOConfig)
    train(args)
