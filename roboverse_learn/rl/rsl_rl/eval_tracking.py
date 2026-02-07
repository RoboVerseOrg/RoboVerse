from __future__ import annotations

import os
import random

try:
    # IsaacGym requires importing its modules before PyTorch.
    from isaacgym import gymapi  # noqa: F401
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

from roboverse_learn.rl.configs.rsl_rl.ppo_tracking import RslRlPPOTrackingConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.factory import make_task_env
from metasim.task.registry import get_task_class


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


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


def make_roboverse_env(args: RslRlPPOTrackingConfig):
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

    env = make_task_env(args.task, scenario=scenario, args=args, device=device)
    return env


def evaluate(args: RslRlPPOTrackingConfig):
    """Evaluate a trained RSL-RL PPO policy"""
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    if args.wandb_path:
        checkpoint_path = args.checkpoint_path

    elif args.resume:
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

        # checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        raise ValueError("Please provide either --wandb-path (WandB run path / model file path) or --resume (timestamp / log dir) for evaluation.")

    # Create environment
    print(f"Creating environment: {args.task} with {args.num_envs} environments")
    env = make_roboverse_env(args)

    # print(f"Loaded training config from task: {task_cls.__name__}")

    # Create environment wrapper
    env_wrapper = RslRlEnvWrapper(env, train_cfg=args.train_cfg)

    runner = OnPolicyRunner(
        env=env_wrapper,
        train_cfg=args.train_cfg,
        log_dir=None,
        device=device
    )
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # Reset environment
    obs = env_wrapper.get_observations()

    print(f"Starting evaluation for 1000000 steps...")
    for i in range(1000000):
        actions = policy(obs)
        obs, _, _, _ = env_wrapper.step(actions)

        if (i + 1) % 1000 == 0:
            print(f"Step {i + 1}/1000000")

    env_wrapper.close()
    print("Evaluation complete!")


if __name__ == "__main__":
    args = tyro.cli(RslRlPPOTrackingConfig)
    evaluate(args)
