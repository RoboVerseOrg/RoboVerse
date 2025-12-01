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

from roboverse_learn.rl.configs.algorithms.rsl_ppo import RslRlPPOConfig
from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper
from metasim.task.registry import get_task_class


def make_roboverse_env(args: RslRlPPOConfig):
    """Create RoboVerse task environment"""
    task_cls = get_task_class(args.task)

    # Load environment configuration from task (like MasterRunner does)
    env_cfg_cls = getattr(task_cls, "env_cfg_cls", None)
    env_cfg = env_cfg_cls() if env_cfg_cls is not None else None

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[]
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")

    # Pass env_cfg to task constructor
    env = task_cls(scenario=scenario, device=device, env_cfg=env_cfg)
    return env


def _merge_train_cfg(base_cfg: dict, args: RslRlPPOConfig) -> dict:
    """Merge base training config with CLI arguments.

    CLI arguments override base config values.
    RSL_RL expects runner params at TOP LEVEL (not under "runner" key),
    with nested 'policy' and 'algorithm' dicts.
    """
    import copy
    train_cfg = copy.deepcopy(base_cfg)

    # Runner settings at TOP LEVEL (not nested under "runner"!)
    train_cfg["seed"] = args.seed
    train_cfg["num_steps_per_env"] = args.num_steps_per_env
    train_cfg["max_iterations"] = args.max_iterations
    train_cfg["save_interval"] = args.save_interval
    train_cfg["experiment_name"] = args.exp_name
    train_cfg["empirical_normalization"] = args.empirical_normalization

    # Set required runner fields if not present
    if "run_name" not in train_cfg:
        train_cfg["run_name"] = ""
    if "resume" not in train_cfg:
        train_cfg["resume"] = False
    if "load_run" not in train_cfg:
        train_cfg["load_run"] = ".*"  # Regex pattern for loading
    if "load_checkpoint" not in train_cfg:
        train_cfg["load_checkpoint"] = "model_.*.pt"  # Not "checkpoint"!
    if "logger" not in train_cfg:
        train_cfg["logger"] = "tensorboard"

    # obs_groups for asymmetric observations (IMPORTANT!)
    if "obs_groups" not in train_cfg:
        train_cfg["obs_groups"] = {
            "policy": ["policy"],
            "critic": ["policy", "critic"]
        }

    # Policy settings - ensure nested dict exists
    if "policy" not in train_cfg:
        train_cfg["policy"] = {}
    policy_cfg = train_cfg["policy"]

    # Override policy settings with CLI args
    policy_cfg["class_name"] = "ActorCritic"  # NOT "policy_class_name"!
    policy_cfg["init_noise_std"] = args.init_noise_std
    policy_cfg["actor_hidden_dims"] = list(args.actor_hidden_dims)
    policy_cfg["critic_hidden_dims"] = list(args.critic_hidden_dims)
    policy_cfg["activation"] = args.activation

    # Set normalization if not present
    if "actor_obs_normalization" not in policy_cfg:
        policy_cfg["actor_obs_normalization"] = False
    if "critic_obs_normalization" not in policy_cfg:
        policy_cfg["critic_obs_normalization"] = False

    # Algorithm settings - ensure nested dict exists
    if "algorithm" not in train_cfg:
        train_cfg["algorithm"] = {}
    algo_cfg = train_cfg["algorithm"]

    # Override algorithm settings with CLI args
    algo_cfg["class_name"] = "PPO"  # NOT "algorithm_class_name"!
    algo_cfg["value_loss_coef"] = args.value_loss_coef
    algo_cfg["use_clipped_value_loss"] = args.use_clipped_value_loss
    algo_cfg["clip_param"] = args.clip_param
    algo_cfg["num_learning_epochs"] = args.num_learning_epochs
    algo_cfg["entropy_coef"] = args.entropy_coef
    algo_cfg["num_mini_batches"] = args.num_mini_batches
    algo_cfg["learning_rate"] = args.learning_rate
    algo_cfg["schedule"] = args.schedule
    algo_cfg["gamma"] = args.gamma
    algo_cfg["lam"] = args.lam
    algo_cfg["desired_kl"] = args.desired_kl
    algo_cfg["max_grad_norm"] = args.max_grad_norm

    return train_cfg


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

    # Load task's default training configuration (like MasterRunner does)
    task_cls = get_task_class(args.task)
    train_cfg_cls = getattr(task_cls, "train_cfg_cls", None)

    if train_cfg_cls is not None:
        # Instantiate task's train config
        task_train_cfg = train_cfg_cls() if callable(train_cfg_cls) else train_cfg_cls
        # Convert to dict if needed
        if not isinstance(task_train_cfg, dict):
            base_train_cfg = task_train_cfg.to_dict()
        else:
            base_train_cfg = task_train_cfg
        print(f"Loaded training config from task: {task_cls.__name__}")
    else:
        # Fallback: build config from CLI args
        base_train_cfg = {}
        print("No task-specific training config found, using CLI arguments")

    # Override with CLI arguments (CLI args take precedence)
    train_cfg = _merge_train_cfg(base_train_cfg, args)

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
