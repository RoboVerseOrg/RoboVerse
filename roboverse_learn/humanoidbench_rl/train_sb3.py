from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import rootutils
import wandb
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim", type=str, default="isaaclab", choices=["isaaclab", "isaacgym", "mujoco"])
    parser.add_argument(
        "--robot",
        type=str,
        default="h1",
        choices=[
            "franka",
            "ur5e_2f85",
            "sawyer",
            "franka_with_gripper_extension",
            "h1_2_without_hand",
            "h1",
            "h1_simple_hand",
        ],
    )
    parser.add_argument("--task", type=str, default="Stand")
    parser.add_argument("--add_table", action="store_true")
    parser.add_argument("--total_timesteps", type=int, default=100000000)
    parser.add_argument("--train_or_eval", type=str, default="train", choices=["train", "eval", "test"])
    parser.add_argument("--model_path", type=str, default=None)
    # PPO specific arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=2)
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="metasim_rl_training")
    parser.add_argument("--wandb_entity", type=str, default="<your_wandb_entity>")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.sim == "isaacgym":
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            name=f"SB3-{args.sim}-{args.robot}-{args.task}-{time.strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    else:
        from collections import namedtuple

        Run = namedtuple("Run", ["id"])
        run = Run(id=int(time.time()))

    # Create scenario config
    from metasim.cfg.scenario import ScenarioCfg

    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        try_add_table=args.add_table,
        sim=args.sim,
        num_envs=args.num_envs,
        headless=True,
        cameras=[],
    )

    from roboverse_learn.humanoidbench_rl.wrapper_sb3 import Sb3EnvWrapper

    if args.sim == "mujoco":
        if args.num_envs > 1:
            log.error("Mujoco does not support multiple environments > 1")
            exit()
        env = Sb3EnvWrapper(scenario=scenario)
    elif args.sim == "isaacgym":
        env = Sb3EnvWrapper(scenario=scenario)
    elif args.sim == "isaaclab":
        env = Sb3EnvWrapper(scenario=scenario)
    else:
        raise ValueError(f"Invalid sim type: {args.sim}")

    # Initialize PPO algorithm
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
        tensorboard_log=f"./ppo_logs/{run.id}",
        device="cpu",
    )

    # Setup wandb callback with additional monitoring options
    from wandb.integration.sb3 import WandbCallback

    wandb_callback = (
        WandbCallback(
            model_save_freq=100000,
            model_save_path=f"models/{run.id}",
            verbose=2,
            gradient_save_freq=100000,  # Added to track gradients
        )
        if args.use_wandb
        else None
    )

    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLogCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.returns_info = {
                "results/return": [],
                "results/episode_length": [],
                "results/success": [],
                "results/success_subtasks": [],
            }

        def _on_step(self) -> bool:
            infos = self.locals["infos"]
            for idx in range(len(infos)):
                curr_info = infos[idx]
                if "episode" in curr_info:
                    self.returns_info["results/return"].append(curr_info["episode"]["r"])
                    self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                    cur_info_success = curr_info.get("success", 0)
                    self.returns_info["results/success"].append(cur_info_success)
                    # cur_info_success_subtasks = curr_info.get("success_subtasks", 0)
                    # self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
            return True

        def _on_rollout_end(self) -> None:
            # Use the model's num_timesteps (cumulative across envs) as the global step.
            global_step = self.model.num_timesteps
            for key in self.returns_info.keys():
                if self.returns_info[key]:
                    self.logger.record(key, np.mean(self.returns_info[key]), global_step)
                    self.returns_info[key] = []

    class EvalCallback(BaseCallback):
        def __init__(self, eval_every: int = 5000, scenario=None, sim_type=None, verbose: int = 0):
            super().__init__(verbose=verbose)
            self.eval_every = eval_every
            self.eval_env = None
            self.scenario = scenario
            self.sim_type = sim_type

        def _on_training_start(self) -> None:
            from wrapper_sb3 import Sb3EnvWrapper

            from metasim.constants import SimType

            if self.sim_type == SimType.MUJOCO:
                self.eval_env = Sb3EnvWrapper(scenario=self.scenario)
            elif self.sim_type == SimType.ISAACGYM:
                self.eval_env = Sb3EnvWrapper(scenario=self.scenario)
            else:
                raise ValueError(f"Invalid sim type: {self.sim_type}")

        def _on_step(self) -> bool:
            if self.num_timesteps % self.eval_every == 0:
                self.record_video()
            return True

        def record_video(self) -> None:
            print("recording video")
            obs, info = self.eval_env.reset()
            video = [self.eval_env.render().transpose(2, 0, 1)]
            for _ in range(1000):
                action = self.model.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                pixels = self.eval_env.render()
                if pixels is not None:
                    video.append(pixels.transpose(2, 0, 1))
                if terminated or truncated:
                    break

            if video:
                video = np.stack(video)
                wandb.log({"results/video": wandb.Video(video, fps=10, format="mp4")}, step=self.model.num_timesteps)
            else:
                log.warning("No video recorded")

    class LogCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0, info_keywords=(), args=None):
            super().__init__(verbose)
            self.aux_rewards = {}
            self.aux_returns = {}
            for key in info_keywords:
                self.aux_rewards[key] = np.zeros(args.num_envs)
                self.aux_returns[key] = deque(maxlen=100)

        def _on_step(self) -> bool:
            infos = self.locals["infos"]
            for idx in range(len(infos)):
                for key in self.aux_rewards.keys():
                    self.aux_rewards[key][idx] += infos[idx][key]

                if self.locals["dones"][idx]:
                    for key in self.aux_rewards.keys():
                        self.aux_returns[key].append(self.aux_rewards[key][idx])
                        self.aux_rewards[key][idx] = 0
            return True

        def _on_rollout_end(self) -> None:
            global_step = self.model.num_timesteps
            for key in self.aux_returns.keys():
                self.logger.record(f"aux_returns_{key}/mean", np.mean(self.aux_returns[key]), global_step)

    if args.train_or_eval == "train":
        # Train the agent with additional callbacks
        log.info("Starting training...")
        # from metasim.constants import SimType
        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=1,
            callback=[
                # EvalCallback(scenario=scenario, sim_type=SimType(args.sim)),
                EpisodeLogCallback(),
                LogCallback(args=args),
            ]
            + ([wandb_callback] if args.use_wandb else []),
            progress_bar=True,
        )

        # Save the trained model
        model_path = f"models/ppo_model_{args.robot}_{args.sim}_{run.id}"
        model.save(model_path)
        log.info(f"Model saved to {model_path}")
    elif args.train_or_eval == "eval":
        # Load the trained model
        model.load(args.model_path)

        # Evaluate the agent
        log.info("Starting evaluation...")
        obs, info = env.reset()
        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
                import matplotlib.pyplot as plt

                plt.plot(rewards)
                plt.savefig("rewards.png")
                plt.clf()
                rewards = []

    elif args.train_or_eval == "test":
        # Test the joints
        log.info("Starting evaluation...")
        obs, info = env.reset()
        rewards = []

        step = 0
        target_joint = 0
        while True:
            step += 1
            action = np.array(
                [0] * 19,
                dtype=np.float32,
            )
            action[target_joint] = np.sin(step / 10)
            # action = env.normalize_action(np.array([0,0,-0.4,0.8,-0.4,0,0,-0.4,0.8,-0.4,0,0,0,0,0,0,0,0,0]))
            # action = env.normalize_action(obs[7:26])
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

    # Close environment and wandb
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
