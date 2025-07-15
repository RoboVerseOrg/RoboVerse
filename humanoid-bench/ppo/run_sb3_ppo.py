import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import wandb
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--wandb_entity", default="robot-learning", type=str)
ARGS = parser.parse_args()


def make_env(rank, seed=0):
    """Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        env = gym.make(ARGS.env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)

        env.action_space.seed(ARGS.seed + rank)

        return env

    return _init


class EvalCallback(BaseCallback):
    def __init__(self, eval_every: int = 100000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)

        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_env(1)])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()

        return True

    def record_video(self) -> None:
        print("recording video")
        video = []

        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render().transpose(2, 0, 1)
            video.append(pixels)

        video = np.stack(video)
        wandb.log({"results/video": wandb.Video(video, fps=100, format="gif")})


class LogCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
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
        for key in self.aux_returns.keys():
            self.logger.record(f"aux_returns_{key}/mean", np.mean(self.aux_returns[key]))


class EpisodeLogCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""

    def __init__(self, verbose=0, info_keywords=()):
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
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


def main(argv):
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])

    steps = 1000

    run = wandb.init(
        entity=ARGS.wandb_entity,
        project="humanoid-bench",
        name=f"ppo_{ARGS.env_name}",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=float(ARGS.learning_rate),
        batch_size=512,
    )
    model.learn(
        total_timesteps=ARGS.max_steps,
        log_interval=1,
        callback=[
            WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
            EvalCallback(),
            LogCallback(info_keywords=[]),
            EpisodeLogCallback(),
        ],
    )

    model.save("ppo")
    print("Training finished")


if __name__ == "__main__":
    #   app.run(main)
    main(None)
