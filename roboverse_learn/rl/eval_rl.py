from __future__ import annotations

import os
import time

import imageio
import numpy as np
import rootutils
from loguru import logger as log
from rich.logging import RichHandler

from roboverse_learn.rl.env import RLEnvWrapper

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def set_np_formatting():
    """Formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    if cfg.environment.sim_name == "isaacgym":
        import isaacgym  # noqa: F401
    import torch

    set_np_formatting()

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.utils.setup_util import SimType, get_robot, get_sim_env_class, get_task
    from roboverse_learn.rl.algos import get_algorithm

    # Environment setup
    cprint("Setting up environment for evaluation", "green", attrs=["bold"])
    task = get_task(cfg.train.task_name)
    robot = get_robot(cfg.train.robot_name)
    scenario = ScenarioCfg(task=task, robots=[robot])
    scenario.cameras = []

    tic = time.time()
    scenario.num_envs = cfg.environment.num_envs
    scenario.headless = cfg.environment.headless

    env_class = get_sim_env_class(SimType(cfg.environment.sim_name))
    env = env_class(scenario)

    env = RLEnvWrapper(
        gym_env=env,
        seed=cfg.experiment.seed,
        verbose=False,
    )

    if hasattr(env, "set_seed"):
        env.set_seed(cfg.experiment.seed)

    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    checkpoint_path = cfg.experiment.checkpoint
    if checkpoint_path is None:
        log.error("No checkpoint specified for evaluation. Use --experiment.checkpoint to specify.")
        return

    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    output_dir = os.path.join("eval_outputs", cfg.experiment.output_name)
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Initialize agent
    algo_name = cfg.train.algo.lower()
    agent = get_algorithm(algo_name, env=env, output_dif=output_dir, full_config=OmegaConf.to_container(cfg))
    log.info(f"Loading policy from checkpoint: {checkpoint_path}")
    agent.restore_test(checkpoint_path)

    num_eval_episodes = cfg.experiment.get("num_eval_episodes", 10)
    max_episode_steps = task.episode_length
    save_video = cfg.experiment.get("save_video", True)

    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    step_counter = 0

    for episode in range(num_eval_episodes):
        log.info(f"Starting evaluation episode {episode + 1}/{num_eval_episodes}")

        obs_dict = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_success = False
        video_frames = [] if save_video else None

        if hasattr(env, "render"):
            frame = env.render()
            if frame is not None and save_video:
                video_frames.append(frame)

        done = torch.zeros(env.num_envs, dtype=torch.bool)
        while not done.all() and episode_steps < max_episode_steps:
            if hasattr(agent, "running_mean_std"):
                obs_normalized = agent.running_mean_std(obs_dict["obs"])
                input_dict = {"obs": obs_normalized}
                action = agent.model.act_inference(input_dict)
                action = torch.clamp(action, -1.0, 1.0)
            else:
                action = agent.predict(obs_dict["rgb"])

            obs_dict, reward, done, timeout, info = env.step(action)
            episode_reward += reward.mean().item() if isinstance(reward, torch.Tensor) else reward
            episode_steps += 1
            step_counter += 1

            if hasattr(env, "render") and save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            if episode_steps % 10 == 0:
                log.info(
                    f"Episode {episode + 1}, Step {episode_steps}, Reward: {reward.mean().item() if isinstance(reward, torch.Tensor) else reward}"
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        if episode_success:
            success_rate += 1

        if save_video and video_frames:
            video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
            imageio.mimsave(video_path, video_frames, fps=30)
            log.info(f"Saved video to {video_path}")
        log.info(
            f"Episode {episode + 1} finished: Steps={episode_steps}, Reward={episode_reward}, Success={episode_success}"
        )

    success_rate = success_rate / num_eval_episodes
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"Success Rate: {success_rate:.2f}")
    log.info(f"Mean Episode Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    log.info(f"Mean Episode Length: {mean_length:.2f}")
    log.info("=" * 50)

    results = {
        "checkpoint": checkpoint_path,
        "task": cfg.train.task_name,
        "robot": cfg.train.robot_name,
        "algorithm": cfg.train.algo,
        "num_episodes": num_eval_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

    results_path = os.path.join(output_dir, "eval_results.pkl")
    import pickle

    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
