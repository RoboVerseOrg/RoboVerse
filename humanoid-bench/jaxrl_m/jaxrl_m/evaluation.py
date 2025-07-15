import time
from collections import defaultdict
from typing import Dict

import gymnasium as gym
import jax
import numpy as np


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    """Evaluates a policy in an environment by running it for some number of episodes,
    and returns average statistics for metrics in the environment's info dict.

    If you wish to log environment returns, you can use the EpisodeMonitor wrapper (see below).

    Arguments:
        policy_fn: A function that takes an observation and returns an action.
            (if your policy needs JAX RNG keys, use supply_rng to supply a random key)
        env: The environment to evaluate in.
        num_episodes: The number of episodes to run for.

    Returns:
        A dictionary of average statistics for metrics in the environment's info dict.

    """
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    """Same as evaluate, but also returns the trajectories of observations, actions, rewards, etc.

    Arguments:
        See evaluate.

    Returns:
        stats: See evaluate.
        trajectories: A list of dictionaries (each dictionary corresponds to an episode),
            where trajectories[i] = {
                'observation': list_of_observations,
                'action': list_of_actions,
                'next_observation': list_of_next_observations,
                'reward': list of rewards,
                'done': list of done flags,
                'info': list of info dicts,
            }
    """
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        done = False
        while not done:
            action = policy_fn(observation)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = self.get_normalized_score(info["episode"]["return"]) * 100.0

        return observation, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
