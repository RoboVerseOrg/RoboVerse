from collections import deque

import gymnasium as gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
    """Wrapper for pixel observations. Compatible with DMControl environments."""

    def __init__(self, cfg, env, num_frames=3, render_size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_frames * 3, render_size, render_size),
            dtype=np.uint8,
        )
        self._frames = deque([], maxlen=num_frames)
        self._render_size = render_size

    def _get_obs(self):
        frame = self.env.render(mode="rgb_array", width=self._render_size, height=self._render_size).transpose(2, 0, 1)
        self._frames.append(frame)
        return torch.from_numpy(np.concatenate(self._frames))

    def reset(self):
        self.env.reset()
        for _ in range(self._frames.maxlen):
            obs = self._get_obs()
        return obs, {}

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        return self._get_obs(), reward, done, truncated, info
