import torch
import numpy as np
from collections import deque


class EpisodeTracker:
    """Track episode returns and lengths without relying on info dict."""

    def __init__(self, num_envs, device, max_episodes=1000):
        self.num_envs = num_envs
        self.device = device
        self.episode_returns = torch.zeros(num_envs, device=device)
        self.episode_lengths = torch.zeros(num_envs, device=device)
        self.episode_count = 0
        self.total_returns = deque(maxlen=max_episodes)
        self.total_lengths = deque(maxlen=max_episodes)

    def update(self, rewards, terminations, truncations):
        """Update episode tracking with new rewards and done flags."""
        dones = torch.logical_or(terminations, truncations)

        # Add rewards to running episode returns
        self.episode_returns += rewards
        self.episode_lengths += 1

        # Check for completed episodes
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            # Record completed episodes
            for idx in done_indices:
                self.total_returns.append(self.episode_returns[idx].item())
                self.total_lengths.append(self.episode_lengths[idx].item())
                self.episode_count += 1

            # Reset completed episodes
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0

    def get_stats(self, window_size=100):
        """Get average return and length over recent episodes."""
        if len(self.total_returns) == 0:
            return 0.0, 0.0

        # Convert deque to list for slicing
        recent_returns = list(self.total_returns)[-window_size:]
        recent_lengths = list(self.total_lengths)[-window_size:]

        avg_return = np.mean(recent_returns)
        avg_length = np.mean(recent_lengths)

        return avg_return, avg_length

    def get_episode_count(self):
        """Get total number of completed episodes."""
        return self.episode_count
