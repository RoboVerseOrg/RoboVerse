from __future__ import annotations
from typing import Union
import torch
from tensordict import TensorDict
from roboverse_pack.tasks.unitree_rl.base import AgentTask


class RslRlEnvWrapper:
    """Wraps RoboVerse environments for RSL-RL OnPolicyRunner compatibility.

    Works with all RoboVerse environments as they
    all provide obs_buf and priv_obs_buf properties.

    Provides the interface expected by rsl_rl.runners.OnPolicyRunner:
    - obs_buf as TensorDict with "policy" and "critic" keys
    - step() returning (obs, rewards, dones, extras)
    - Properties: num_envs, num_actions, max_episode_length, device, cfg
    """

    def __init__(self, env: AgentTask, train_cfg: dict | object = None):
        self.env = env
        self.train_cfg = train_cfg

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute actions and return observations, rewards, dones, extras.

        RSL-RL expects combined done flags (terminated OR truncated).
        """
        # Call step and get Gymnasium format returns
        # Note: We use obs_buf property instead of returned obs_tensor for consistency
        _, rewards, terminated, truncated, info = self.env.step(actions)

        # RSL-RL expects combined done flags (terminated OR truncated)
        dones = torch.logical_or(terminated, truncated)

        # Merge info into extras
        extras = {**getattr(self.env, 'extras', {}), **info}

        # Return RSL-RL format with TensorDict observations
        return self.obs_buf, rewards, dones, extras

    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict."""
        return self.obs_buf

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def max_episode_length(self) -> int:
        return self.env.max_episode_steps

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env._episode_steps = value

    @property
    def device(self) -> torch.device:
        return self.env.device

    @property
    def cfg(self) -> dict | object:
        return self.train_cfg

    @property
    def obs_buf(self) -> TensorDict:
        """Return observations as TensorDict with 'policy' and 'critic' keys.

        RSL-RL expects asymmetric observations:
        - policy: observations for actor network
        - critic: privileged observations for critic network

        All RoboVerse RL environments provide
        obs_buf and priv_obs_buf properties, so we simply wrap them.
        """
        return TensorDict(
            policy=self.env.obs_buf,
            critic=self.env.priv_obs_buf
        )
