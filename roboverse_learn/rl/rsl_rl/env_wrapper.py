from __future__ import annotations
from typing import Union
import torch
from tensordict import TensorDict
from roboverse_pack.tasks.unitree_rl.base import AgentTask


class RslRlEnvWrapper:
    """Wraps RoboVerse AgentTask for RSL-RL OnPolicyRunner compatibility.

    Provides the interface expected by rsl_rl.runners.OnPolicyRunner:
    - obs_buf as TensorDict with "policy" and "critic" keys
    - step() returning (obs, rewards, dones, extras)
    - Properties: num_envs, num_actions, max_episode_length, device, cfg
    """

    def __init__(self, env: AgentTask, train_cfg: dict | object = None):
        self.env = env
        self.train_cfg = train_cfg

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute actions and return observations, rewards, dones, extras."""
        _ = self.env.step(actions)
        return self.obs_buf, self.env.rew_buf, self.env.reset_buf, self.env.extras

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
        """
        return TensorDict(
            policy=self.env.obs_buf,
            critic=self.env.priv_obs_buf
        )
