from __future__ import annotations
from typing import Union
import torch
from tensordict import TensorDict

from roboverse_pack.tasks.unitree_rl.base import AgentTask
from .master import BaseRunnerWrapper


'''
class RslRlEnvWrapper:
    def __init__(self, env: AgentTask):
        self.env = env

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        _ = self.env.reset(list(range(self.num_envs)))
        _ = self.step(torch.zeros(size=(self.num_envs, self.num_actions), device=self.device, requires_grad=False))
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.privileged_obs_buf

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_obs(self):
        return self.env.num_obs

    @property
    def num_privileged_obs(self):
        return self.env.num_priv_obs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def max_episode_length(self):
        return self.env.max_episode_steps

    @property
    def privileged_obs_buf(self):
        return self.env.priv_obs_buf

    @property
    def obs_buf(self):
        return self.env.obs_buf

    @property
    def rew_buf(self):
        return self.env.rew_buf

    @property
    def reset_buf(self):
        return self.env.reset_buf

    @property
    def episode_length_buf(self):
        return self.env._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env._episode_steps = value

    @property
    def extras(self):
        return self.env.extras

    @property
    def device(self):
        return self.env.device
'''

class RslRlEnvWrapper:
    def __init__(self, env: AgentTask, train_cfg: dict | object=None):
        self.env = env
        self.train_cfg = train_cfg

    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            observations (TensorDict): Observations from the environment.
        """
        raise

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf, self.env.rew_buf, self.env.reset_buf, self.env.extras

    def get_observations(self) -> TensorDict:
        return self.obs_buf

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def max_episode_length(self):
        return self.env.max_episode_steps

    @property
    def episode_length_buf(self):
        return self.env._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env._episode_steps = value

    @property
    def device(self):
        return self.env.device

    @property
    def cfg(self):
        return self.train_cfg

    @property
    def obs_buf(self) -> TensorDict:
        return TensorDict(policy=self.env.obs_buf,
                          critic=self.env.priv_obs_buf)


class RslRlWrapper(BaseRunnerWrapper):
    def __init__(self, env: AgentTask, train_cfg: dict, log_dir:str):
        super().__init__(env, train_cfg, log_dir)
        from rsl_rl.runners import OnPolicyRunner, DistillationRunner

        self.env_wrapper = RslRlEnvWrapper(self.env)
        self.runner = OnPolicyRunner(
            env=self.env_wrapper,
            train_cfg=self.train_cfg,
            device=self.device,
            log_dir=log_dir,
        )

    def learn(self, max_iterations=10000):
        self.runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    def load(self, path):
        self.runner.load(path)

    def get_policy(self):
        return self.runner.get_inference_policy()
