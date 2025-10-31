from __future__ import annotations

import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler
from tensordict.tensordict import TensorDict
from torch import Tensor
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class Buffer:
	## Not supported for multi-task setting
	def __init__(
		self,
		obs_shape: dict,
		action_size: int,
		task_embed_size: int,
		device: str | torch.device,
		num_envs: int = 1,
		capacity: int = 5000000,
		batch_size: int = 256,
		horizon: int = 15,
  		max_length: int = 1000,
	):
		self.device = device
		self.num_envs = num_envs
		self.capacity = capacity
		self._batch_size = batch_size * (horizon + 1)

		self._current_episode_obs = {
			key: torch.zeros((num_envs, max_length + 1, *obs_shape[key]), dtype=torch.float32 if "rgb" not in key else torch.uint8)
			for key in obs_shape.keys()
		}
		self._current_episode_action = torch.zeros((num_envs, max_length + 1, action_size), dtype=torch.float32)
		self._current_episode_reward = torch.zeros((num_envs, max_length + 1, 1), dtype=torch.float32)
		self._current_episode_done = torch.zeros((num_envs, max_length + 1, 1), dtype=torch.float32)
		self._current_episode_length = torch.zeros((num_envs,), dtype=torch.int32)
		if task_embed_size > 0:
			self._current_episode_task = torch.zeros((num_envs, task_embed_size), dtype=torch.float32)
		self.task_embed_size = task_embed_size

		self.buffer_index = 0
		self.full = False
		self._sampler = SliceSampler(
			num_slices=batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=False,
		)
		self.num_eps = 0
		self._buffer = self._reserve_buffer(
			LazyTensorStorage(capacity, device="cpu")
		)
		self.obs_shape = obs_shape
		
	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
	 
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def __len__(self):
		return self.capacity if self.full else self.buffer_index

	def add(
		self,
		observation: TensorDict,
		next_observation: TensorDict,
		action: Tensor,
		reward: Tensor,
		done: Tensor,
		terminated: Tensor,
		task: Tensor | None,
	):
		i = torch.arange(self.num_envs)
		t = self._current_episode_length
		for key in self.obs_shape.keys():
			if "rgb" in key:
				self._current_episode_obs[key][i, t, ...] = (observation[key] * 255.0).detach().cpu().to(torch.uint8)
				self._current_episode_obs[key][i, t+1, ...] = (next_observation[key] * 255.0).detach().clone().cpu().to(torch.uint8)
			else:
				self._current_episode_obs[key][i, t, ...] = observation[key].detach().clone().cpu()
				self._current_episode_obs[key][i, t+1, ...] = next_observation[key].detach().clone().cpu()
		self._current_episode_action[i, t, ...] = action.detach().cpu()
		self._current_episode_reward[i, t, ...] = reward.unsqueeze(-1).detach().cpu()
		self._current_episode_done[i, t, ...] = done.unsqueeze(-1).detach().cpu().float()
		if self.task_embed_size > 0 and task is not None:
			self._current_episode_task[i, ...] = task.detach().cpu()
		
		self._current_episode_length += 1
		
		if done.any():
			for env_idx in done.nonzero(as_tuple=False).squeeze(-1).tolist():
				L  = self._current_episode_length[env_idx] + 1
				episode = TensorDict({}, batch_size=[L])
				episode['observation'] = TensorDict({}, batch_size=[L])	 
				for key in self.obs_shape.keys():
					episode['observation'][key] = self._current_episode_obs[key][env_idx, :L, ...].clone()
				episode['action'] = self._current_episode_action[env_idx, :L, ...].clone()
				episode['reward'] = self._current_episode_reward[env_idx, :L, ...].clone()
				episode['done'] = self._current_episode_done[env_idx, :L, ...].clone()
				if self.task_embed_size > 0 and task is not None:
					episode['task'] = self._current_episode_task[env_idx, ...].clone()
				self._store_episode(episode)
				self._current_episode_length[env_idx] = 0
				
	def _store_episode(self, episode):
		"""
		Add an episode to the buffer.
		"""
		episode_idx = torch.ones_like(episode['reward'], dtype=torch.int32) * self.num_eps
		episode_idx = episode_idx.squeeze(-1)
		episode['episode'] = episode_idx	
		self._buffer.extend(episode)
		self.num_eps += 1
		return self.num_eps

	def _prepare_batch(self, batch):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `batch` to be a TensorDict with batch size TxB.
		"""
		batch = batch.to(self.device, non_blocking=True)
		observation = {
			key: batch["observation"][key].contiguous().float() / 255.0 if "rgb" in key else batch["observation"][key].contiguous()
			for key in self.obs_shape.keys()
		}
		action = batch["action"][:-1].contiguous()
		reward = batch["reward"][:-1].contiguous()
		done = batch["done"][:-1].contiguous()
		terminated = batch["done"][:-1].contiguous()
		task = batch.get("task", None)
		if task is not None:
			task = task[0].contiguous()
		return {
			"observation": observation,
			"action": action,
			"reward": reward,
			"done": done,
			"terminated": terminated,
			"task": task,
		}

	def sample(self, chunk_size) -> TensorDict[str, Tensor]:
		"""Sample a batch of subsequences from the buffer."""
		batch = self._buffer.sample().view(-1, chunk_size+1).permute(1, 0)
		return self._prepare_batch(batch)