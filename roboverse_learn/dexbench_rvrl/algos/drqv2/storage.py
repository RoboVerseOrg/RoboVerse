
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import IterableDataset
import sys
from torch.multiprocessing import get_context
from rich.logging import RichHandler
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


def episode_len(episode):
	# subtract -1 because the dummy first transition
	return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
	with io.BytesIO() as bs:
		np.savez_compressed(bs, **episode)
		bs.seek(0)
		with fn.open('wb') as f:
			f.write(bs.read())


def load_episode(fn):
	with fn.open('rb') as f:
		episode = np.load(f)
		episode = {k: episode[k] for k in episode.keys()}
		return episode


class ReplayBufferStorage:
	def __init__(self, obs_shape: dict, action_size: int, replay_dir: str, num_envs: int, max_length: int):
		self._obs_shape = obs_shape
		self._action_size = action_size
		self._replay_dir = Path(replay_dir)
		if self._replay_dir.exists():
			shutil.rmtree(self._replay_dir)

		self._replay_dir.mkdir(parents=True, exist_ok=False)
		self._current_episode_obs = {
			key: np.zeros((num_envs, max_length + 1, *obs_shape[key]), dtype=np.uint8 if "rgb" in key else np.float32)
			for key in obs_shape.keys()
		}
		self._current_episode_action = np.zeros((num_envs, max_length, action_size), dtype=np.float32)
		self._current_episode_reward = np.zeros((num_envs, max_length, 1), dtype=np.float32)
		self._current_episode_length = np.zeros((num_envs,), dtype=np.int32)
		self.num_envs = num_envs
		self._preload()

	def __len__(self):
		return self._num_transitions

	def add(self, observation, action, reward, next_observation, done):
		i = np.arange(self.num_envs)
		t = self._current_episode_length
		for key in self._obs_shape.keys():
			if "rgb" in key:
				self._current_episode_obs[key][i, t, ...] = (observation[key] * 255.0).detach().cpu().numpy().astype(np.uint8)
				self._current_episode_obs[key][i, t + 1, ...] = (next_observation[key] * 255.0).detach().cpu().numpy().astype(np.uint8)
			else:
				self._current_episode_obs[key][i, t, ...] = observation[key].detach().cpu().numpy()
				self._current_episode_obs[key][i, t + 1, ...] = next_observation[key].detach().cpu().numpy()
		self._current_episode_action[i, t, ...] = action.detach().cpu().numpy()
		self._current_episode_reward[i, t, ...] = reward.unsqueeze(-1).detach().cpu().numpy()
		self._current_episode_length += 1
		
		if done.any():
			for env_idx in done.nonzero(as_tuple=False).squeeze(-1).tolist():
				L  = self._current_episode_length[env_idx]
				episode = {}
				for key in self._obs_shape.keys():
					episode[f"observation_{key}"] = self._current_episode_obs[key][env_idx, :L+1, ...]
				episode['action'] = self._current_episode_action[env_idx, :L, ...]
				episode['reward'] = self._current_episode_reward[env_idx, :L, ...]
				self._store_episode(episode)
				self._current_episode_length[env_idx] = 0

	def _preload(self):
		self._num_episodes = 0
		self._num_transitions = 0
		for fn in self._replay_dir.glob('*.npz'):
			_, _, eps_len = fn.stem.split('_')
			self._num_episodes += 1
			self._num_transitions += int(eps_len)

	def _store_episode(self, episode):
		eps_idx = self._num_episodes
		eps_len = episode_len(episode)
		self._num_episodes += 1
		self._num_transitions += eps_len
		ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
		save_episode(episode, self._replay_dir / eps_fn)


class IterableReplayBuffer(IterableDataset):
	def __init__(self, obs_shape: dict, action_size: int, replay_dir: str, max_size: int, num_workers: int, nstep: int, discount: float,
				 fetch_every: int, save_snapshot: bool):
		self._obs_shape = obs_shape
		self._action_size = action_size
		self._replay_dir = Path(replay_dir)
		self._size = 0
		self._max_size = max_size
		self._num_workers = max(1, num_workers)
		print(f"Initializing ReplayBuffer with max_size={max_size}, num_workers={num_workers}, nstep={nstep}, discount={discount}, fetch_every={fetch_every}, save_snapshot={save_snapshot}")
		self._episode_fns = []
		self._episodes = {}
		self._nstep = nstep
		self._discount = discount
		self._fetch_every = fetch_every
		self._samples_since_last_fetch = fetch_every
		self._save_snapshot = save_snapshot

	def _sample_episode(self):
		eps_fn = random.choice(self._episode_fns)
		return self._episodes[eps_fn]
			
	def _store_episode(self, eps_fn):
		try:
			episode = load_episode(eps_fn)
		except:
			return False
		eps_len = episode_len(episode)
		while eps_len + self._size > self._max_size:
			early_eps_fn = self._episode_fns.pop(0)
			early_eps = self._episodes.pop(early_eps_fn)
			self._size -= episode_len(early_eps)
			Path(early_eps_fn).unlink(missing_ok=True)
		str_eps_fn = str(eps_fn)
		self._episode_fns.append(str_eps_fn)
		self._episode_fns.sort()
		self._episodes[str_eps_fn] = episode
		self._size += eps_len

		if not self._save_snapshot:
			eps_fn.unlink(missing_ok=True)
		return True

	def _try_fetch(self):
		if self._samples_since_last_fetch < self._fetch_every:
			return
		self._samples_since_last_fetch = 0
		try:
			worker_id = torch.utils.data.get_worker_info().id
		except:
			worker_id = 0
		eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
		fetched_size = 0
		for eps_fn in eps_fns:
			eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
			if eps_idx % self._num_workers != worker_id:
				continue
			if str(eps_fn) in self._episodes.keys():
				continue
			if fetched_size + eps_len > self._max_size:
				break
			fetched_size += eps_len
			if not self._store_episode(eps_fn):
				break

	def _sample(self):
		try:
			self._try_fetch()
		except:
			traceback.print_exc()
		self._samples_since_last_fetch += 1
		episode = self._sample_episode()
		# add +1 for the first dummy transition
		idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
		obs = (
			episode[f'observation_{key}'][idx] 
			if "rgb" not in key 
			else episode[f'observation_{key}'][idx].astype(np.float32) / 255.0
			for key in self._obs_shape.keys()
		)
		next_obs = (
			episode[f'observation_{key}'][idx + self._nstep]
			if "rgb" not in key 
			else episode[f'observation_{key}'][idx + self._nstep].astype(np.float32) / 255.0
			for key in self._obs_shape.keys()
		)
		action = episode['action'][idx]
		reward = np.zeros_like(episode['reward'][idx])
		discount = np.ones_like(episode['reward'][idx])
		for i in range(self._nstep):
			step_reward = episode['reward'][idx + i]
			reward += discount * step_reward
			discount *= self._discount
		return (*obs, *next_obs, action, reward, discount)

	def __iter__(self):
		while True:
			yield self._sample()


def _worker_init_fn(worker_id):
	seed = int(np.random.get_state()[1][0]) + worker_id
	np.random.seed(seed)
	random.seed(seed)


def make_replay_loader(obs_shape, action_size, replay_dir, max_size, batch_size, num_workers,
					   save_snapshot, nstep, discount):
	max_size_per_worker = max_size // max(1, num_workers)

	iterable = IterableReplayBuffer(obs_shape,
							action_size,
							replay_dir,
							max_size_per_worker,
							num_workers,
							nstep,
							discount,
							fetch_every=1000,
							save_snapshot=save_snapshot)

	loader = torch.utils.data.DataLoader(iterable,
										 batch_size=batch_size,
										 num_workers=num_workers,
										 pin_memory=True,
										 worker_init_fn=_worker_init_fn,
										 multiprocessing_context=get_context('spawn')
										 )
	return loader

class Buffer:
	## Not supported for multi-task setting
	def __init__(
		self,
		obs_shape: dict,
		action_size: int,
		device: str | torch.device,
		num_envs: int = 1,
		capacity: int = 5000000,
		batch_size: int = 256,
		nstep:int = 3,
		discount: float = 0.99,
  		max_length: int = 1000,
	):
		self.device = device
		self.num_envs = num_envs
		self.capacity = capacity
		self._batch_size = batch_size * (nstep + 1)

		self._current_episode_obs = {
			key: torch.zeros((num_envs, max_length + 1, *obs_shape[key]), dtype=torch.float32 if "rgb" not in key else torch.uint8)
			for key in obs_shape.keys()
		}
		self._current_episode_action = torch.zeros((num_envs, max_length + 1, action_size), dtype=torch.float32)
		self._current_episode_reward = torch.zeros((num_envs, max_length + 1, 1), dtype=torch.float32)
		self._current_episode_length = torch.zeros((num_envs,), dtype=torch.int32)

		self.buffer_index = 0
		self.nstep = nstep
		self._discount = discount
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
		action: Tensor,
		reward: Tensor,
		next_observation: Tensor,
		done: Tensor,
	):
		i = torch.arange(self.num_envs)
		t = self._current_episode_length
		for key in self.obs_shape.keys():
			if "rgb" in key:
				self._current_episode_obs[key][i, t, ...] = (observation[key] * 255.0).detach().cpu().to(torch.uint8)
				self._current_episode_obs[key][i, t + 1, ...] = (next_observation[key] * 255.0).detach().cpu().to(torch.uint8)
			else:
				self._current_episode_obs[key][i, t, ...] = observation[key].detach().cpu()
				self._current_episode_obs[key][i, t + 1, ...] = next_observation[key].detach().cpu()
		self._current_episode_action[i, t, ...] = action.detach().cpu()
		self._current_episode_reward[i, t, ...] = reward.unsqueeze(-1).detach().cpu()
		
		self._current_episode_length += 1
		
		if done.any():
			for env_idx in done.nonzero(as_tuple=False).squeeze(-1).tolist():
				L  = self._current_episode_length[env_idx] + 1  # +1 for the next_observation
				episode = TensorDict({}, batch_size=[L])
				episode['observation'] = TensorDict({}, batch_size=[L])	 
				for key in self.obs_shape.keys():
					episode['observation'][key] = self._current_episode_obs[key][env_idx, :L, ...].clone()
				episode['action'] = self._current_episode_action[env_idx, :L, ...].clone()
				episode['reward'] = self._current_episode_reward[env_idx, :L, ...].clone()
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
			key: batch["observation"][key][0].contiguous().float() / 255.0 if "rgb" in key else batch["observation"][key][0].contiguous()
			for key in self.obs_shape.keys()
		}
		next_observation = {
			key: batch["observation"][key][-1].contiguous().float() / 255.0 if "rgb" in key else batch["observation"][key][-1].contiguous()
			for key in self.obs_shape.keys()
		}
		action = batch["action"][0].contiguous()
		reward = torch.zeros_like(batch["reward"][0]).contiguous()
		discount = torch.ones_like(batch["reward"][0]).contiguous()
		for i in range(self.nstep):
			step_reward = batch["reward"][i].contiguous()
			reward += discount * step_reward
			discount *= self._discount
   
		return {
			"observation": observation,
			"next_observation": next_observation,
			"action": action,
			"reward": reward,
			"discount": discount,
		}

	def sample(self, nstep) -> TensorDict[str, Tensor]:
		"""Sample a batch of subsequences from the buffer."""
		batch = self._buffer.sample().view(-1, nstep + 1).permute(1, 0)
		return self._prepare_batch(batch)