from typing import Dict
import torch
import numpy as np
import copy
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from typing import Dict, List

class HumanoidValDataset(BaseLowdimDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='obs',
                 #state_key='state',
                 action_key='actions',
                 seed=42,
                #  #val_ratio=0.0,
                #  max_train_episodes=None
                ):
        super().__init__()
        print(f"Initializing HumanoidValDataset with zarr_path: {zarr_path}")
        if not zarr_path:
            raise ValueError("zarr_path is empty.")
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr path {zarr_path} does not exist.")

        # 加载指定的 keys 以及 motion_id
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key,  action_key],
        #     chunks={
        # 'actions': (100, 19),
        # 'obs': (100, 71)}
        )
        print("ReplayBuffer loaded successfully.")

        # 加载 motion_id
        # self.motion_ids = self.replay_buffer.meta['motion_id'][:]
        # print(f"Loaded {len(self.motion_ids)} motion_ids.")

        # 初始化采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=downsample_mask(
                mask=np.ones(self.replay_buffer.n_episodes, dtype=bool),
                max_n=None,
                seed=seed
            )
        )
        print(f"Sampler initialized with {len(self.sampler)} samples.")

        # 预先计算每个样本对应的 episode index
        self.episode_indices = self._compute_episode_indices()
        print("Episode indices computed.")

        # 保存其他参数
        self.obs_key = obs_key
        #self.state_key = state_key
        self.action_key = action_key
        #self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _compute_episode_indices(self):
        """
        预计算每个样本对应的 episode index。
        """
        #import ipdb;ipdb.set_trace()
        episode_ends = self.replay_buffer.episode_ends[:]
        indices = self.sampler.indices  # shape: (n_samples, 4)
        buffer_end_idxs = indices[:,1]
        # 使用 searchsorted 找到每个 buffer_end_idx 所属的 episode index
        # side='right' 确保 buffer_end_idx <= episode_ends[i]
        episode_indices = np.searchsorted(episode_ends, buffer_end_idxs, side='left')
        #print("Diff:",episode_indices[250:260],":",buffer_end_idxs[250:260],":")
        return episode_indices

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]
        data = {
            'obs': obs,
            'action': sample[self.action_key],
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['obs'] = torch_data['obs'].squeeze(1)        # Shape: [10,1,1665] -> [10, 1665]
        torch_data['action'] = torch_data['action'].squeeze(1)  # Shape: [10,1 , 19] -> [10,19]
        return torch_data
