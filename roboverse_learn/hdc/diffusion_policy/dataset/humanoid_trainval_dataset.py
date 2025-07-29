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
from diffusion_policy.dataset.humanoid_val_dataset import HumanoidValDataset
from typing import Dict, List

class HumanoidTrainValDataset(BaseLowdimDataset):
    def __init__(self, 
                 train_zarr_path, 
                 val_zarr_path=None,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='obs',
                 action_key='actions',
                 seed=42,
                 max_train_episodes=None,
                 max_val_episodes=None):
        super().__init__()
        print(f"Initializing HumanoidTrainValDataset with train_zarr_path: {train_zarr_path}")
        
        if not train_zarr_path:
            raise ValueError("train_zarr_path is empty.")
        if not os.path.exists(train_zarr_path):
            raise FileNotFoundError(f"Train Zarr path {train_zarr_path} does not exist.")
        
        # 加载训练集
        self.train_replay_buffer = ReplayBuffer.copy_from_path(
            train_zarr_path, keys=[obs_key, action_key])
        #self.train_replay_buffer = ReplayBuffer.create_from_path(train_zarr_path)
        print("Training ReplayBuffer loaded successfully.")
        
        #print(f"Loaded {len(self.train_replay_buffer.meta['motion_id'][:])} motion_ids.")
        
        # 加载验证集
        self.val_replay_buffer = None
        if val_zarr_path:
            print(f"Initializing HumanoidTrainValDataset with val_zarr_path: {val_zarr_path}")
            if not os.path.exists(val_zarr_path):
                raise FileNotFoundError(f"Validation Zarr path {val_zarr_path} does not exist.")
            self.val_replay_buffer = ReplayBuffer.copy_from_path(
                val_zarr_path, keys=[obs_key, action_key])
            #self.val_replay_buffer = ReplayBuffer.create_from_path(val_zarr_path)
            print("Validation ReplayBuffer loaded successfully.")
        
        #print(f"Loaded {len(self.val_replay_buffer.meta['motion_id'][:])} motion_ids.")

        # 初始化采样器
        self.train_sampler = SequenceSampler(
            replay_buffer=self.train_replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=downsample_mask(
                mask=np.ones(self.train_replay_buffer.n_episodes, dtype=bool),
                max_n=max_train_episodes,
                seed=seed
            )
        )
        print(f"Train Sampler initialized with {len(self.train_sampler)} samples.")
        
        self.val_sampler = None
        if self.val_replay_buffer:
            self.val_sampler = SequenceSampler(
                replay_buffer=self.val_replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=downsample_mask(
                    mask=np.ones(self.val_replay_buffer.n_episodes, dtype=bool),
                    max_n=max_val_episodes,
                    seed=seed
                )
            )
            print(f"Validation Sampler initialized with {len(self.val_sampler)} samples.")
        
        # 保存其他参数
        self.obs_key = obs_key
        self.action_key = action_key
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_zarr_path = val_zarr_path
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.horizon = horizon

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]
        data = {
            'obs': obs,
            'action': sample[self.action_key],
        }
        return data

    def _compute_episode_indices(self, sampler, replay_buffer):
        """
        根据给定的采样器和 ReplayBuffer 计算 episode 索引。
        """
        episode_ends = replay_buffer.episode_ends[:]
        indices = sampler.indices  # shape: (n_samples, 4)
        buffer_end_idxs = indices[:, 1]
        episode_indices = np.searchsorted(episode_ends, buffer_end_idxs, side='left')
        return episode_indices

    def get_validation_dataset(self):
        return HumanoidValDataset(
                 zarr_path = self.val_zarr_path, 
                 horizon=self.horizon,
                 pad_before=self.pad_before,
                 pad_after=self.pad_after,
                 obs_key='obs',
                 action_key='actions',
                 seed=42,
                )

    def get_normalizer(self, mode='limits', **kwargs):
        """
        构建 Normalizer，默认基于所有数据（训练集 + 验证集）。
        """

        train_data = self._sample_to_data(self.train_replay_buffer)
        if self.val_replay_buffer:
            val_data = self._sample_to_data(self.val_replay_buffer)
            # 合并训练和验证数据
            combined_data = {
                'obs': np.concatenate([train_data['obs'], val_data['obs']], axis=0),
                'action': np.concatenate([train_data['action'], val_data['action']], axis=0)
            }
        else:
            combined_data = train_data


            
        # 基于合并后的数据构建 Normalizer
        normalizer = LinearNormalizer()
        normalizer.fit(data=combined_data, last_n_dims=1, mode=mode, **kwargs)
        self.normalizer = normalizer
        return self.normalizer
    # def get_normalizer(self, mode='limits', n_sample=4000, **kwargs):
    #     """
    #     构建 Normalizer，仅采样前 n_sample 个样本用于统计，避免 OOM。
    #     """
    #     obs_list = []
    #     action_list = []

    #     for i in range(min(n_sample, len(self))):
    #         sample = self.train_sampler.sample_sequence(i)
    #         data = self._sample_to_data(sample)
    #         obs_list.append(data['obs'])       # numpy
    #         action_list.append(data['action'])

    #     obs_arr = np.stack(obs_list, axis=0)       # [N, H, D]
    #     action_arr = np.stack(action_list, axis=0) # [N, H, D]

    #     combined_data = {
    #         'obs': obs_arr,
    #         'action': action_arr
    #     }

    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=combined_data, last_n_dims=1, mode=mode, **kwargs)
    #     self.normalizer = normalizer
    #     return self.normalizer

    def __len__(self) -> int:
        return len(self.train_sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.train_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['obs'] = torch_data['obs'].squeeze(1)        # Shape: [10,1,1665] -> [10, 1665]
        torch_data['action'] = torch_data['action'].squeeze(1) 
        return torch_data
