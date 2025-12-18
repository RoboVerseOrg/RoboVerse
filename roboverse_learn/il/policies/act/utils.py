import sys
import os

import numpy as np
import torch
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
from roboverse_learn.il.utils.replay_buffer import ReplayBuffer

import IPython
e = IPython.embed

class ZarrEpisodicRoboVerseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(ZarrEpisodicRoboVerseDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = False

        # Load zarr data
        zarr_path = dataset_dir
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["head_camera", "state", "action"]
        )


    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        # Get episode id from the index
        episode_id = self.episode_ids[index]

        # Get the episode slice directly
        episode_slice = self.replay_buffer.get_episode_slice(episode_id)

        # Get data for this episode
        state_sequence = self.replay_buffer["state"][episode_slice]
        action_sequence = self.replay_buffer["action"][episode_slice]
        head_camera_sequence = self.replay_buffer["head_camera"][episode_slice]


        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(len(state_sequence))
        # Take the first frame as the starting point
        state = state_sequence[start_ts]

        head_camera = head_camera_sequence[start_ts:start_ts+1]
        # Assumes the zarr stores images in NCHW format, convert to NHWC
        # Get actions starting from the first time step
        action = action_sequence[start_ts:]
        action_len = len(action)

        # Pad action sequence to max episode length
        padded_action = np.zeros([self.norm_stats['max_episode_len'], action_sequence.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.norm_stats['max_episode_len'])
        is_pad[action_len:] = 1

        # Format observations
        image_data = torch.from_numpy(head_camera).float()
        state_data = torch.from_numpy(state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Channel last to channel first
        # image_data = torch.einsum('k h w c -> k c h w', image_data)

        # Normalize data
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        state_data = (state_data - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]

        return image_data, state_data, action_data, is_pad

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    # Load the zarr data
    zarr_path = dataset_dir
    replay_buffer = ReplayBuffer.copy_from_path(
        zarr_path,
        keys=["state", "action"]
    )
    actual_episodes = replay_buffer.n_episodes
    if num_episodes > actual_episodes:
        print(f"Warning: Requested {num_episodes} episodes, but dataset only has {actual_episodes}")
        print(f"Using all {actual_episodes} available episodes")
        num_episodes = actual_episodes

    # Calculate max episode length
    max_episode_len = int(np.max(replay_buffer.episode_lengths))

    # Get all state and action data
    all_state_data = []
    all_action_data = []

    # Process each episode up to num_episodes
    for episode_idx in range(num_episodes):
        episode_slice = replay_buffer.get_episode_slice(episode_idx)
        state = replay_buffer["state"][episode_slice]
        action = replay_buffer["action"][episode_slice]

        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))

    all_state_data = torch.vstack(all_state_data)
    all_action_data = torch.vstack(all_action_data)

    # Normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # Normalize state data
    state_mean = all_state_data.mean(dim=0, keepdim=True)
    state_std = all_state_data.std(dim=0, keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "state_mean": state_mean.numpy().squeeze(),
        "state_std": state_std.numpy().squeeze(),
        "example_state": state,
        "max_episode_len": max_episode_len
    }

    return stats, num_episodes



def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    norm_stats, num_episodes = get_norm_stats(dataset_dir, num_episodes)

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # construct dataset and dataloader
    train_dataset = ZarrEpisodicRoboVerseDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = ZarrEpisodicRoboVerseDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
