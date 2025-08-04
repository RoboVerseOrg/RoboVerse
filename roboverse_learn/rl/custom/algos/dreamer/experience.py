import torch
from torch.utils.data import DataLoader, Dataset


def transform_op(arr):
    """
    Swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(self, num_envs, horizon_length, batch_size, batch_length, obs_dim, act_dim, device, capacity=10000):
        """
        Initialize an experience buffer for Dreamer algorithm with efficient tensor-based storage

        Args:
            num_envs: Number of parallel environments
            horizon_length: Number of steps to collect per iteration
            batch_size: Number of sequences for batch training
            batch_length: Length of each sequence in batch
            obs_dim: Observation dimension (int or tuple)
            act_dim: Action dimension
            device: Device to store data on
            capacity: Maximum number of transitions to store (default: 1M)
        """
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.capacity = capacity

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Preallocate tensors for storage (more efficient than lists)
        self.storage_dict = {
            "obses": {
                # Handle both flat and image observations
                "obs": torch.zeros((capacity, *self.obs_dim), dtype=torch.float32, device=self.device)
            },
            "rewards": torch.zeros((capacity, 1), dtype=torch.float32, device=self.device),
            "actions": torch.zeros((capacity, self.act_dim), dtype=torch.float32, device=self.device),
            "dones": torch.zeros((capacity), dtype=torch.bool, device=self.device),
        }

        # Position trackers
        self.current_size = 0
        self.insert_index = 0

        # Calculate length for Dataset interface
        self.length = self.batch_size // self.batch_length
        if self.batch_size % self.batch_length != 0:
            self.length += 1

    def __len__(self):
        return self.length

    def add_data(self, obs, action=None, reward=None, done=None):
        """Add transition(s) to the buffer - efficiently processes batches"""
        # Convert inputs to tensors if they aren't already
        if not isinstance(obs, torch.Tensor):
            if isinstance(obs, list):
                # Stack a list of observations into a single tensor
                obs = torch.stack(
                    [
                        o if isinstance(o, torch.Tensor) else torch.tensor(o, device=self.device, dtype=torch.float32)
                        for o in obs
                    ],
                    dim=0,
                ).float()
            else:
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        # Handle batch dimension
        is_batch = obs.dim() > 1 and obs.size(0) > 1
        batch_size = obs.size(0) if is_batch else 1

        # Prepare other tensors if provided
        if action is not None and not isinstance(action, torch.Tensor):
            if isinstance(action, list):
                action = torch.stack(
                    [
                        a if isinstance(a, torch.Tensor) else torch.tensor(a, device=self.device, dtype=torch.float32)
                        for a in action
                    ],
                    dim=0,
                ).float()
            else:
                action = torch.tensor(action, device=self.device, dtype=torch.float32)

        if reward is not None and not isinstance(reward, torch.Tensor):
            if isinstance(reward, list):
                reward = torch.stack(
                    [
                        r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device, dtype=torch.float32)
                        for r in reward
                    ],
                    dim=0,
                ).float()
            else:
                reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            # Add a singleton dimension if reward is a scalar
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
            # Make sure reward has a final dimension of 1
            if reward.dim() == 1:
                reward = reward.unsqueeze(-1)

        if done is not None and not isinstance(done, torch.Tensor):
            if isinstance(done, list):
                done = torch.stack(
                    [
                        d if isinstance(d, torch.Tensor) else torch.tensor(d, device=self.device, dtype=torch.uint8)
                        for d in done
                    ],
                    dim=0,
                )
            else:
                done = torch.tensor(done, device=self.device, dtype=torch.uint8)
            # Add a singleton dimension if done is a scalar
            if done.dim() == 0:
                done = done.unsqueeze(0)
            # Make sure done has a final dimension of 1
            if done.dim() == 1 and is_batch:
                done = done.unsqueeze(-1)

        # Calculate indices for insertion
        if batch_size + self.current_size <= self.capacity:
            # Simple case: all new data fits without wrapping
            indices = torch.arange(self.insert_index, self.insert_index + batch_size, device=self.device)
            self.current_size += batch_size
        else:
            # Complex case: some data must wrap around the circular buffer
            remaining_space = self.capacity - self.insert_index
            first_part = min(remaining_space, batch_size)
            second_part = batch_size - first_part

            # Create indices for both parts
            indices = torch.cat([
                torch.arange(self.insert_index, self.insert_index + first_part, device=self.device),
                torch.arange(0, second_part, device=self.device),
            ])

            self.current_size = self.capacity

        # Insert data in batch
        self.storage_dict["obses"]["obs"][indices] = obs if is_batch else obs.unsqueeze(0)

        if action is not None:
            self.storage_dict["actions"][indices] = action if is_batch else action.unsqueeze(0)

        if reward is not None:
            self.storage_dict["rewards"][indices] = reward if is_batch else reward.unsqueeze(0)

        if done is not None:
            self.storage_dict["dones"][indices] = done if is_batch else done.unsqueeze(0)

        # Update insert index for next addition
        self.insert_index = (self.insert_index + batch_size) % self.capacity

    def update_data(self, key, step_index, value):
        """
        Update method to store data from multiple environments for a specific timestep

        Args:
            key: The type of data ('obses', 'actions', 'rewards', 'dones')
            step_index: The timestep index within the current collection period
            value: Batch of values to store (shape: [num_envs, ...])
        """
        if key == "obses":
            self.add_data(value)
        else:
            # Calculate the starting index for this batch of transitions
            # If we're collecting horizon_length steps from num_envs environments,
            # each step's data should go into a contiguous block
            base_idx = self.insert_index - self.num_envs if self.current_size >= self.num_envs else 0

            # For each timestep in horizon_length, we store num_envs transitions
            # So the actual position is: base_idx + step_index * num_envs
            start_idx = (base_idx + step_index * self.num_envs) % self.capacity

            # Get appropriate indices, handling potential wrap-around
            if start_idx + self.num_envs <= self.capacity:
                # Simple case: all indices are contiguous
                indices = torch.arange(start_idx, start_idx + self.num_envs, device=self.device)
            else:
                # Handle wrap-around at the end of the buffer
                first_part = self.capacity - start_idx
                second_part = self.num_envs - first_part
                indices = torch.cat([
                    torch.arange(start_idx, self.capacity, device=self.device),
                    torch.arange(0, second_part, device=self.device),
                ])

            # Store the batch of values at the calculated indices
            if key == "actions":
                self.storage_dict["actions"][indices] = value
            elif key == "rewards":
                self.storage_dict["rewards"][indices] = value
            elif key == "dones":
                self.storage_dict["dones"][indices] = value

    def __getitem__(self, idx):
        """Sample a batch of sequences for training"""
        # Make sure data is prepared for batch sampling
        if self.data_dict is None:
            self.prepare_batch_data()

        batch_data = {}
        for k, v in self.data_dict.items():
            if isinstance(v, dict):
                batch_data[k] = {kd: vd[idx] for kd, vd in v.items()}
            else:
                batch_data[k] = v[idx]
        return batch_data

    def prepare_batch_data(self):
        """Convert stored transitions to batch format for training"""
        if self.current_size == 0:
            raise ValueError("No data in buffer to sample from")

        # No need to stack since data is already in tensors
        all_obs = self.storage_dict["obses"]["obs"][: self.current_size]
        all_actions = self.storage_dict["actions"][: self.current_size]
        all_rewards = self.storage_dict["rewards"][: self.current_size]
        all_dones = self.storage_dict["dones"][: self.current_size]

        # Limit the number of sequences to sample from to prevent out-of-memory
        valid_range = self.current_size - self.batch_length + 1
        if valid_range <= 0:
            raise ValueError(f"Not enough transitions in buffer to create sequences of length {self.batch_length}")

        # Sample random starting indices for sequences
        num_seqs = min(self.batch_size, valid_range)
        seq_indices = torch.randint(0, valid_range, (num_seqs,), device=self.device)

        # Create sequences efficiently with vectorized operations
        seq_data = {
            "obs": self._create_sequences(all_obs, seq_indices),
            "action": self._create_sequences(all_actions, seq_indices),
            "reward": self._create_sequences(all_rewards, seq_indices),
            "done": self._create_sequences(all_dones, seq_indices),
        }

        self.data_dict = seq_data
        return seq_data

    def _create_sequences(self, data, seq_indices):
        """Create sequences of length batch_length from data using vectorized operations"""
        # Create a batch of indices for all sequences at once
        # Shape: [num_seqs, batch_length]
        batch_indices = seq_indices.unsqueeze(1) + torch.arange(self.batch_length, device=self.device).unsqueeze(0)

        # Get shape info
        num_seqs = len(seq_indices)

        # Flatten the batch indices for efficient gather operation
        flat_indices = batch_indices.reshape(-1)

        # Gather all sequence elements at once
        if data.dim() == 2:  # For 2D data like actions, rewards
            # Shape: [num_seqs * batch_length, feature_dim]
            flat_data = data[flat_indices]

            # Reshape to [num_seqs, batch_length, feature_dim]
            return flat_data.reshape(num_seqs, self.batch_length, -1)

        elif data.dim() == 1:  # For 1D data like dones
            # Shape: [num_seqs * batch_length]
            flat_data = data[flat_indices]

            # Reshape to [num_seqs, batch_length]
            return flat_data.reshape(num_seqs, self.batch_length)

        else:  # For higher dimensional data like images
            # For example, if data is [buffer_size, height, width, channels]
            feature_shape = data.shape[1:]

            # Shape: [num_seqs * batch_length, *feature_shape]
            flat_data = data[flat_indices]

            # Reshape to [num_seqs, batch_length, *feature_shape]
            return flat_data.reshape(num_seqs, self.batch_length, *feature_shape)

    def get_dataloader(self, shuffle=True):
        """Get a dataloader for the experience data"""
        return DataLoader(self, batch_size=None, shuffle=shuffle)

    def prepare_training(self):
        """
        Prepare the data for training.
        This method converts stored transitions to batch format and adds
        any metadata required by the Dreamer algorithm.
        """
        # Use the existing method to create batched sequences
        seq_data = self.prepare_batch_data()

        # Add is_first flags (needed by Dreamer)
        # First entry in each sequence is first, others depend on done flags
        batch_size = seq_data["obs"].shape[0]
        seq_len = seq_data["obs"].shape[1]

        # Initialize is_first tensor (all zeros)
        is_first = torch.zeros((batch_size, seq_len, 1), device=self.device, dtype=torch.float32)

        # First step in each sequence is always first
        is_first[:, 0, 0] = 1.0

        # For the rest of the sequence, a step is first if the previous step was done
        for t in range(1, seq_len):
            # Previous timestep's done flag determines if current step is first
            is_first[:, t, 0] = seq_data["done"][:, t - 1].float()

        # Add is_first to the sequence data
        seq_data["is_first"] = is_first

        # Create is_terminal from dones
        seq_data["is_terminal"] = seq_data["done"].unsqueeze(-1).float()

        return seq_data
