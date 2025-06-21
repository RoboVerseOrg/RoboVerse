import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.actions = torch.zeros((n_env, buffer_size, n_act), device=device, dtype=torch.float)
        self.rewards = torch.zeros((n_env, buffer_size), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.next_observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
        self.ptr = 0

    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
            next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)

            rewards = torch.gather(self.rewards, 1, indices).reshape(self.n_env * batch_size)
            dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
            truncations = torch.gather(self.truncations, 1, indices).reshape(self.n_env * batch_size)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.privileged_obs_size)
                    privileged_observations = torch.gather(self.privileged_observations, 1, priv_obs_indices).reshape(
                        self.n_env * batch_size, self.privileged_obs_size
                    )
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat([observations, privileged_observations], dim=1)
                    next_critic_observations = torch.cat([next_observations, next_privileged_observations], dim=1)
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
                    critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # Sample base indices
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.privileged_obs_size)
                    privileged_observations = torch.gather(self.privileged_observations, 1, priv_obs_indices).reshape(
                        self.n_env * batch_size, self.privileged_obs_size
                    )

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat([observations, privileged_observations], dim=1)
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
                    critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_dones = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards after first done
            # This creates a cumulative product that zeroes out rewards after the first done
            done_masks = torch.cumprod(1.0 - all_dones, dim=2)  # [n_env, batch_size, n_step]

            # Create discount factors
            discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(1, 1, -1)  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax((all_dones > 0).float(), dim=2)  # [n_env, batch_size]
            first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(first_done, first_trunc)  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = self.next_privileged_observations.gather(
                        1,
                        final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.privileged_obs_size),
                    )

                    # Reshape for output
                    next_privileged_observations = final_next_privileged_observations.reshape(
                        self.n_env * batch_size, self.privileged_obs_size
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(self.n_env * batch_size, self.n_obs)
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = self.next_critic_observations.gather(
                        1,
                        final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs),
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(self.n_env * batch_size, self.n_obs)

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations
        return out


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        if self.training:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values using Welford's online algorithm"""
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Welford's parallel algorithm
        if self.count > 0:  # Ensure we're not dividing by zero
            # Compute batch variance
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)

            # Combine variances using parallel algorithm
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self._var = torch.mean((x - self._mean) ** 2, dim=0, keepdim=True)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        gamma: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * gamma * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()  # noqa: E741
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)  # noqa: E741
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            gamma,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            gamma,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        action = self.fc_mu(x)
        return action

    def explore(self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = torch.rand(self.n_envs, 1, device=obs.device) * (self.std_max - self.std_min) + self.std_min

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise
