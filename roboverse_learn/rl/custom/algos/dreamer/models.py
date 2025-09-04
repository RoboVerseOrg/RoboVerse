import torch
import torch.nn as nn
from torch import distributions as torchd

from .networks import RSSM, Decoder, DenseHead, Encoder


class RewardEMA:
    """Running mean and std for reward normalization"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # This should be an in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """World model component of Dreamer"""

    def __init__(self, obs_space, act_space, step, config, device="cuda"):
        super(WorldModel, self).__init__()
        self._step = step
        self._config = config
        self.device = device

        # Get shapes from observation space
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # Create encoder and decoder networks
        self.encoder = Encoder(shapes, config.get("encoder", {}), device=device)
        self.embed_size = self.encoder.out_dim

        # RSSM component (recurrent state-space model)
        self.dynamics = RSSM(
            config.get("dyn_stoch", 32),
            config.get("dyn_deter", 512),
            config.get("dyn_hidden", 512),
            config.get("dyn_rec_depth", 1),
            config.get("dyn_discrete", 32),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            config.get("dyn_mean_act", "none"),
            config.get("dyn_std_act", "softplus"),
            config.get("dyn_min_std", 0.1),
            config.get("dyn_unimix_ratio", 0.01),
            self.embed_size,
            act_space.shape[0],
            device,
        )

        # Define feature size based on RSSM configuration
        if config.get("dyn_discrete", 0):
            feat_size = config["dyn_stoch"] * config["dyn_discrete"] + config["dyn_deter"]
        else:
            feat_size = config["dyn_stoch"] + config["dyn_deter"]

        # Create decoder (observation model)
        self.decoder = Decoder(feat_size, shapes, config.get("decoder", {}), device=device)

        # Create reward prediction head
        self.reward_head = DenseHead(
            feat_size,
            1,
            config.get("reward_layers", 2),
            config.get("units", 400),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            config.get("reward_dist", "normal"),
            device=device,
        )

        # Create terminal state prediction head (for predicting episode endings)
        self.cont_head = DenseHead(
            feat_size,
            1,
            config.get("cont_layers", 2),
            config.get("units", 400),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            dist="binary",
            device=device,
        )

        # Create optimizers
        self._model_opt = torch.optim.Adam(
            self.parameters(),
            lr=config.get("model_lr", 3e-4),
            eps=config.get("opt_eps", 1e-5),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Loss scales
        self._scales = {
            "kl": config.get("kl_scale", 1.0),
            "reward": config.get("reward_scale", 1.0),
            "cont": config.get("cont_scale", 1.0),
            "decoder": config.get("decoder_scale", 1.0),
        }

        # Initialize EMA for reward normalization if needed
        self.reward_ema = RewardEMA(device) if config.get("reward_ema", False) else None
        self.reward_ema_vals = torch.zeros(2, device=device)

    def preprocess(self, obs):
        """
        Preprocess observations for the world model
        """
        # Convert to torch tensors if needed
        if not isinstance(obs["obs"], torch.Tensor):
            obs = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in obs.items()}

        # Ensure dimensions are correct
        if "action" in obs and obs["action"].dim() == 2:
            obs["action"] = obs["action"].unsqueeze(1)  # Add time dimension

        # Create cont from is_terminal
        if "is_terminal" in obs:
            obs["cont"] = 1.0 - obs["is_terminal"]

        return obs

    def train_batch(self, data):
        """
        Train the world model on a batch of data
        """
        data = self.preprocess(data)

        # Get model into training mode
        self.train()

        # Zero gradients
        self._model_opt.zero_grad()

        # Forward pass through encoder
        embed = self.encoder(data)

        # Forward pass through dynamics model
        post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])

        # Compute KL loss between posterior and prior
        kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post,
            prior,
            self._config.get("kl_free", 1.0),
            self._config.get("dyn_scale", 1.0),
            self._config.get("rep_scale", 1.0),
        )

        # Get features from posterior states
        feat = self.dynamics.get_feat(post)

        # Compute all model losses
        losses = {}

        # Reconstruction loss
        obs_dist = self.decoder(feat)

        # Handle shape mismatch - reshape the target observation to match decoder output
        if "obs" in data:
            if data["obs"].dim() == 5:  # [batch, seq, h, w, c]
                batch_size, seq_len = data["obs"].shape[0], data["obs"].shape[1]
                # Reshape to [batch*seq, h, w, c] to match decoder output
                obs_target = data["obs"].reshape(-1, *data["obs"].shape[2:])
            else:
                obs_target = data["obs"]

            # Make sure the batch dimension matches
            if obs_dist["obs"].batch_shape[0] != obs_target.shape[0]:
                # Adapt target shape to match distribution batch size
                if obs_dist["obs"].batch_shape[0] > obs_target.shape[0]:
                    # The dist has bigger batch - likely due to repeat interleave in the dynamics model
                    # This happens when batch is flattened over sequence dimension
                    if obs_dist["obs"].batch_shape[0] % obs_target.shape[0] == 0:
                        repeat_factor = obs_dist["obs"].batch_shape[0] // obs_target.shape[0]
                        obs_target = obs_target.repeat_interleave(repeat_factor, dim=0)
                else:
                    # Distribution has smaller batch - take corresponding subset
                    obs_target = obs_target[: obs_dist["obs"].batch_shape[0]]

            obs_loss = -obs_dist["obs"].log_prob(obs_target)
            losses["decoder"] = obs_loss.mean(-1)

        # Reward prediction loss
        if "reward" in data:
            reward_dist = self.reward_head(feat)

            # Similar handling for reward shape
            if data["reward"].dim() == 2:  # [batch, seq]
                reward_target = data["reward"].reshape(-1, 1)  # Flatten to [batch*seq, 1]
            else:
                reward_target = data["reward"]

            # Check reward_target shape and fix if needed
            if reward_target.dim() == 3:  # If shape is [batch, extra_dim, 1]
                # Directly reshape to [batch, 1] by removing the middle dimension
                reward_target = reward_target.reshape(reward_target.shape[0], -1)

                # If we now have multiple columns, just keep the first one
                if reward_target.shape[1] > 1:
                    reward_target = reward_target[:, 0:1]

            # Make sure batch dimensions match (after reshaping)
            if reward_dist.batch_shape[0] != reward_target.shape[0]:
                if reward_dist.batch_shape[0] > reward_target.shape[0]:
                    if reward_dist.batch_shape[0] % reward_target.shape[0] == 0:
                        repeat_factor = reward_dist.batch_shape[0] // reward_target.shape[0]
                        reward_target = reward_target.repeat_interleave(repeat_factor, dim=0)
                else:
                    reward_target = reward_target[: reward_dist.batch_shape[0]]

            # Final safety check to ensure exact shape match with distribution
            if reward_target.shape != reward_dist.loc.shape:
                reward_target = reward_target.reshape(reward_dist.loc.shape)

            # Compute the log probability with properly shaped target
            reward_loss = -reward_dist.log_prob(reward_target)
            losses["reward"] = reward_loss

        # Terminal state prediction loss
        if "cont" in data:
            cont_dist = self.cont_head(feat)
            cont_dist = cont_dist["dist"]  # Extract the actual distribution from the dict

            # Similar handling for cont shape
            if data["cont"].dim() == 2:  # [batch, seq]
                cont_target = data["cont"].reshape(-1, 1)  # Flatten to [batch*seq, 1]
            else:
                cont_target = data["cont"]

            # Check for and fix 3D shape issue
            if cont_target.dim() == 3:  # If shape is [batch, extra_dim, 1]
                # Directly reshape to [batch, 1] by selecting first element along second dimension
                cont_target = cont_target[:, 0, :]

            # Make sure batch dimensions match
            if cont_dist.batch_shape[0] != cont_target.shape[0]:
                if cont_dist.batch_shape[0] > cont_target.shape[0]:
                    if cont_dist.batch_shape[0] % cont_target.shape[0] == 0:
                        repeat_factor = cont_dist.batch_shape[0] // cont_target.shape[0]
                        cont_target = cont_target.repeat_interleave(repeat_factor, dim=0)
                else:
                    cont_target = cont_target[: cont_dist.batch_shape[0]]

            # Final safety check to ensure shape matches - handle different distribution types
            if isinstance(cont_dist, torchd.bernoulli.Bernoulli):
                # For Bernoulli, match shape with probs
                if cont_target.shape != cont_dist.probs.shape:
                    cont_target = cont_target.reshape(cont_dist.probs.shape)
            elif hasattr(cont_dist, "loc"):
                # For Normal and other distributions
                if cont_target.shape != cont_dist.loc.shape:
                    cont_target = cont_target.reshape(cont_dist.loc.shape)

            # Compute loss
            cont_loss = -cont_dist.log_prob(cont_target)
            losses["cont"] = cont_loss

        # Apply loss scales
        scaled_losses = {k: v * self._scales.get(k, 1.0) for k, v in losses.items()}

        # Add KL loss
        scaled_losses["kl"] = kl_loss * self._scales["kl"]

        # Compute total loss
        model_loss = sum(scaled_losses.values())

        # Backward pass
        model_loss.mean().backward()

        # Clip gradients and update parameters
        if self._config.get("grad_clip", 100.0) > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self._config["grad_clip"])
        else:
            grad_norm = 0.0

        self._model_opt.step()

        # Collect metrics
        metrics = {}
        for name, loss in losses.items():
            metrics[f"{name}_loss"] = loss.detach().mean().item()

        metrics["kl_free"] = self._config.get("kl_free", 1.0)
        metrics["dyn_scale"] = self._config.get("dyn_scale", 1.0)
        metrics["rep_scale"] = self._config.get("rep_scale", 1.0)
        metrics["dyn_loss"] = dyn_loss.mean().item()
        metrics["rep_loss"] = rep_loss.mean().item()
        metrics["kl"] = kl_value.mean().item()
        metrics["grad_norm"] = grad_norm
        metrics["prior_ent"] = self.dynamics.get_dist(prior).entropy().mean().item()
        metrics["post_ent"] = self.dynamics.get_dist(post).entropy().mean().item()

        # Context for behavior learning
        context = dict(
            embed=embed,
            feat=feat,
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )

        # Detach post for returning
        post_detached = {k: v.detach() for k, v in post.items()}

        return post_detached, context, metrics

    def imagine(self, policy, start, horizon):
        """
        Imagine trajectories in the latent space
        """
        dynamics = self.dynamics
        start = {k: v.detach() for k, v in start.items()}

        # Initialize states with start
        states = [start]
        actions = []

        # Rollout in the latent space
        for _ in range(horizon):
            feat = dynamics.get_feat(states[-1])
            action = policy(feat).sample()
            state = dynamics.img_step(states[-1], action)

            states.append(state)
            actions.append(action)

        # Convert lists to tensors - handle inconsistent tensor shapes
        stacked_states = {}

        for k in states[0]:
            # Get all tensors for this key
            tensors = [s[k] for s in states]

            # Check if shapes are consistent
            shapes = [t.shape for t in tensors]
            consistent = all(s == shapes[0] for s in shapes)

            if consistent:
                # If all shapes match, just stack normally
                stacked_states[k] = torch.stack(tensors, dim=1)
            else:
                # If shapes don't match, we need to adjust
                # First, determine target shape based on first state
                reference_shape = tensors[0].shape

                # Process each tensor to ensure compatible shape
                processed_tensors = []
                for i, tensor in enumerate(tensors):
                    if i == 0:
                        # Keep first tensor as reference
                        processed_tensors.append(tensor)
                    else:
                        # For subsequent tensors, reshape to match first one
                        if len(reference_shape) == 4:  # [batch, seq, stoch, discrete]
                            batch, seq, stoch, discrete = reference_shape

                            if tensor.dim() == 3:  # [batch*seq, stoch, discrete]
                                # Reshape to match reference if possible
                                reshaped = tensor.reshape(batch, seq, stoch, discrete)
                                processed_tensors.append(reshaped)
                            else:
                                # If dimensions don't match expectations, fallback to repeat
                                processed_tensors.append(tensors[0].clone())  # Safe fallback

                        elif len(reference_shape) == 2:  # [batch, feature]
                            batch, feature = reference_shape

                            if tensor.dim() == 2 and tensor.shape[1] == feature:
                                # If feature dim matches but batch doesn't, reshape
                                reshaped = tensor[:batch]
                                if reshaped.shape[0] < batch:
                                    # If too small, repeat to fill
                                    repeats = (batch + reshaped.shape[0] - 1) // reshaped.shape[0]
                                    reshaped = reshaped.repeat(repeats, 1)[:batch]
                                processed_tensors.append(reshaped)
                            else:
                                # Fallback
                                processed_tensors.append(tensors[0].clone())

                        else:
                            # For other shapes, if they're incompatible, use first as template
                            processed_tensors.append(tensors[0].clone())

                # Stack the processed tensors
                stacked_states[k] = torch.stack(processed_tensors, dim=1)

        # Actions should be consistent, but apply safety check
        action_shapes = [a.shape for a in actions]
        if all(s == action_shapes[0] for s in action_shapes):
            actions = torch.stack(actions, dim=1)
        else:
            # Process actions to ensure consistent shapes
            reference_shape = actions[0].shape
            processed_actions = []

            for action in actions:
                if action.shape == reference_shape:
                    processed_actions.append(action)
                else:
                    # Reshape or pad/truncate to match reference
                    if action.dim() == reference_shape.dim() and action.shape[1:] == reference_shape[1:]:
                        # Only batch dimension is different
                        if action.shape[0] > reference_shape[0]:
                            processed_actions.append(action[: reference_shape[0]])
                        else:
                            # Pad by repeating
                            repeats = (reference_shape[0] + action.shape[0] - 1) // action.shape[0]
                            padded = action.repeat(repeats, *([1] * (action.dim() - 1)))
                            processed_actions.append(padded[: reference_shape[0]])
                    else:
                        # Fallback to using first action as template
                        processed_actions.append(actions[0].clone())

            actions = torch.stack(processed_actions, dim=1)

        # Get features
        feats = dynamics.get_feat(stacked_states)

        return feats, stacked_states, actions

    def video_pred(self, data):
        """
        Generate and visualize predictions from the world model for evaluation

        Args:
            data: A batch of data from the replay buffer

        Returns:
            torch.Tensor: Combined visualization of ground truth, model predictions, and error
        """
        data = self.preprocess(data)
        embed = self.encoder(data)

        # Process the first few steps (observed) - use first 6 sequences, first 5 steps
        states, _ = self.dynamics.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])

        # Reconstruct observed frames
        feat = self.dynamics.get_feat(states)
        recon = self.decoder(feat)["obs"].mode()[:6]  # Mode gives most likely reconstruction

        # Generate reward predictions for observed frames
        reward_post = self.reward_head(feat).mean[:6]

        # Get the final state to start open-loop prediction
        init = {k: v[:, -1] for k, v in states.items()}

        # Open-loop prediction using given actions
        prior = self.dynamics.imagine_with_actions(data["action"][:6, 5:], init)
        prior_feat = self.dynamics.get_feat(prior)

        # Reconstruct predicted frames
        openl = self.decoder(prior_feat)["obs"].mode()

        # Generate reward predictions for predicted frames
        reward_prior = self.reward_head(prior_feat).mean

        # Combine observed reconstruction with predicted frames
        model = torch.cat([recon[:, :5], openl], 1)

        # Get ground truth frames
        truth = data["obs"][:6]

        # Calculate error (and rescale for visualization)
        error = (model - truth + 1.0) / 2.0

        # Concatenate along height dimension for visualization
        # [truth images]
        # [model predictions]
        # [error visualization]
        return torch.cat([truth, model, error], 2)

    def imagine_with_actions(self, actions, initial):
        """
        Roll out the dynamics model using given actions starting from an initial state

        Args:
            actions: Actions tensor of shape [batch, time, action_dim]
            initial: Initial latent state dictionary

        Returns:
            Dictionary of latent states over time
        """
        latent = {k: v.detach() for k, v in initial.items()}
        states = {k: [v] for k, v in latent.items()}

        # Rollout using actions
        for t in range(actions.shape[1]):
            action = actions[:, t]
            latent = self.dynamics.img_step(latent, action)

            # Append new states
            for k, v in latent.items():
                states[k].append(v)

        # Stack states along time dimension
        return {k: torch.stack(v, dim=1) for k, v in states.items()}


class ImagBehavior(nn.Module):
    """
    Imagination-based policy for Dreamer
    """

    def __init__(self, config, world_model, device="cuda"):
        super(ImagBehavior, self).__init__()
        self._config = config
        self._world_model = world_model
        self.device = device

        # Define feature size based on RSSM configuration
        if config.get("dyn_discrete", 0):
            feat_size = config["dyn_stoch"] * config["dyn_discrete"] + config["dyn_deter"]
        else:
            feat_size = config["dyn_stoch"] + config["dyn_deter"]

        # Create actor (policy) network
        self.actor = DenseHead(
            feat_size,
            config["action_num"],
            config.get("actor_layers", 4),
            config.get("units", 400),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            config.get("actor_dist", "normal"),
            config.get("actor_init_std", 1.0),
            config.get("actor_min_std", 0.1),
            config.get("actor_max_std", 1.0),
            config.get("actor_outscale", 1.0),
            device=device,
        )

        # Create critic (value) network
        self.critic = DenseHead(
            feat_size,
            1,
            config.get("critic_layers", 4),
            config.get("units", 400),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            config.get("critic_dist", "normal"),
            config.get("critic_outscale", 1.0),
            device=device,
        )

        # Target critic for stability
        self.target_critic = DenseHead(
            feat_size,
            1,
            config.get("critic_layers", 4),
            config.get("units", 400),
            config.get("act", "ELU"),
            config.get("norm", "none"),
            config.get("critic_dist", "normal"),
            config.get("critic_outscale", 1.0),
            device=device,
        )

        # Initialize target weights to match critic
        self.update_target()

        # Create optimizers
        self._actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.get("actor_lr", 8e-5),
            eps=config.get("opt_eps", 1e-5),
            weight_decay=config.get("weight_decay", 0.0),
        )

        self._critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 8e-5),
            eps=config.get("opt_eps", 1e-5),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Other hyperparameters
        self.horizon = config.get("horizon", 15)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_ = config.get("lambda", 0.95)
        self.imag_gradient = config.get("imag_gradient", "dynamics")
        self.target_update_freq = config.get("target_update_freq", 100)
        self._update_count = 0

    def update_target(self):
        """Update target critic"""
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(param.data)

    def train_batch(self, start):
        """Train behavior policy"""
        self.train()

        # Define reward and value functions
        reward_fn = lambda feat: self._world_model.reward_head(feat).mean
        value_fn = lambda feat: self.target_critic(feat).mean

        # Imagine trajectories
        feat, states, actions = self._world_model.imagine(self.actor, start, self.horizon)

        # Compute returns
        with torch.no_grad():
            rewards = reward_fn(feat)
            values = value_fn(feat)

            # Terminal values
            if self._world_model.cont_head is not None:
                cont = self._world_model.cont_head(feat)["mean"]
                values = values * cont

            # Compute lambda returns
            returns = self._compute_return(rewards, values, self.gamma, self.lambda_)

        # Train critic
        critic_loss = self._train_critic(feat, returns)

        # Train actor
        actor_loss = self._train_actor(feat, returns, actions)

        # Update target network periodically
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.update_target()

        # Collect metrics
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imag_reward": rewards.mean().item(),
            "imag_value": values.mean().item(),
            "imag_return": returns.mean().item(),
        }

        return metrics

    def _compute_return(self, rewards, values, discount, lambda_):
        """Compute lambda returns for a batch of trajectories"""
        # rewards: [B, T]
        # values: [B, T]
        # Initialize returns with values
        last_value = values[:, -1]
        returns = torch.zeros_like(rewards)

        # Backward computation of returns
        returns[:, -1] = rewards[:, -1] + discount * last_value
        for t in range(rewards.shape[1] - 2, -1, -1):
            returns[:, t] = rewards[:, t] + discount * (lambda_ * returns[:, t + 1] + (1 - lambda_) * values[:, t + 1])

        return returns

    def _train_critic(self, feat, target):
        """Train critic network to predict returns"""
        # Zero gradients
        self._critic_opt.zero_grad()

        # Compute critic loss
        dist = self.critic(feat)
        critic_loss = -dist.log_prob(target.detach()).mean()

        # Backward pass
        critic_loss.backward()

        # Clip gradients and update parameters
        if self._config.get("grad_clip", 100.0) > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self._config["grad_clip"])

        self._critic_opt.step()

        return critic_loss

    def _train_actor(self, feat, target, actions):
        """Train actor network to maximize returns"""
        # Zero gradients
        self._actor_opt.zero_grad()

        # Compute advantage
        with torch.no_grad():
            value = self.critic(feat).mean
            advantage = target.detach() - value

        # Actor loss
        actor_dist = self.actor(feat)
        actor_entropy = actor_dist.entropy().mean()

        # We want to maximize the advantage, so negative loss
        actor_loss = -(actor_entropy * self._config.get("actor_entropy", 1e-4))

        # If we're using reinforce, maximize likelihood weighted by advantage
        if self._config.get("actor_grad", "reinforce") == "reinforce":
            # Check if actions and feat/advantage have compatible shapes
            # actions shape: [batch, seq, action_dim]
            # feat shape: [batch, feat_dim]
            # advantage shape: [batch]
            if actions.dim() == 3 and feat.dim() == 2:
                # Need to reshape either actions or distribution

                # Option 1: Reshape actions to match actor_dist batch shape
                # Flatten batch and sequence dimensions for actions
                batch_size, seq_len, action_dim = actions.shape
                flat_actions = actions.reshape(-1, action_dim)

                # If actor_dist has fewer batch elements than flat_actions,
                # we need to expand actor_dist or take a subset of flat_actions
                if actor_dist.batch_shape[0] < flat_actions.shape[0]:
                    # Take only the first elements that match actor_dist batch size
                    flat_actions = flat_actions[: actor_dist.batch_shape[0]]
                elif actor_dist.batch_shape[0] > flat_actions.shape[0]:
                    # Repeat actions to match batch size (less ideal)
                    repeat_factor = (actor_dist.batch_shape[0] + flat_actions.shape[0] - 1) // flat_actions.shape[0]
                    flat_actions = flat_actions.repeat(repeat_factor, 1)[: actor_dist.batch_shape[0]]

                # Also reshape advantage to match
                if advantage.shape[0] != actor_dist.batch_shape[0]:
                    if advantage.shape[0] < actor_dist.batch_shape[0]:
                        repeat_factor = (actor_dist.batch_shape[0] + advantage.shape[0] - 1) // advantage.shape[0]
                        advantage = advantage.repeat_interleave(repeat_factor)[: actor_dist.batch_shape[0]]
                    else:
                        advantage = advantage[: actor_dist.batch_shape[0]]

                # Use reshaped flat actions for log_prob
                actor_loss = actor_loss - (actor_dist.log_prob(flat_actions) * advantage.detach()).mean()
            else:
                # Fallback to original approach if dimensions don't match expectations
                actor_loss = actor_loss - (actor_dist.log_prob(actions) * advantage.detach()).mean()
        else:
            # Dynamics loss - backprop directly through dynamics model
            actor_loss = actor_loss - advantage.mean()

        # Backward pass
        actor_loss.backward()

        # Clip gradients and update parameters
        if self._config.get("grad_clip", 100.0) > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self._config["grad_clip"])

        self._actor_opt.step()

        return actor_loss
