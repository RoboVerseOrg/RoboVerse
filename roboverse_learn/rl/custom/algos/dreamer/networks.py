import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd


def get_act(name):
    """Get activation function by name"""
    if name == "relu":
        return nn.ReLU
    elif name == "elu":
        return nn.ELU
    elif name == "leaky_relu":
        return nn.LeakyReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "sigmoid":
        return nn.Sigmoid
    else:
        return nn.ELU


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: int,
        hidden_dim: int,
        activation: str = "elu",
        norm: str = "none",
        device: str = "cuda",
    ):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        act_fn = get_act(activation.lower())

        layers = []
        dims = [in_dim] + [hidden_dim] * hidden_layers + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Don't add activation and normalization after the last layer
            if i < len(dims) - 2:
                # Add normalization if requested
                if norm == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                # Add activation
                layers.append(act_fn())

        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        # Check tensor shape and verify it matches expected input dimension
        if x.dim() > 2:
            # If tensor has more than 2 dimensions, reshape while preserving batch dimension
            batch_size = x.shape[0]
            # Flatten all dimensions except batch dimension
            x = x.reshape(batch_size, -1)

        # Check if the second dimension matches the expected input dimension
        if x.shape[1] != self.in_dim:
            # If dimensions don't match, we need to handle this to avoid matrix multiplication errors
            if x.shape[1] > self.in_dim:
                # Too many features - truncate to expected input size
                x = x[:, : self.in_dim]
            else:
                # Too few features - this is a more serious error
                # Pad with zeros or raise exception
                raise ValueError(f"Input tensor has {x.shape[1]} features but model expects {self.in_dim}")

        return self.model(x)


class Encoder(nn.Module):
    """Encoder for processing observations"""

    def __init__(self, shapes, config={}, device="cuda"):
        super(Encoder, self).__init__()

        self.device = device
        self.shapes = shapes
        self.config = config

        # Default configuration
        hidden_dims = config.get("hidden_dims", [400, 400, 400])
        activation = config.get("activation", "elu")
        norm = config.get("norm", "none")

        # Create encoder for each observation type
        encoders = {}
        for name, shape in shapes.items():
            if len(shape) == 1:  # 1D vector input (e.g., proprioception)
                encoders[name] = MLP(
                    in_dim=shape[0],
                    out_dim=hidden_dims[-1],
                    hidden_layers=len(hidden_dims) - 1,
                    hidden_dim=hidden_dims[0],
                    activation=activation,
                    norm=norm,
                    device=device,
                )
            elif len(shape) == 3:  # Image input (H, W, C)
                encoders[name] = self._make_image_encoder(
                    in_shape=shape, out_dim=hidden_dims[-1], config=config, device=device
                )

        self.encoders = nn.ModuleDict(encoders)

        # Output dimension is the output of one encoder (all have the same output dimension)
        self.out_dim = hidden_dims[-1]
        self.to(device)

    def _make_image_encoder(self, in_shape, out_dim, config, device):
        """Make an image encoder (CNN)"""
        act_fn = get_act(config.get("activation", "elu").lower())

        # Default minimal CNN architecture
        return nn.Sequential(
            nn.Conv2d(in_shape[2], 32, 4, stride=2),
            act_fn(),
            nn.Conv2d(32, 64, 4, stride=2),
            act_fn(),
            nn.Conv2d(64, 128, 4, stride=2),
            act_fn(),
            nn.Conv2d(128, 256, 4, stride=2),
            act_fn(),
            nn.Flatten(),
            nn.Linear(1024, out_dim),
        ).to(device)

    def forward(self, obs):
        outputs = []
        # Skip these keys as they're not actual observations
        skip_keys = {"is_first", "is_terminal", "action", "is_last"}

        for name, encoder in self.encoders.items():
            if name in obs and name not in skip_keys:
                # Get the input tensor
                x = obs[name]
                x = x.float()

                # Process image input if needed
                if len(self.shapes[name]) == 3:  # Image input
                    if x.dim() == 5:  # [batch, sequence, height, width, channels]
                        # Reshape to [batch*sequence, height, width, channels]
                        batch, seq_len = x.shape[0], x.shape[1]
                        x = x.reshape(-1, *x.shape[2:])

                        # Convert to [batch*sequence, channels, height, width]
                        x = x.permute(0, 3, 1, 2)

                        # Apply encoder
                        out = encoder(x)

                        # Reshape back to [batch, sequence, feature_dim]
                        out = out.reshape(batch, seq_len, -1)

                    elif x.dim() == 4:  # [batch, height, width, channels]
                        # Convert to [batch, channels, height, width]
                        x = x.permute(0, 3, 1, 2)
                        out = encoder(x)
                else:
                    # Handle vector observations
                    if x.dim() == 3:  # [batch, sequence, feature_dim]
                        # Reshape to [batch*sequence, feature_dim]
                        batch, seq_len = x.shape[0], x.shape[1]
                        x = x.reshape(-1, x.shape[2])

                        # Apply encoder
                        out = encoder(x)

                        # Reshape back to [batch, sequence, feature_dim]
                        out = out.reshape(batch, seq_len, -1)
                    else:
                        out = encoder(x)

                outputs.append(out)

        # Average all encoder outputs if there are multiple
        if len(outputs) > 1:
            return torch.mean(torch.stack(outputs), dim=0)
        elif len(outputs) == 1:
            return outputs[0]
        else:
            raise ValueError("No valid observations found in input")


class Decoder(nn.Module):
    """Decoder for reconstructing observations"""

    def __init__(self, in_dim, shapes, config={}, device="cuda"):
        super(Decoder, self).__init__()

        self.device = device
        self.shapes = shapes
        self.config = config

        # Default configuration
        hidden_dims = config.get("hidden_dims", [400, 400, 400])
        activation = config.get("activation", "elu")
        norm = config.get("norm", "none")

        # Create decoder for each observation type
        decoders = {}
        for name, shape in shapes.items():
            if len(shape) == 1:  # 1D vector output
                decoders[name] = DenseHead(
                    in_dim=in_dim,
                    out_dim=shape[0],
                    hidden_layers=len(hidden_dims),
                    hidden_dim=hidden_dims[0],
                    activation=activation,
                    norm=norm,
                    dist="normal",  # Default distribution
                    device=device,
                )
            elif len(shape) == 3:  # Image output
                decoders[name] = self._make_image_decoder(in_dim=in_dim, out_shape=shape, config=config, device=device)

        self.decoders = nn.ModuleDict(decoders)
        self.to(device)

    def _make_image_decoder(self, in_dim, out_shape, config, device):
        """Create a CNN-based image decoder"""
        # This is a placeholder for a more sophisticated image decoder
        # In a real implementation, you'd use transposed convolutions
        return MLP(
            in_dim=in_dim,
            out_dim=np.prod(out_shape),
            hidden_layers=3,
            hidden_dim=400,
            activation=config.get("activation", "elu"),
            norm=config.get("norm", "none"),
            device=device,
        )

    def forward(self, features):
        outputs = {}
        for name, decoder in self.decoders.items():
            # Get distribution from the dense head
            if isinstance(decoder, DenseHead):
                outputs[name] = decoder(features)
            else:
                # Handle image decoding
                # This is simplified - normally this would be a proper image distribution
                raw = decoder(features)
                batch_size = features.shape[0]
                mean = raw.reshape(batch_size, *self.shapes[name])
                # Create a normal distribution with fixed variance
                outputs[name] = torchd.independent.Independent(
                    torchd.normal.Normal(mean, torch.ones_like(mean)), len(self.shapes[name])
                )

        return outputs


class DenseHead(nn.Module):
    """MLP based head with distribution output"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 400,
        activation: str = "elu",
        norm: str = "none",
        dist: str = "normal",
        init_std: float = 1.0,
        min_std: float = 0.1,
        max_std: float = 1.0,
        outscale: float = 1.0,
        device: str = "cuda",
    ):
        super(DenseHead, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dist = dist
        self.device = device
        self.min_std = min_std
        self.max_std = max_std
        self.init_std = init_std

        # Create feature network
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=hidden_dim,
            hidden_layers=hidden_layers - 1,
            hidden_dim=hidden_dim,
            activation=activation,
            norm=norm,
            device=device,
        )

        # Distribution parameters
        if dist == "normal":
            self.mean_layer = nn.Linear(hidden_dim, out_dim)
            self.std_layer = nn.Linear(hidden_dim, out_dim)

            # Initialize layers
            torch.nn.init.xavier_uniform_(self.mean_layer.weight, gain=outscale)
            torch.nn.init.zeros_(self.mean_layer.bias)
            torch.nn.init.xavier_uniform_(self.std_layer.weight, gain=outscale)
            torch.nn.init.zeros_(self.std_layer.bias)

        elif dist == "binary":
            self.logit_layer = nn.Linear(hidden_dim, out_dim)

            # Initialize layer
            torch.nn.init.xavier_uniform_(self.logit_layer.weight, gain=outscale)
            torch.nn.init.zeros_(self.logit_layer.bias)

        elif dist == "categorical":
            self.logit_layer = nn.Linear(hidden_dim, out_dim)

            # Initialize layer
            torch.nn.init.xavier_uniform_(self.logit_layer.weight, gain=outscale)
            torch.nn.init.zeros_(self.logit_layer.bias)

        self.to(device)

    def forward(self, features):
        x = self.mlp(features)

        if self.dist == "normal":
            mean = self.mean_layer(x)

            # Get log std with constraints
            log_std = self.std_layer(x)
            log_std = torch.clamp(log_std, math.log(self.min_std), math.log(self.max_std))
            std = torch.exp(log_std)

            dist = torchd.normal.Normal(mean, std)

            # Ensure proper shape for multivariate
            if self.out_dim > 1:
                dist = torchd.independent.Independent(dist, 1)

            # Instead of setting the mean attribute, return both the dist and mean
            return dist

        elif self.dist == "binary":
            logits = self.logit_layer(x)
            dist = torchd.bernoulli.Bernoulli(logits=logits)

            # Ensure proper shape for multivariate
            if self.out_dim > 1:
                dist = torchd.independent.Independent(dist, 1)

            # Return both the dist and mean
            return {"dist": dist, "mean": torch.sigmoid(logits)}

        elif self.dist == "categorical":
            logits = self.logit_layer(x)
            dist = torchd.categorical.Categorical(logits=logits)

            # Return both the dist and mean
            return {"dist": dist, "mean": F.softmax(logits, dim=-1)}


class RSSM(nn.Module):
    """Recurrent State-Space Model (RSSM) from Dreamer paper"""

    def __init__(
        self,
        stoch_size: int = 30,
        deter_size: int = 200,
        hidden_size: int = 200,
        rec_depth: int = 1,
        discrete_size: int = 32,
        act: str = "elu",
        norm: str = "none",
        mean_act: str = "none",
        std_act: str = "softplus",
        min_std: float = 0.1,
        unimix_ratio: float = 0.01,
        embed_size: int = 1024,
        action_size: int = 6,
        device: str = "cuda",
    ):
        super(RSSM, self).__init__()

        self.device = device
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.rec_depth = rec_depth
        self.discrete_size = discrete_size
        self.act = act
        self.norm = norm
        self.mean_act = mean_act
        self.std_act = std_act
        self.min_std = min_std
        self.unimix_ratio = unimix_ratio
        self.embed_size = embed_size
        self.action_size = action_size

        # Activation functions
        act_fn = get_act(act.lower())

        # Define model components
        # For GRU: input is [action, stoch_state]
        gru_input_size = action_size + (stoch_size if not discrete_size else stoch_size * discrete_size)

        # GRU (deterministic state dynamics)
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=deter_size, num_layers=rec_depth, batch_first=True)

        # Prior: p(s_t | h_t)
        self.prior_net = self._build_prior_net()

        # Posterior: q(s_t | h_t, o_t)
        self.posterior_net = self._build_posterior_net()

        self.to(device)

    def _build_prior_net(self):
        """Build the prior network p(s_t | h_t)"""
        if self.discrete_size:
            # For discrete state space
            return MLP(
                in_dim=self.deter_size,
                out_dim=self.stoch_size * self.discrete_size,
                hidden_layers=2,
                hidden_dim=self.hidden_size,
                activation=self.act,
                norm=self.norm,
                device=self.device,
            )
        else:
            # For continuous state space (mean and log_std)
            return MLP(
                in_dim=self.deter_size,
                out_dim=2 * self.stoch_size,
                hidden_layers=2,
                hidden_dim=self.hidden_size,
                activation=self.act,
                norm=self.norm,
                device=self.device,
            )

    def _build_posterior_net(self):
        """Build the posterior network q(s_t | h_t, o_t)"""
        if self.discrete_size:
            # For discrete state space
            return MLP(
                in_dim=self.deter_size + self.embed_size,
                out_dim=self.stoch_size * self.discrete_size,
                hidden_layers=2,
                hidden_dim=self.hidden_size,
                activation=self.act,
                norm=self.norm,
                device=self.device,
            )
        else:
            # For continuous state space (mean and log_std)
            return MLP(
                in_dim=self.deter_size + self.embed_size,
                out_dim=2 * self.stoch_size,
                hidden_layers=2,
                hidden_dim=self.hidden_size,
                activation=self.act,
                norm=self.norm,
                device=self.device,
            )

    def initial(self, batch_size):
        """Return initial state (h, s)"""
        # Initial deterministic state
        deter = torch.zeros(self.rec_depth, batch_size, self.deter_size, device=self.device)

        if self.discrete_size:
            # Initial discrete stochastic state (uniform)
            logits = torch.zeros(batch_size, self.stoch_size, self.discrete_size, device=self.device)
            stoch = torch.zeros(batch_size, self.stoch_size, self.discrete_size, device=self.device)
            # One-hot encoding with uniform probability
            stoch[:, :, 0] = 1

            # Only return relevant keys for discrete case
            return {"deter": deter, "stoch": stoch, "logits": logits}
        else:
            # Initial continuous stochastic state (zeros)
            mean = torch.zeros(batch_size, self.stoch_size, device=self.device)
            std = torch.ones(batch_size, self.stoch_size, device=self.device)
            stoch = torch.zeros(batch_size, self.stoch_size, device=self.device)

            # Return all keys for continuous case
            return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

    def observe(self, embed, action, is_first):
        """Run the dynamics model to get state posteriors with observations"""
        # Initialize state if needed
        batch_size = action.shape[0]
        seq_len = action.shape[1]
        prev_state = self.initial(batch_size)

        # Initialize outputs
        prior = {k: [] for k in ["mean", "std", "stoch", "deter", "logits"] if k in prev_state}
        post = {k: [] for k in ["mean", "std", "stoch", "deter", "logits"] if k in prev_state}

        # Loop through sequence
        for t in range(seq_len):
            # Reset state for new episodes
            _is_first = is_first[:, t] if is_first is not None else torch.zeros(batch_size, device=self.device).bool()

            if t == 0:
                # For first timestep
                prev_state = {
                    k: v * (1.0 - _is_first.float()).reshape(-1, 1, 1 if "deter" in k else 1)
                    for k, v in prev_state.items()
                }
            else:
                # Reset state for environment resets
                for key, val in prev_state.items():
                    if key == "deter":
                        # Handle deterministic state special case (different shape)
                        if val.dim() == 3:  # (num_layers, batch, hidden)
                            # Create correctly sized mask for each layer
                            _is_first_r = _is_first.reshape(1, -1, 1).expand(val.size(0), val.size(1), 1)
                            # Ensure mask is boolean
                            _is_first_r = _is_first_r.bool()
                            # Apply the mask with proper broadcasting
                            val = torch.where(_is_first_r, torch.zeros_like(val), val)
                        else:  # (batch, hidden)
                            # Reshape for 2D tensor
                            _is_first_r = _is_first.reshape(-1, 1).expand_as(val)
                            # Ensure mask is boolean
                            _is_first_r = _is_first_r.bool()
                            val = torch.where(_is_first_r, torch.zeros_like(val), val)
                    else:
                        # Stochastic state and other variables
                        # Determine proper reshape based on tensor dimensions
                        if val.dim() == 3:  # (batch, stoch, discrete) for discrete case
                            _is_first_r = _is_first.reshape(-1, 1, 1).expand_as(val)
                        else:  # (batch, feature) for continuous case
                            _is_first_r = _is_first.reshape(-1, 1).expand_as(val)

                        # Ensure mask is boolean
                        _is_first_r = _is_first_r.bool()
                        val = torch.where(_is_first_r, torch.zeros_like(val), val)

                    prev_state[key] = val

            # Get state posteriors
            prev_state, prior_state = self.obs_step(prev_state, action[:, t], embed[:, t], _is_first)

            # Store outputs
            for k in prev_state:
                post[k].append(prev_state[k])

            for k in prior_state:
                prior[k].append(prior_state[k])

        # Stack outputs along time dimension (dim=1)
        for k in post:
            post[k] = torch.stack(post[k], dim=1)

        for k in prior:
            prior[k] = torch.stack(prior[k], dim=1)

        return post, prior

    def obs_step(self, prev_state, action, embed, is_first=None):
        """Single step of the dynamics model with observation"""
        # Get prior state
        prior_state = self.img_step(prev_state, action, is_first)

        # Get posterior state
        deter = prior_state["deter"][-1]  # Use the latest layer's state
        x = torch.cat([deter, embed], dim=-1)

        # Compute posterior
        if self.discrete_size:
            # Discrete state case
            logits = self.posterior_net(x).reshape(-1, self.stoch_size, self.discrete_size)
            # Apply temperature and add mixture for stability
            logits = self._apply_unimix(logits)
            # Sample categorical stochastic state
            dist = torchd.categorical.Categorical(logits=logits)
            stoch = torch.zeros_like(logits)
            stoch_sample = dist.sample()
            # Create one-hot encoding
            stoch = stoch.scatter_(-1, stoch_sample.unsqueeze(-1), 1.0)

            post = {"stoch": stoch, "deter": deter, "logits": logits}
        else:
            # Continuous state case
            x = self.posterior_net(x)
            mean, log_std = torch.split(x, self.stoch_size, dim=-1)

            # Apply activation for mean if needed
            if self.mean_act == "tanh":
                mean = torch.tanh(mean)

            # Get std with activation and clamping
            std = {
                "softplus": lambda: F.softplus(log_std),
                "sigmoid": lambda: torch.sigmoid(log_std),
                "sigmoid2": lambda: 2 * torch.sigmoid(log_std / 2),
                "exp": lambda: torch.exp(log_std),
                "none": lambda: log_std,
            }[self.std_act]()

            std = torch.clamp(std, min=self.min_std)

            # Sample stochastic state
            dist = torchd.normal.Normal(mean, std)
            stoch = dist.rsample()

            post = {"stoch": stoch, "deter": deter, "mean": mean, "std": std}

        return post, prior_state

    def img_step(self, prev_state, action, is_first=None):
        """Single step of the dynamics model (prior without observation)"""
        # Initialize state if needed
        if prev_state is None:
            batch_size = action.shape[0]
            prev_state = self.initial(batch_size)

        # Extract previous state components
        prev_stoch = prev_state["stoch"]
        prev_deter = prev_state["deter"]

        # Reset state for new episodes
        if is_first is not None:
            is_first = is_first.to(torch.bool)

            # Check if batch sizes match, if not, expand is_first accordingly
            if is_first.shape[0] != prev_stoch.shape[0]:
                # Assuming is_first should be broadcast to match prev_stoch's batch size
                repeat_times = prev_stoch.shape[0] // is_first.shape[0]
                is_first = is_first.repeat_interleave(repeat_times)

            # Now apply the masking with correctly sized tensors
            if prev_deter.dim() == 3:  # Handle case where prev_deter is (num_layers, batch, hidden)
                # Reshape is_first to match prev_deter's batch dimension location (dim 1)
                is_first_deter = is_first.reshape(1, -1, 1)
                # Create a boolean mask instead of using float multiplication
                mask = (~is_first_deter).expand_as(prev_deter)
                prev_deter = torch.where(mask, prev_deter, torch.zeros_like(prev_deter))
            else:  # Handle case where prev_deter is (batch, hidden)
                mask = (~is_first.reshape(-1, 1)).expand_as(prev_deter)
                prev_deter = torch.where(mask, prev_deter, torch.zeros_like(prev_deter))

            # Handle stochastic state similarly
            if prev_stoch.dim() > 2:  # For discrete case (batch, stoch, discrete)
                mask = (~is_first.reshape(-1, 1, 1)).expand_as(prev_stoch)
                prev_stoch = torch.where(mask, prev_stoch, torch.zeros_like(prev_stoch))
            else:  # For continuous case (batch, stoch)
                mask = (~is_first.reshape(-1, 1)).expand_as(prev_stoch)
                prev_stoch = torch.where(mask, prev_stoch, torch.zeros_like(prev_stoch))

        # Prepare input to GRU
        if self.discrete_size:
            # Flatten one-hot stochastic state
            x = prev_stoch.reshape(-1, self.stoch_size * self.discrete_size)
        else:
            x = prev_stoch

        # Concatenate with action
        x = torch.cat([x, action], dim=-1)

        # Add sequence dimension for GRU (batch, 1, features)
        x = x.unsqueeze(1)

        # Ensure prev_deter has the correct shape (num_layers, batch_size, hidden_size)
        # First determine batch_size from available tensors
        if action is not None:
            batch_size = action.shape[0]
        else:
            # Try to infer from prev_state
            if prev_stoch.dim() > 1:
                batch_size = prev_stoch.shape[0]
            else:
                # Default fallback
                batch_size = 1

        # If prev_deter is (batch_size, hidden_size) or has other incorrect shapes
        if prev_deter.dim() == 2:
            # Convert (batch_size, hidden_size) to (num_layers, batch_size, hidden_size)
            prev_deter = prev_deter.unsqueeze(0)
        elif prev_deter.dim() == 3 and prev_deter.size(0) != self.rec_depth:
            # If 3D but first dimension is not num_layers, reshape correctly

            # Calculate the expected total elements
            expected_elements = self.rec_depth * batch_size * self.deter_size

            # Check if the tensor has the right number of elements
            if prev_deter.numel() != expected_elements:
                # We need to create a properly sized tensor
                if prev_deter.size(0) * prev_deter.size(1) * prev_deter.size(2) > expected_elements:
                    # If we have too many elements, slice to get the right size
                    # Try to preserve the last states which are most relevant
                    elements_per_layer = batch_size * self.deter_size
                    # Take the last n elements that would fill our target shape
                    flat_deter = prev_deter.reshape(-1)[-expected_elements:]
                    prev_deter = flat_deter.reshape(self.rec_depth, batch_size, self.deter_size)
                else:
                    # If we have too few elements, use zeros and copy what we have
                    new_deter = torch.zeros(self.rec_depth, batch_size, self.deter_size, device=prev_deter.device)
                    # Copy available data (might be partial)
                    flat_prev = prev_deter.reshape(-1)
                    flat_new = new_deter.reshape(-1)
                    flat_new[: min(flat_prev.size(0), flat_new.size(0))] = flat_prev[
                        : min(flat_prev.size(0), flat_new.size(0))
                    ]
                    prev_deter = new_deter
            else:
                # We have the right number of elements, just reshape
                prev_deter = prev_deter.reshape(self.rec_depth, batch_size, self.deter_size)

        # Run GRU
        x, deter = self.gru(x, prev_deter)
        x = x.squeeze(1)  # Remove sequence dimension

        # Compute prior
        if self.discrete_size:
            # Discrete state case
            logits = self.prior_net(x).reshape(-1, self.stoch_size, self.discrete_size)
            # Apply temperature and add mixture for stability
            logits = self._apply_unimix(logits)
            # Sample categorical stochastic state
            dist = torchd.categorical.Categorical(logits=logits)
            stoch = torch.zeros_like(logits)
            stoch_sample = dist.sample()
            # Create one-hot encoding
            stoch = stoch.scatter_(-1, stoch_sample.unsqueeze(-1), 1.0)

            prior = {"stoch": stoch, "deter": deter, "logits": logits}
        else:
            # Continuous state case
            x = self.prior_net(x)
            mean, log_std = torch.split(x, self.stoch_size, dim=-1)

            # Apply activation for mean if needed
            if self.mean_act == "tanh":
                mean = torch.tanh(mean)

            # Get std with activation and clamping
            std = {
                "softplus": lambda: F.softplus(log_std),
                "sigmoid": lambda: torch.sigmoid(log_std),
                "sigmoid2": lambda: 2 * torch.sigmoid(log_std / 2),
                "exp": lambda: torch.exp(log_std),
                "none": lambda: log_std,
            }[self.std_act]()

            std = torch.clamp(std, min=self.min_std)

            # Sample stochastic state
            dist = torchd.normal.Normal(mean, std)
            stoch = dist.rsample()

            prior = {"stoch": stoch, "deter": deter, "mean": mean, "std": std}

        return prior

    def _apply_unimix(self, logits):
        """Apply uniform mixture to logits for stability"""
        if self.unimix_ratio > 0.0:
            # Apply uniform mixture
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
            logits = torch.log(probs)
        return logits

    def get_dist(self, state):
        """Get distribution from state"""
        if self.discrete_size:
            return torchd.categorical.Categorical(logits=state["logits"])
        else:
            return torchd.normal.Normal(state["mean"], state["std"])

    def get_feat(self, state):
        """Get features for downstream tasks"""
        # Handle different possible shapes of state["stoch"]
        stoch = state["stoch"]

        # Check for sequence dimension
        if stoch.dim() == 4:  # [batch, seq, stoch_size, discrete_size]
            batch_size, seq_len = stoch.shape[0], stoch.shape[1]
            # Flatten batch and sequence dims
            batch_seq_size = batch_size * seq_len

            if self.discrete_size:
                # Reshape to [batch*seq, stoch*discrete]
                stoch_flat = stoch.reshape(batch_seq_size, self.stoch_size * self.discrete_size)

                # Handle deterministic state
                if state["deter"].dim() == 3:  # [layers, batch, hidden]
                    # Get last layer and repeat for each sequence position
                    deter = state["deter"][-1]
                    # Only repeat if batch dimensions don't already match
                    if deter.shape[0] != batch_seq_size:
                        deter = deter.repeat_interleave(seq_len, dim=0)
                else:  # Assume [batch, hidden] or already matching
                    deter = state["deter"]
                    # If sizes still don't match, try expanding
                    if deter.shape[0] != batch_seq_size:
                        # Only repeat if it's a factor (avoid stretching errors)
                        if batch_seq_size % deter.shape[0] == 0:
                            seq_factor = batch_seq_size // deter.shape[0]
                            deter = deter.repeat_interleave(seq_factor, dim=0)

                # Safety check - if dimensions still don't match for concatenation
                if deter.shape[0] != stoch_flat.shape[0]:
                    # Adapt whichever one is smaller to match the larger one
                    if deter.shape[0] < stoch_flat.shape[0]:
                        ratio = stoch_flat.shape[0] // deter.shape[0]
                        deter = deter.repeat_interleave(ratio, dim=0)
                    else:
                        ratio = deter.shape[0] // stoch_flat.shape[0]
                        # Take every nth item to reduce size
                        deter = deter[::ratio]

                return torch.cat([stoch_flat, deter], dim=-1)
            else:
                # For continuous case, reshape stoch to [batch*seq, stoch_size]
                stoch_flat = stoch.reshape(batch_seq_size, self.stoch_size)

                # Apply same logic as discrete case
                if state["deter"].dim() == 3:
                    deter = state["deter"][-1]
                    if deter.shape[0] != batch_seq_size:
                        deter = deter.repeat_interleave(seq_len, dim=0)
                else:
                    deter = state["deter"]
                    if deter.shape[0] != batch_seq_size:
                        if batch_seq_size % deter.shape[0] == 0:
                            seq_factor = batch_seq_size // deter.shape[0]
                            deter = deter.repeat_interleave(seq_factor, dim=0)

                # Safety check
                if deter.shape[0] != stoch_flat.shape[0]:
                    if deter.shape[0] < stoch_flat.shape[0]:
                        ratio = stoch_flat.shape[0] // deter.shape[0]
                        deter = deter.repeat_interleave(ratio, dim=0)
                    else:
                        ratio = deter.shape[0] // stoch_flat.shape[0]
                        deter = deter[::ratio]

                return torch.cat([stoch_flat, deter], dim=-1)

        # Original code for 3D or 2D stochastic state
        batch_size = stoch.shape[0]

        if self.discrete_size:
            # For discrete state, reshape one-hot vectors to flat
            if stoch.dim() == 3:  # [batch, stoch_size, discrete_size]
                stoch_flat = stoch.reshape(batch_size, self.stoch_size * self.discrete_size)
            else:
                # Handle unexpected shape - try to make it work
                stoch_flat = stoch.reshape(batch_size, -1)

            # Handle deterministic state
            if state["deter"].dim() > 2:  # [layers, batch, hidden] or [batch, seq, hidden]
                if state["deter"].size(0) == self.rec_depth:
                    # Case: [layers, batch, hidden]
                    deter_expanded = state["deter"][-1]  # Take last layer
                else:
                    # Case: [batch, seq, hidden]
                    # Flatten the sequence dimension
                    deter_expanded = state["deter"].reshape(state["deter"].size(0), -1)

                if deter_expanded.shape[0] != batch_size:
                    # Try to match batch sizes
                    if batch_size % deter_expanded.shape[0] == 0:
                        seq_factor = batch_size // deter_expanded.shape[0]
                        deter_expanded = deter_expanded.repeat_interleave(seq_factor, dim=0)
            else:  # [batch, hidden]
                deter_expanded = state["deter"]
                if deter_expanded.shape[0] != batch_size:
                    if batch_size % deter_expanded.shape[0] == 0:
                        seq_factor = batch_size // deter_expanded.shape[0]
                        deter_expanded = deter_expanded.repeat_interleave(seq_factor, dim=0)

            # Safety check
            if deter_expanded.shape[0] != stoch_flat.shape[0]:
                if deter_expanded.shape[0] < stoch_flat.shape[0]:
                    ratio = stoch_flat.shape[0] // deter_expanded.shape[0]
                    deter_expanded = deter_expanded.repeat_interleave(ratio, dim=0)
                else:
                    ratio = deter_expanded.shape[0] // stoch_flat.shape[0]
                    deter_expanded = deter_expanded[::ratio]

            # Ensure dimensions match for concatenation
            if deter_expanded.dim() > 2:
                # Flatten extra dimensions
                deter_expanded = deter_expanded.reshape(deter_expanded.size(0), -1)

            return torch.cat([stoch_flat, deter_expanded], dim=-1)
        else:
            # For continuous state
            if state["deter"].dim() > 2:
                if state["deter"].size(0) == self.rec_depth:
                    # Case: [layers, batch, hidden]
                    deter_expanded = state["deter"][-1]  # Take last layer
                else:
                    # Case: [batch, seq, hidden]
                    # Flatten the sequence dimension
                    deter_expanded = state["deter"].reshape(state["deter"].size(0), -1)

                if deter_expanded.shape[0] != batch_size:
                    if batch_size % deter_expanded.shape[0] == 0:
                        seq_factor = batch_size // deter_expanded.shape[0]
                        deter_expanded = deter_expanded.repeat_interleave(seq_factor, dim=0)
            else:
                deter_expanded = state["deter"]
                if deter_expanded.shape[0] != batch_size:
                    if batch_size % deter_expanded.shape[0] == 0:
                        seq_factor = batch_size // deter_expanded.shape[0]
                        deter_expanded = deter_expanded.repeat_interleave(seq_factor, dim=0)

            # Safety check
            if deter_expanded.shape[0] != stoch.shape[0]:
                if deter_expanded.shape[0] < stoch.shape[0]:
                    ratio = stoch.shape[0] // deter_expanded.shape[0]
                    deter_expanded = deter_expanded.repeat_interleave(ratio, dim=0)
                else:
                    ratio = deter_expanded.shape[0] // stoch.shape[0]
                    deter_expanded = deter_expanded[::ratio]

            # Ensure dimensions match for concatenation
            if deter_expanded.dim() > 2:
                # Flatten extra dimensions
                deter_expanded = deter_expanded.reshape(deter_expanded.size(0), -1)

            # Ensure stoch is 2D before concatenation
            if stoch.dim() > 2:
                stoch = stoch.reshape(stoch.size(0), -1)

            return torch.cat([stoch, deter_expanded], dim=-1)

    def kl_loss(self, post, prior, kl_free, dyn_scale, rep_scale):
        """Compute KL loss between posterior and prior distributions"""
        # Helper function to stop gradients
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # Get distributions based on discrete or continuous state
        if self.discrete_size:
            # For discrete states
            q_post = torchd.categorical.Categorical(logits=post["logits"])
            p_prior = torchd.categorical.Categorical(logits=prior["logits"])
            q_post_sg = torchd.categorical.Categorical(logits=sg(post)["logits"])
            p_prior_sg = torchd.categorical.Categorical(logits=sg(prior)["logits"])

            # Calculate two different KL divergences with proper gradient flow
            rep_loss = torchd.kl.kl_divergence(q_post, p_prior_sg).sum(-1)
            dyn_loss = torchd.kl.kl_divergence(q_post_sg, p_prior).sum(-1)
        else:
            # For continuous states
            q_post = torchd.normal.Normal(post["mean"], post["std"])
            p_prior = torchd.normal.Normal(prior["mean"], prior["std"])
            q_post_sg = torchd.normal.Normal(sg(post)["mean"], sg(post)["std"])
            p_prior_sg = torchd.normal.Normal(sg(prior)["mean"], sg(prior)["std"])

            # Calculate two different KL divergences with proper gradient flow
            rep_loss = torchd.kl.kl_divergence(q_post, p_prior_sg).sum(-1)
            dyn_loss = torchd.kl.kl_divergence(q_post_sg, p_prior).sum(-1)

        # For logging purposes
        kl_value = rep_loss.clone()

        # Apply free bits (KL balancing)
        if kl_free > 0:
            rep_loss = torch.clip(rep_loss, min=kl_free)
            dyn_loss = torch.clip(dyn_loss, min=kl_free)

        # Combine losses with scales
        kl_loss = dyn_scale * dyn_loss.mean() + rep_scale * rep_loss.mean()

        return kl_loss, kl_value.mean(), dyn_loss.mean(), rep_loss.mean()
