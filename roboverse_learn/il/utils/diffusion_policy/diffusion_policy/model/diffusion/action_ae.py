import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.diffusion.layers import Mlp


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class CNNActionEncoder(nn.Module):
    def __init__(
        self,
        pred_horizon: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # CNN encoder layers
        layers = []
        current_dim = action_dim

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            else:
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        conv_output_length = pred_horizon // (2 ** num_layers)
        conv_output_dim = hidden_dim * conv_output_length

        self.latent_proj = nn.Linear(conv_output_dim, latent_dim)

        self.apply(weights_init_encoder)

    def forward(self, actions, deterministic=False):
        batch_size = actions.shape[0]

        x = actions.transpose(1, 2)  # (B, action_dim, pred_horizon)
        x = self.encoder(x)  # (B, hidden_dim, conv_output_length)
        x = x.view(batch_size, -1)  # (B, hidden_dim * conv_output_length)

        z = self.latent_proj(x)  # (B, latent_dim)

        return z


class SimpleActionDecoder(nn.Module):
    def __init__(
        self,
        dec_hidden_dim: int,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = pred_horizon * action_dim

        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.output_proj = nn.Linear(dec_hidden_dim, output_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                Mlp(
                    in_features=dec_hidden_dim,
                    hidden_features=dec_hidden_dim,
                    out_features=dec_hidden_dim,
                    norm_layer=None,
                    bias=True,
                    drop=dropout
                )
            )

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        actions = x.view(-1, self.pred_horizon, self.action_dim)
        return actions
