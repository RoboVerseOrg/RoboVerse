import torch
import torch.nn as nn

from .fast_td3 import Actor
from .fast_td3_utils import EmpiricalNormalization


class Policy(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        actor_hidden_dim: int,
    ):
        super().__init__()
        self.actor = Actor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device="cpu",
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
        )
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")

        self.actor.eval()
        self.obs_normalizer.eval()

    @torch.no_grad
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        actions = self.actor(norm_obs)
        return actions

    @torch.no_grad
    def act(self, obs: torch.Tensor) -> torch.distributions.Normal:
        actions = self.forward(obs)
        return torch.distributions.Normal(actions, torch.ones_like(actions) * 1e-8)


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(f"{checkpoint_path}", map_location="cpu", weights_only=False)
    args = torch_checkpoint["args"]

    n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
    n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]

    policy = Policy(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args["num_envs"],
        init_scale=args["init_scale"],
        actor_hidden_dim=args["actor_hidden_dim"],
    )

    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])

    return policy
