# diffusion_policy/model/bc/actor_mlp_predictor.py
import torch
import torch.nn as nn
from typing import Sequence, List, Union, Callable

_ACTS = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

def get_activation(name: str) -> Callable[[], nn.Module]:
    if name not in _ACTS:
        raise ValueError(f"Unsupported activation: {name}")
    return _ACTS[name]

class MLPActionPredictor(nn.Module):
    """
    输入 : (B, 4, Do)         输出 : (B, Da)
    结构与 legged-gym Actor 一致：Linear-Act-...-Linear
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_obs_steps: int = 4,
        hidden_dims: Union[Sequence[int], List[int]] = (256, 256, 256),
        activation: str = "elu",
        p_dropout: float = 0.0,            # RL 常用 0；若想 regularize 可调
    ):
        super().__init__()
        act = get_activation(activation)()

        in_dim = obs_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]
            if p_dropout > 0:
                layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(hidden_dims[-1], action_dim))     # 输出 mean

        self.net = nn.Sequential(*layers)

        # ---- 若你想像 RL 一样加可学习 std，可以解注释以下三行 ----
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        # self.log_std.requires_grad = True
        # self.tanh_out = False   # 若需 tanh 压缩动作空间

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)  # (B, 4*Do)
        mean = self.net(x)          # (B, Da)
        # if self.tanh_out:          # 需要时开启
        #     mean = torch.tanh(mean)
        return mean
