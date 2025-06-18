import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Actor(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs["actions_num"]
        input_shape = kwargs["input_shape"]
        self.units = kwargs["actor_units"]
        mlp_input_shape = input_shape[0]

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        last_layer = list(self.actor_mlp.modules())[-2].out_features
        self.actor_mlp = nn.Sequential(*list(self.actor_mlp.children()), nn.Linear(last_layer, actions_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        last_linear = list(self.actor_mlp.modules())[-1]
        torch.nn.init.uniform_(last_linear.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(last_linear.bias, -3e-3, 3e-3)

    def forward(self, input_dict):
        obs_original = input_dict["obs"]
        obs = obs_original
        x = self.actor_mlp(obs)
        return torch.tanh(x)

    @torch.no_grad()
    def act_inference(self, obs_dict):
        return self.forward(obs_dict)

    def act(self, obs_dict, exploration_noise=0.1, target_noise=None, noise_clip=None):
        actions = self.forward(obs_dict)

        if target_noise is not None and noise_clip is not None:
            noise = torch.randn_like(actions) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            actions = torch.clamp(actions + noise, -1.0, 1.0)
        elif exploration_noise > 0:
            noise = torch.randn_like(actions) * exploration_noise
            actions = torch.clamp(actions + noise, -1.0, 1.0)

        return actions


class Critic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        input_shape = kwargs.get("input_shape")
        self.units = kwargs.get("critic_units")
        mlp_input_shape = input_shape[0] + kwargs["actions_num"]

        self.Q1 = MLP(units=self.units, input_size=mlp_input_shape)
        last_layer = list(self.Q1.modules())[-2].out_features
        self.Q1 = nn.Sequential(*list(self.Q1.children()), nn.Linear(last_layer, 1))

        self.Q2 = MLP(units=self.units, input_size=mlp_input_shape)
        last_layer = list(self.Q2.modules())[-2].out_features
        self.Q2 = nn.Sequential(*list(self.Q2.children()), nn.Linear(last_layer, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_dict, action):
        obs_original = input_dict["obs"]
        obs = obs_original
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

    def Q1_forward(self, input_dict, action):
        obs_original = input_dict["obs"]
        obs = obs_original
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        return q1
