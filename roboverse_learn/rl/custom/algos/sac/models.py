import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
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

        out_size = self.units[-1]

        self.log_std_bounds = kwargs.get("log_std_bounds", [-5, 2])
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        last_layer = list(self.actor_mlp.modules())[-2].out_features
        self.actor_mlp = nn.Sequential(*list(self.actor_mlp.children()), nn.Linear(last_layer, actions_num * 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_dict):
        obs_original = input_dict["obs"]
        obs = obs_original
        mu, log_std = torch.chunk(self.actor_mlp(obs), 2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_bounds[0] + 0.5 * (self.log_std_bounds[1] - self.log_std_bounds[0]) * (log_std + 1)
        return mu, log_std

    @torch.no_grad()
    def act_inference(self, obs_dict):
        mu, log_std = self.forward(obs_dict)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        return action

    def get_action(self, obs_dict, action_scale=1.0, action_bias=0.0, stage2=False):
        mu, log_std = self.forward(obs_dict)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * action_scale + action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mu) * action_scale + action_bias
        return action, log_prob, mean


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
