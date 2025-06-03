# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# import os
# from copy import deepcopy

# import numpy as np
# import torch
# import torch.nn as nn
# from legged_gym import PPO_ROOT_DIR
# from torch.distributions import Normal
# from torch.nn.modules import rnn

# from metasim.utils.dict import class_to_dict


# class ActorCriticHierarchical(nn.Module):
#     is_recurrent = False

#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         obs_context_len=1,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="elu",
#         init_noise_std=1.0,
#         **kwargs,
#     ):
#         # if kwargs:
#         #     print("ActorCriticHierarchical.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
#         super(ActorCriticHierarchical, self).__init__()

#         self.obs_context_len = obs_context_len

#         activation = get_activation(activation)

#         mlp_input_dim_a = num_actor_obs
#         mlp_input_dim_c = num_critic_obs

#         # Get low-level skills
#         self.args = kwargs["args"]
#         self.device = kwargs["device"]
#         self.frame_stack = kwargs["frame_stack"]
#         self.command_dim = kwargs["command_dim"]
#         self.num_dofs = kwargs["num_dofs"]
#         self._get_low_level_policies(self.args, self.device, kwargs)
#         num_output = self._get_num_output(kwargs["frame_stack"])

#         # Policy
#         actor_layers = []
#         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
#         actor_layers.append(activation)
#         for l in range(len(actor_hidden_dims)):
#             if l == len(actor_hidden_dims) - 1:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_output))
#             else:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
#                 actor_layers.append(activation)
#         self.actor = nn.Sequential(*actor_layers)

#         # Value function
#         critic_layers = []
#         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
#         critic_layers.append(activation)
#         for l in range(len(critic_hidden_dims)):
#             if l == len(critic_hidden_dims) - 1:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
#             else:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
#                 critic_layers.append(activation)
#         self.critic = nn.Sequential(*critic_layers)

#         print(f"Actor Network: {self.actor}")
#         print(f"Critic Network: {self.critic}")

#         # Action noise
#         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         self.distribution = None
#         # disable args validation for speedup
#         Normal.set_default_validate_args = False

#         # seems that we get better performance without init
#         # self.init_memory_weights(self.memory_a, 0.001, 0.)
#         # self.init_memory_weights(self.memory_c, 0.001, 0.)

#     @staticmethod
#     # not used at the moment
#     def init_weights(sequential, scales):
#         [
#             torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
#             for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
#         ]

#     def _get_one_policy(self, args, device, task, experiment_name, load_run, checkpoint):
#         from legged_gym.utils import task_registry
#         from legged_gym.utils.helpers import get_load_path

#         from rsl_rl.modules import ActorCritic

#         # get skill arguments
#         skill_args = deepcopy(args)
#         assert task == experiment_name
#         skill_args.task = task
#         skill_args.experiment_name = experiment_name
#         skill_args.load_run = load_run
#         skill_args.checkpoint = checkpoint
#         skill_env_cfg, skill_train_cfg = task_registry.get_cfgs(
#             name=skill_args.task, load_run=skill_args.load_run, experiment_name=skill_args.experiment_name
#         )
#         # load skill policy
#         skill_policy = ActorCritic(
#             skill_env_cfg.env.num_observations,
#             skill_env_cfg.env.num_privileged_obs,
#             skill_env_cfg.env.num_actions,
#             obs_context_len=1,
#             **class_to_dict(skill_train_cfg)["policy"],
#         ).to(device)
#         log_root = os.path.join(PPO_ROOT_DIR, "logs", skill_train_cfg.runner.experiment_name)
#         skill_resume_path = get_load_path(log_root, load_run=skill_args.load_run, checkpoint=skill_args.checkpoint)
#         print(f"Loading {skill_args.task} policy from: {skill_resume_path}")
#         try:
#             loaded_dict = torch.load(skill_resume_path, map_location=device)
#         except:
#             loaded_dict = torch.load(skill_resume_path, map_location="cuda:0")
#         skill_policy.load_state_dict(loaded_dict["model_state_dict"])
#         skill_train_cfg.runner.resume_path = skill_resume_path
#         skill_policy.freeze()
#         skill_policy = skill_policy.actor
#         return skill_policy, skill_env_cfg, skill_train_cfg

#     def _get_low_level_policies(self, args, device, kwargs):
#         skill_dict = kwargs["skill_dict"]
#         self.skill_names = list(skill_dict.keys())
#         self.policy_list = []
#         self.env_cfg_list = []
#         self.train_cfg_list = []
#         self.low_high_list = []
#         for key, value in skill_dict.items():
#             policy, env_cfg, train_cfg = self._get_one_policy(
#                 args, device, key, value["experiment_name"], value["load_run"], value["checkpoint"]
#             )
#             self.policy_list.append(policy)
#             self.env_cfg_list.append(env_cfg)
#             self.train_cfg_list.append(train_cfg)
#             self.low_high_list.append(value["low_high"])
#         self.num_skills = len(self.policy_list)

#     def _get_num_output(self, frame_stack=1):
#         num_output = 0
#         for i in range(self.num_skills):
#             num_output += self.env_cfg_list[i].env.command_dim + self.env_cfg_list[i].env.num_actions
#         return num_output * frame_stack

#     def reset(self, dones=None):
#         pass

#     def forward(self):
#         raise NotImplementedError

#     @property
#     def action_mean(self):
#         return self.distribution.mean

#     @property
#     def action_std(self):
#         return self.distribution.stddev

#     @property
#     def entropy(self):
#         return self.distribution.entropy().sum(dim=-1)

#     def _replace_observations(self, observations, command, low_high=None):
#         """
#         observations dim: [4096 (num_envs), frame_stack*num_single_obs]
#         command_dim: [4096 (num_envs), frame_stack*new_command_dim]
#         state_dim (not include command): 63
#         num_single_obs = state_dim + command_dim
#         """
#         # import pdb; pdb.set_trace()
#         if low_high is not None:
#             low, high = low_high
#             command = torch.clamp(command, low, high)
#         new_observations = observations.clone().reshape(observations.shape[0], self.frame_stack, -1)
#         num_envs, frame_stack, num_single_obs = new_observations.shape
#         state_dim = num_single_obs - self.command_dim  # 63
#         new_command = command.reshape(num_envs, frame_stack, -1)
#         replaced_observations = torch.zeros(
#             (num_envs, frame_stack, state_dim + new_command.shape[-1]), device=self.device
#         )  # command: [num_envs, frame_stack*command_dim], e.g. [4096, 15*3]
#         replaced_observations[:, :, new_command.shape[-1] :] = new_observations[:, :, self.command_dim :]
#         replaced_observations[:, :, : new_command.shape[-1]] = new_command
#         return replaced_observations.reshape(num_envs, -1)

#     def _actor(self, observations):
#         raw_mean = self.actor(observations)  # [input_to_low_level_policies, weight_for_low_level_policies]
#         input_to_low_level_policies = raw_mean[:, : -(self.num_skills * self.num_dofs)]
#         mask_to_low_level_policies = raw_mean[:, -(self.num_skills * self.num_dofs) :]
#         masks = []
#         for i in range(self.num_skills):
#             mask = mask_to_low_level_policies[:, i * self.num_dofs : (i + 1) * self.num_dofs]
#             masks.append(mask)
#         masks = torch.stack(masks, dim=1)  # [4096, num_skills, 19]
#         masks = torch.softmax(masks, dim=1)  # [4096, num_skills, 19]
#         means = []
#         for i in range(self.num_skills):
#             prev_command_dim_sum = sum([self.env_cfg_list[j].env.command_dim for j in range(i)])
#             curr_command_dim = self.env_cfg_list[i].env.command_dim
#             input_to_low_level_policies_i = input_to_low_level_policies[
#                 :, prev_command_dim_sum : prev_command_dim_sum + curr_command_dim
#             ]
#             input_to_low_level_policies_i = self._replace_observations(
#                 observations, input_to_low_level_policies_i, low_high=self.low_high_list[i]
#             )
#             mean = self.policy_list[i](input_to_low_level_policies_i) * masks[:, i]
#             means.append(mean)
#         actions_mean = sum(means)
#         return {"actions_mean": actions_mean, "masks": masks}

#     def update_distribution(self, observations):
#         mean = self._actor(observations)["actions_mean"]
#         self.distribution = Normal(mean, mean * 0.0 + self.std)

#     def act(self, observations, **kwargs):
#         if self.obs_context_len != 1:
#             observations = observations[..., -1, :]
#         self.update_distribution(observations)
#         return self.distribution.sample()

#     def get_actions_log_prob(self, actions):
#         return self.distribution.log_prob(actions).sum(dim=-1)

#     def act_inference(self, observations):
#         if self.obs_context_len != 1:
#             observations = observations[..., -1, :]
#         actions_mean = self._actor(observations)["actions_mean"]
#         return actions_mean

#     def act_inference_hrl(self, observations):
#         if self.obs_context_len != 1:
#             observations = observations[..., -1, :]
#         output = self._actor(observations)
#         # print(output['weight_for_low_level_policies'][:5])
#         return output

#     def evaluate(self, critic_observations, **kwargs):
#         if self.obs_context_len != 1:
#             critic_observations = critic_observations[..., -1, :]
#         value = self.critic(critic_observations)
#         return value


# def get_activation(act_name):
#     if act_name == "elu":
#         return nn.ELU()
#     elif act_name == "selu":
#         return nn.SELU()
#     elif act_name == "relu":
#         return nn.ReLU()
#     elif act_name == "crelu":
#         return nn.ReLU()
#     elif act_name == "lrelu":
#         return nn.LeakyReLU()
#     elif act_name == "tanh":
#         return nn.Tanh()
#     elif act_name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         print("invalid activation function!")
#         return None
