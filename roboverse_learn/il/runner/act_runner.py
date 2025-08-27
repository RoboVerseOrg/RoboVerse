import os
import pickle

import torch
import yaml

from metasim.cfg.policy import ACTPolicyCfg
from il.utils.act.policy import ACTPolicy

from .base_runner import PolicyRunner


class ACTRunner(PolicyRunner):
    def _init_policy(self, **kwargs):
        self.policy_cfg = ACTPolicyCfg()
        self.model_path = kwargs.get("checkpoint_path")

        config_path = os.path.join(self.model_path, "cfg.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract policy configuration
        policy_config = config["policy_config"]
        act_ckpt_name = "policy_best.ckpt"

        policy = ACTPolicy(policy_config)
        ckpt_path = os.path.join(self.model_path, act_ckpt_name)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda()
        policy.eval()
        self.policy = policy
        print(f"Loaded: {ckpt_path}")
        stats_path = os.path.join(self.model_path, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        # Override self.policy_cfg
        self.policy_cfg.obs_config.obs_type = config["data"]["observation_space"]
        self.policy_cfg.action_config.action_type = config["data"]["action_space"]
        self.policy_cfg.action_config.action_chunk_steps = policy_config["num_queries"]
        self.policy_cfg.obs_config.obs_padding = config["data"].get("joint_pos_padding", 0)
        self.policy_cfg.action_config.delta = config["data"]["delta_ee"]
        self.policy_cfg.obs_config.obs_dim = policy_config["state_dim"]

    def pre_process(self, s_state):
        return (s_state - torch.tensor(self.stats["state_mean"], device=self.device)) / torch.tensor(
            self.stats["state_std"], device=self.device
        )

    def post_process(self, a):
        return a * torch.tensor(self.stats["action_std"], device=self.device) + torch.tensor(
            self.stats["action_mean"], device=self.device
        )

    def predict_action(self, observation):
        state = self.pre_process(observation["agent_pos"]).cuda()
        curr_image = observation["head_cam"].unsqueeze(1).cuda()
        with torch.no_grad():
            action_chunk = self.policy(state, curr_image)
        action = self.post_process(action_chunk)
        action = action[:, :, : self.policy_cfg.action_config.action_dim]
        return action.transpose(0, 1)  # Expects (action_chunk_steps, n_envs, action_dim)