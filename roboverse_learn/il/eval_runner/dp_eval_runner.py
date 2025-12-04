from collections import deque

import dill
import hydra
import numpy as np
import torch
from roboverse_learn.il.runner.base_policy import DiffusionPolicyCfg
from roboverse_learn.il.base.base_eval_runner import BaseEvalRunner
from roboverse_learn.il.runner.dp_runner import DPRunner


class DPEvalRunner(BaseEvalRunner):
    """Runner for a diffusion policy, loads in a workspace and policy from checkpoint, and overrides some of the
    PolicyCFG attributes to match how the policy was trained
    """

    def _init_policy(self, dprunner: DPRunner, **kwargs):
        self.task_name = kwargs.get("task_name")
        payload = torch.load(open(kwargs["checkpoint_path"], "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        dprunner.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = dprunner.model
        if cfg.train_config.training_params.use_ema:
            policy = dprunner.ema_model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        self.policy = policy
        self.yaml_cfg = cfg
        self.policy_cfg = DiffusionPolicyCfg()

        if "policy_runner" in cfg.eval_config:
            self.policy_cfg.obs_config.from_dict(cfg.eval_config.policy_runner.obs)
            self.policy_cfg.action_config.from_dict(
                cfg.eval_config.policy_runner.action
            )
            self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
            self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        self.policy_cfg.action_config.action_chunk_steps = cfg.n_action_steps
        self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
        self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        self.obs = deque(maxlen=cfg.n_obs_steps + 1)
        self.env = None

    def _stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
            result = np.swapaxes(
                result, 0, 1
            )  # Policy expects (Batch_size, n_steps, ...)
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros(
                (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
            )
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
            result = result.transpose(0, 1)  # Policy expects (Batch_size, n_steps, ...)
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset(self):
        self.obs.clear()
        super().reset()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def _get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self._stack_last_n_obs(
                [obs[key] for obs in self.obs], self.yaml_cfg.n_obs_steps
            )

        return result

    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self._get_n_steps_obs()

        with torch.no_grad():
            action_chunk = (
                self.policy.predict_action(obs)["action"].detach().to(torch.float32)
            )
            action_chunk = action_chunk.transpose(0, 1)
        return action_chunk
