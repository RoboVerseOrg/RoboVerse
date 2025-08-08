import json
import os
from collections import deque

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from scenario_cfg.policy import VLAPolicyCfg

from .base_runner import PolicyRunner

# import jax
# from octo.model.octo_model import OctoModel


class OpenVLARunner(PolicyRunner):
    def _init_policy(self, **kwargs):
        self.model_path = kwargs.get("checkpoint_path")
        self.task = kwargs.get("task_name")
        self.subset = kwargs.get("subset")
        # Initialize VLA policy configuration
        self.policy_cfg = VLAPolicyCfg()
        self.policy_cfg.obs_config.obs_type = "no_proprio"
        # Load dataset info
        self.dataset_info_path = os.path.join(self.model_path, "dataset_statistics.json")
        self.DATA_STAT = json.load(open(self.dataset_info_path))
        # Initialize OpenVLA model
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            # attn_implementation="flash_attention_2", # Uncomment if your GPU supports flash_attn
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.norm_stats[self.task.lower()] = self.DATA_STAT[self.subset]

        # Initialize observation deque
        self.obs = deque(maxlen=2)

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def predict_action(self, observation=None):
        if observation is not None:
            self.update_obs(observation)

        # Get the latest observation
        latest_obs = self.obs[-1]

        # Convert numpy array to PIL Image for the model
        image = Image.fromarray(np.array(latest_obs["head_cam"][0].cpu()))

        # Create prompt for the model
        try:
            instruction = self.scenario.task.task_language
        except AttributeError:
            instruction = self.task_name
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process the input for the model
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        # Get action from the model
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.task.lower(), do_sample=False)

        action = torch.tensor(action, dtype=torch.float32).to(self.device)

        if self.num_envs == 1:
            action = action.unsqueeze(0)

        # Add dimension for action chunk of 1
        action_chunk = action.unsqueeze(1)

        return action_chunk


class OctoVLARunner(PolicyRunner):
    def _init_policy(self, **kwargs):
        self.task_name = kwargs.get("task_name")
        self.model_path = kwargs.get("checkpoint_path")


#         # Initialize VLA policy configuration
#         self.policy_cfg = VLAPolicyCfg()

#         # Load dataset statistics
#         dataset_info_path = os.path.join(self.model_path, "dataset_statistics.json")
#         self.data_stat = json.load(open(dataset_info_path, "r"))

#         # Format data stats for Octo
#         self.octo_data_stat = {
#             "action": {
#                 "min": np.array(self.data_stat['action']["min"]),
#                 "max": np.array(self.data_stat['action']["max"]),
#                 "mean": np.array(self.data_stat['action']["mean"]),
#                 "std": np.array(self.data_stat['action']["std"]),
#                 "p01": np.array(self.data_stat['action']["p01"]),
#                 "p99": np.array(self.data_stat['action']["p99"]),
#             }
#         }

#         # Import JAX for Octo model


#         # Initialize Octo model
#         self.model = OctoModel.load_pretrained(self.model_path)
#         self.vla_task = self.model.create_tasks(texts=[f"{self.task_name}"])
#         self.rng_key = jax.random.PRNGKey(0)

#         # Initialize observation deque
#         self.obs = deque(maxlen=2)


#     def update_obs(self, current_obs):
#         self.obs.append(current_obs)

#     def predict_action(self, observation=None):
#         if observation is not None:
#             self.update_obs(observation)

#         # Get the latest observation
#         latest_obs = self.obs[-1]

#         # Format observation for Octo model
#         img = np.array(latest_obs['rgb'])
#         img = img[np.newaxis, ...]
#         proprio = np.array(latest_obs['joint_qpos'])
#         proprio = proprio[np.newaxis, ...]
#         octo_observation = {
#             "image_primary": img,
#             "timestep_pad_mask": np.array([[True]]),
#             "proprio": proprio
#         }

#         # Import JAX here to update RNG key
#         import jax

#         # Get action from the model
#         action = self.model.sample_actions(
#             octo_observation,
#             self.vla_task,
#             unnormalization_statistics=self.octo_data_stat['action'],
#             rng=self.rng_key
#         )

#         # Update RNG key for next prediction
#         self.rng_key, _ = jax.random.split(self.rng_key)

#         # Convert to torch tensor and format
#         action = torch.tensor(np.array(action[:,0,:]), dtype=torch.float32, device=self.device)

#         # Format action for the expected return type (list of tensors for each step)
#         action_chunk = [action for _ in range(self.policy_cfg.action_config.action_chunk_steps)]

#         return action_chunk
