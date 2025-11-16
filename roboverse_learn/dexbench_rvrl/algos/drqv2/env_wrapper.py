from ...envs.dex_env import DexEnv
from collections import deque
import torch
from gymnasium import spaces
import numpy as np

class DrqDexEnv:
    """Dex Environment Wrapper for Drqv2 tasks.
    Stacks num_frames frames for each observation."""

    def __init__(self, obs_shape: dict, num_frames: int, env: DexEnv, device: str):
        self._frames = {}
        self.device = device
        for key in obs_shape.keys():
            if "rgb" in key:
                self._frames[key] = torch.zeros((env.num_envs, num_frames, *obs_shape[key]), dtype=torch.float32, device=self.device)
        self._num_frames = num_frames
        self.env = env
        self.num_envs = env.num_envs
        self.obs_shape = obs_shape
        self.single_observation_space = spaces.Dict({})
        for k, v in env.single_observation_space.spaces.items():
            if "rgb" in k:
                shape = (num_frames * v.shape[0], *v.shape[1:])
                self.single_observation_space.spaces[k] = spaces.Box(low=-5.0, high=5.0, shape=shape, dtype=np.float32)
            else:
                self.single_observation_space.spaces[k] = v
            
    def reset(self):
        observations = self.env.reset()
        for key, value in observations.items():
            if "rgb" in key:
                for i in range(self._num_frames):
                    self._frames[key][:, i] = value.to(self.device).clone()
                observations[key] = self._frames[key].reshape(self.num_envs, self._num_frames * self.obs_shape[key][0], *self.obs_shape[key][1:])
        return observations

    def step(
        self, 
        actions: torch.Tensor,
        save_image_path=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_actions = actions
        observations, rewards, dones, dones, info = self.env.step(step_actions)
        for key, value in observations.items():
            observations[key] = value
            if "rgb" in key:
                if save_image_path is not None:
                    import cv2
                    import numpy as np
                    img0 = value[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
                    img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_image_path, img_bgr)
                if dones.any():
                    for i in range(self._num_frames):
                        self._frames[key][dones, i, ...] = value[dones].to(self.device).clone()
                    self._frames[key][~dones, :-1, ...] = self._frames[key][~dones, 1:, ...].clone()
                    self._frames[key][~dones, -1, ...] = value[~dones].to(self.device).clone()
                else:
                    self._frames[key][:, :-1] = self._frames[key][:, 1:].clone()
                    self._frames[key][:, -1] = value.to(self.device).clone()
                observations[key] = self._frames[key].reshape(self.num_envs, self._num_frames * self.obs_shape[key][0], *self.obs_shape[key][1:])
        return observations, rewards, dones, dones, info