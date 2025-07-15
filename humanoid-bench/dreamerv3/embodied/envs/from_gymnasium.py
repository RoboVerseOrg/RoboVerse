import functools
import os
import sys

import embodied
import gymnasium as gym
import numpy as np


class FromGymnasium(embodied.Env):
    def __init__(self, env, obs_key="image", act_key="action", is_eval=False, **kwargs):
        if isinstance(env, str):
            if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
                os.environ["MUJOCO_GL"] = "egl"
            if "SLURM_STEP_GPUS" in os.environ:
                os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
                print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
            if "SLURM_JOB_GPUS" in os.environ:
                os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
                print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

            self._env = gym.make(env, **kwargs)
        else:
            assert not kwargs, kwargs
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
        self._is_eval = is_eval

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        if self._is_eval:
            spaces["image"] = embodied.Space(np.uint8, (256, 256, 3))
        return {
            **spaces,
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "success": embodied.Space(np.float32),
            "success_subtasks": embodied.Space(np.float32),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs, _ = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, terminated, truncated, self._info = self._env.step(action)
        self._done = terminated or truncated
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get("is_terminal", terminated)),
            success=self._info.get("success", 0.0),
            success_subtasks=self._info.get("success_subtasks", 0.0),
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False, success=0.0, success_subtasks=0.0):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            success=np.float32(success),
            success_subtasks=np.float32(success_subtasks),
        )
        if self._is_eval:
            obs["image"] = self.render()
        return obs

    def render(self):
        # image = self._env.render("rgb_array")
        image = self._env.render()
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)
