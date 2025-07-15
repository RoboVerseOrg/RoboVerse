import os
import sys

import gymnasium as gym
import numpy as np


class HumanoidWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()


def make_env(cfg):
    """Make Humanoid environment."""
    if not cfg.task.startswith("humanoid_"):
        raise ValueError("Unknown task:", cfg.task)

    policy_path = cfg.get("policy_path", None)
    mean_path = cfg.get("mean_path", None)
    var_path = cfg.get("var_path", None)
    policy_type = cfg.get("policy_type", None)
    small_obs = cfg.get("small_obs", None)
    if small_obs is not None:
        small_obs = str(small_obs)

    print("small obs start:", small_obs)

    env = gym.make(
        cfg.task.removeprefix("humanoid_"),
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
    )
    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    return env
