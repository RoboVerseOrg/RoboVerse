import functools
import os

import embodied
import numpy as np

from . import gym


class MetaWorld(embodied.Env):
    def __init__(
        self,
        task,
        mode="train",
        repeat=1,
        render=True,
        size=(64, 64),
        camera=None,
        seed=None,
    ):
        assert mode in ("train", "eval")
        if camera in (None, -1):
            camera = "corner2"
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        )

        os.environ["MUJOCO_GL"] = "egl"
        task = task.replace("_", "-")
        task = f"{task}-v2-goal-observable"
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task](seed=seed)
        env._freeze_rand_vec = False
        self._env = gym.Gym(env)
        self._env = embodied.wrappers.TimeLimit(self._env, 500)
        self._mode = mode
        self._repeat = repeat
        self._size = size
        self._render = render
        self._camera = camera
        self._once = None

    @functools.cached_property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        if self._render:
            spaces["image"] = embodied.Space(np.uint8, self._size + (3,))
        spaces["log_success"] = embodied.Space(bool)
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        if action["reset"]:
            obs = self._env.step(action)
            obs["log_success"] = False
            self._once = True
            if self._camera == "corner2":
                # self._env._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
                self._env._env.model.cam_pos[2][:] = [0.4, 0.0, 0.4]
                orient = quat([0, 0, 0], 0)
                orient = mult(quat([0, 0, 1], 180), orient)  # roll
                orient = mult(quat([1, 0, 0], 70), orient)  # up
                orient = mult(quat([0, 0, 1], 25), orient)  # side
                self._env._env.model.cam_quat[2] = orient
        else:
            reward, success = 0.0, False
            for _ in range(self._repeat):
                obs = self._env.step(action)
                success = success or self._env.info["success"]
                reward += obs["reward"]
                if obs["is_last"] or obs["is_terminal"]:
                    break
            obs["reward"] = reward
            obs["log_success"] = success
        if self._render:
            obs["image"] = self._env._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera)
        if self._mode == "eval":
            if obs["log_success"] and self._once:
                obs["reward"] = 1.0
                self._once = False
            else:
                obs["reward"] = 0.0
        return obs


def quat(axis, angle):
    angle = angle / 360 * (2 * np.pi)
    axis = np.array(axis)
    return [np.cos(angle / 2), *np.sin(angle / 2) * axis]


def mult(quat1, quat2):
    w0, x0, y0, z0 = quat2
    w1, x1, y1, z1 = quat1
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ])
