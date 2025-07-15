import numpy as np


class Task:
    qpos0_robot = {}
    dof = 0
    frame_skip = 10
    camera_name = "cam_default"
    max_episode_steps = 1000
    kwargs = {}  # Default kwargs for a task

    def __init__(self, robot=None, env=None, **kwargs):
        self.robot = robot
        if env:
            self._env = env

        self.unwrapped = self

        if env is None:
            return

        if self._env.render_mode is not None:
            self._env.viewer = self._env.mujoco_renderer._get_viewer(self._env.render_mode)

    @property
    def observation_space(self):
        return None

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        state = np.concatenate((position, velocity))
        return state

    def get_reward(self):
        return 0, {}

    def get_terminated(self):
        return False, {}

    def reset_model(self):
        return self.get_obs()

    def normalize_action(self, action):
        return 2 * (action - self._env.action_low) / (self._env.action_high - self._env.action_low) - 1

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (self._env.action_high - self._env.action_low) + self._env.action_low

    def step(self, action):
        action = self.unnormalize_action(action)
        self._env.do_simulation(action, self._env.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.get_reward()
        terminated, terminated_info = self.get_terminated()

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info

    def render(self):
        return self._env.mujoco_renderer.render(self._env.render_mode, self._env.camera_id, self._env.camera_name)
