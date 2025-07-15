import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task


class Reach(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0",
    }
    htarget_low = np.array([-2.5, -2.5, 0.2])
    htarget_high = np.array([25, 2.5, 2.0])

    success_bar = 12000

    def __init__(
        self,
        robot=None,
        env=None,
        **kwargs,
    ):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        self.target_low = np.array([-2, -2, 0.2])
        self.target_high = np.array([2, 2, 2.0])
        self.goal = np.zeros(3)

        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.robot.dof * 2 - 1) + 6,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        left_hand = self.robot.left_hand_position()
        target = self.goal.copy()

        return np.concatenate((position, velocity, left_hand, target))

    def get_reward(self):
        hand_dist = np.sqrt(np.square(self.robot.left_hand_position() - self.goal).sum())

        healthy_reward = self._env.data.xmat[1, -1] * 5.0
        motion_penalty = np.square(self._env.data.qvel[: self.robot.dof - 1]).sum()
        reward_close = 5 if hand_dist < 1 else 0
        reward_success = 10 if hand_dist < 0.05 else 0
        reward = healthy_reward - 0.0001 * motion_penalty + reward_close + reward_success

        info = {
            "hand_dist": hand_dist,
            "healthy_reward": healthy_reward,
            "motion_penalty": motion_penalty,
            "reward_close": reward_close,
            "reward_success": reward_success,
        }
        return reward, info

    def reset_model(self):
        self.goal = np.random.uniform(self.target_low, self.target_high, size=(3,))
        return self.get_obs()

    def render(self):
        found = False
        for i in range(len(self._env.viewer._markers)):
            if self._env.viewer._markers[i]["objid"] == 789:
                self._env.viewer._markers[i]["pos"] = self.goal
                found = True
                break

        if not found:
            self._env.viewer.add_marker(
                pos=self.goal,
                size=0.05,
                objid=789,
                rgba=(0.8, 0.28, 0.28, 1.0),
                label="",
            )

        return self._env.mujoco_renderer.render(self._env.render_mode, self._env.camera_id, self._env.camera_name)


if __name__ == "__main__":
    env = gym.make("Reaching-v0", render_mode="human", policy_path=None)
    env.reset()
    env.render()
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            env.reset()
    env.close()
