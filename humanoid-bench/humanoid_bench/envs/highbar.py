import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task


class HighBarBase(Task):
    high_bar_mode = None
    success_bar = 750

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64)

    def get_reward(self):
        upright_reward = rewards.tolerance(
            -self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )

        feet_reward = rewards.tolerance(
            (self.robot.left_foot_height() + self.robot.right_foot_height()) / 2,
            bounds=(4.8, float("inf")),
            sigmoid="linear",
            margin=2.0,
            value_at_margin=0,
        )
        feet_reward = (1 + feet_reward) / 2

        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        reward = upright_reward * feet_reward * small_control

        return reward, {
            "upright_reward": upright_reward,
            "feet_reward": feet_reward,
            "small_control": small_control,
        }

    def get_terminated(self):
        terminated = False
        if self.robot.head_height() < 2.0:
            terminated = True
        return terminated, {}

    def reset_model(self):
        self._env.randomness = 0
        return super().reset_model()


class HighBarSimple(HighBarBase):
    high_bar_mode = "simple"
    qpos0_robot = {
        "h1": [
            '<key name="qpos0" qpos="0.27 0 1.74 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3.11 0 1.49 0 -3.11 0 1.49" ctrl="0 0 0 0 0 0 0 0 0 0 0 0 3.11 0 1.49 0 -3.11 0 1.49"/>',
        ]
    }


class HighBarHard(HighBarBase):
    high_bar_mode = "hard"
    qpos0_robot = {
        "h1strong": [
            """<key qpos='0.27 0. 1.62 1 0 0 0
            0 0 0 0 0
            0 0 0 0 0
            0
            0 3.11 0 1.6 0
            0.11 0.33
            0 1.3 0.7 1.7
            0 1.3 0.7 1.7
            0 1.3 0.7 1.7
            0.1 0 1.2 0.77 1.62
            0.31 1.35 0.27 0.27 1.45
            0 -3.11 0 1.6 0
            0.11 0.33
            0 1.3 0.7 1.7
            0 1.3 0.7 1.7
            0 1.3 0.7 1.7
            0.1 0 1.2 0.77 1.62
            0.31 1.35 0.27 0.27 1.45'
            ctrl='0 0 0 0 0
            0 0 0 0 0
            0
            0 3.11 0 1.6 0
            0 -3.11 0 1.6 0
            0.11 0.33
            1.05 1.22 0.209 0.698 1.57
            0 1.57 2.5
            0 1.57 2.5
            0 1.57 2.5
            0.785 0 1.57 2.5
            0.11 0.33
            1.05 1.22 0.209 0.698 1.57
            0 1.57 2.5
            0 1.57 2.5
            0 1.57 2.5
            0.785 0 1.57 2.5'/>""",
        ],
    }
