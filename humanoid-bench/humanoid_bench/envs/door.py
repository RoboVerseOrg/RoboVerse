import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Door(Task):
    qpos0_robot = {
        "h1": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0
            0
        """,
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0
            0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0
            0
        """,
        "g1": """
            0 0 0.75
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
            0
            0
        """,
    }
    dof = 2
    success_bar = 600

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2,),
            dtype=np.float64,
        )

    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        door_openness_reward = min(1, (self._env.data.qpos[-2] / 1) * abs(self._env.data.qpos[-2] / 1))
        door_hatch_openness_reward = rewards.tolerance(
            self._env.data.qpos[-1], bounds=(0.75, 2), margin=0.75, sigmoid="linear"
        )

        left_hand_hatch_closeness = np.linalg.norm(
            self._env.data.body("door_hatch").xpos - self._env.named.data.site_xpos["left_hand"]
        )
        right_hand_hatch_closeness = np.linalg.norm(
            self._env.data.body("door_hatch").xpos - self._env.named.data.site_xpos["right_hand"]
        )
        hand_hatch_proximity_reward = rewards.tolerance(
            min(right_hand_hatch_closeness, left_hand_hatch_closeness),
            bounds=(0, 0.25),
            margin=1,
            sigmoid="linear",
        )

        passage_reward = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "x"],
            bounds=(1.2, float("inf")),
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )

        reward = (
            0.1 * stand_reward * small_control
            + 0.45 * door_openness_reward
            + 0.05 * door_hatch_openness_reward
            + 0.05 * hand_hatch_proximity_reward
            + 0.35 * passage_reward
        )

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "door_openness_reward": door_openness_reward,
            "door_hatch_openness_reward": door_hatch_openness_reward,
            "hand_hatch_proximity_reward": hand_hatch_proximity_reward,
            "passage_reward": passage_reward,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.58, {}
