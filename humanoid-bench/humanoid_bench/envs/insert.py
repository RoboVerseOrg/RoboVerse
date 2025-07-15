import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Insert(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.8 -0.2 0.95 1 0 0 0
            0.8 0.2 0.95 1 0 0 0
            0.6 0 0.95 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.8 -0.2 0.95 1 0 0 0
            0.8 0.2 0.95 1 0 0 0
            0.6 0 0.95 1 0 0 0
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
            0.8 -0.2 0.95 1 0 0 0
            0.8 0.2 0.95 1 0 0 0
            0.6 0 0.95 1 0 0 0
        """,
    }
    dof = 21
    max_episode_steps = 500
    camera_name = "cam_inhand"
    success_bar = 350

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
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 3,),
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
            margin=1.9,
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

        cube_targets = [
            rewards.tolerance(
                np.linalg.norm(
                    self._env.named.data.site_xpos[f"block_peg_{ch}"] - self._env.named.data.site_xpos[f"peg_{ch}"]
                ),
                margin=0.5,
                sigmoid="linear",
            )
            for ch in ["a", "b"]
        ]
        cube_target_reward = np.mean(cube_targets)
        peg_heights = [
            rewards.tolerance(
                self._env.named.data.site_xpos[f"peg_{ch}", "z"] - 1.1,
                margin=0.15,
                sigmoid="linear",
            )
            for ch in ["a", "b"]
        ]
        peg_height_reward = np.mean(peg_heights)

        left_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"] - self._env.named.data.site_xpos["peg_a"]
        )
        right_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"] - self._env.named.data.site_xpos["peg_b"]
        )
        hand_tool_proximity_reward = rewards.tolerance(
            min(left_hand_tool_distance, right_hand_tool_distance),
            bounds=(0, 0.2),
            margin=0.5,
        )

        reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) * (
            0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward
        )

        return reward, {
            "small_control": small_control,
            "stand_reward": stand_reward,
            "cube_target_reward": cube_target_reward,
            "hand_tool_proximity_reward": hand_tool_proximity_reward,
            "peg_height_reward": peg_height_reward,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.5:
            return True, {"terminated_reason": 0}
        if self._env.named.data.site_xpos["block_peg_a", "z"] < 0.5:
            return True, {"terminated_reason": 1}
        if self._env.named.data.site_xpos["block_peg_b", "z"] < 0.5:
            return True, {"terminated_reason": 1}
        if self._env.named.data.site_xpos["peg_a", "z"] < 0.5:
            return True, {"terminated_reason": 1}
        if self._env.named.data.site_xpos["peg_b", "z"] < 0.5:
            return True, {"terminated_reason": 1}
        return False, {}
