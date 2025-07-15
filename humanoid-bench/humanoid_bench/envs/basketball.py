import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Basketball(Task):
    qpos0_robot = {
        "h1": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0.35 0.9 2 1 0 0 0
            """,
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.35 0.9 2 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.35 0.9 2 1 0 0 0
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
            0.35 0.9 2 1 0 0 0
        """,
    }
    dof = 7
    max_episode_steps = 500
    camera_name = "cam_basketball"
    success_bar = 1200

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        self.stage = "catch"
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.robot.dof * 2 - 1) + self.dof * 2 - 1,),
            dtype=np.float64,
        )

    def get_reward(self):
        self.ball_collision_id = self._env.named.data.geom_xpos.axes.row.names.index("basketball_collision")

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

        basketball_pos = self._env.named.data.xpos["basketball"]
        left_hand_distance = self._env.named.data.site_xpos["left_hand"] - basketball_pos
        right_hand_distance = self._env.named.data.site_xpos["right_hand"] - basketball_pos
        reward_hand_proximity = rewards.tolerance(
            max([
                np.linalg.norm(left_hand_distance),
                np.linalg.norm(right_hand_distance),
            ]),
            bounds=(0, 0.2),
            margin=1,
        )
        reward_ball_success = 0
        ball_hoop_distance = np.linalg.norm(basketball_pos - self._env.named.data.site_xpos["hoop_center"])
        reward_ball_success = rewards.tolerance(
            ball_hoop_distance,
            margin=7,
            sigmoid="linear",
        )

        if self.stage == "catch":
            for pair in self._env.data.contact.geom:
                if self.ball_collision_id in pair:
                    self.stage = "throw"
                    break
        if self.stage == "throw":
            reward = 0.15 * (stand_reward * small_control) + 0.05 * reward_hand_proximity + 0.8 * reward_ball_success
        elif self.stage == "catch":
            reward = 0.5 * (stand_reward * small_control) + 0.5 * reward_hand_proximity

        if ball_hoop_distance < 0.05:
            reward += 1000

        return reward, {
            "reward_hand_proximity": reward_hand_proximity,
            "reward_ball_success": reward_ball_success,
            "stand_reward": stand_reward,
            "small_control": small_control,
            "success_subtasks": 1 if self.stage == "throw" else 0,
            "success": ball_hoop_distance < 0.05,
        }

    def get_terminated(self):
        if self._env.data.qpos.flat[-5] < 0.5:
            return True, {}
        if self._env.data.qpos[2] < 0.5:
            return True, {}
        if (
            np.linalg.norm(self._env.named.data.xpos["basketball"] - self._env.named.data.site_xpos["hoop_center"])
            < 0.05
        ):
            return True, {}
        return False, {}

    def reset_model(self):
        ball_init_angle = np.random.random() * 1.45 * np.random.choice([1, -1])
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        velocity[-6] = -7.5 * np.cos(ball_init_angle)
        velocity[-5] = -7.5 * np.sin(ball_init_angle)
        position[-7] = 1.5 * np.cos(ball_init_angle)
        position[-6] = 1.5 * np.sin(ball_init_angle)
        position[-5] = 0.3924 + self._env.named.data.site_xpos["left_hand", "z"]
        self._env.set_state(position, velocity)
        self.stage = "catch"
        return super().reset_model()
