import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Cube(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.45 0.21 1.125 1 0 0 0
            0.45 -0.21 1.125 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.45 0.21 1.125 1 0 0 0
            0.45 -0.21 1.125 1 0 0 0
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
            0.35 0.16 0.895 1 0 0 0
            0.35 -0.16 0.895 1 0 0 0
        """,
    }
    dof = 14
    max_episode_steps = 500
    camera_name = "cam_inhand"
    success_bar = 370

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
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 2 + 4,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        target_cube_orientation = self._env.data.body("target_cube").xquat

        return np.concatenate((
            position,
            velocity,
            target_cube_orientation,
        ))

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

        horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
        dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()

        left_cube_orientation = self._env.data.body("left_cube_to_rotate").xquat
        right_cube_orientation = self._env.data.body("right_cube_to_rotate").xquat
        target_cube_orientation = self._env.data.body("target_cube").xquat

        left_orientation_alignment_reward = rewards.tolerance(
            np.linalg.norm(left_cube_orientation - target_cube_orientation), margin=0.3
        )
        right_orientation_alignment_reward = rewards.tolerance(
            np.linalg.norm(right_cube_orientation - target_cube_orientation), margin=0.3
        )
        orientation_alignment_reward = (left_orientation_alignment_reward + right_orientation_alignment_reward) / 2

        left_hand_cube_distance = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"] - self._env.named.data.xpos["left_cube_to_rotate"]
        )
        right_hand_cube_distance = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"] - self._env.named.data.xpos["right_cube_to_rotate"]
        )
        left_hand_cube_proximity = rewards.tolerance(left_hand_cube_distance, bounds=(0, 0.1), margin=0.5)
        right_hand_cube_proximity = rewards.tolerance(right_hand_cube_distance, bounds=(0, 0.1), margin=0.5)

        cube_closeness_reward = (left_hand_cube_proximity + right_hand_cube_proximity) / 2

        reward = (
            0.2 * (small_control * stand_reward * dont_move)
            + 0.5 * orientation_alignment_reward
            + 0.3 * cube_closeness_reward
        )

        return reward, {
            "small_control": small_control,
            "stand_reward": stand_reward,
            "dont_move": dont_move,
            "standing": standing,
            "upright": upright,
            "orientation_alignment_reward": orientation_alignment_reward,
            "cube_closeness_reward": cube_closeness_reward,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {"terminated_reason": 0}
        if self._env.named.data.xpos["left_cube_to_rotate", "z"] < 0.6:
            return True, {"terminated_reason": 1}
        if self._env.named.data.xpos["right_cube_to_rotate", "z"] < 0.6:
            return True, {"terminated_reason": 1}
        return False, {}

    @staticmethod
    def euler_to_quat(angles):
        cr, cp, cy = np.cos(angles[0] / 2), np.cos(angles[1] / 2), np.cos(angles[2] / 2)
        sr, sp, sy = np.sin(angles[0] / 2), np.sin(angles[1] / 2), np.sin(angles[2] / 2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    def reset_model(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        position[-11:-7] = self.euler_to_quat(np.random.uniform(-3.14, 3.14, 3))
        position[-4:] = self.euler_to_quat(np.random.uniform(-3.14, 3.14, 3))
        self._env.model.body_quat[-1] = self.euler_to_quat(np.random.uniform(-3.14, 3.14, 3))
        self._env.set_state(position, velocity)
        return super().reset_model()
