import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Package(Task):
    qpos0_robot = {
        "h1": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0.75 0 0.35 1 0 0 0
            """,
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.75 0 0.35 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.75 0 0.35 1 0 0 0
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
            0.75 0 0.35 1 0 0 0
        """,
    }
    dof = 7
    frame_skip = 10
    htarget_low = np.array([-2.5, -2.5, 0.3])
    htarget_high = np.array([2.5, 2.5, 1.8])
    success_bar = 1500

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

        if env is None:
            return

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 1 + 9,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        package_destination = self._env.named.data.site_xpos["destination_loc"]
        box_pos = self._env.data.qpos.flat.copy()[-7:]
        box_vel = self._env.data.qvel.flat.copy()[-6:]
        left_hand = self.robot.left_hand_position()
        right_hand = self.robot.right_hand_position()

        return np.concatenate([
            position,
            velocity,
            package_destination,
            box_pos,
            box_vel,
            left_hand,
            right_hand,
        ])

    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.8, float("inf")),
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

        package_destination = self._env.named.data.site_xpos["destination_loc"]
        package_location = self._env.named.data.qpos["free_package"][:3]

        dist_package_destination = np.linalg.norm(package_location - package_destination)
        dist_hand_package_right = np.linalg.norm(self._env.named.data.site_xpos["right_hand"] - package_location)
        dist_hand_package_left = np.linalg.norm(self._env.named.data.site_xpos["left_hand"] - package_location)
        package_height = np.min((package_location[2], 1))

        reward_success = dist_package_destination < 0.1

        reward = (
            stand_reward * small_control
            - 3 * dist_package_destination * 1
            - (dist_hand_package_left + dist_hand_package_right) * 0.1
            + package_height
            + reward_success * 1000
        )

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "dist_package_destination": dist_package_destination,
            "dist_hand_package_right": dist_hand_package_right,
            "dist_hand_package_left": dist_hand_package_left,
            "package_height": package_height,
            "success": reward_success > 0,
        }

    def get_terminated(self):
        dist_package_destination = np.linalg.norm(
            self._env.named.data.qpos["free_package"][:3] - self._env.named.data.site_xpos["destination_loc"]
        )

        return dist_package_destination < 0.1, {}

    def reset_model(self):
        q_pos = self._env.data.qpos.copy()
        q_pos[-7] = np.random.uniform(-2, 2)
        q_pos[-6] = np.random.uniform(-2, 2)
        q_pos[-5] = 0.35

        q_vel = self._env.data.qvel.copy()
        self._env.set_state(q_pos, q_vel)
        self._env.model.body_pos[-1] = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0])
        return super().reset_model()
