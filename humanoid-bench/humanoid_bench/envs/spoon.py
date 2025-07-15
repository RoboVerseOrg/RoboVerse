import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Spoon(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.75 0.2 0.9 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.75 0.2 0.9 1 0 0 0
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
            0.75 0.2 0.9 1 0 0 0
        """,
    }
    dof = 7
    frame_skip = 10
    camera_name = "cam_tabletop"
    success_bar = 650

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        self.step_counter = 0
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 1 + 3,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()

        current_spin_angle = self.step_counter * (2 * np.pi / 40)
        spoon_target_pos = np.array([0.75, -0.1, 0.95]) + np.array([
            np.cos(current_spin_angle) * 0.06,
            np.sin(current_spin_angle) * 0.06,
            0,
        ])

        return np.concatenate((position, velocity, spoon_target_pos))

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

        left_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"] - self._env.named.data.geom_xpos["spoon_handle"]
        )
        right_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"] - self._env.named.data.geom_xpos["spoon_handle"]
        )
        hand_tool_proximity_reward = rewards.tolerance(
            min(left_hand_tool_distance, right_hand_tool_distance),
            bounds=(0, 0.2),
            margin=0.5,
        )

        current_spin_angle = self.step_counter * (2 * np.pi / 40)
        spoon_target_pos = np.array([0.75, -0.1, 0.95]) + np.array([
            np.cos(current_spin_angle) * 0.06,
            np.sin(current_spin_angle) * 0.06,
            0,
        ])
        self._env.named.data.site_xpos["goal"] = spoon_target_pos
        # spoon_velocity = self._env.named.data.sensordata["spoon_gyro"][2]
        spoon_plate_pos = self._env.named.data.geom_xpos["spoon_plate"]
        cup_pos = self._env.named.data.xpos["cup"]
        spoon_spinning_reward = rewards.tolerance(
            np.linalg.norm(spoon_plate_pos - spoon_target_pos),
            margin=0.15,
        )

        spoon_in_cup_x = abs(spoon_plate_pos[0] - cup_pos[0]) < 0.1
        spoon_in_cup_y = abs(spoon_plate_pos[1] - cup_pos[1]) < 0.1
        spoon_in_cup_z = abs(spoon_plate_pos[2] - (cup_pos[2] + 0.1)) < 0.1
        reward_spoon_in_cup = (int(spoon_in_cup_x) + int(spoon_in_cup_y) + int(spoon_in_cup_z)) // 3

        self.step_counter += 1

        reward = (
            0.15 * (stand_reward * small_control)
            + 0.25 * hand_tool_proximity_reward
            + 0.25 * reward_spoon_in_cup
            + 0.35 * spoon_spinning_reward
        )

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "hand_tool_proximity_reward": hand_tool_proximity_reward,
            "reward_spoon_in_cup": reward_spoon_in_cup,
            "spoon_spinning_reward": spoon_spinning_reward,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {}
        return False, {}
