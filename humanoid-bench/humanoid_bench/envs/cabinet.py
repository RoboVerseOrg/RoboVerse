import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class Cabinet(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0
            0.9 0 0.58 1 0 0 0
            0.9 0 0.87 1 0 0 0
            0.9 0 1.16 1 0 0 0
            0.9 0 1.45 1 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0
            0.9 0 0.58 1 0 0 0
            0.9 0 0.87 1 0 0 0
            0.9 0 1.16 1 0 0 0
            0.9 0 1.45 1 0 0 0
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
            0 0 0 0 0
            0.9 0 0.58 1 0 0 0
            0.9 0 0.87 1 0 0 0
            0.9 0 1.16 1 0 0 0
            0.9 0 1.45 1 0 0 0
        """,
    }
    dof = 5 * 1 + 4 * 7
    camera_name = "cam_hand_visible"
    success_bar = 2500

    def __init__(self, robot=None, env=None, **kwargs):
        self.current_subtask = 1
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 1 - 3,),
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
        stabilization_reward = stand_reward * small_control

        subtask_get_reward = [
            self.get_reward_subtask_one,
            self.get_reward_subtask_two,
            self.get_reward_subtask_three,
            self.get_reward_subtask_four,
            lambda: (1000, {}, False),
        ]

        reward, reward_info, subtask_complete = subtask_get_reward[self.current_subtask - 1]()
        if self.current_subtask < 5:
            reward = 0.2 * stabilization_reward + 0.8 * reward

        if subtask_complete:
            print("Completed subtask", self.current_subtask)
            reward += 100 * (self.current_subtask)
            self.current_subtask += 1

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "success_subtasks": self.current_subtask - 1,  # The first subtask is not automatically completed.
            "success": self.current_subtask == 5,
            **reward_info,
        }

    def get_reward_subtask_one(self):
        pulling_cabinet_joint_pos = self._env.data.qpos[-(4 * 7) - 2]
        door_openness_reward = abs(pulling_cabinet_joint_pos / 0.4)
        subtask_complete = door_openness_reward > 0.95
        return (
            door_openness_reward,
            {
                "door_openness_reward": door_openness_reward,
                "subtask_complete": subtask_complete,
            },
            subtask_complete,
        )

    def get_reward_subtask_two(self):
        drawer_joint_pos = self._env.data.qpos[-(4 * 7) - 5]
        door_openness_reward = abs(drawer_joint_pos / 0.45)
        subtask_complete = door_openness_reward > 0.95
        return (
            door_openness_reward,
            {
                "door_openness_reward": door_openness_reward,
                "subtask_complete": subtask_complete,
            },
            subtask_complete,
        )

    def get_reward_subtask_three(self):
        drawer_cube_pos = self._env.named.data.xpos["drawer_cube"]
        normal_cabinet_left_joint_pos = self._env.data.qpos[-(4 * 7) - 4]
        normal_cabinet_right_joint_pos = self._env.data.qpos[-(4 * 7) - 3]
        left_door_openness_reward = min(1, abs(normal_cabinet_left_joint_pos))
        right_door_openness_reward = min(1, abs(normal_cabinet_right_joint_pos))
        door_openness_reward = max(left_door_openness_reward, right_door_openness_reward)  # Any open door is sufficient

        cube_proximity_horizontal = (
            rewards.tolerance(
                drawer_cube_pos[0] - 0.9,
                bounds=(-0.3, 0.3),
                margin=0.3,
                sigmoid="linear",
            )
            + rewards.tolerance(drawer_cube_pos[1], bounds=(-0.6, 0.6), margin=0.3, sigmoid="linear")
        ) / 2
        cube_proximity_vertical = rewards.tolerance(
            drawer_cube_pos[2] - 0.94,
            bounds=(-0.15, 0.15),
            margin=0.3,
            sigmoid="linear",
        )

        in_cabinet_x = 0.9 - 0.3 <= drawer_cube_pos[0] <= 0.9 + 0.3
        in_cabinet_y = 0 - 0.6 <= drawer_cube_pos[1] <= 0 + 0.6
        in_cabinet_z = 0.94 - 0.15 <= drawer_cube_pos[2] <= 0.94 + 0.15
        task_completed = in_cabinet_x and in_cabinet_y and in_cabinet_z

        drawer_cube_proximity_reward = 0.3 * cube_proximity_horizontal + 0.7 * cube_proximity_vertical
        reward = 0.5 * (drawer_cube_proximity_reward) + 0.5 * door_openness_reward
        return (
            reward,
            {
                "door_openness_reward": door_openness_reward,
                "drawer_cube_proximity_reward": drawer_cube_proximity_reward,
                "subtask_complete": task_completed,
            },
            task_completed,
        )

    def get_reward_subtask_four(self):
        pullup_drawer_cube_pos = self._env.named.data.xpos["lateral_cabinet_cube"]
        pullup_drawer_joint_pos = self._env.data.qpos[-(4 * 7) - 1]
        door_openness_reward = min(1, abs(pullup_drawer_joint_pos))

        normal_cabinet_left_joint_pos = self._env.data.qpos[-(4 * 7) - 4]
        normal_cabinet_right_joint_pos = self._env.data.qpos[-(4 * 7) - 3]
        left_door_openness_reward = min(1, abs(normal_cabinet_left_joint_pos))
        right_door_openness_reward = min(1, abs(normal_cabinet_right_joint_pos))
        secondary_door_openness_reward = max(left_door_openness_reward, right_door_openness_reward)

        cube_proximity_horizontal = rewards.tolerance(
            pullup_drawer_cube_pos[0] - 0.9,
            bounds=(-0.3, 0.3),
            margin=0.3,
            sigmoid="linear",
        ) + rewards.tolerance(pullup_drawer_cube_pos[1], bounds=(-0.6, 0.6), margin=0.3, sigmoid="linear")
        cube_proximity_vertical = rewards.tolerance(
            pullup_drawer_cube_pos[2] - 1.54,
            bounds=(-0.15, 0.15),
            margin=0.3,
            sigmoid="linear",
        )

        in_cabinet_x = 0.9 - 0.3 <= pullup_drawer_cube_pos[0] <= 0.9 + 0.3
        in_cabinet_y = 0 - 0.6 <= pullup_drawer_cube_pos[1] <= 0 + 0.6
        in_cabinet_z = 1.54 - 0.15 <= pullup_drawer_cube_pos[2] <= 1.54 + 0.15
        task_completed = in_cabinet_x and in_cabinet_y and in_cabinet_z

        drawer_cube_proximity_reward = 0.3 * cube_proximity_horizontal / 2 + 0.7 * cube_proximity_vertical
        reward = 0.5 * (drawer_cube_proximity_reward) + 0.5 * door_openness_reward

        return (
            reward,
            {
                "door_openness_reward": door_openness_reward,
                "secondary_door_openness_reward": secondary_door_openness_reward,
                "drawer_cube_proximity_reward": drawer_cube_proximity_reward,
                "subtask_complete": task_completed,
            },
            task_completed,
        )

    def get_terminated(self):
        if self.current_subtask == 5:
            return True, {}
        return False, {}

    def reset_model(self):
        self.current_subtask = 1
        return super().reset_model()
