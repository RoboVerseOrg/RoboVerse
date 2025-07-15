import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65

_MOVE_SPEED = 2.0


class MazeBase(Task):
    camera_name = "cam_maze"
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
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
        """,
    }
    htarget_low = np.array([-1, -1, 0.8])
    htarget_high = np.array([7, 7, 1.2])

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64)

    def update_move_direction(self):
        self.move_direction = np.array([1, 0, 0])
        self.maze_stage = 0

    def get_reward(self):
        self.begining_wall_id = self._env.named.data.geom_xpos.axes.row.names.index("block_collision_00")

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

        wall_collision_discount = 1

        for pair in self._env.data.contact.geom:
            if pair[0] >= self.begining_wall_id or pair[1] >= self.begining_wall_id:
                wall_collision_discount = 0.1
                # print(pair)
                break

        stage_convert_reward = self.update_move_direction()

        move = rewards.tolerance(
            self.robot.center_of_mass_velocity()[0] - self.move_direction[0] * _MOVE_SPEED,
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        ) * rewards.tolerance(
            self.robot.center_of_mass_velocity()[1] - self.move_direction[1] * _MOVE_SPEED,
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )

        if self.maze_stage == len(self.checkpoints) - 1:
            move = 1

        move = (5 * move + 1) / 6

        checkpoint_proximity = np.linalg.norm(
            self.checkpoints[self.maze_stage][:2] - self._env.named.data.site_xpos["imu"][:2]
        )

        checkpoint_proximity_reward = rewards.tolerance(checkpoint_proximity, margin=1)

        reward = (
            0.2 * (stand_reward * small_control) + 0.4 * move + 0.4 * checkpoint_proximity_reward
        ) * wall_collision_discount + stage_convert_reward

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "wall_collision_discount": wall_collision_discount,
            "stage_convert_reward": stage_convert_reward,
            "checkpoint_proximity_reward": checkpoint_proximity_reward,
            "success_subtasks": self.maze_stage,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}


class Maze(MazeBase):
    checkpoints = [
        np.array([0, 0, 1]),
        np.array([3, 0, 1]),
        np.array([3, 6, 1]),
        np.array([6, 6, 1]),
        np.array([6, 6, 1]),
    ]

    success_bar = 1200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_direction = np.array([1, 0, 0])
        self.maze_stage = 0

    def update_move_direction(self):
        prev_maze_stage = self.maze_stage

        if self.maze_stage == 0:
            self._env.named.model.geom_rgba["intersection_a"] = np.array([0, 1, 0, 1])

        dist_cp = np.linalg.norm(self._env.named.data.xpos["pelvis"][:2] - self.checkpoints[self.maze_stage][:2])
        if dist_cp < 0.4:
            if self.maze_stage == 0:
                self.move_direction = np.array([1, 0, 0])
                self._env.named.model.geom_rgba["intersection_a"] = np.array([0, 1, 0, 1])
                self._env.named.model.geom_rgba["intersection_b"] = np.array([1, 0, 0, 1])
            elif self.maze_stage == 1:
                self.move_direction = np.array([0, 1, 0])
                self._env.named.model.geom_rgba["intersection_b"] = np.array([0, 1, 0, 1])
                self._env.named.model.geom_rgba["intersection_c"] = np.array([1, 0, 0, 1])
            elif self.maze_stage == 2:
                self.move_direction = np.array([1, 0, 0])
                self._env.named.model.geom_rgba["intersection_c"] = np.array([0, 1, 0, 1])
                self._env.named.model.geom_rgba["intersection_d"] = np.array([1, 0, 0, 1])
            elif self.maze_stage == 3:
                self.move_direction = np.array([0, 1, 0])
                self._env.named.model.geom_rgba["intersection_d"] = np.array([0, 1, 0, 1])

            self.maze_stage += 1
            self.maze_stage = min(self.maze_stage, 4)

        if prev_maze_stage < self.maze_stage:
            return self.maze_stage * 100

        return 0

    def reset_model(self):
        self.move_direction = np.array([1, 0, 0])
        self.maze_stage = 0
        return super().reset_model()
