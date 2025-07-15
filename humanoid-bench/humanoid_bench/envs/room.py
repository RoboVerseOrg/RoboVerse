import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1


class Room(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0
            0 0 0 0.0733422 0.0519076 -0.240058 -0.966591
            0 0 0.1 0 0 0 0
            0 0 0.1 0 0 0 0
            0 0 0 0 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0
            0 0 0 0.0733422 0.0519076 -0.240058 -0.966591
            0 0 0.1 0 0 0 0
            0 0 0.1 0 0 0 0
            0 0 0 0 0 0 0
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
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0
            0 0 0 0.0733422 0.0519076 -0.240058 -0.966591
            0 0 0.1 0 0 0 0
            0 0 0.1 0 0 0 0
            0 0 0 0 0 0 0
        """,
    }
    dof = 7 * 6

    success_bar = 400

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
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 6,),
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

        room_object_positions = np.vstack([
            self._env.named.data.xpos[obj_name]
            for obj_name in [
                "chair",
                "trophy",
                "headphone",
                "package_a",
                "package_b",
                "snow_globe",
            ]
        ])
        room_object_entropies = np.array([np.var(room_object_positions[:, col_id]) for col_id in range(2)])

        room_object_organized = rewards.tolerance(
            np.max(room_object_entropies),
            margin=3,
        )

        reward = 0.2 * (small_control * stand_reward) + 0.8 * room_object_organized
        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "standing": standing,
            "upright": upright,
            "room_object_organized": room_object_organized,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.3, {}

    def reset_model(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        for i in range(-7, 0):
            position[i * 7] = np.random.uniform(-3.5 + (i + 7), -3.5 + (i + 8))
            position[i * 7 + 1] = np.random.uniform(1.2, 3.5) * np.random.choice([1, -1])
            if i == -4:
                position[i * 7 + 3 : i * 7 + 7] = np.array([0.0733422, 0.0519076, -0.240058, -0.966591])
        self._env.set_state(position, velocity)
        return super().reset_model()
