import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 5


class Walk(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0",
    }
    _move_speed = _WALK_SPEED
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = 700

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT, _CRAWL_HEIGHT
            _STAND_HEIGHT = 1.28
            _CRAWL_HEIGHT = 0.6

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64)

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
        if self._move_speed == 0:
            horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move, {
                "small_control": small_control,
                "stand_reward": stand_reward,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = self.robot.center_of_mass_velocity()[0]
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            reward = small_control * stand_reward * move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}


class Stand(Walk):
    _move_speed = 0
    success_bar = 800


class Run(Walk):
    _move_speed = _RUN_SPEED


class Crawl(Walk):
    def get_reward(self):
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(1, float("inf")),
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        crawling_head = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        crawling = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "z"],
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        reward_xquat = rewards.tolerance(
            np.linalg.norm(self._env.data.body("pelvis").xquat - np.array([0.75, 0, 0.65, 0])),
            margin=1,
        )

        in_tunnel = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "y"],
            bounds=(-1, 1),
            margin=0,
        )

        reward = (
            0.1 * small_control + 0.25 * min(crawling, crawling_head) + 0.4 * move + 0.25 * reward_xquat
        ) * in_tunnel
        return reward, {
            "crawling": crawling,
            "crawling_head": crawling_head,
            "small_control": small_control,
            "move": move,
            "in_tunnel": in_tunnel,
        }

    def get_terminated(self):
        return False, {}


class ClimbingUpwards(Walk):
    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height() - self.robot.left_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        ) * rewards.tolerance(
            self.robot.head_height() - self.robot.right_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.5, float("inf")),
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

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return stand_reward * small_control * move, {  # small_control *
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }

    def get_terminated(self):
        return self.robot.torso_upright() < 0.1, {}


class Stair(ClimbingUpwards):
    pass


class Slide(ClimbingUpwards):
    pass


class Hurdle(Walk):
    _move_speed = _RUN_SPEED
    camera_name = "cam_hurdle"

    def get_reward(self):
        self.wall_collision_ids = [
            self._env.named.data.geom_xpos.axes.row.names.index(wall_name)
            for wall_name in [
                "left_barrier_collision",
                "right_barrier_collision",
                "behind_barrier_collision",
            ]
        ]

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
        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(self._move_speed, float("inf")),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        wall_collision_discount = 1

        for pair in self._env.data.contact.geom:
            if any([
                wall_collision_id in pair for wall_collision_id in self.wall_collision_ids
            ]):  # for no hand. if for hand, > 155
                wall_collision_discount = 0.1
                # print(pair)
                break

        reward = small_control * stand_reward * move * wall_collision_discount

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
            "wall_collision_discount": wall_collision_discount,
        }


class Sit(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0",
    }
    dof = 0
    vels = 0
    success_bar = 750

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof + self.vels,),
            dtype=np.float64,
        )

    def get_reward(self):
        sitting = rewards.tolerance(self._env.data.qpos[2], bounds=(0.68, 0.72), margin=0.2)
        chair_location = self._env.named.data.xpos["chair"]
        on_chair = rewards.tolerance(
            self._env.data.qpos[0] - chair_location[0], bounds=(-0.19, 0.19), margin=0.2
        ) * rewards.tolerance(self._env.data.qpos[1] - chair_location[1], margin=0.1)
        sitting_posture = rewards.tolerance(
            self.robot.head_height() - self._env.named.data.site_xpos["imu", "z"],
            bounds=(0.35, 0.45),
            margin=0.3,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.95, float("inf")),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
        dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
        return small_control * sit_reward * dont_move, {
            "small_control": small_control,
            "sit_reward": sit_reward,
            "dont_move": dont_move,
            "sitting": sitting,
            "upright": upright,
            "sitting_posture": sitting_posture,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.5, {}

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


class SitHard(Sit):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
    }

    dof = 7
    vels = 6

    def reset_model(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        position[0] = np.random.uniform(0.2, 0.4)
        position[1] = np.random.uniform(-0.15, 0.15)
        rotation_angle = np.random.uniform(-1.8, 1.8)
        position[3:7] = self.euler_to_quat(np.array([0, 0, rotation_angle]))
        self._env.set_state(position, velocity)
        return super().reset_model()
