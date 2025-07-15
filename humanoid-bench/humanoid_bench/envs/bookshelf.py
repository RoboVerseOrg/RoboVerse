import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class BookshelfBase(Task):
    fixed = None
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.95 -0.36 1.25 0 0 0 0
            0.95 -0.3 1.25 0 0 0 0
            0.95 -0.24 1.25 0 0 0 0
            0.95 -0.18 1.25 0 0 0 0
            0.95 -0.12 1.25 0 0 0 0
            0.95 0.02 1.25 0 0 0 0
            0.95 0.12 1.255 0.7073883 -0.7068252 0 0
            0.95 -0.3 0.65 0 0 0 0
            0.95 -0.13 0.65 0 0 0 0
            0.95 0 0.65 0 0 0 0
            0.95 0.2 0.65 0 0 0 0
            0.85 0 1.58 0.8689313 -0.06692531 0.12250874 0.47483787
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.95 -0.36 1.25 0 0 0 0
            0.95 -0.3 1.25 0 0 0 0
            0.95 -0.24 1.25 0 0 0 0
            0.95 -0.18 1.25 0 0 0 0
            0.95 -0.12 1.25 0 0 0 0
            0.95 0.02 1.25 0 0 0 0
            0.95 0.12 1.255 0.7073883 -0.7068252 0 0
            0.95 -0.3 0.65 0 0 0 0
            0.95 -0.13 0.65 0 0 0 0
            0.95 0 0.65 0 0 0 0
            0.95 0.2 0.65 0 0 0 0
            0.85 0 1.58 0.8689313 -0.06692531 0.12250874 0.47483787
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
            0.95 -0.36 1.25 0 0 0 0
            0.95 -0.3 1.25 0 0 0 0
            0.95 -0.24 1.25 0 0 0 0
            0.95 -0.18 1.25 0 0 0 0
            0.95 -0.12 1.25 0 0 0 0
            0.95 0.02 1.25 0 0 0 0
            0.95 0.12 1.255 0.7073883 -0.7068252 0 0
            0.95 -0.3 0.65 0 0 0 0
            0.95 -0.13 0.65 0 0 0 0
            0.95 0 0.65 0 0 0 0
            0.95 0.2 0.65 0 0 0 0
            0.85 0 1.58 0.8689313 -0.06692531 0.12250874 0.47483787
        """,
    }
    dof = 12 * 7
    camera_name = "cam_hand_visible"
    object_names = {
        "book_a": ["book_a_vision_a"],
        "book_b": ["book_b_vision_a"],
        "book_c": ["book_c_vision_a"],
        "book_d": ["book_d_vision_a"],
        "book_e": ["book_e_vision_a"],
        "trophy": ["trophy_vision_a", "trophy_vision_b", "trophy_vision_c"],
        "book_flat": ["book_flat_vision_a"],
        "soda_can": ["soda_can_vision_a"],
        "book_thick": ["book_thick_vision_a"],
        "chalk": ["chalk_vision_a"],
        "snow_globe": ["snow_globe_vision_a", "snow_globe_vision_b"],
        "headphone": [
            "headphone_vision_a",
            "headphone_vision_b",
            "headphone_vision_c",
            "headphone_vision_d",
            "headphone_vision_e",
        ],
    }
    carried_order_rgb = np.array([[1, 0.8, 0.8], [1, 0.5, 0.5], [0.9, 0.2, 0.2], [0.5, 0, 0], [0.2, 0, 0]])
    bookshelf_objects = [-19, -22, -15, -20, -23, -23]  # placeholder last element
    success_bar = 2000
    htarget_low = np.array([0.5, -0.5, 0.0])
    htarget_high = np.array([1.5, 0.5, 2.0])

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        self.task_index = 0
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 12 + 1,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()

        return np.concatenate((position, velocity, np.array([self.bookshelf_objects[self.task_index]])))

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

        curr_reach_obj_pos = self._env.named.data.xpos[self.bookshelf_objects[self.task_index]]
        curr_placement_goal = self.placement_goals[self.task_index]
        obj_goal_dist = np.linalg.norm(curr_reach_obj_pos - curr_placement_goal)

        reward_proximity = rewards.tolerance(obj_goal_dist, bounds=(0, 0.15), margin=1, sigmoid="linear")
        left_hand_distance = self._env.named.data.site_xpos["left_hand"] - curr_reach_obj_pos
        righ_hand_distance = self._env.named.data.site_xpos["right_hand"] - curr_reach_obj_pos
        reward_hand_proximity = np.exp(-min([np.linalg.norm(left_hand_distance), np.linalg.norm(righ_hand_distance)]))

        reward = 0.2 * (stand_reward * small_control) + 0.4 * reward_proximity + 0.4 * reward_hand_proximity
        if obj_goal_dist < 0.15:
            self.task_index += 1
            reward += 100 * self.task_index

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "reward_proximity": reward_proximity,
            "reward_hand_proximity": reward_hand_proximity,
            "obj_goal_dist": obj_goal_dist,
            "success_subtasks": self.task_index,
            "success": self.task_index == 5,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {"terminated_reason": 0}
        if self.task_index == 5:
            return True, {"terminated_reason": 1}
        if self._env.named.data.xpos[self.bookshelf_objects[self.task_index], "z"] < 0.5:
            return True, {"terminated_reason": 2}
        return False, {}

    def reset_model(self):
        self.task_index = 0
        self.placement_goals = [
            [0.75, -0.25, 1.55],
            [0.8, 0.05, 0.95],
            [0.8, -0.25, 0.95],
            [0.85, 0.05, 0.35],
            [0.85, -0.25, 0.35],
        ]

        if self.fixed == False:
            self.placement_goals = np.random.permutation(self.placement_goals)
            self.placement_goals = np.vstack([self.placement_goals, self.placement_goals[-1]])

            self.bookshelf_objects = np.random.choice(np.arange(-24, -12), 5, replace=False)
            self.bookshelf_objects = np.append(self.bookshelf_objects, self.bookshelf_objects[-1])

        for obj_name in self.object_names:
            for geom_name in self.object_names[obj_name]:
                self._env.named.model.geom_rgba[geom_name] = np.array([0.8, 0.8, 0.8, 0])
                self._env.named.model.geom_rgba[geom_name.replace("_vision", "")] = np.array([0.8, 0.8, 0.8, 1])

        for obj_idx, obj_dest, obj_color in zip(self.bookshelf_objects, self.placement_goals, self.carried_order_rgb):
            self._env.model.body_pos[obj_idx + 12] = obj_dest
            for geom_name in self.object_names[list(self.object_names.keys())[obj_idx + 12]]:
                self._env.named.model.geom_rgba[geom_name] = np.append(obj_color, 0.2)
                self._env.named.model.geom_rgba[geom_name.replace("_vision", "")] = np.append(obj_color, 1)

        return super().reset_model()


class BookshelfSimple(BookshelfBase):
    fixed = True


class BookshelfHard(BookshelfBase):
    fixed = False
