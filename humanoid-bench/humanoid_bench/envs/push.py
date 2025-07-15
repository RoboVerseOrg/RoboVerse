import numpy as np
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task


class Push(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
    }
    dof = 7
    max_episode_steps = 500
    camera_name = "cam_tabletop"
    # Below args are only used for reaching-based hierarchical control
    htarget_low = np.array([0, -1, 0.8])
    htarget_high = np.array([2.0, 1, 1.2])

    success_bar = 700

    def __init__(
        self,
        robot=None,
        env=None,
        **kwargs,
    ):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        self.reward_dict = {
            "hand_dist": 0.1,
            "target_dist": 1,
            "success": 1000,
            "terminate": True,
        }

        self.goal = np.array([1.0, 0.0, 1.0])

        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + 12,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        left_hand = self.robot.left_hand_position()
        target = self.goal.copy()
        box = self._env.data.qpos.flat.copy()[-7:-4]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_vel = self._env.data.qvel.flat.copy()[dofadr : dofadr + 3]

        return np.concatenate((position, velocity, left_hand, target, box, box_vel))

    def goal_dist(self):
        box = self._env.data.qpos.flat.copy()[-7:-4]
        return np.sqrt(np.square(box - self.goal).sum())

    def get_reward(self):
        goal_dist = self.goal_dist()
        penalty_dist = self.reward_dict["target_dist"] * goal_dist
        reward_success = self.reward_dict["success"] if goal_dist < 0.05 else 0

        left_hand = self.robot.left_hand_position()
        # box = self._env.data.qpos.flat.copy()[-7:-4]
        box = self._env.named.data.qpos["free_object"][:3]

        hand_dist = np.sqrt(np.square(left_hand - box).sum())
        hand_penalty = self.reward_dict["hand_dist"] * hand_dist

        reward = -hand_penalty - penalty_dist + reward_success
        info = {
            "target_dist": goal_dist,
            "hand_dist": hand_dist,
            "reward_success": reward_success,
            "success": reward_success > 0,
        }
        return reward, info

    def get_terminated(self):
        if self.reward_dict["terminate"]:
            terminated = self.goal_dist() < 0.05
        else:
            terminated = False

        return terminated, {}

    def reset_model(self):
        self.goal[0] = np.random.uniform(0.7, 1.0)
        self.goal[1] = np.random.uniform(-0.5, 0.5)

        return self.get_obs()

    def render(self):
        found = False
        for i in range(len(self._env.viewer._markers)):
            if self._env.viewer._markers[i]["objid"] == 789:
                self._env.viewer._markers[i]["pos"] = self.goal
                found = True
                break

        if not found:
            self._env.viewer.add_marker(
                pos=self.goal,
                size=0.05,
                objid=789,
                rgba=(0.8, 0.28, 0.28, 1.0),
                label="",
            )

        return self._env.mujoco_renderer.render(self._env.render_mode, self._env.camera_id, self._env.camera_name)
