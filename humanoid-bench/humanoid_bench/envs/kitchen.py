import numpy as np
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([2, 3]),
    "top burner": np.array([6, 7]),
    "light switch": np.array([8, 9]),
    "slide cabinet": np.array([10]),
    "hinge cabinet": np.array([11, 12]),
    "microwave": np.array([13]),
    "kettle": np.array([14, 15, 16, 17, 18, 19, 20]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3


class Kitchen(Task):
    qpos0_robot = {
        "h1hand": [
            '<key name="qpos0" qpos="0 -0.6 0.98 1 0 0 1.57 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="kettle" qpos="0.127969 -0.220115 0.979132 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -0.3731 0.60875 -0.17875 0.2554 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="microwave" qpos="-0.251416 -0.155807 0.990415 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -0.8323 0.902 -0.81125 0.7572 0 0 0 0 1.39688 0 0 0 1.29608 0 0 0 1.20444 0 0 0 0 1.10364 0 0 -0.90042 0.26884 0.087948 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="light switch" qpos="-0.251416 -0.155807 0.990415 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -1.722 0.4535 -0.84 1.3748 0 0 0 0 1.39688 0 0 0 1.29608 0 0 0 1.20444 0 0 0 0 1.10364 0 0 -0.90042 0.26884 0.087948 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="bottom burner" qpos="-0.0400544 -0.192299 0.987716 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.0705 -1.5498 0.81575 -0.84 1.066 1.0282 0.069785 -0.235248 0 0.205564 0.589125 0 0 0.1964 0.526285 0 0 0.33386 0.400605 0 0 0 0.315532 0.40846 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="slide cabinet" qpos="0.186633 -0.135474 0.97774 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.2585 -1.9516 1.05725 -0.84 1.3748 0 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0 0.003956 0 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="hinge cabinet" qpos="-0.216207 -0.145177 0.987971 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.2585 -1.9516 1.05725 -0.84 1.3748 0 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0 0.003956 0 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
        ],
        "h1touch": [
            '<key name="qpos0" qpos="0 -0.6 0.98 1 0 0 1.57 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="kettle" qpos="0.127969 -0.220115 0.979132 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -0.3731 0.60875 -0.17875 0.2554 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="microwave" qpos="-0.251416 -0.155807 0.990415 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -0.8323 0.902 -0.81125 0.7572 0 0 0 0 1.39688 0 0 0 1.29608 0 0 0 1.20444 0 0 0 0 1.10364 0 0 -0.90042 0.26884 0.087948 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="light switch" qpos="-0.251416 -0.155807 0.990415 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 -1.722 0.4535 -0.84 1.3748 0 0 0 0 1.39688 0 0 0 1.29608 0 0 0 1.20444 0 0 0 0 1.10364 0 0 -0.90042 0.26884 0.087948 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="bottom burner" qpos="-0.0400544 -0.192299 0.987716 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.0705 -1.5498 0.81575 -0.84 1.066 1.0282 0.069785 -0.235248 0 0.205564 0.589125 0 0 0.1964 0.526285 0 0 0.33386 0.400605 0 0 0 0.315532 0.40846 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="slide cabinet" qpos="0.186633 -0.135474 0.97774 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.2585 -1.9516 1.05725 -0.84 1.3748 0 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0 0.003956 0 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
            '<key name="hinge cabinet" qpos="-0.216207 -0.145177 0.987971 0.679513 -0.00743181 -0.010674 0.733549 0 0 -0.4 0.72175 -0.4 0 0 -0.4 0.8 -0.4 -0.2585 -1.9516 1.05725 -0.84 1.3748 0 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0.003956 0 0 0 0 0.003956 0 0 -0.90042 0.26884 0 0.265278 0.682092 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
        ],
        "g1": [
            '<key name="qpos0" qpos="0 -0.6 0.75 1 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 1.25561e-05 1.57437e-07 8.24417e-05 9.48283e-05 0 0 0 0 -0.269 0.35 0.994 1 1.34939e-19 -3.51612e-05 -7.50168e-19"/>',
        ],
    }

    dof = 21
    camera_name = "cam_kitchen"
    max_episode_steps = 500
    success_bar = 4

    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    ENFORCE_TASK_ORDER = True

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)

        assert self.REMOVE_TASKS_WHEN_COMPLETE
        assert self.ENFORCE_TASK_ORDER

        if env is None:
            return

        self.tasks_to_complete = dict.fromkeys(self.TASK_ELEMENTS)
        self.obs_dict = {}
        self.goal = np.zeros(self.dof)
        self.goal = self._get_task_goal()

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof + self.dof,),
            dtype=np.float64,
        )

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def get_obs(self):
        qp = self._env.data.qpos[: self.robot.dof].copy()
        qv = self._env.data.qvel[: self.robot.dof - 1].copy()
        obj_qp = self._env.data.qpos[-self.dof :].copy()
        obj_qv = self._env.data.qvel[-self.dof + 1 :].copy()

        self.obs_dict = {}
        self.obs_dict["qp"] = qp
        self.obs_dict["qv"] = qv
        self.obs_dict["obj_qp"] = obj_qp
        self.obs_dict["obj_qv"] = obj_qv
        return np.concatenate([qp, obj_qp])

    def get_reward(self):
        reward_dict = {}
        next_obj_obs = self.obs_dict["obj_qp"]
        completions = []
        all_completed_so_far = True
        for i, element in enumerate(self.tasks_to_complete):
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(next_obj_obs[..., element_idx] - self.goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete and (all_completed_so_far or not self.ENFORCE_TASK_ORDER):
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            for element in completions:
                del self.tasks_to_complete[element]
            # [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))

        reward_dict["success_subtasks"] = len(self.TASK_ELEMENTS) - len(self.tasks_to_complete)
        reward_dict["success"] = 0
        if len(self.tasks_to_complete) == 0:
            reward_dict["success"] = 1

        return bonus, reward_dict

    def get_terminated(self):
        terminated = False
        if self.TERMINATE_ON_TASK_COMPLETE:
            terminated = not self.tasks_to_complete
        return terminated, {}

    def reset_model(self):
        self.tasks_to_complete = dict.fromkeys(self.TASK_ELEMENTS)
        return super().reset_model()
