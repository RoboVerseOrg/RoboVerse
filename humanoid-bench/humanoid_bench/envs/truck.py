import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box
from humanoid_bench.tasks import Task


class Truck(Task):
    qpos0_robot = {
        "h1": """
            2 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            5 0 1.1 0.707105 -0.707108 0 0 5.1 0 1.3 0.706752 0 0 0.707462 5 0.35 1.3 0.707105 0.707108 0 0 5 -0.5 1.1 1 0 0 0 5.1 -0.6 1.4 1 0 0 0
        """,
        "h1hand": """
            2 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            5 0 1.1 0.707105 -0.707108 0 0 5.1 0 1.3 0.706752 0 0 0.707462 5 0.35 1.3 0.707105 0.707108 0 0 5 -0.5 1.1 1 0 0 0 5.1 -0.6 1.4 1 0 0 0
        """,
        "h1touch": """
            2 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            5 0 1.1 0.707105 -0.707108 0 0 5.1 0 1.3 0.706752 0 0 0.707462 5 0.35 1.3 0.707105 0.707108 0 0 5 -0.5 1.1 1 0 0 0 5.1 -0.6 1.4 1 0 0 0
        """,
        "g1": """
            2 0 0.75
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
            5 0 1.1 0.707105 -0.707108 0 0 5.1 0 1.3 0.706752 0 0 0.707462 5 0.35 1.3 0.707105 0.707108 0 0 5 -0.5 1.1 1 0 0 0 5.1 -0.6 1.4 1 0 0 0
        """,
    }
    dof = 5 * 7

    camera_name = "cam_hurdle"

    success_bar = 3000

    htarget_low = np.array([-2.5, -2.5, 0.3])
    htarget_high = np.array([10.0, 2.5, 1.8])

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        self.package_list = [
            "package_a",
            "package_b",
            "package_c",
            "package_d",
            "package_e",
        ]

        self.packages_on_truck = [x for x in self.package_list]
        self.packages_picked_up = []
        self.packages_on_table = []

        self.initialized = False

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.robot.dof + self.dof) * 2 - 6,),
            dtype=np.float64,
        )

    def upon_table(self, package):
        return (
            self._env.named.data.xpos[package][0] < 2.35
            and self._env.named.data.xpos[package][1] < -1.35
            and self._env.named.data.xpos[package][0] > 1.65
            and self._env.named.data.xpos[package][1] > -2.05
            and self._env.named.data.xpos[package][2] > 0.5
        )

    def get_reward(self):
        reward = 0

        # Store initial z positions of packages
        if self.initialized == False:
            self.initialized = True
            self.initial_zs = {}
            for package in self.package_list:
                self.initial_zs[package] = self._env.named.data.xpos[package][2]

        # Check if packages have been picked up from truck
        for package in self.packages_on_truck:
            if self._env.named.data.xpos[package][2] > self.initial_zs[package] + 0.1:
                self.packages_picked_up.append(package)
                self.packages_on_truck.remove(package)
                reward += 100

        # Check if packages have been placed on table
        for package in self.packages_picked_up:
            # print('Package: ', package, self._env.named.data.xpos[package])
            if self.upon_table(package):
                self.packages_on_table.append(package)
                self.packages_picked_up.remove(package)
                reward += 100

        # Check if packages are no longer on table
        for package in self.packages_on_table:
            if not self.upon_table(package):
                self.packages_on_table.remove(package)
                self.packages_picked_up.append(package)
                reward -= 100

        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )

        # minimize distance between robot and packages on truck
        reward_robot_package_truck = 0
        if len(self.packages_on_truck) > 0:
            dist_robot_package_truck = [
                np.linalg.norm(self._env.named.data.xpos[package] - self._env.named.data.qpos["free_base"][:3])
                for package in self.packages_on_truck
            ]
            reward_robot_package_truck = rewards.tolerance(
                np.min(dist_robot_package_truck),
                bounds=(0, 0.2),
                margin=4,
                value_at_margin=0,
                sigmoid="linear",
            )

        # minimize distance between robot and packages picked up
        reward_robot_package_picked_up = 0
        if len(self.packages_picked_up) > 0:
            dist_robot_package_picked_up = [
                np.linalg.norm(self._env.named.data.xpos[package] - self._env.named.data.qpos["free_base"][:3])
                for package in self.packages_picked_up
            ]
            reward_robot_package_picked_up = rewards.tolerance(
                np.min(dist_robot_package_picked_up),
                bounds=(0, 0.2),
                margin=4,
                value_at_margin=0,
                sigmoid="linear",
            )

        # minimize distance between picked up packages and table
        reward_package_table = 0
        if len(self.packages_picked_up) > 0:
            dist_package_table = [
                np.linalg.norm(self._env.named.data.xpos[package] - self._env.named.data.xpos["table"])
                for package in self.packages_picked_up
            ]
            reward_package_table = rewards.tolerance(
                np.min(dist_package_table),
                bounds=(0, 0.2),
                margin=4,
                value_at_margin=0,
                sigmoid="linear",
            )

        reward += upright * (1 + reward_robot_package_truck + reward_robot_package_picked_up + reward_package_table)

        reward_dict = {
            "upright": upright,
            "reward_robot_package_truck": reward_robot_package_truck,
            "reward_robot_package_picked_up": reward_robot_package_picked_up,
            "reward_package_table": reward_package_table,
            "packages_on_truck": len(self.packages_on_truck),
            "packages_picked_up": len(self.packages_picked_up),
            "packages_on_table": len(self.packages_on_table),
            "success": 0,
        }

        reward_dict["success_subtasks"] = len(self.packages_on_table)

        if len(self.packages_on_table) == len(self.package_list):
            reward_dict["success"] = 1
            reward += 1000

        return reward, reward_dict

    def get_terminated(self):
        terminated = False
        if len(self.packages_on_table) == len(self.package_list):
            terminated = True
        return terminated, {}
