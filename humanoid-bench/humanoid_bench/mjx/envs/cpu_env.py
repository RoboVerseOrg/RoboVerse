from typing import Dict, Tuple

import mujoco
import numpy as np


class HumanoidNumpyEnv:
    """A humanoid environment with PyTorch-compatible observations."""

    def __init__(self, path: str, task: str = "stabilize", physics_steps_per_control_step: int = 1):
        mj_model = mujoco.MjModel.from_xml_path(path)

        physics_steps_per_control_step = 10

        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

        self.body_idxs = []
        self.body_vel_idxs = []
        curr_idx = 0
        curr_vel_idx = 0
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name.startswith("free_"):
                if joint_name == "free_base":
                    self.body_idxs.extend(list(range(curr_idx, curr_idx + 7)))
                    self.body_vel_idxs.extend(list(range(curr_vel_idx, curr_vel_idx + 6)))
                curr_idx += 7
                curr_vel_idx += 6
                continue
            elif not joint_name.startswith("lh_") and not joint_name.startswith("rh_") and "wrist" not in joint_name:
                self.body_idxs.append(curr_idx)
                self.body_vel_idxs.append(curr_vel_idx)
            curr_idx += 1
            curr_vel_idx += 1

        if self.model.nu > 19:
            self.act_idxs = list(range(15)) + list(range(16, 20))  # reaching is trained without hands
        else:
            self.act_idxs = list(range(19))

        action_range = self.model.actuator_ctrlrange[self.act_idxs]
        self.low_action = np.array(action_range[:, 0])
        self.high_action = np.array(action_range[:, 1])

        self.left_target_idxs = range(self.model.nq - 14, self.model.nq - 11)
        self.right_target_idxs = range(self.model.nq - 7, self.model.nq - 4)

        print("Body Idxs: ", self.body_idxs)
        print("Body Vel Idxs: ", self.body_vel_idxs)
        print("Act Idxs: ", self.act_idxs)
        print("Left Target Idxs: ", self.left_target_idxs)
        print("Right Target Idxs: ", self.right_target_idxs)

        self.task = task

    def _sample_from_sphere(self, center, radius):
        phi = np.random.uniform(low=0, high=2 * np.pi)
        costheta = np.random.uniform(low=-1, high=1)
        u = np.random.uniform(low=0, high=1)

        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]

        return np.array([x, y, z])

    def reset(self, seed=0):
        """Resets the environment."""
        try:
            self.data.qpos = np.array([
                0,
                0,
                0.98,
                1,
                0,
                0,
                0,
                0,
                0,
                -0.4,
                0.8,
                -0.4,
                0,
                0,
                -0.4,
                0.8,
                -0.4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0.05,  # for reaching
                1,
                0,
                0,
                0,  # for reaching
                0,
                -0.1,
                0.05,  # for reaching
                1,
                0,
                0,
                0,  # for reaching
            ])
        except:
            self.data.qpos = np.array([
                0,
                0,
                0.98,
                1,
                0,
                0,
                0,
                0,
                0,
                -0.4,
                0.8,
                -0.4,
                0,
                0,
                -0.4,
                0.8,
                -0.4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0.05,  # for reaching
                1,
                0,
                0,
                0,  # for reaching
                0,
                -0.1,
                0.05,  # for reaching
                1,
                0,
                0,
                0,  # for reaching
            ])
        self.data.qvel[:] = 0

        if self.task == "reach":
            target = np.random.uniform(np.array([-2, -2, 0.2]), np.array([2, 2, 2]), size=(3,))
            self.data.qpos[self.left_target_idxs] = target
        elif self.task == "reach_two_hands":
            left_target = np.random.uniform(np.array([-2, -2, 0.5]), np.array([2, 2, 1.8]), size=(3,))
            right_target = self._sample_from_sphere(left_target, 1.0)
            right_target = np.clip(right_target, np.array([-2, -2, 0.5]), np.array([2, 2, 1.8]))
            self.data.qpos[self.left_target_idxs] = left_target
            self.data.qpos[self.right_target_idxs] = right_target

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(self.data)

    def get_info(self):
        torso_pos = self.data.body("torso_link").xpos
        target_dist = np.sqrt(np.square(self.data.site("left_hand").xpos - self.data.body("target").xpos).sum())
        max_joint_vel = np.max(np.abs(self.data.qvel[:25]))
        return {"torso_pos": torso_pos, "target_dist": target_dist, "max_joint_vel": max_joint_vel}

    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action

    def compute_reward(self, data):
        lef_hand_dist = np.sqrt(np.square(data.site("left_hand").xpos - data.qpos[self.left_target_idxs]).sum())
        right_hand_dist = np.sqrt(np.square(data.site("right_hand").xpos - data.qpos[self.right_target_idxs]).sum())
        if self.task == "reach":
            dist = lef_hand_dist
        elif self.task == "reach_two_hands":
            dist = np.max((lef_hand_dist, right_hand_dist))
        reward = float(
            dist < 0.1
        )  # Trained with 0.05, but 0.1 allows to evaluate the policy for a wider range of targets
        height = data.qpos[2]
        terminated = (height < 0.3) or (height > 1.2)
        return reward, terminated

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Applies an action to the environment."""
        action = np.clip(action, -1.0, 1.0)
        action = self.unnorm_action(action)
        self.data.ctrl[self.act_idxs] = action
        mujoco.mj_step(self.model, self.data, nstep=self._physics_steps_per_control_step)

        reward, done = self.compute_reward(self.data)

        if self.task == "reach":
            if reward > 0.5:
                target = np.random.uniform(np.array([-2, -2, 0.2]), np.array([2, 2, 2]), size=(3,))
                self.data.qpos[self.left_target_idxs] = target
                mujoco.mj_forward(self.model, self.data)
        elif self.task == "reach_two_hands":
            if reward > 0.5:
                left_target = np.random.uniform(np.array([-2, -2, 0.5]), np.array([2, 2, 1.8]), size=(3,))
                right_target = self._sample_from_sphere(left_target, 1.0)
                right_target = np.clip(right_target, np.array([-2, -2, 0.5]), np.array([2, 2, 1.8]))
                self.data.qpos[self.left_target_idxs] = left_target
                self.data.qpos[self.right_target_idxs] = right_target
                mujoco.mj_forward(self.model, self.data)

        return self._get_obs(self.data), reward, done, {}

    def _get_obs(self, data) -> np.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        offset = np.array([data.qpos[0], data.qpos[1], 0])
        if self.task == "reach":
            return np.concatenate((
                data.qpos.copy()[self.body_idxs][2:],
                data.qvel.copy()[self.body_vel_idxs],
                data.site("left_hand").xpos - offset,
                data.qpos.copy()[self.left_target_idxs] - offset,
            ))
        elif self.task == "reach_two_hands":
            return np.concatenate((
                data.qpos.copy()[self.body_idxs][2:],
                data.qvel.copy()[self.body_vel_idxs],
                data.site("left_hand").xpos - offset,
                data.site("right_hand").xpos - offset,
                data.qpos.copy()[self.left_target_idxs] - offset,
                data.qpos.copy()[self.right_target_idxs] - offset,
            ))
        else:
            raise NotImplementedError
