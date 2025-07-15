import jax
from brax.envs.base import State
from humanoid_bench.mjx.envs.base import Humanoid
from jax import numpy as jp


class HumanoidReachContinualTwoHands(Humanoid):
    def __init__(self, path="./unitree_h1/scene.xml", **kwargs):
        super().__init__(path, **kwargs)

    def _sample_from_sphere(self, rng, center, radius):
        phi = jax.random.uniform(rng, shape=(1,), minval=0, maxval=2 * jp.pi)
        costheta = jax.random.uniform(rng, shape=(1,), minval=-1, maxval=1)
        u = jax.random.uniform(rng, shape=(1,), minval=0, maxval=1)

        theta = jp.arccos(costheta)
        r = radius * jp.cbrt(u)

        x = r * jp.sin(theta) * jp.cos(phi) + center[0]
        y = r * jp.sin(theta) * jp.sin(phi) + center[1]
        z = r * jp.cos(theta) + center[2]

        return jp.concatenate([x, y, z])

    def _sample_target(self, rng):
        left_rng, right_rng, flag_rng = jax.random.split(rng, 3)

        min_target = jp.array([-2, -2, 0.3])
        max_target = jp.array([2, 2, 1.8])

        target_left = jax.random.uniform(left_rng, shape=(3,), minval=min_target, maxval=max_target)

        target_right = self._sample_from_sphere(right_rng, target_left, 1.0)
        target_right = jp.where(target_right > max_target, max_target, target_right)
        target_right = jp.where(target_right < min_target, min_target, target_right)

        return target_left, target_right

    def _resample_target(self, state, log_info):
        old_rng, old_target_left, old_target_right = (
            state.info["rng"],
            state.info["target_left"],
            state.info["target_right"],
        )
        new_rng, target_rng, flag_rng = jax.random.split(old_rng, 3)
        new_target_left, new_target_right = self._sample_target(target_rng)

        # resample_ratio = 0.2  # Expoential decay
        # switch_flag = log_info['success'] * jax.random.bernoulli(flag_rng, p=resample_ratio)
        switch_flag = log_info["success"]

        state.info["rng"] = switch_flag * new_rng + (1 - switch_flag) * old_rng
        state.info["target_left"] = switch_flag * new_target_left + (1 - switch_flag) * old_target_left
        state.info["target_right"] = switch_flag * new_target_right + (1 - switch_flag) * old_target_right

        # Should not update the target dist as the reward from last time step should be counted first
        # left_hand_pos = data.data.site_xpos[1]
        # target_pos = state.info['target']
        # target_dist = jp.sqrt(jp.square(left_hand_pos - target_pos).sum())
        # state.info['target_dist'] = target_dist

        return state

    def reset(self, rng):
        step_counter = 0

        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()

        rng, subkey = jax.random.split(rng)
        target_left, target_right = self._sample_target(subkey)

        # center = np.array([0.25, 0.5, 1.0])
        # range = 0.05
        # # target = np.random.uniform(np.array([0, -0.5, 0.5]), np.array([0.5, 0.5, 1.5]), size=(3,))
        # target = np.random.uniform(center - range, center + range, size=(3,))

        qpos = qpos.at[self.left_target_idxs].set(target_left)
        qpos = qpos.at[self.right_target_idxs].set(target_right)

        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data.data, target_left, target_right)
        state = State(
            data,
            obs,
            reward,
            done,
            {"reward": zero},
            {
                "rng": rng,
                "step_counter": step_counter,
                "last_xfrc_applied": jp.zeros((self.sys.nbody, 6)),
                "target_left": target_left,
                "target_right": target_right,
                "success": 0.0,
                "total_successes": 0.0,
            },
        )
        info = self.get_info(state, data)
        state.info.update(**info)
        return state

    def check_out_of_range(self, xpos):
        xpos_min = jp.array([-1.0, -0.7, 0.1])
        xpos_max = jp.array([1.5, 1.5, 1.1])
        return jp.any(jp.logical_or(xpos < xpos_min, xpos > xpos_max))

    def compute_reward(self, data, info):
        # healthy_reward = data.data.xmat[1, -1, -1] * 5.0
        motion_penalty = jp.square(data.data.qvel[self.body_vel_idxs]).sum()

        angle_targets = info["angle_targets"]
        angle_desired = angle_targets + jp.pi / 2
        quat_desired = jp.array([jp.cos(angle_desired / 2), 0.0, 0.0, jp.sin(angle_desired / 2)])
        quat_torso = data.data.xquat[12] / jp.linalg.norm(data.data.xquat[12])
        quat_pelvis = data.data.xquat[1] / jp.linalg.norm(data.data.xquat[1])
        healthy_reward = 1.0 * jp.square(jp.dot(quat_desired, quat_torso)) + 1.0 * jp.square(
            jp.dot(quat_desired, quat_pelvis)
        )

        dist_left = info["target_dist_left"]
        dist_right = info["target_dist_right"]

        # reaching_reward_l1 = jp.where(dist_left < 1, x=1.0, y=0.0) + jp.where(dist_right < 1, x=1.0, y=0.0)
        reaching_reward_l1 = jp.where(jp.logical_and(dist_left < 1, dist_right < 1), x=1.0, y=0.0)
        reaching_reward_l1_bis = jp.where(jp.logical_and(dist_left < 0.5, dist_right < 0.5), x=1.0, y=0.0)
        reaching_reward_l1_tris = jp.where(jp.logical_and(dist_left < 0.25, dist_right < 0.25), x=1.0, y=0.0)
        reached_single_reward = jp.where(dist_left < 0.05, x=1.0, y=0.0) + jp.where(dist_right < 0.05, x=1.0, y=0.0)

        # reward = healthy_reward - 0.0001 * motion_penalty - 0*dist_left - 0*dist_right + reaching_reward_l1 + 5 * reaching_reward_l1_bis + 0 * reached_single_reward + 1000.0 * info['success'] #+ 1e-4 * penalty
        reward = (
            healthy_reward
            - 0.0001 * motion_penalty
            + 1 * reaching_reward_l1
            + 1 * reaching_reward_l1_bis
            + 1 * reaching_reward_l1_tris
            + 1000.0 * info["success"]
        )
        # reward = healthy_reward - 0.0001 * motion_penalty - jp.max(jp.array([dist_left, dist_right])) + 1 * reaching_reward_l1 + 5 * reaching_reward_l1_bis + 1000.0 * info['success']

        # terminate if torso is out of range or body height is out of range
        height = data.data.qpos[2]
        terminated = jp.where(height < 0.3, x=1.0, y=0.0)
        terminated = jp.where(height > 1.8, x=1.0, y=terminated)
        # out_of_range = self.check_out_of_range(info['hand_pos'])
        # terminated = jp.where(out_of_range, x=1.0, y=terminated)
        reward = jp.where(jp.isnan(reward), x=-1, y=reward)
        return reward, terminated

    def _get_obs(self, data, target_left, target_right) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        offset = jp.array([data.qpos[0], data.qpos[1], 0])
        return jp.concatenate((
            data.qpos[self.body_idxs][2:],
            data.qvel[self.body_vel_idxs],
            data.site_xpos[4] - offset,
            data.site_xpos[5] - offset,
            target_left - offset,
            target_right - offset,
        ))
