import jax
from brax.envs.base import State
from humanoid_bench.mjx.envs.base import Humanoid
from jax import numpy as jp


class HumanoidReachContinual(Humanoid):
    def __init__(self, path="./unitree_h1/scene.xml", **kwargs):
        super().__init__(path, **kwargs)

    def _sample_target(self, rng):
        min_target = jp.array([-2, -2, 0.2])
        max_target = jp.array([2, 2, 2.0])
        target = jax.random.uniform(rng, shape=(3,), minval=min_target, maxval=max_target)
        return target

    def _resample_target(self, state, log_info):
        old_rng, old_target = state.info["rng"], state.info["target_left"]
        new_rng, target_rng, flag_rng = jax.random.split(old_rng, 3)
        new_target = self._sample_target(target_rng)

        # resample_ratio = 0.2  # Expoential decay
        # switch_flag = log_info['success'] * jax.random.bernoulli(flag_rng, p=resample_ratio)
        switch_flag = log_info["success_left"]

        state.info["rng"] = switch_flag * new_rng + (1 - switch_flag) * old_rng
        state.info["target_left"] = switch_flag * new_target + (1 - switch_flag) * old_target

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
        target = self._sample_target(subkey)

        # center = np.array([0.25, 0.5, 1.0])
        # range = 0.05
        # # target = np.random.uniform(np.array([0, -0.5, 0.5]), np.array([0.5, 0.5, 1.5]), size=(3,))
        # target = np.random.uniform(center - range, center + range, size=(3,))

        qpos = qpos.at[self.left_target_idxs].set(target)

        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data.data, target, None)
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
                "target_left": target,
                "target_right": target,
                "success": 0.0,
                "success_left": 0.0,
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
        healthy_reward = 5.0
        healthy_reward = data.data.xmat[1, -1, -1] * 5.0
        motion_penalty = jp.square(data.data.qvel[self.body_vel_idxs]).sum()

        dist = info["target_dist_left"]
        reaching_reward_l1 = jp.where(dist < 1, x=1.0, y=0.0)

        reward = (
            healthy_reward - 0.0001 * motion_penalty + 5 * reaching_reward_l1 + 1000.0 * info["success_left"]
        )  # + 1e-4 * penalty

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
            # data.site_xpos[5],
            target_left - offset,
            # target_right
        ))
