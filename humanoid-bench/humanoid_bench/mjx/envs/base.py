import jax
import mujoco
from brax.envs.base import MjxEnv, State
from humanoid_bench.mjx.envs.utils import perturbed_pipeline_step
from jax import numpy as jp


class Humanoid(MjxEnv):
    def __init__(self, path="./unitree_h1/scene.xml", reward_weights_dict=None, **kwargs):
        collisions = kwargs.get("collisions", "feet")
        act_control = kwargs.get("act_control", "pos")
        hands = kwargs.get("hands", "both")

        path = "./humanoid_bench/assets/mjx/scene_mjx_feet_collisions_two_targets_pos.xml"

        del kwargs["collisions"]
        del kwargs["act_control"]
        del kwargs["hands"]

        mj_model = mujoco.MjModel.from_xml_path(path)

        physics_steps_per_control_step = 10
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)

        self.body_idxs = []
        self.body_vel_idxs = []
        curr_idx = 0
        curr_vel_idx = 0
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name.startswith("free_"):
                if joint_name == "free_base":
                    self.body_idxs.extend(list(range(curr_idx, curr_idx + 7)))
                    self.body_vel_idxs.extend(list(range(curr_vel_idx, curr_vel_idx + 6)))
                curr_idx += 7
                curr_vel_idx += 6
                continue
            elif (
                not joint_name.startswith("lh_") and not joint_name.startswith("rh_") and "wrist" not in joint_name
            ):  # NOTE: excluding hands here
                self.body_idxs.append(curr_idx)
                self.body_vel_idxs.append(curr_vel_idx)
            curr_idx += 1
            curr_vel_idx += 1

        print("Body idxs: ", self.body_idxs)
        print("Body vel idxs: ", self.body_vel_idxs)

        self.left_target_idxs = range(mj_model.nq - 14, mj_model.nq - 11)
        self.right_target_idxs = range(mj_model.nq - 7, mj_model.nq - 4)

        print("Left target idxs: ", self.left_target_idxs)
        print("Right target idxs: ", self.right_target_idxs)

        self.body_idxs = jp.array(self.body_idxs)
        self.body_vel_idxs = jp.array(self.body_vel_idxs)
        self.left_target_idxs = jp.array(self.left_target_idxs)
        self.right_target_idxs = jp.array(self.right_target_idxs)

        super().__init__(model=mj_model, **kwargs)

        self.q_pos_init = jp.array([
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
            -0.1,
            0.05,  # for reaching
            1,
            0,
            0,
            0,
            0,
            0.1,
            0.05,  # for reaching
            1,
            0,
            0,
            0,
        ])  # for reaching
        self.q_vel_init = jp.zeros(self.sys.nv)

        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        data = self.pipeline_init(
            self.q_pos_init,
            self.q_vel_init,
        )

        self.state_dim = self._get_obs(data.data, jp.array([0, 0, 0]), jp.array([0, 0, 0])).shape[-1]
        self.action_dim = self.sys.nu

        assert reward_weights_dict is not None
        self.reward_weight_dict = reward_weights_dict

    def reset(self, rng):
        # implemented in task subclass
        raise NotImplementedError

    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action

    def compute_reward(self, data, info):
        # implemented in task subclass
        raise NotImplementedError

    def get_info(self, state, data):
        left_hand_pos = data.data.site_xpos[4]
        right_hand_pos = data.data.site_xpos[5]

        target_pos_left = state.info["target_left"]
        target_dist_left = jp.sqrt(jp.square(left_hand_pos - target_pos_left).sum())

        target_pos_right = state.info["target_right"]
        target_dist_right = jp.sqrt(jp.square(right_hand_pos - target_pos_right).sum())

        angle_targets = jp.arctan2(target_pos_right[1] - target_pos_left[1], target_pos_right[0] - target_pos_left[0])

        reached = jp.logical_and(target_dist_left < 0.05, target_dist_right < 0.05)

        max_joint_vel = jp.max(jp.abs(data.data.qvel[self.body_vel_idxs]))

        success = jp.where(reached, 1.0, 0.0)
        success_left = jp.where(target_dist_left < 0.05, 1.0, 0.0)

        total_successes = state.info["total_successes"] + success

        return {
            "hand_pos": left_hand_pos,
            "angle_targets": angle_targets,
            "target_dist_left": target_dist_left,
            "target_dist_right": target_dist_right,
            "max_joint_vel": max_joint_vel,
            "success": success,
            "success_left": success_left,
            "total_successes": total_successes,  # TODO Use final and change goal after reaching
        }

    def _resample_target(self, state, log_info):
        # By default we don't resample target
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        apply_every = 1
        hold_for = 1
        magnitude = 1

        # Reset the applied forces every 200 steps
        rng, subkey = jax.random.split(state.info["rng"])
        xfrc_applied = jp.zeros((self.sys.nbody, 6))
        xfrc_applied = jax.lax.cond(
            state.info["step_counter"] % apply_every == 0,
            lambda _: jax.random.normal(subkey, shape=(self.sys.nbody, 6)) * magnitude,
            lambda _: state.info["last_xfrc_applied"],
            operand=None,
        )
        # Reset to 0 every 50 steps
        perturb = jax.lax.cond(
            state.info["step_counter"] % apply_every < hold_for, lambda _: 1, lambda _: 0, operand=None
        )
        xfrc_applied = xfrc_applied * perturb

        action = self.unnorm_action(action)

        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, xfrc_applied, self._n_frames)
        observation = self._get_obs(data.data, state.info["target_left"], state.info["target_right"])

        log_info = self.get_info(state, data)
        state = self._resample_target(state, log_info)
        reward, terminated = self.compute_reward(data, log_info)

        state.metrics.update(reward=reward)
        state.info.update(
            rng=rng,
            step_counter=state.info["step_counter"] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**log_info)

        return state.replace(pipeline_state=data, obs=observation, reward=reward, done=terminated)

    def _get_obs(self, data, target_left, target_right=None) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        raise NotImplementedError
