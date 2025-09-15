"""Open a cabinet task converted from MuJoCo Playground to RoboVerse."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat
from roboverse_pack.queries.contact import ContactData
from roboverse_pack.queries.site import SiteMat, SitePos

# Default configuration adapted from MuJoCo Playground
DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            # Gripper goes to the handle.
            "gripper_box": 4.0,
            # Handle goes to the target position.
            "box_target": 8.0,
            # Do not collide with the barrier.
            "no_barrier_collision": 0.25,
            # Arm stays close to target pose.
            "robot_target_qpos": 0.3,
        }
    },
    "nconmax": 12 * 8192,
    "njmax": 96,
}


@register_task("mujoco_playground.open_cabinet", "open_cabinet_rl")
class PandaOpenCabinet(RLTaskEnv):
    """Environment for training the Franka Panda robot to open a cabinet.

    Reward shaping and task design adapted from DeepMind's MuJoCo Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

    scenario = ScenarioCfg(
        objects=[
            # Cabinet handle (the object the robot needs to interact with)
            ArticulationObjCfg(
                name="handle",
                mjcf_path="roboverse_data/assets/playground/open_cabinet/mjcf/handle.xml",
                fix_base_link=True,
            ),
            # Target position for the handle
            PrimitiveSphereCfg(
                name="target",
                mass=0.01,
                radius=0.02,
                physics=PhysicStateType.XFORM,
                color=(0.7, 1.0, 0.7),  # Green color for target
            ),
            # Barrier to avoid collision
            RigidObjCfg(
                name="barrier",
                mjcf_path="roboverse_data/assets/playground/open_cabinet/mjcf/barrier.xml",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
            nconmax=12 * 8192,
            njmax=96,
        ),
        decimation=4,
    )
    max_episode_steps = 150

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None  # Initialize _last_action to None
        self._reached_handle = torch.zeros(self.scenario.num_envs, device=device)
        super().__init__(scenario, device)
        self._action_scale = DEFAULT_CONFIG.get("action_scale", 0.04)
        # Latched gate: whether gripper has reached the handle (per-env)
        # mj: info["reached_box"] = 1.0 * jp.maximum(info["reached_box"], cond)
        self.reward_functions = [
            self._reward_handle_target,
            self._reward_gripper_handle,
            self._reward_no_barrier_collision,
            self._reward_robot_target_qpos,
        ]
        self.reward_weights = [
            DEFAULT_CONFIG["reward_config"]["scales"]["box_target"],
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_box"],
            DEFAULT_CONFIG["reward_config"]["scales"]["no_barrier_collision"],
            DEFAULT_CONFIG["reward_config"]["scales"]["robot_target_qpos"],
        ]

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges."""
        # mj: Initialize target position with small random x-variation only
        # mj: target_pos = [0.3, 0.0, 0.5]; target_pos[0] += uniform(-0.1, 0.1)
        x_jitter = (torch.rand(self.num_envs, 1, device=self.device) * 0.2) - 0.1  # [-0.1, 0.1]
        target_pos = torch.cat(
            [
                0.3 + x_jitter,
                torch.zeros(self.num_envs, 1, device=self.device),
                0.5 * torch.ones(self.num_envs, 1, device=self.device),
            ],
            dim=-1,
        )

        # Set orientations (identity quaternions)
        target_quat = torch.tensor([0, 0.707, 0, 0.707], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        # Zero velocities
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Update object states
        states.objects["target"].root_state = torch.cat([target_pos, target_quat, zero_vel, zero_ang_vel], dim=-1)

        return states

    def reset(self, env_ids=None):
        """Reset environment and last actions."""
        if env_ids is None or self._last_action is None:
            # mj: Set initial qpos. Arm joints only with perturbation
            # eps = jp.deg2rad(30)
            # perturb_mins = jp.array([-eps, -eps, -eps, -2 * eps, -eps, 0, -eps])
            # perturb_maxs = jp.array([eps, eps, eps, 0, eps, 2 * eps, eps])
            # perturb_arm = jax.random.uniform(rng_arm, (7,), minval=perturb_mins, maxval=perturb_maxs)
            # init_q = jp.array(self._init_q).at[:7].set(self._init_q[:7] + perturb_arm)

            # Add random perturbation to arm joints
            eps = torch.deg2rad(torch.tensor(30.0, device=self.device))
            perturb_mins = torch.tensor([-eps, -eps, -eps, -2 * eps, -eps, 0, -eps], device=self.device)
            perturb_maxs = torch.tensor([eps, eps, eps, 0, eps, 2 * eps, eps], device=self.device)

            # Get initial joint positions and add perturbation
            init_joint_pos = self._initial_states.robots[self.robot_name].joint_pos.clone()
            perturb_arm = (
                torch.rand(self.num_envs, 7, device=self.device) * (perturb_maxs - perturb_mins) + perturb_mins
            )
            init_joint_pos[:, 2:9] += perturb_arm  # Apply perturbation to arm joints (skip finger joints)

            self._last_action = init_joint_pos
            # Reset latched gate for all envs
            self._reached_handle.zero_()
        else:
            # Apply same perturbation logic for specific env_ids
            eps = torch.deg2rad(torch.tensor(30.0, device=self.device))
            perturb_mins = torch.tensor([-eps, -eps, -eps, -2 * eps, -eps, 0, -eps], device=self.device)
            perturb_maxs = torch.tensor([eps, eps, eps, 0, eps, 2 * eps, eps], device=self.device)

            init_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[env_ids].clone()
            perturb_arm = torch.rand(len(env_ids), 7, device=self.device) * (perturb_maxs - perturb_mins) + perturb_mins
            init_joint_pos[:, 2:9] += perturb_arm
            self._last_action[env_ids] = init_joint_pos
            # Reset latched gate for selected envs
            self._reached_handle[env_ids] = 0.0

        return super().reset(env_ids=env_ids)

    def step(self, actions):
        """Step with delta control."""
        # mj: delta = action * self._config.action_scale
        # mj: ctrl = state.data.ctrl + delta
        # mj: ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)
        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions
        return obs, reward, terminated, time_out, info

    def _reward_handle_target(self, env_states) -> torch.Tensor:
        """Reward for bringing handle to target position."""
        # mj: box_target = 1 - jp.tanh(5 * jp.linalg.norm(target_pos - box_pos))
        # mj: info["reached_box"] = 1.0 * jp.maximum(info["reached_box"], (jp.linalg.norm(box_pos - gripper_pos) < 1.0 * 0.012))
        # mj: return {"box_target": box_target * info["reached_box"], ...}

        handle_pos = env_states.objects["handle"].root_state[:, 0:3]  # [num_envs, 3]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)

        pos_err = torch.norm(target_pos - handle_pos, dim=-1)  # [num_envs]
        handle_target = 1 - torch.tanh(5 * pos_err)

        # Check if gripper has reached the handle and latch across steps
        reached_now = (torch.norm(handle_pos - gripper_pos, dim=-1) < 0.012).float()
        self._reached_handle = torch.maximum(self._reached_handle, reached_now)

        return handle_target * self._reached_handle

    def _reward_gripper_handle(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the handle."""
        # mj: gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
        handle_pos = env_states.objects["handle"].root_state[:, 0:3]  # [num_envs, 3]
        gripper_pos = env_states.extras["gripper_pos"]  # [num_envs, 3]
        gripper_handle_dist = torch.norm(handle_pos - gripper_pos, dim=-1)  # [num_envs]
        approach_reward = 1 - torch.tanh(5 * gripper_handle_dist)
        return approach_reward

    def _reward_no_barrier_collision(self, env_states) -> torch.Tensor:
        """Reward for not colliding with the barrier."""
        # mj: hand_barrier_collision = [data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0 for sensor_id in self._barrier_hand_found_sensor]
        # mj: barrier_collision = sum(hand_barrier_collision) > 0
        # mj: no_barrier_collision = 1 - barrier_collision

        extra_data = env_states.extras
        left_collision = extra_data["left_finger_barrier_collision"]  # [num_envs]
        right_collision = extra_data["right_finger_barrier_collision"]  # [num_envs]
        hand_collision = extra_data["hand_capsule_barrier_collision"]  # [num_envs]
        any_collision = (left_collision > 0) | (right_collision > 0) | (hand_collision > 0)  # [num_envs]
        no_barrier_collision = (~any_collision).float()
        return no_barrier_collision

    def _reward_robot_target_qpos(self, env_states) -> torch.Tensor:
        """Reward for robot staying close to target joint positions."""
        # mj: robot_target_qpos = 1 - jp.tanh(jp.linalg.norm(data.qpos[self._robot_arm_qposadr] - self._init_q[self._robot_arm_qposadr]))
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos[
            :, 2:
        ]  # [num_envs, num_joints], skip finger joints
        target_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[:, 2:]

        joint_error = torch.norm(robot_joint_pos - target_joint_pos, dim=-1)  # [num_envs]
        return 1 - torch.tanh(joint_error)

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        # mj: obs = jp.concatenate([
        # mj:     data.qpos, data.qvel, gripper_pos, gripper_mat[3:],
        # mj:     data.xmat[self._obj_body].ravel()[3:], data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        # mj:     info["target_pos"] - data.xpos[self._obj_body], target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        # mj:     data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        # mj: ])

        handle_pos = env_states.objects["handle"].root_state[:, 0:3]  # [num_envs, 3]
        handle_quat = env_states.objects["handle"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        target_quat = env_states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)
        gripper_mat = env_states.extras["gripper_mat"]  # (B, 9)
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert quaternions to rotation matrices for handle and target
        handle_mat = matrix_from_quat(handle_quat)  # [num_envs, 3, 3]
        target_mat = matrix_from_quat(target_quat)  # [num_envs, 3, 3]

        handle_mat_flat = handle_mat.view(self.num_envs, -1)  # [num_envs, 9]
        target_mat_flat = target_mat.view(self.num_envs, -1)  # [num_envs, 9]

        # Ensure gripper_mat has correct shape [num_envs, 9]
        if gripper_mat.dim() == 3:
            gripper_mat = gripper_mat.view(self.num_envs, -1)  # Reshape to [num_envs, 9]

        handle_to_gripper = handle_pos - gripper_pos  # [num_envs, 3]
        target_to_handle = target_pos - handle_pos  # [num_envs, 3]
        target_handle_rot_diff = target_mat_flat[:, :6] - handle_mat_flat[:, :6]  # [num_envs, 6]

        # Get joint position targets and calculate control error (target - actual)
        robot_joint_pos_target = env_states.robots[self.robot_name].joint_pos_target  # [num_envs, num_joints]
        joint_pos_error = robot_joint_pos_target - robot_joint_pos  # [num_envs, num_joints]

        obs = torch.cat(
            [
                robot_joint_pos,  # Joint positions [num_envs, num_joints]
                robot_joint_vel,  # Joint velocities [num_envs, num_joints]
                gripper_pos,  # Gripper position [num_envs, 3]
                gripper_mat[:, 3:],  # Gripper orientation (last 6 elements) [num_envs, 6]
                handle_mat_flat[:, 3:],  # Handle orientation (last 6 elements) [num_envs, 6]
                handle_to_gripper,  # Handle to gripper vector [num_envs, 3]
                target_to_handle,  # Target to handle vector [num_envs, 3]
                target_handle_rot_diff,  # Target-handle orientation difference [num_envs, 6]
                joint_pos_error,  # Joint position error (target - actual) [num_envs, num_joints]
            ],
            dim=-1,
        )  # [num_envs, obs_dim]

        return obs

    def _terminated(self, env_states) -> torch.Tensor:
        """Check if episode should be terminated."""
        # mj: box_pos = data.xpos[self._obj_body]
        # mj: out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        # mj: out_of_bounds |= box_pos[2] < 0.0
        # mj: done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        handle_pos = env_states.objects["handle"].root_state[:, 0:3]  # [num_envs, 3]
        out_of_bounds = torch.any(torch.abs(handle_pos) > 1.0, dim=-1)  # [num_envs]
        out_of_bounds |= handle_pos[:, 2] < 0.0  # Handle below ground

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]
        nan_joints = torch.isnan(robot_joint_pos).any(dim=-1) | torch.isnan(robot_joint_vel).any(dim=-1)

        nan_objects = torch.isnan(handle_pos).any(dim=-1)
        terminated = out_of_bounds | nan_joints | nan_objects

        return terminated

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        # mj: keyframe="upright" - this corresponds to the home position
        init = [
            {
                "objects": {
                    "handle": {
                        "pos": torch.tensor([0.5, 0.0, 0.5]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {"handle_joint": 0.0},
                    },
                    "target": {
                        "pos": torch.tensor([0.3, 0.0, 0.5]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "barrier": {
                        "pos": torch.tensor([0.52, 0.0, 0.5]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": 0.0,
                            "panda_joint3": 0.0,
                            "panda_joint4": 0.0,
                            "panda_joint5": 0.0,
                            "panda_joint6": 0.0,
                            "panda_joint7": 0.0,
                            "panda_finger_joint1": 0.0,
                            "panda_finger_joint2": 0.0,
                        },
                    }
                },
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]

        return init

    def _extra_spec(self) -> dict:
        """Register extra queries for collision detection and gripper state."""
        return {
            # mj: self._barrier_hand_found_sensor = [self._mj_model.sensor(f"barrier_{geom}_found").id for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]]
            "left_finger_barrier_collision": ContactData(f"{self.robot_name}/left_finger_pad", "barrier/barrier_geom"),
            "right_finger_barrier_collision": ContactData(
                f"{self.robot_name}/right_finger_pad", "barrier/barrier_geom"
            ),
            "hand_capsule_barrier_collision": ContactData(f"{self.robot_name}/hand_capsule", "barrier/barrier_geom"),
            # Gripper site queries for position and orientation
            "gripper_pos": SitePos("gripper"),
            "gripper_mat": SiteMat("gripper"),
        }

    def _get_ee_state(self, states):
        """Return EE state using site queries."""
        ee_pos_world = states.extras["gripper_pos"]  # (B, 3)
        ee_mat_world = states.extras["gripper_mat"]  # (B, 9)
        return ee_pos_world, ee_mat_world
