"""A humanoid base wrapper for skillBench tasks"""

# ruff: noqa: F405
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from metasim.utils.math import quat_apply, quat_rotate_inverse

try:
    from isaacgym.torch_utils import torch_rand_float
except ImportError:
    pass

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.tasks.skillblender.base_legged_cfg import BaseLeggedTaskCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.humanoid_robot_util import *
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper


class HumanoidBaseWrapper(RslRlWrapper):
    """
    Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)

        # FIXME hardcode. read names from config, get from handler.

        self.use_vision = scenario.task.use_vision
        self.up_axis_idx = 2

        self._parse_joint_indices(scenario.robots[0])
        self._get_cfg_from_handler()
        self._prepare_reward_function(scenario.task)
        self._init_buffers()

    def _parse_joint_indices(self, robot):
        """
        Parse humanoid rigid body indices from robot cfg.
        """
        feet_names = robot.feet_links
        knee_names = robot.knee_links
        elbow_names = robot.elbow_links
        termination_contact_names = robot.terminate_contacts_links
        penalised_contact_names = robot.penalized_contacts_links

        # TODO get alphabet order
        self.feet_indices = self.env.handler.get_body_reindexed_indices_from_substring(robot.name, feet_names)
        self.knee_indices = self.env.handler.get_body_reindexed_indices_from_substring(robot.name, knee_names)
        self.elbow_indices = self.env.handler.get_body_reindexed_indices_from_substring(robot.name, elbow_names)
        self.termination_contact_indices = self.env.handler.get_body_reindexed_indices_from_substring(
            robot.name, termination_contact_names
        )
        self.penalised_contact_indices = self.env.handler.get_body_reindexed_indices_from_substring(
            robot.name, penalised_contact_names
        )

        # attach to cfg for reward computation.
        self.cfg.feet_indices = self.feet_indices
        self.cfg.knee_indices = self.knee_indices
        self.cfg.elbow_indices = self.elbow_indices
        self.cfg.termination_contact_indices = self.termination_contact_indices
        self.cfg.penalised_contact_indices = self.penalised_contact_indices

    def _parse_cfg(self, scenario):
        super()._parse_cfg(scenario)
        self.dt = scenario.decimation * scenario.sim_params.dt
        self.num_commands = scenario.task.command_dim

    def _get_init_states(self, scenario):
        """Get initial states from handler."""
        self.init_states, _, _ = get_traj(scenario.task, scenario.robots[0], self.env.handler)
        if len(self.init_states) < self.num_envs:
            self.init_states = (
                self.init_states * (self.num_envs // len(self.init_states))
                + self.init_states[: self.num_envs % len(self.init_states)]
            )
        else:
            self.init_states = self.init_states[: self.num_envs]

    def _get_cfg_from_handler(self):
        """
        Get cfg from handler for reward computation.
        """
        # TODO read name specified in config, add indices here
        self.cfg.default_joint_pd_target = self.env.handler.default_dof_pos
        self.cfg.torque_limits = self.env.handler.torque_limits

    def _init_buffers(self):
        """
        Init all buffer for reward computation

        TODO: move those var into
        """
        self.base_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )  # TODO align it with metasim quaternion format [wxyz]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.gravity_vec = torch.tensor(
            self.get_axis_params(-1.0, self.up_axis_idx), device=self.device, dtype=torch.float32
        ).repeat((
            self.num_envs,
            1,
        ))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
            self.num_envs,
            1,
        ))

        # TODO implement it
        # self.neg_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # self.pos_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.common_step_counter = 0
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # TODO read obs from cfg and auto concatenate
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float
            )
        else:
            self.privileged_obs_buf = None

        self.contact_forces = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.extras = {}
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # store globally for reset update and pass to obs and privileged_obs
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        # reference dof position
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.env.handler.robot_num_dof, device=self.device, requires_grad=False
        )

        # history buffer for reward computation
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(
            self.num_envs, self.env.handler.robot_num_dof, device=self.device, requires_grad=False
        )
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)

        # TODO move it into config
        self.last_feet_z = 0.05 * torch.ones(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, requires_grad=False)
        self.feet_height = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, requires_grad=False)

        # TODO add height buffer, random push force
        # TODO add history manager, read from config.
        self.obs_history = deque(maxlen=self.cfg.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.c_frame_stack)
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(
                torch.zeros(self.num_envs, self.cfg.num_single_obs, dtype=torch.float, device=self.device)
            )
        for _ in range(self.cfg.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, self.cfg.single_num_privileged_obs, dtype=torch.float, device=self.device)
            )

        # random push force
        self.rand_push_force = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )  # TODO now set 0
        self.rand_push_torque = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )  # TODO now set 0
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)  # TODO now set 0
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

    def _get_phase(
        self,
    ):
        # FIXME cycle_time definition and access
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _update_history(self, envstate):
        """update history buffer at the the of the frame, called after reset"""
        # we should always make a copy here
        # check whether torch.clone is necessary
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel_tensor(envstate, self.robot.name)[:]
        self.last_root_vel[:] = robot_root_state_tensor(envstate, self.robot.name)[:, 7:13]

    def _parse_gait_phase(self, envstate):
        envstate.robots[self.robot.name].extra["gait_phase"] = self._get_gait_phase()

    def _parse_action(self, envstate):
        envstate.robots[self.robot.name].extra["actions"] = self.actions

    def _parse_projected_gravity(self, envstate):
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        envstate.robots[self.robot.name].extra["projected_gravity"] = self.projected_gravity

    def _parse_base_euler_xyz(self, envstate):
        self.base_euler_xyz = get_euler_xyz_tensor(envstate.robots[self.robot.name].root_state[:, 3:7])
        envstate.robots[self.robot.name].extra["base_euler_xyz"] = self.base_euler_xyz

    def _parse_foot_all(self, envstate):
        """
        Run all the parse foot function sequentially. foot pos update must run first.

        Note that orders matters here, since some of the foot states are computed based on the previous foot states.
        """
        self._parse_feet_air_time(envstate)
        self._parse_feet_clearance(envstate)

    def _parse_feet_air_time(self, envstate):
        # TODO contact is computed for servaral times. maybe precompute it as a class var?
        contact = contact_force_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 5.0
        stance_mask = gait_phase_tensor(envstate, self.robot.name)
        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        envstate.robots[self.robot.name].extra["feet_air_time"] = air_time

    def _parse_feet_clearance(self, envstate):
        """Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.


        Directly calculates reward since no intermediate variables are reused for other reward.
        """
        contact = contact_forces_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 5.0
        feet_z = envstate.robots[self.robot.name].body_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.reward_cfg.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        envstate.robots[self.robot.name].extra["feet_clearance"] = rew_pos

    def _parse_history_state(self, envstate):
        # TODO check integration
        """update history buffer, must be called after reset"""
        # envstate.robots[self.robot.name].extra["last_contact_forces"] = self.last_contact_forces

        envstate.robots[self.robot.name].extra["last_root_vel"] = self.last_root_vel
        envstate.robots[self.robot.name].extra["last_dof_vel"] = self.last_dof_vel
        envstate.robots[self.robot.name].extra["last_actions"] = self.last_actions
        envstate.robots[self.robot.name].extra["last_last_actions"] = self.last_last_actions

    def _parse_local_base_vel(self, envstate):
        """Add local base velocity into states"""
        envstate.robots[self.robot.name].extra["base_lin_vel"] = self.base_lin_vel
        envstate.robots[self.robot.name].extra["base_ang_vel"] = self.base_ang_vel

    def _parse_command(self, envstate):
        """Add command into states"""
        envstate.robots[self.robot.name].extra["command"] = self.commands

    def _parse_epsidoe_legth(self, envstate):
        """parse episode length into states"""
        envstate.robots[self.robot.name].extra["episode_length_buf"] = self.episode_length_buf

    def _parse_state_for_reward(self, envstate):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        The

        Eg., offset the observation by default obs, compute input rewards.
        """
        # TODO read from config
        self._parse_gait_phase(envstate)
        self._parse_action(envstate)
        self._parse_history_state(envstate)
        self._parse_base_euler_xyz(envstate)
        self._parse_foot_all(envstate)
        self._parse_command(envstate)
        self._parse_projected_gravity(envstate)
        self._parse_local_base_vel(envstate)

    def _prepare_reward_function(self, task: BaseLeggedTaskCfg):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.reward_scales = task.reward_weights
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "reward_" + name
            self.reward_functions.append(self.get_reward_fn(name, task.reward_functions))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }
        self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def compute_reward(self, envstates):
        """Compute all the reward from the states provided."""
        self.rew_buf[:] = 0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew_func_return = self.reward_functions[i](envstates, self.robot.name, self.cfg)
            if isinstance(rew_func_return, tuple):
                unscaled_rew, metric = rew_func_return
                self.episode_metrics[name] = metric.mean().item()
            else:
                unscaled_rew = rew_func_return
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.reward_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # TODO add termination reward checking

    def _get_gait_phase(self):
        """Add phase into states"""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def _compute_observations(self, envstates):
        """compute observations and priviledged observation"""
        raise NotImplementedError

    def _update_refreshed_tensors(self, env_states):
        """Update tensors from are refreshed tensors after physics step."""
        self.base_quat[:] = robot_rotation_tensor(env_states, self.robot.name)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, robot_velocity_tensor(env_states, self.robot.name))
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, robot_ang_velocity_tensor(env_states, self.robot.name)
        )
        # print(self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

    def _post_physics_step(self, env_states):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        # update episode length from env_wrapper
        self.episode_length_buf = self.env.episode_length_buf_tensor
        self.common_step_counter += 1

        self._post_physics_step_callback()
        # update refreshed tensors from simulaor
        self._update_refreshed_tensors(env_states)
        # prepare all the states for reward computation
        self._parse_state_for_reward(env_states)
        # compute the reward
        self.compute_reward(env_states)
        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        self.reset(reset_env_idx)

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations(env_states)
        self._update_history(env_states)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

    def wrap_action_as_dict(self, actions):
        """
        wrap actions as a dict for the env handler.
        """
        joint_names = list(self.robot.actuators.keys())
        return [
            {"dof_pos_target": {joint_name: float(pos) for joint_name, pos in zip(joint_names, actions[env_id])}}
            for env_id in range(len(actions))
        ]

    def clip_actions(self, actions):
        """Clip actions based on cfg."""
        clip_action_limit = self.cfg.normalization.clip_actions
        return torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

    def _pre_physics_step(self, actions):
        """Apply action smoothing and wrap actions as dict before physics step."""
        # action smoothing
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions.to(self.device) + delay * self.actions
        clipped_actions = self.clip_actions(actions)
        self.actions = clipped_actions
        # action_dict = self.wrap_action_as_dict(clipped_actions)
        return self.actions

    def _physics_step(self, action_dict):
        """
        Task physics step
        """
        env_states, _, terminated, time_out, _ = self.env.step(action_dict)
        self.reset_buf = terminated | time_out
        return env_states

    def step(self, actions):
        """
        Input: actions
        Output: obs, privileged_obs, rewards, dones, infos
        """
        # FIXME return code back
        action_dict = self._pre_physics_step(actions)
        env_states = self._physics_step(action_dict)
        obs, privileged_obs, rewards = self._post_physics_step(env_states)
        return obs, privileged_obs, rewards, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        """
        Reset state in the env and buffer in this wrapper
        """
        # if env_ids is None, reset all envs
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        # if env_ids is empty, do nothing
        if len(env_ids) == 0:
            return

        # TODO
        # update terrain curriculum
        # update command curriculum

        _, _ = self.env.reset(self.init_states, env_ids)

        self._resample_commands(env_ids)

        # reset state in the wrapper
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0.0
        self.base_quat[env_ids] = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(len(env_ids), 1)
        )
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.cfg.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        # log metrics
        self.extras["episode_metrics"] = deepcopy(self.episode_metrics)

        # reset env handler state buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.cfg.command_ranges.lin_vel_x[0],
            self.cfg.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.cfg.command_ranges.lin_vel_y[0],
            self.cfg.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.cfg.command_ranges.heading[0],
                self.cfg.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg.command_ranges.ang_vel_yaw[0],
                self.cfg.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        # TODO modified this name
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * self.wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # TODO: implement terrain height measurement
        # TODO: implement random push force

    # TODO implement this
    def _push_robots(self):
        pass

    # TODO implement this
    def _get_heights(self):
        pass

    @staticmethod
    def get_reward_fn(target: str, reward_functions: list[Callable]) -> Callable:
        fn = next((f for f in reward_functions if f.__name__ == target), None)
        if fn is None:
            raise KeyError(f"No reward function named '{target}'")
        return fn

    # TODO move the utils file
    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float64, n_dims=3):
        """construct arguments to `Vec` according to axis index."""
        zs = np.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = np.where(zs == 1.0, value, zs)
        params[0] = x_value
        return list(params.astype(dtype))

    # TODO move .utils file
    @staticmethod
    def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
        angles %= 2 * np.pi
        angles -= 2 * np.pi * (angles > np.pi)
        return angles
