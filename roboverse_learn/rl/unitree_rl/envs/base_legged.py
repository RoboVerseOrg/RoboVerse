from __future__ import annotations

from collections import deque
from typing import Callable

import torch

import metasim.types as mstypes
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.humanoid_robot_util import (
    contact_forces_tensor,
    dof_vel_tensor,
    get_euler_xyz_tensor,
    robot_ang_velocity_tensor,
    robot_position_tensor,
    robot_rotation_tensor,
    robot_velocity_tensor,
)
from metasim.queries.base import BaseQueryType
from metasim.utils.math import quat_apply, quat_rotate_inverse, wrap_to_pi
from metasim.utils.state import TensorState
from metasim.task.base import BaseTaskEnv
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper
from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg
from roboverse_learn.rl.unitree_rl.helper.utils import get_body_reindexed_indices_from_substring, torch_rand_float
from metasim.constants import SimType

class LeggedRobot(RslRlWrapper):
    """
    This env define the legged robot base env,
    which canbe put into the RslRlWrapper to be used in the RL training.
    Note that Training only for Gym, Lab, Genesis
    Mujoco can be used fvaluation/render only.
    """

    cfg: BaseLeggedTaskCfg

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._parse_cfg(scenario)
        self._parse_rigid_body_indices(self.cfg.robots[0])
        self._parse_joint_cfg(self.cfg)
        self._prepare_reward_function(self.cfg)
        self._init_buffers()

    def reset(self, env_ids=None):
        """
        Reset state in the env and buffer in this wrapper
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if len(env_ids) == 0:
            return

        env_states, _ = BaseTaskEnv.reset(self, self.init_states, env_ids)

        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._resample_commands(env_ids)

        # reset state buffer in the wrapper
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0.0
        self.base_quat[env_ids] = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
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
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # # log metrics
        # self.extras["episode_metrics"] = deepcopy(self.episode_metrics)

        # reset env handler state buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        return None, None

    def step(self, actions: torch.Tensor):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions: torch.Tensor | list[mstypes.Action] = self._pre_physics_step(actions)
        env_states = self._physics_step(self.actions)
        obs, privileged_obs, reward = self._post_physics_step(env_states)
        ## clip observations
        return obs, privileged_obs, reward, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    # Termination logic
    # ------------------------------------------------------------------
    def _terminated(self, env_states: TensorState) -> torch.Tensor:
        """
        Judge early termination based on contacts and base orientation, matching
        the logic used in RoboVerse. An episode is terminated when either:
        - Any body in `termination_contact_indices` has contact force above a threshold.
        - Base roll/pitch exceeds configured thresholds (approximately fall-over).
        """
        # Contact-based termination
        contact_forces = contact_forces_tensor(env_states, self.robot.name)
        # termination_contact_indices is computed from robot config in _parse_rigid_body_indices
        term_idx = self.termination_contact_indices
        contact_mag = torch.norm(contact_forces[:, term_idx, :], dim=-1)  # [N, K]
        contact_violation = torch.any(contact_mag > 1.0, dim=1)  # [N]

        # Orientation-based termination (roll/pitch)
        base_quat = robot_rotation_tensor(env_states, self.robot.name)
        rpy = get_euler_xyz_tensor(base_quat)  # [N, 3] -> roll (x), pitch (y), yaw (z)
        roll_excess = torch.abs(rpy[:, 0]) > 0.8
        pitch_excess = torch.abs(rpy[:, 1]) > 1.0
        orient_violation = torch.logical_or(roll_excess, pitch_excess)

        return torch.logical_or(contact_violation, orient_violation)

    def _extra_spec(self) -> dict[str, BaseQueryType]:
        return self.cfg.extra_spec

    def compute_reward(self, envstate: TensorState):
        """Compute all the reward from the states provided."""
        self.rew_buf[:] = 0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew_func_return = self.reward_functions[i](envstate, self.robot.name, self.cfg)
            if isinstance(rew_func_return, tuple):
                unscaled_rew, metric = rew_func_return
                # self.episode_metrics[name] = metric.mean().item()
            else:
                unscaled_rew = rew_func_return
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.reward_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    """The necessary functions for the child class to implement"""

    def compute_observations(self, envstate: TensorState):
        """compute observations and priviledged observation"""
        raise NotImplementedError(
            "compute_observations should be implemented in the child class, "
            "e.g. HumanoidWalkingTask, LeggedWalkingTask, etc."
        )

    # region: For step function
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply action smoothing and wrap actions as dict before physics step."""
        # low frequency action smoothing
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions.to(self.device) + delay * self.actions
        # clip actions
        clip_action_limit = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)


        # TODO: add the support of multi-embodiments
        # should return actions_list, [List, Action:[str, RobotAction:[...]]]
        return actions

    def _physics_step(self, actions: torch.Tensor | list[mstypes.Action]):
        """
        Task physics step
        """
        env_states = self.handler.get_states()
        for i in range(self.decimation):
            # Apply PD control if needed
            if self.manual_pd_on:
                # Get current environment states for PD control
                effort = self._apply_pd_control(actions, env_states)
                self.handler._effort = effort
                send_action = effort
            else:
                send_action = actions
            # reverse_reindex = self.handler.get_joint_reindex(obj_name=self.handler.robots[0].name, inverse=True)
            # self.handler.gym.set_dof_position_target_tensor(self.handler.sim, gymtorch.unwrap_tensor(actions[:, reverse_reindex]))
            env_states, _, terminated, self.time_out_buf, _ = BaseTaskEnv.step(self, send_action)
        self.reset_buf = torch.logical_or(terminated, self.time_out_buf)
        return env_states

    def _post_physics_step(self, env_states):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        # update episode length from env_wrapper
        # self.episode_length_buf = self.handler.episode_length_buf
        self.common_step_counter += 1
        # unpack the estimated states from env_states, currently, it only support quaternions
        self._update_odometry_tensors(env_states)
        # update commands & randomizations
        self._post_physics_step_callback()
        # prepare all the states for reward computation
        self._parse_state_for_reward(env_states)
        # compute the reward
        self.compute_reward(env_states)
        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        self.reset(reset_env_idx)
        # simulate the push operation
        if self.cfg.random.push.enabled and self.common_step_counter % self.cfg.random.push.push_interval == 0:
            self._push_robots()
        # compute obs for actor,  privileged_obs for critic network
        self.compute_observations(env_states)
        # clip the observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # copy the last observations
        self._update_history(env_states)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (self._episode_steps % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # quat:[w, x, y, z], forward:[x, y, z]
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

    # endregion

    # region: Randomizations
    def _push_robots(self):
        """Randomly set robot's root velocity to simulate a push."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self._episode_steps[env_ids] % self.cfg.random.push.push_interval == 0]
        if len(push_env_ids) == 0:
            return
        env_states = self.handler.get_states()

        max_vel = self.cfg.random.push.max_push_vel_xy
        env_states.robots[self.robot.name].root_state[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.rand_push_force[:, :2] = env_states.robots[self.robot.name].root_state[:, 7:9]
        max_angular = self.cfg.random.push.max_push_ang_vel
        max_angular = 0
        env_states.robots[self.robot.name].root_state[:, 10:13] = torch_rand_float(
            -max_angular, max_angular, (self.num_envs, 3), device=self.device
        )
        self.rand_push_torque = env_states.robots[self.robot.name].root_state[:, 10:13]
        self.handler.set_states(env_states, push_env_ids.tolist())

    # endregion

    # region: PD Control
    def _compute_effort(self, actions: torch.Tensor, env_states: TensorState) -> torch.Tensor:
        """Compute effort from actions using PD control"""
        # Scale the actions (generally output from policy)
        action_scaled = self.cfg.control.action_scale * actions

        # Get current joint positions and velocities
        sorted_dof_pos = env_states.robots[self.robot.name].joint_pos
        sorted_dof_vel = env_states.robots[self.robot.name].joint_vel


        # Compute PD control effort
        if self.cfg.control.action_offset:
            effort = (
                self.p_gains * (action_scaled + self.cfg.default_joint_pd_target - sorted_dof_pos)
                - self.d_gains * sorted_dof_vel
            )
        else:
            effort = self.p_gains * (action_scaled - sorted_dof_pos) - self.d_gains * sorted_dof_vel

        # Apply torque limits
        effort = torch.clip(effort, -self.cfg.torque_limits, self.cfg.torque_limits)
        return effort.to(torch.float32)

    def _apply_pd_control(self, actions: torch.Tensor, env_states: TensorState) -> torch.Tensor:
        """
        Compute torque using PD controller for effort actuator and return torques.
        """
        effort = self._compute_effort(actions, env_states)
        return effort

    # endregion

    # region: Utilities
    def _update_odometry_tensors(self, env_states):
        """Update tensors from are refreshed tensors after physics step."""
        self.base_pos[:] = robot_position_tensor(env_states, self.robot.name)
        self.base_quat[:] = robot_rotation_tensor(env_states, self.robot.name)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, robot_velocity_tensor(env_states, self.robot.name))
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, robot_ang_velocity_tensor(env_states, self.robot.name)
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _update_history(self, envstate):
        """update history buffer at the the of the frame, called after reset"""
        # we should always make a copy here
        # check whether torch.clone is necessary
        self.last_last_actions[:] = self.last_actions[:].clone()
        self.last_actions[:] = self.actions[:].clone()
        self.last_dof_vel[:] = dof_vel_tensor(envstate, self.robot.name)[:].clone()
        self.last_root_vel[:] = torch.cat((self.base_lin_vel, self.base_ang_vel), dim=1).clone()
        # robot_root_state_tensor(envstate, self.robot.name)[:, 7:13]

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges.lin_vel_x[0],
            self.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges.lin_vel_y[0],
            self.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges.heading[0],
                self.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges.ang_vel_yaw[0],
                self.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    @staticmethod
    def get_reward_fn(target: str, reward_functions: list[Callable] | str) -> Callable:
        if isinstance(reward_functions, (list, tuple)):
            fn = next((f for f in reward_functions if f.__name__ == target), None)
        elif isinstance(reward_functions, str):
            reward_module = __import__(reward_functions, fromlist=[target])
            fn = getattr(reward_module, target, None)
        else:
            raise ValueError("reward_functions should be a list of functions or a string module path")
        if fn is None:
            raise KeyError(f"No reward function named '{target}'")
        return fn

    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
        """construct arguments to `Vec` according to axis index."""
        zs = torch.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = torch.where(zs == 1.0, value, zs)
        params[0] = x_value
        return params.tolist()

    # endregion

    # region: Parse configs & Get the necessary parametres
    def _parse_cfg(self, scenario):
        super()._parse_cfg(scenario)
        self.decimation = self.cfg.decimation = scenario.decimation
        self.dt = self.cfg.dt = self.cfg.sim_params.dt
        self.command_ranges = self.cfg.commands.ranges
        self.num_commands = self.cfg.commands.commands_dim
        self.use_vision = self.cfg.use_vision

    def _parse_rigid_body_indices(self, robot):
        """
        Parse rigid body indices from robot cfg.
        """
        feet_names = robot.feet_links
        termination_contact_names = robot.terminate_contacts_links
        penalised_contact_names = robot.penalized_contacts_links

        # get sorted indices for specific body links
        self.feet_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, feet_names, device=self.device
        )
        if SimType(self.scenario.simulator) is SimType.ISAACSIM:
            names = self.handler.contact_sensor.body_names
            termination_contact_indices = []
            for i, body_name in enumerate(names):
                for term_name in termination_contact_names:
                    if term_name in body_name:
                        termination_contact_indices.append(i)
            self.termination_contact_indices = torch.tensor(termination_contact_indices, device=self.device)
        elif SimType(self.scenario.simulator) is SimType.ISAACGYM:
            self.termination_contact_indices = get_body_reindexed_indices_from_substring(
                self.handler, robot.name, termination_contact_names, device=self.device
            )
        elif SimType(self.scenario.simulator) is SimType.MUJOCO:
            self.termination_contact_indices = get_body_reindexed_indices_from_substring(
                self.handler, robot.name, termination_contact_names, device=self.device
            )
        else:
            raise NotImplementedError(f"Simulator {self.scenario.simulator} not supported yet.")
        self.penalised_contact_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, penalised_contact_names, device=self.device
        )
        # attach to cfg for reward computation.
        self.cfg.feet_indices = self.feet_indices
        self.cfg.termination_contact_indices = self.termination_contact_indices
        self.cfg.penalised_contact_indices = self.penalised_contact_indices

    def _parse_joint_cfg(self, cfg: BaseLeggedTaskCfg):
        """
        parse default joint positions and torque limits from cfg.
        """
        torque_limits = (
            cfg.robots[0].torque_limits
            if hasattr(cfg.robots[0], "torque_limits")
            else {name: actuator_cfg.torque_limit for name, actuator_cfg in cfg.robots[0].actuators.items()}
        )
        # sorted_joint_names = sorted(torque_limits.keys())
        sorted_joint_names = self.handler.get_joint_names(self.robot.name, sort=True)
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.cfg.torque_limits = (
            torch.tensor(sorted_limits, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            * self.cfg.control.torque_limit_scale
        )

        dof_pos_limits = cfg.robots[0].joint_limits
        sorted_dof_pos_limits = [dof_pos_limits[joint] for joint in sorted_joint_names]
        self.cfg.dof_pos_limits = torch.tensor(sorted_dof_pos_limits, device=self.device)  # [n_joints, 2]
        # soft constraints
        _mid = (self.cfg.dof_pos_limits[:, 0] + self.cfg.dof_pos_limits[:, 1]) / 2.0
        _diff = self.cfg.dof_pos_limits[:, 1] - self.cfg.dof_pos_limits[:, 0]
        self.cfg.dof_pos_limits[:, 0] = _mid - 0.5 * _diff * self.cfg.reward_cfg.soft_dof_pos_limit
        self.cfg.dof_pos_limits[:, 1] = _mid + 0.5 * _diff * self.cfg.reward_cfg.soft_dof_pos_limit

        default_joint_pos = cfg.robots[0].default_joint_positions
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.cfg.default_joint_pd_target = torch.tensor(sorted_joint_pos, device=self.device).unsqueeze(0)

        # Parse PD gains for manual PD control, in sorted joint order
        actuators = cfg.robots[0].actuators
        p_gains = []
        d_gains = []
        for name in sorted_joint_names:
            actuator_cfg = actuators[name]
            p_gains.append(actuator_cfg.stiffness if actuator_cfg.stiffness is not None else 0.0)
            d_gains.append(actuator_cfg.damping if actuator_cfg.damping is not None else 0.0)

        self.p_gains = torch.tensor(p_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.d_gains = torch.tensor(d_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Check if manual PD control is needed (if any joints use effort control)
        control_types = cfg.robots[0].control_type
        self.manual_pd_on = any(mode == "effort" for mode in control_types.values()) if control_types else False

    def _init_buffers(self):
        """
        Init all buffer for reward computation
        """
        self.up_axis_idx = 2
        self.base_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
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

        self.common_step_counter = 0
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        ## align episode_steps with episode_length_buf, they are equal
        self._episode_steps = self.episode_length_buf
        self.max_episode_steps = self.max_episode_length

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

        # history buffer for reward computation
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)

        self.last_feet_z = 0.05 * torch.ones(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, requires_grad=False)
        self.feet_height = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, requires_grad=False)

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

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

        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)  # TODO now set 0
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)


    # endregion

    # region: Parse states for reward computation
    def _prepare_reward_function(self, task: BaseLeggedTaskCfg):
        """Prepares a list of reward functions, which will be called to compute the total reward."""

        self.reward_scales = task.reward_weights
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
        # self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def _parse_state_for_reward(self, envstate: TensorState):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """
        _state = envstate.robots[self.robot.name]  # weak reference

        """Adds the current action to state."""
        _state.extra["actions"] = self.actions

        """update history buffer, must be called after reset"""
        _state.extra["last_root_vel"] = self.last_root_vel
        _state.extra["last_dof_vel"] = self.last_dof_vel
        _state.extra["last_actions"] = self.last_actions
        _state.extra["last_last_actions"] = self.last_last_actions

        """Adds the current base euler angles to state."""
        # self.base_euler_xyz = get_euler_xyz_tensor(envstate.robots[self.robot.name].root_state[:, 3:7])
        _state.extra["base_euler_xyz"] = self.base_euler_xyz

        """Adds the scaled command to state."""
        _state.extra["command"] = self.commands

        """add the project gravity to state"""
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        _state.extra["projected_gravity"] = self.projected_gravity

        """Add local base velocity into states"""
        _state.extra["base_lin_vel"] = self.base_lin_vel
        _state.extra["base_ang_vel"] = self.base_ang_vel

        self._parse_feet_air_time(envstate)

    def _parse_feet_air_time(self, envstate: TensorState):
        contact = contact_forces_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt * self.decimation
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        envstate.robots[self.robot.name].extra["feet_air_time"] = air_time

    # endregion
