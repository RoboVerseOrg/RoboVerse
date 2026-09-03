from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Info, StateMode, StateOutput
from metasim.utils.state import TensorState, list_state_to_tensor
from roboverse_pack.robots.g1_tracking import G1TrackingCfg
from roboverse_pack.tasks.beyondmimic.metasim.configs.cfg_base import BaseEnvCfg, CallbacksCfg
from roboverse_pack.tasks.beyondmimic.metasim.mdp.commands import MotionCommand
from roboverse_pack.tasks.beyondmimic.metasim.utils.misc import get_axis_params
from roboverse_pack.tasks.beyondmimic.metasim.utils.string import find_bodies, pattern_match


class LeggedRobotTask(RLTaskEnv):
    """Base environment for legged robots."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: BaseEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = config
        _callbacks_cfg = asdict(getattr(self.cfg, "callbacks", CallbacksCfg()))
        self._query: dict = _callbacks_cfg.pop("query", {})
        self.robot = scenario.robots[0]
        BaseTaskEnv.__init__(self, scenario=scenario, device=device)
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)

        self.extras: dict[str, Any] = {}
        self.extras_buffer: dict[str, any] = {}

        # callbacks
        self._bind_callbacks(callbacks=_callbacks_cfg)

        self.name = self.robot.name
        self.total_action_dim = len(self.robot.actuators)
        self.sim_dt = self.scenario.sim_params.dt
        self.sorted_body_names = self.handler.get_body_names(self.name, sort=True)
        self.sorted_joint_names = self.handler.get_joint_names(self.name, sort=True)
        self.original_joint_names = self.handler.get_joint_names(self.name, sort=False)

        self._init_joint_cfg()
        self._instantiate_cfg()
        self._init_buffers()
        self._setup()
        # self.reset()

    def _bind_callbacks(self, callbacks: dict | None = None):
        for _callbacks in callbacks.values():
            for _key, _val in _callbacks.items():
                if not isinstance(_val, tuple):
                    assert callable(_val) or isinstance(_val, object)
                    _callbacks[_key] = (_val, {})
                if hasattr(_callbacks[_key][0], "bind_handler"):
                    _callbacks[_key][0].bind_handler(self.handler)

        self.setup_callback = callbacks.pop("setup", {})
        assert isinstance(self.setup_callback, dict)
        self.reset_callback = callbacks.pop("reset", {})
        assert isinstance(self.reset_callback, dict)
        self.pre_physics_step_callback = callbacks.pop("pre_step", {})
        assert isinstance(self.pre_physics_step_callback, dict)
        self.post_physics_step_callback = callbacks.pop("post_step", {})
        assert isinstance(self.post_physics_step_callback, dict)
        self.terminate_callback = callbacks.pop("terminate", {})
        assert isinstance(self.terminate_callback, dict)

    def _init_joint_cfg(self):
        """Set position limits, default joint positions, and default joint velocities."""
        robot: G1TrackingCfg = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names
        original_joint_names: list[str] = self.original_joint_names

        self.sorted_to_original_joint_indexes = torch.tensor(
            find_bodies(original_joint_names, sorted_joint_names, preserve_order=True)[0], device=self.device
        )
        self.original_to_sorted_joint_indexes = torch.tensor(
            find_bodies(sorted_joint_names, original_joint_names, preserve_order=True)[0], device=self.device
        )

        # set position limits
        dof_pos_limits = robot.joint_limits
        sorted_dof_pos_limits = [dof_pos_limits[joint] for joint in sorted_joint_names]
        self.dof_pos_limits = torch.tensor(sorted_dof_pos_limits, device=self.device)  # (n_dofs, 2)

        soft_limit_factor = getattr(robot, "soft_joint_pos_limit_factor", 0.9)
        _mid = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2.0
        _diff = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]

        # NOTE same as `ArticulationData.soft_joint_pos_limits` in Isaac Lab
        soft_dof_pos_limits = torch.zeros_like(self.dof_pos_limits, device=self.device)
        soft_dof_pos_limits[:, 0] = _mid - 0.5 * _diff * soft_limit_factor
        soft_dof_pos_limits[:, 1] = _mid + 0.5 * _diff * soft_limit_factor
        self.soft_dof_pos_limits_sorted = soft_dof_pos_limits.unsqueeze(0).repeat(
            self.num_envs, 1, 1
        )  # (n_envs, n_dofs, 2)
        self.soft_dof_pos_limits_original = self.soft_dof_pos_limits_sorted[:, self.sorted_to_original_joint_indexes, :]

        # set default joint positions
        default_joint_pos = robot.default_joint_positions
        default_joint_pos = pattern_match(default_joint_pos, sorted_joint_names)
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        default_dof_pos = torch.tensor(sorted_joint_pos, device=self.device)  # (n_dofs,)
        self.default_dof_pos_sorted = default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)  # (n_envs, n_dofs)
        self.default_dof_pos_original = self.default_dof_pos_sorted[:, self.sorted_to_original_joint_indexes]

        # set default joint velocities
        default_joint_vel = robot.default_joint_velocities
        default_joint_vel = pattern_match(default_joint_vel, sorted_joint_names)
        sorted_joint_vel = [default_joint_vel[name] for name in sorted_joint_names]
        default_dof_vel = torch.tensor(sorted_joint_vel, device=self.device)  # (n_dofs,)
        self.default_dof_vel_sorted = default_dof_vel.unsqueeze(0).repeat(self.num_envs, 1)  # (n_envs, n_dofs)

    def _instantiate_cfg(self):
        self.decimation = self.cfg.decimation
        self.step_dt = self.sim_dt * self.decimation

        # NOTE actions, action scale, and action offset are in the original order
        self.action_clip = self.robot.action_clip
        self.action_offset = self.default_dof_pos_original.clone() if self.robot.action_offset else 0.0

        # per-actuator action scale
        self.action_scale = (
            torch.tensor(list(self.robot.action_scale.values()), device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.common_step_counter = 0  # counter for curriculum
        self.commands = MotionCommand(env=self, cfg=self.cfg.commands)

    def _init_buffers(self):
        self._action = torch.zeros(size=(self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # prepare extra info to store individual termination term information
        self._term_dones = dict()
        for term_name in asdict(self.cfg.terminations).keys():
            self._term_dones[term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # record terminated envs for adapting sampling
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # set gravity vector
        self.up_axis_idx = 2
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx),
            dtype=torch.float,
            device=self.device,
        ).repeat((self.num_envs, 1))

        # reset commands

        # for logging
        self.episode_rewards = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in asdict(self.cfg.rewards).keys()
        }

    def _setup(self):
        """Apply domain randomization of start-up mode."""
        for _setup_fn, _params in self.setup_callback.values():
            _setup_fn(env=self, **_params)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply pre-physics callbacks."""
        for pre_fn, _params in self.pre_physics_step_callback.values():
            pre_fn(self, **_params)
        # NOTE corresponds to action clipping in `RslRlVecEnvWrapper.step()` in Isaac Lab
        return torch.clip(actions, -self.action_clip, self.action_clip) if self.action_clip else actions

    def reset(
        self,
        states: TensorState | None = None,
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        """Reset the specified environments.

        Args:
            states: Optional external states to set for the selected envs, per the
                ``BaseTaskEnv.reset`` contract. If None, the reset events own the initial
                state. IL/VLA eval runners replay demos through this argument.
            env_ids: The environment ids to reset (defaults to all).
            seed: Optional reproducibility seed, forwarded to the handler when supported
                (this base reimplements ``reset`` and does not call ``super().reset``),
                honoring the ``env.reset(seed=)`` contract.
        """
        if seed is not None:
            set_seed = getattr(self.handler, "set_seed", None)
            if callable(set_seed):
                set_seed(seed)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # reset state of scene (reset events randomize the initial state)
        self._reset_idx(env_ids)

        # caller-supplied states win over the reset events, so demo replay is exact
        if states is not None:
            self.handler.set_states(states=states, env_ids=env_ids)

        # update articulation kinematics
        # self.handler.scene.write_data_to_sim()
        # self.handler.sim.forward()

        # update commands
        self.commands.compute(self.handler.get_states(mode="tensor"))
        env_states = self.handler.get_states(mode="tensor")

        # step interval events
        for _step_fn, _params in self.post_physics_step_callback.values():
            _step_fn(self, env_states, **_params)

        # compute observations after resets
        self._obs_buf = self._observation(self.handler.get_states(mode="tensor"))

    def _reset_idx(self, env_ids: torch.Tensor | list[int] | None = None):
        """Reset selected envs (defaults to all)."""
        if env_ids is None:
            env_ids = torch.tensor(list(range(self.num_envs)), device=self.device)

        self.extras["episode"] = {}

        # update the curriculum for environments that need a reset

        # reset the internal buffers of the scene elements (actions, sensors, etc.)
        # TODO reset contact sensor (currently not supported by MetaSim)

        # apply events such as randomizations for environments that need a reset
        for _reset_fn, _params in self.reset_callback.values():
            _ = _reset_fn(self, env_ids, **_params)

        # reset actions
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0

        # reset rewards
        for key in self.episode_rewards.keys():
            self.extras["episode"]["Episode_Reward/" + key] = (
                torch.mean(self.episode_rewards[key][env_ids]) / self.cfg.max_episode_length_s
            )
            self.episode_rewards[key][env_ids] = 0.0

        # reset curriculum

        # reset commands
        metrics = self.commands.reset(env_ids=env_ids)
        for metric_name, metric_value in metrics.items():
            self.extras["episode"][f"Metrics_Motion/{metric_name}"] = metric_value

        # reset events

        # reset terminations
        for key in self._term_dones.keys():
            self.extras["episode"]["Episode_Termination/" + key] = torch.count_nonzero(
                self._term_dones[key][env_ids]
            ).item()

        # reset the episode length buffer
        self._episode_steps[env_ids] = 0

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]:
        """Apply actions, simulate for `decimation` steps, and compute RLTask-style outputs."""
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # NOTE `actions` is in the original order since it's computed inside `OnPolicyRunner`
        actions = self._pre_physics_step(actions)
        self._prev_action[:] = self._action
        self._action[:] = actions.to(self.device)

        processed_actions = self._action * self.action_scale + self.action_offset
        if self.action_clip is not None:
            processed_actions = processed_actions.clip(-self.action_clip, self.action_clip)
        processed_actions = processed_actions.clone()[:, self.original_to_sorted_joint_indexes]  # sorted order

        for _ in range(self.decimation):
            env_states = self._physics_step(processed_actions)

        self._post_physics_step(env_states)

        # NOTE for RSL-RL v2.3.0 (needed in env wrapper)
        self.extras["observations"] = self._obs_buf

        return (
            self._obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _post_physics_step(self, env_states: TensorState):
        self._episode_steps += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # check termination conditions and compute rewards
        self.reset_buf = self._terminated(env_states)
        self.reward_buf = self._reward(env_states)

        # reset envs and MDP
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(env_ids=reset_env_ids)

            # update articulation kinematics
            # self.handler.scene.write_data_to_sim()
            # self.handler.sim.forward()

        # update commands
        self.commands.compute(self.handler.get_states(mode="tensor"))
        env_states = self.handler.get_states(mode="tensor")

        # step interval events
        for _step_fn, _params in self.post_physics_step_callback.values():
            _step_fn(self, env_states, **_params)

        # compute observations after resets
        self._obs_buf = self._observation(self.handler.get_states(mode="tensor"))

    def _physics_step(self, actions: torch.Tensor) -> TensorState:
        """Issue low-level actions and simulate one physics step."""
        # FIXME both `set_dof_targets()` and `simulate()` call `articulation.write_data_to_sim()`
        self.handler.set_dof_targets(actions)
        self.handler.simulate()
        return self.handler.get_states(mode="tensor")

    def _reward(self, env_states: TensorState):
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, term_cfg in asdict(self.cfg.rewards).items():
            value = term_cfg["func"](self, env_states, **term_cfg["params"]) * term_cfg["weight"] * self.step_dt
            rew_buf += value  # update total reward
            self.episode_rewards[name] += value  # update episodic sum

        return rew_buf

    def _terminated(self, env_states: TensorState | None) -> torch.BoolTensor:
        self.reset_time_outs[:] = False
        self.reset_terminated[:] = False
        for name, term_cfg in asdict(self.cfg.terminations).items():
            value = term_cfg["func"](self, env_states, **term_cfg["params"])
            # store timeout signal separately
            if term_cfg["time_out"]:
                self.reset_time_outs |= value
            else:
                self.reset_terminated |= value
            # add to episode dones
            self._term_dones[name][:] = value

        return self.reset_time_outs | self.reset_terminated

    def _observation(self, env_states: TensorState) -> dict[str, torch.Tensor]:
        """Return a dictionary with keys "policy" and "critic", each corresponding to a tensor."""
        raise NotImplementedError

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """For compatibility with Isaac Lab native RSL-RL env wrapper."""
        return self._observation(self.handler.get_states(mode="tensor"))

    def _get_initial_states(self):
        """Return list of per-env initial states derived from config."""
        sorted_joint_names = self.handler.get_joint_names(self.robot.name, sort=True)

        pos = self.robot.default_pos
        rot = self.robot.default_rot
        joint_pos = self.robot.default_joint_positions
        joint_pos = pattern_match(joint_pos, sorted_joint_names)

        joint_vel = self.robot.default_joint_velocities
        joint_vel = pattern_match(joint_vel, sorted_joint_names)

        template = {
            "objects": {},
            "robots": {
                self.robot.name: {
                    "pos": torch.tensor(pos, dtype=torch.float32),
                    "rot": torch.tensor(rot, dtype=torch.float32),
                    "dof_pos": {name: joint_pos[name] for name in joint_pos},
                    "dof_vel": {name: joint_vel[name] for name in joint_vel},
                }
            },
        }
        return [deepcopy(template) for _ in range(self.scenario.num_envs)]

    def get_states(self, env_ids: list[int] | None = None, mode: StateMode = "dict") -> StateOutput:
        """Get the current simulator state through the handler state-mode contract."""
        return self.handler.get_states(env_ids=env_ids, mode=mode)

    def set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set simulator state for selected env indexes."""
        self.handler.set_states(states=states, env_ids=env_ids)

    def _extra_spec(self) -> dict:
        """Expose optional sensor queries to the simulator handler."""
        return self._query

    @property
    def max_episode_steps(self):
        """Maximum episode length in steps."""
        return math.ceil(self.cfg.max_episode_length_s / self.step_dt)

    @property
    def obs_buf(self) -> torch.Tensor:
        """Policy (actor) observations in shape (num_envs, num_obs)."""
        return self._obs_buf["policy"]

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Critic observations in shape (num_envs, num_priv_obs)."""
        return self._obs_buf["critic"]

    # for backward compatibility with RSL-RL env wrapper

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Current episode lengths of each env. Used in time-out computation."""
        return self._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._episode_steps = value

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return self.max_episode_steps

    @property
    def num_actions(self) -> int:
        """Total dimension of actions."""
        return self.total_action_dim
