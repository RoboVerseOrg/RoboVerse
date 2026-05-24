"""Manager-based RL environment base for RoboVerse.

This class is the sim-agnostic generalization of
``roboverse_pack/tasks/beyondmimic/metasim/envs/base_legged_robot.py``.
The legged-robot machinery (sorted-joint maps, action scale/offset,
gravity vector, motion-command instantiation) stays in
``LeggedRobotTask``; this base keeps only the manager loop so non-legged
tasks (cartpole, manipulation, mjlab port) can share the declarative
``ObsTerm/RewTerm/DoneTerm`` config style.

Backward-compat
---------------
- ``LeggedRobotTask`` is **not** modified by this introduction. Existing
  motion-tracking / locomotion tasks keep their current import paths and
  behavior unchanged.
- The class implements the same public surface
  (``obs_buf``, ``priv_obs_buf``, ``num_envs``, ``num_actions``,
  ``max_episode_length``, ``episode_length_buf``, ``device``, ``cfg``,
  ``step`` return tuple, ``extras``) that ``RslRlEnvWrapper`` reads, so
  existing rsl_rl trainers work without modification.

Subclass responsibilities
-------------------------
- Provide a ``ManagerBasedRVEnvCfg``-derived ``cfg`` with
  ``observations / rewards / terminations`` declared via Term classes.
- Implement ``_get_initial_states()``.
- Optionally override ``_process_action()`` and ``_apply_action()``
  if raw policy actions need scaling/offset before reaching the handler.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Sequence

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Info
from metasim.utils import configclass
from metasim.utils.state import TensorState, list_state_to_tensor


@configclass
class ManagerBasedRVEnvCfg:
    """Base config for manager-based RV envs.

    Subclasses declare:
        observations: any ``@configclass`` that exposes named groups
            (each group is itself a ``@configclass`` of ``ObsTerm`` fields,
            e.g. ``observations.policy``, ``observations.critic``).
        rewards: ``@configclass`` of ``RewTerm`` fields.
        terminations: ``@configclass`` of ``DoneTerm`` fields.
        events: optional ``@configclass`` of ``EventTerm`` fields.
        curriculum: optional ``@configclass`` of ``CurriculumTerm`` fields.

    Episode / physics:
        decimation: physics substeps per env step.
        max_episode_length_s: episode time budget in seconds (used to
            derive ``max_episode_steps`` once ``step_dt`` is known).
        is_finite_horizon: passed through to RSL-RL wrapper for time-out
            bootstrap handling.

    Observation group names default to ``("policy", "critic")``. Override
    ``observation_group_names`` to ship more (e.g. an asymmetric setup
    with separate ``rgb`` group).
    """

    decimation: int = 4
    max_episode_length_s: float = 10.0
    is_finite_horizon: bool = False
    observation_group_names: tuple[str, ...] = ("policy", "critic")


class ManagerBasedRVEnv(RLTaskEnv):
    """Sim-agnostic manager-based RL env.

    Lifecycle::

        __init__
            ├─ bind cfg + scenario
            ├─ BaseTaskEnv.__init__ (constructs handler)
            ├─ derive num_actions from handler joints (or subclass override)
            ├─ _init_buffers (term_dones, episode_rewards, ...)
            ├─ _bind_events (parse cfg.events into per-mode dicts)
            ├─ _apply_setup_events
            └─ reset(all envs)

        step(actions):
            ├─ pre_step events
            ├─ _process_action → raw policy action to handler-friendly form
            ├─ for _ in range(decimation): _apply_action + handler.simulate
            ├─ _post_physics_step
            │     ├─ _terminated (manager loop over cfg.terminations)
            │     ├─ _reward (manager loop over cfg.rewards)
            │     ├─ if any reset → _reset_idx + post_step events
            │     ├─ post_step events
            │     └─ _observation (manager loop over cfg.observations groups)
            └─ return (obs_buf, reward, reset_terminated, reset_time_outs, extras)

    No physics is bypassed and no public RLTaskEnv method is overridden
    with semantically different behavior; this is purely a generalization
    of the existing beyondmimic manager loop.
    """

    cfg: ManagerBasedRVEnvCfg

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        scenario: ScenarioCfg,
        cfg: ManagerBasedRVEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.robot = scenario.robots[0] if scenario.robots else None

        # Use BaseTaskEnv init (bypassing RLTaskEnv's reset-on-construction
        # path; managers need to be initialized before we can _observation).
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._observation_space = None
        self._action_space = None
        self.asymmetric_obs = False

        BaseTaskEnv.__init__(self, scenario=scenario, device=device)

        self.num_envs = scenario.num_envs
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.sim_dt = scenario.sim_params.dt
        self.step_dt = self.sim_dt * self.cfg.decimation

        self.extras: dict[str, Any] = {}
        self.common_step_counter = 0

        # Command managers — subclasses may register one or more via
        # ``self.command_managers[name] = manager_instance``. Each manager
        # must implement ``.update()`` (called post-physics) and
        # ``.reset(env_ids)`` (called from _reset_idx). The reward / obs
        # functions read commands via ``env.command_managers[name].current()``.
        self.command_managers: dict[str, Any] = {}

        # Sensor registry — subclasses register ContactSensor /
        # TerrainHeightSensor / BuiltinSensor instances here for reward
        # functions to read. Each sensor must implement ``.update(dt)``
        # (called post-physics, after physics step but before _reward) and
        # ``.reset(env_ids)``.
        self._mjlab_sensors: dict[str, Any] = {}

        self._init_action_dim()
        self._init_buffers()
        self._bind_events()
        self._apply_setup_events()

        # Initial states are computed by the subclass (which knows the
        # robot's default joint pose etc.); cache as tensor state.
        self._initial_states = list_state_to_tensor(
            self.handler, self._get_initial_states(), self.device
        )

        # First reset must happen after managers are wired so we get a
        # valid observation tensor for the wrapper's lazy obs_buf path.
        self.reset(env_ids=list(range(self.num_envs)))

    # ------------------------------------------------------------------
    # subclass hooks (override these)
    # ------------------------------------------------------------------

    def _get_initial_states(self) -> list[dict]:
        """Return a list of per-env initial state dicts.

        Default builds a minimal robot-only template from ``scenario.robots[0]``
        with its ``default_pos``, ``default_rot``, ``default_joint_positions``,
        ``default_joint_velocities``. Subclasses with multiple robots or
        object placement should override.
        """
        if self.robot is None:
            return [{"objects": {}, "robots": {}}] * self.scenario.num_envs

        joint_names = self.handler.get_joint_names(self.robot.name, sort=True)
        joint_pos = dict(self.robot.default_joint_positions)
        joint_vel = dict(self.robot.default_joint_velocities)
        # Restrict to joints the handler actually reports (handles aliases).
        joint_pos = {n: joint_pos.get(n, 0.0) for n in joint_names}
        joint_vel = {n: joint_vel.get(n, 0.0) for n in joint_names}

        template = {
            "objects": {},
            "robots": {
                self.robot.name: {
                    "pos": torch.tensor(self.robot.default_pos, dtype=torch.float32),
                    "rot": torch.tensor(self.robot.default_rot, dtype=torch.float32),
                    "dof_pos": joint_pos,
                    "dof_vel": joint_vel,
                }
            },
        }
        return [deepcopy(template) for _ in range(self.scenario.num_envs)]

    def _process_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map raw policy output to handler-ready action.

        Default: identity (clamp by handler-side joint limits, no scaling).
        Subclasses can implement scale/offset/clip (e.g. legged-robot
        ``action_scale * raw + default_dof_pos``).
        """
        return raw_action

    def _apply_action(self, processed_action: torch.Tensor) -> None:
        """Push processed action into the simulator handler.

        Default: ``handler.set_dof_targets`` (joint-position control,
        the most common case across mjlab tasks). Subclasses that use
        torque or velocity control should override.
        """
        self.handler.set_dof_targets(processed_action)

    # ------------------------------------------------------------------
    # init helpers
    # ------------------------------------------------------------------

    def _init_action_dim(self) -> None:
        """Compute action dim. Subclasses may set ``self.num_actions`` themselves first."""
        if hasattr(self, "num_actions") and self.num_actions is not None:
            return
        if self.robot is None:
            self.num_actions = 0
            return
        actuators = getattr(self.robot, "actuators", None)
        if actuators:
            self.num_actions = len(actuators)
        else:
            self.num_actions = len(self.handler.get_joint_names(self.robot.name, sort=True))

    def _init_buffers(self) -> None:
        # Per-term termination flags
        self._term_dones: dict[str, torch.Tensor] = {}
        for term_name in self._term_names(self.cfg.terminations):
            self._term_dones[term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-term episode reward sums (for logging)
        self.episode_rewards: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self._term_names(self.cfg.rewards)
        }

        # Last action for action-rate-style rewards
        self._prev_action: torch.Tensor | None = None
        self._action: torch.Tensor | None = None

    def _bind_events(self) -> None:
        """Parse ``cfg.events`` (optional) into per-mode callable dicts.

        Format mirrors beyondmimic's callback dicts: each entry is either
        a callable or a ``(callable, params_dict)`` tuple. Supports modes
        ``setup``, ``reset``, ``pre_step``, ``post_step``, ``terminate``.

        If ``cfg.events`` is absent, all per-mode dicts are empty.
        """
        empty: dict = {}
        self.setup_events: dict = dict(empty)
        self.reset_events: dict = dict(empty)
        self.pre_step_events: dict = dict(empty)
        self.post_step_events: dict = dict(empty)
        self.terminate_events: dict = dict(empty)
        self.curriculum_terms: dict = dict(empty)

        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None:
            for name in self._term_names(events_cfg):
                term = getattr(events_cfg, name)
                mode = getattr(term, "mode", "reset")
                entry = (term.func, term.params or {})
                if mode == "setup":
                    self.setup_events[name] = entry
                elif mode == "reset":
                    self.reset_events[name] = entry
                elif mode == "pre_step":
                    self.pre_step_events[name] = entry
                elif mode == "post_step":
                    self.post_step_events[name] = entry
                elif mode == "terminate":
                    self.terminate_events[name] = entry
                else:
                    raise ValueError(f"Unknown event mode '{mode}' for term '{name}'")

        curriculum_cfg = getattr(self.cfg, "curriculum", None)
        if curriculum_cfg is not None:
            for name in self._term_names(curriculum_cfg):
                term = getattr(curriculum_cfg, name)
                self.curriculum_terms[name] = (term.func, term.params or {})

    def _initial_states_has_entities(self) -> bool:
        """Whether ``self._initial_states`` carries any robot or object state.

        Scene-MJCF tasks (e.g. mjlab cartpole loaded as the scene rather
        than as a robot) return empty robot/object dicts; calling
        ``handler.set_states`` with an empty TensorState raises
        ``StopIteration`` inside the handler's state-shape inference,
        so skip the call in that case and let the reset event handle
        state restoration.
        """
        s = self._initial_states
        robots = getattr(s, "robots", {}) or {}
        objects = getattr(s, "objects", {}) or {}
        return bool(robots) or bool(objects)

    @staticmethod
    def _term_names(group_cfg) -> list[str]:
        """Return the ordered list of term-attribute names on a ``@configclass`` group.

        Using ``dataclasses.fields()`` instead of ``asdict()`` preserves
        object identity for term ``params`` entries (otherwise nested
        dataclass params like ``SceneEntityCfg`` get flattened to dicts
        before the term function sees them).
        """
        if not is_dataclass(group_cfg):
            raise TypeError(f"Term group must be a @configclass dataclass, got {type(group_cfg)}")
        return [f.name for f in fields(group_cfg)]

    def _apply_setup_events(self) -> None:
        for func, params in self.setup_events.values():
            func(self, **params)

    # ------------------------------------------------------------------
    # env api
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset selected envs (or all)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.int64, device=self.device)

        self._reset_idx(env_ids)

        env_states = self.handler.get_states(mode="tensor")
        # post-reset events (e.g. push_robot interval reset)
        for func, params in self.post_step_events.values():
            func(self, env_states, **params)
        # Observations after reset
        self._obs_buf = self._observation(self.handler.get_states(mode="tensor"))
        # Lazy: extras['observations'] keeps last obs around for rsl-rl
        self.extras["observations"] = self._obs_buf

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Per-env reset (set initial state, run reset events, zero buffers)."""
        self.extras["episode"] = self.extras.get("episode", {})

        # Reset state to initial (uses cached tensor state from __init__).
        # Skip when no robots/objects are configured — e.g. scene-MJCF tasks
        # (mjlab cartpole) that rely on a reset event for state setup.
        if self._initial_states_has_entities():
            self.handler.set_states(self._initial_states, env_ids=env_ids.tolist())

        # Reset command managers (resample fresh velocity command for the
        # newly-reset envs).
        for mgr in self.command_managers.values():
            try:
                mgr.reset(env_ids)
            except Exception:
                pass

        # Reset sensors (zero air-time / contact-time / history).
        for sensor in self._mjlab_sensors.values():
            try:
                sensor.reset(env_ids)
            except Exception:
                pass

        # Reset events (e.g. random base velocity, randomize joint default).
        for func, params in self.reset_events.values():
            func(self, env_ids, **params)

        # Reset action history.
        if self._prev_action is not None and len(env_ids):
            self._prev_action[env_ids] = 0.0
            self._action[env_ids] = 0.0

        # Log + zero per-term episode reward sums.
        max_ep_s = max(self.cfg.max_episode_length_s, 1e-6)
        for key, buf in self.episode_rewards.items():
            self.extras["episode"][f"Episode_Reward/{key}"] = (
                torch.mean(buf[env_ids]) / max_ep_s
            )
            buf[env_ids] = 0.0

        # Log + leave term-dones — they get overwritten next step.
        for key, buf in self._term_dones.items():
            self.extras["episode"][f"Episode_Termination/{key}"] = (
                torch.count_nonzero(buf[env_ids]).item()
            )

        # Curriculum: each term receives env_ids and returns a scalar/dict.
        for name, (func, params) in self.curriculum_terms.items():
            result = func(self, env_ids, **params)
            if result is not None:
                if isinstance(result, dict):
                    for k, v in result.items():
                        self.extras["episode"][f"Curriculum/{name}/{k}"] = v
                else:
                    self.extras["episode"][f"Curriculum/{name}"] = result

        # Episode step counter reset.
        self._episode_steps[env_ids] = 0

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]:
        """Apply one env step (decimation physics) and return RL signals."""
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.to(self.device)

        if self._action is None:
            self._action = torch.zeros_like(actions)
            self._prev_action = torch.zeros_like(actions)
        self._prev_action[:] = self._action
        self._action[:] = actions

        # Pre-step events (e.g. external force application).
        for func, params in self.pre_step_events.values():
            func(self, **params)

        processed = self._process_action(actions)

        # NB: ``handler.simulate()`` already does ``scenario.decimation`` MuJoCo
        # substeps internally. If ``cfg.decimation`` was also looped here, we'd
        # double-decimate (e.g. 4 × 4 = 16 substeps per env.step) causing
        # multi-DOF robots (Go1) to diverge before the first action lands.
        # The convention is: scenario.decimation = physics substeps per
        # env.step, cfg.decimation = same value (for step_dt computation).
        # Loop once; rely on handler to do all substeps.
        self._apply_action(processed)
        self.handler.simulate()

        self._post_physics_step()

        # rsl-rl expects observation under extras["observations"] too.
        self.extras["observations"] = self._obs_buf
        return (
            self._obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _post_physics_step(self) -> None:
        self._episode_steps += 1
        self.common_step_counter += 1

        # Command managers update (e.g. velocity command resample) — must
        # happen before reward computation so reward sees fresh commands.
        for mgr in self.command_managers.values():
            try:
                mgr.update()
            except Exception:
                pass  # Tolerate misbehaving managers; logged elsewhere.

        # Sensors update — must happen post-physics, before reward + obs.
        for sensor in self._mjlab_sensors.values():
            try:
                sensor.update(self.step_dt)
            except Exception:
                pass

        env_states = self.handler.get_states(mode="tensor")

        # Term / reward first; reset happens before observation so policy sees fresh state.
        self.reset_buf = self._terminated(env_states)
        self.reward_buf = self._reward(env_states)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            self._reset_idx(reset_env_ids)
            for func, params in self.terminate_events.values():
                func(self, reset_env_ids, **params)
            env_states = self.handler.get_states(mode="tensor")

        # Post-step events (interval push, friction noise, ...).
        for func, params in self.post_step_events.values():
            func(self, env_states, **params)

        self._obs_buf = self._observation(self.handler.get_states(mode="tensor"))

    # ------------------------------------------------------------------
    # manager loops
    # ------------------------------------------------------------------

    def _reward(self, env_states: TensorState) -> torch.Tensor:
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name in self._term_names(self.cfg.rewards):
            term = getattr(self.cfg.rewards, name)
            params = term.params or {}
            value = term.func(self, env_states, **params) * term.weight * self.step_dt
            rew_buf += value
            self.episode_rewards[name] += value
        return rew_buf

    def _terminated(self, env_states: TensorState) -> torch.Tensor:
        self.reset_time_outs[:] = False
        self.reset_terminated[:] = False
        for name in self._term_names(self.cfg.terminations):
            term = getattr(self.cfg.terminations, name)
            params = term.params or {}
            value = term.func(self, env_states, **params).bool()
            if getattr(term, "time_out", False):
                self.reset_time_outs |= value
            else:
                self.reset_terminated |= value
            self._term_dones[name][:] = value
        return self.reset_time_outs | self.reset_terminated

    def _observation(self, env_states: TensorState) -> dict[str, torch.Tensor]:
        """Compute observation for every configured group."""
        out: dict[str, torch.Tensor] = {}
        for group_name in self.cfg.observation_group_names:
            out[group_name] = self._observation_group(group_name, env_states)
        return out

    def _observation_group(self, group_name: str, env_states: TensorState) -> torch.Tensor:
        """Concatenate ObsTerm outputs within one group, applying noise."""
        group = getattr(self.cfg.observations, group_name, None)
        if group is None:
            # Permit policy-only observations: critic falls back to policy.
            if group_name == "critic":
                return self._observation_group("policy", env_states)
            raise AttributeError(
                f"cfg.observations has no group '{group_name}' "
                f"(expected one of {self.cfg.observation_group_names})"
            )
        parts = []
        for name in self._term_names(group):
            term = getattr(group, name)
            params = term.params or {}
            obs = term.func(self, env_states, **params).clone()
            # Pipeline mirrors mjlab ObservationManager: noise -> clip -> scale.
            noise_range = getattr(term, "noise_range", None)
            if noise_range is not None:
                lo, hi = noise_range
                obs = obs + (torch.rand_like(obs) * (hi - lo) + lo)
            clip = getattr(term, "clip", None)
            if clip is not None:
                obs = obs.clip(min=clip[0], max=clip[1])
            scale = getattr(term, "scale", None)
            if scale is not None:
                obs = obs * scale
            parts.append(obs)
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # rsl-rl compat properties
    # ------------------------------------------------------------------

    @property
    def obs_buf(self) -> torch.Tensor:
        """Actor (policy) observation tensor for RSL-RL wrapper.

        Accepts either of the two group-naming conventions:
        ``policy`` (RoboVerse beyondmimic legacy) or ``actor`` (mjlab/IsaacLab).
        """
        if isinstance(self._obs_buf, dict):
            for k in ("actor", "policy"):
                if k in self._obs_buf:
                    return self._obs_buf[k]
            return next(iter(self._obs_buf.values()))
        return self._obs_buf

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Critic (privileged) observation tensor, falling back to actor obs if absent."""
        if isinstance(self._obs_buf, dict):
            if "critic" in self._obs_buf:
                return self._obs_buf["critic"]
            return self.obs_buf
        return self._obs_buf

    @property
    def max_episode_steps(self) -> int:
        return math.ceil(self.cfg.max_episode_length_s / self.step_dt)

    @property
    def max_episode_length(self) -> int:
        return self.max_episode_steps

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._episode_steps = value
