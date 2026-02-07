from __future__ import annotations

import copy
from typing import Any, Sequence

import gymnasium as gym
import torch

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.setup_util import get_sim_handler_class

from .contract import WarnOnce
from .managers import (
    CompatActionManager,
    CompatCommandManager,
    CompatCurriculumManager,
    CompatEventManager,
    CompatObservationManager,
    CompatRecorderManager,
    CompatRewardManager,
    CompatTerminationManager,
)
from .scene import CompatScene
from .sensor_registry import SensorRegistry
from .utils import resolve_scene_entity_cfgs


class HandlerBackedManagerBasedRLEnv(BaseTaskEnv, gym.Env):
    """A MetaSim handler-backed runner for IsaacLab manager-based env cfgs.

    This env executes IsaacLab-style term functions/classes against a MetaSim
    `BaseSimHandler`, enabling cross-simulator execution.
    """

    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(
        self,
        *,
        scenario: ScenarioCfg,
        cfg: Any,
        args: Any,
        device: str | torch.device | None = None,
        asset_name_map: dict[str, str] | None = None,
        strict: bool = False,
        reset_in_env_wrapper: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.args = args
        self.strict = strict
        self.render_mode = render_mode
        self._warn_once = WarnOnce()

        # Patch cfg knobs from scenario (num_envs is the source of truth in MetaSim training scripts).
        if getattr(self.cfg, "scene", None) is not None and hasattr(self.cfg.scene, "num_envs"):
            self.cfg.scene.num_envs = int(scenario.num_envs)

        # Patch scenario runtime knobs from cfg (dt/decimation/env_spacing).
        scenario_copy = copy.deepcopy(scenario)
        try:
            scenario_copy.__post_init__()
        except Exception:
            pass

        # In IsaacLab, `decimation` is handled in the env step-loop. To match that semantics across
        # MetaSim handlers (which otherwise implement decimation internally), we instantiate handlers
        # with `decimation=1` and run an explicit decimation loop in `step()`.
        self.decimation = int(getattr(self.cfg, "decimation", getattr(scenario_copy, "decimation", 1)) or 1)
        scenario_copy.decimation = 1

        sim_cfg = getattr(self.cfg, "sim", None)
        if sim_cfg is not None and getattr(sim_cfg, "dt", None) is not None:
            scenario_copy.sim_params.dt = float(sim_cfg.dt)
        elif getattr(scenario_copy.sim_params, "dt", None) is None:
            # Keep a common IsaacLab convention: dt * decimation ≈ 0.015s per env step.
            scenario_copy.sim_params.dt = 0.015 / float(max(1, self.decimation))

        scene_cfg = getattr(self.cfg, "scene", None)
        if scene_cfg is not None and getattr(scene_cfg, "env_spacing", None) is not None:
            scenario_copy.env_spacing = float(scene_cfg.env_spacing)

        # Optional queries / sensors (backend-gated)
        self._sensor_registry = SensorRegistry(strict=self.strict, warn_once=self._warn_once)
        self._extra_queries, self._contact_sensor_plan = self._sensor_registry.plan_optional_queries(
            cfg=self.cfg, scenario=scenario_copy
        )

        # BaseTaskEnv creates the handler and launches it.
        super().__init__(scenario=scenario_copy, device=device)

        # Timing
        dt = getattr(scenario_copy.sim_params, "dt", None)
        if dt is None:
            # Fall back to a common IsaacLab convention (IsaacSim handler uses dt * decimation = 0.015 by default).
            dt = 0.015 / float(max(1, self.decimation))
        self.physics_dt = float(dt)
        self.step_dt = float(dt) * float(max(1, self.decimation))

        # Compatibility surface
        self.scene = CompatScene(
            handler=self.handler, scenario=scenario_copy, asset_name_map=asset_name_map, device=self.device
        )

        self._sensor_registry.setup_scene_sensors(env=self, cfg=self.cfg, plan=self._contact_sensor_plan)

        # Resolve SceneEntityCfg patterns into indices once up-front (used by many IsaacLab term functions).
        resolve_scene_entity_cfgs(self.cfg, scene=self.scene)

        # Managers
        self.command_manager = CompatCommandManager(
            getattr(self.cfg, "commands", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.action_manager = CompatActionManager(
            getattr(self.cfg, "actions", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.observation_manager = CompatObservationManager(
            getattr(self.cfg, "observations", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.reward_manager = CompatRewardManager(
            getattr(self.cfg, "rewards", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.termination_manager = CompatTerminationManager(
            getattr(self.cfg, "terminations", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.curriculum_manager = CompatCurriculumManager(
            getattr(self.cfg, "curriculum", object()), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.event_manager = CompatEventManager(
            getattr(self.cfg, "events", None), env=self, strict=self.strict, warn_once=self._warn_once
        )
        self.recorder_manager = CompatRecorderManager(
            getattr(self.cfg, "recorders", None), env=self, strict=self.strict, warn_once=self._warn_once
        )

        # IsaacLab-like observation buffers
        self._obs_buf: dict[str, torch.Tensor] = {"policy": torch.zeros((self.num_envs, 0), device=self.device)}

        # Public extras dict (used by training wrappers)
        self.extras: dict[str, Any] = {}

        if not reset_in_env_wrapper:
            self.reset()

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """Instantiate the MetaSim handler.

        For the `isaacsim` backend, we may have already created a shared IsaacLab/IsaacSim
        SimulationApp to make IsaacLab configs importable. If present, pass it into the
        handler so we don't create a second app instance.
        """
        handler_class = get_sim_handler_class(SimType(scenario.simulator))
        self.handler = handler_class(scenario, self.extra_spec)

        runtime = getattr(self, "_isaaclab_runtime", None)
        if scenario.simulator == "isaacsim" and runtime is not None:
            try:
                self.handler.launch(simulation_app=runtime.simulation_app)
                return
            except TypeError:
                pass
        self.handler.launch()

    # ------------------------------------------------------------------
    # BaseTaskEnv hooks
    # ------------------------------------------------------------------
    def _extra_spec(self):
        return self._extra_queries

    def _get_initial_states(self):
        """Default initial state: use scenario robot defaults."""
        if not self.scenario.robots:
            raise ValueError("Scenario has no robots; cannot build initial states.")

        robot = self.scenario.robots[0]
        robot_name = robot.name
        joint_names = self.handler.get_joint_names(robot_name, sort=True)

        # Some robot cfgs use `default_pos/default_rot` rather than BaseObjCfg fields.
        pos = getattr(robot, "default_pos", None) or getattr(robot, "default_position", (0.0, 0.0, 0.0))
        rot = getattr(robot, "default_rot", None) or getattr(robot, "default_orientation", (1.0, 0.0, 0.0, 0.0))

        # Default joint positions/velocities may be regex-pattern dicts.
        default_joint_pos = getattr(robot, "default_joint_positions", None) or {}
        default_joint_vel = getattr(robot, "default_joint_velocities", None) or {".*": 0.0}

        def _resolve(pattern_dict: dict[str, float], names: list[str]) -> dict[str, float]:
            out = {n: 0.0 for n in names}
            for pat, val in pattern_dict.items():
                for n in names:
                    if torch.jit.is_scripting():  # pragma: no cover
                        continue
                    import re

                    if re.fullmatch(pat, n):
                        out[n] = float(val)
            return out

        jpos = _resolve(default_joint_pos, joint_names)
        jvel = _resolve(default_joint_vel, joint_names)

        template = {
            "objects": {},
            "robots": {
                robot_name: {
                    "pos": torch.tensor(pos, dtype=torch.float32),
                    "rot": torch.tensor(rot, dtype=torch.float32),
                    "dof_pos": {jn: jpos[jn] for jn in joint_names},
                    "dof_vel": {jn: jvel[jn] for jn in joint_names},
                }
            },
            "cameras": {},
            "extras": {},
        }
        return [copy.deepcopy(template) for _ in range(self.scenario.num_envs)]

    # ------------------------------------------------------------------
    # IsaacLab / RSL-RL compatibility properties
    # ------------------------------------------------------------------
    @property
    def obs_buf(self) -> torch.Tensor:
        return self._obs_buf["policy"]

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        # IsaacLab tasks commonly name privileged observations as "critic" or "privileged".
        if "critic" in self._obs_buf:
            return self._obs_buf["critic"]
        if "privileged" in self._obs_buf:
            return self._obs_buf["privileged"]
        return torch.zeros((self.num_envs, 0), device=self.device)

    @property
    def num_actions(self) -> int:
        return int(getattr(self.action_manager, "total_action_dim", 0))

    @property
    def max_episode_steps(self) -> int:
        # IsaacLab uses `episode_length_s` / step_dt when present.
        ep_s = getattr(self.cfg, "episode_length_s", None)
        if ep_s is None:
            return int(getattr(BaseTaskEnv, "max_episode_steps", 100))
        return int(torch.ceil(torch.tensor(float(ep_s) / float(self.step_dt))).item())

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds (IsaacLab surface)."""
        ep_s = getattr(self.cfg, "episode_length_s", None)
        if ep_s is not None:
            return float(ep_s)
        return float(self.max_episode_steps) * float(self.step_dt)

    # IsaacLab compatibility aliases
    @property
    def max_episode_length(self) -> int:
        return self.max_episode_steps

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._episode_steps = value

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None, env_ids: Sequence[int] | None = None):
        if seed is not None:
            torch.manual_seed(int(seed))

        # Clear any stale episodic logs (IsaacLab uses extras["log"] for reset stats).
        self.extras.pop("log", None)

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids_list = list(env_ids)
        ids = torch.tensor(env_ids_list, dtype=torch.long, device=self.device)

        # Recorder hook: pre-reset (before state changes)
        self.recorder_manager.record_pre_reset(ids)

        # Reset simulator state
        self.handler.set_states(states=self._initial_states, env_ids=env_ids_list)
        try:
            self.handler.refresh_render()
        except Exception:
            pass

        # Reset query histories (important for contact history terms)
        for query in getattr(self.handler, "optional_queries", {}).values():
            reset_fn = getattr(query, "reset", None)
            if callable(reset_fn):
                reset_fn(env_ids_list)

        # Reset episode length buf for selected envs
        self._episode_steps[ids] = 0

        # Refresh state before any command sampling that touches `env.scene[...]`.
        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)

        # Reset command terms (sampling uses termination_manager flags, which are false on manual reset)
        cmd_metrics = self.command_manager.reset(ids)
        if cmd_metrics:
            metrics = self.extras.setdefault("metrics", {})
            if isinstance(metrics, dict):
                for k, v in cmd_metrics.items():
                    metrics[f"command/{k}"] = v
        self.command_manager.compute()

        # Refresh state after possible command-side state writes
        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)

        # Ensure action buffers exist at reset so obs terms like `mdp.last_action` have stable shapes.
        self._reset_action_buffers(ids)

        # Apply startup events once (best-effort) after the compat scene is initialized.
        if getattr(self.event_manager, "_startup_applied", False) is False:
            self.event_manager.apply_startup()
            # Refresh state after startup events may have written to sim.
            full_state = self.handler.get_states()
            self.scene.update_from_states(full_state)

        self._obs_buf = self.observation_manager.compute()
        self.extras["observations"] = self._obs_buf
        self.recorder_manager.record_post_reset(ids)
        info = {"privileged_observation": self.priv_obs_buf}
        return self._obs_buf, info

    def step(self, actions: torch.Tensor):
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # Clear any stale episodic logs from previous resets.
        self.extras.pop("log", None)

        # Cache prev_action/action for MDP terms (IsaacLab semantics)
        action_payload = self.action_manager.process(actions)

        self.recorder_manager.record_pre_step()

        # Apply low-level action and step physics with explicit decimation (IsaacLab semantics).
        for _ in range(int(max(1, self.decimation))):
            self.handler.set_dof_targets(action_payload)
            self.handler.simulate()
            self.recorder_manager.record_post_physics_decimation_step()

        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)

        # Reward/termination evaluated against current commands (updated last step/reset).
        rewards = self.reward_manager.compute()
        terminated, time_outs = self.termination_manager.compute()

        self._episode_steps = self._episode_steps + 1

        # Auto-reset terminated envs (IsaacLab-style vector env semantics).
        done = torch.logical_or(terminated, time_outs)
        reset_ids = torch.where(done)[0]

        # Recorder hook: post-step before any resets.
        self.recorder_manager.record_post_step()

        if reset_ids.numel() > 0:
            # Mark termination flags for sampling-based command terms (e.g., MotionCommand adaptive sampling).
            # termination_manager already holds per-env flags.
            self._reset_envs(reset_ids)

        # Update commands for next-step observations (may write to sim).
        self.command_manager.compute()

        # Refresh state after command-side writes, then apply interval events and compute obs.
        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)
        self.event_manager.step()

        self._obs_buf = self.observation_manager.compute()
        self.extras["observations"] = self._obs_buf

        info = {"privileged_observation": self.priv_obs_buf}
        return self._obs_buf, rewards, terminated, time_outs, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_envs(self, env_ids: torch.Tensor):
        env_ids_list = env_ids.detach().cpu().to(dtype=torch.long).tolist()
        if not env_ids_list:
            return

        # Allocate a fresh episodic log dict (IsaacLab semantics: returned on reset).
        self.extras["log"] = {}

        # Recorder hook: pre-reset
        self.recorder_manager.record_pre_reset(env_ids)

        # Curriculum: update before reset (best-effort IsaacLab semantics)
        try:
            self.curriculum_manager.compute(env_ids=env_ids)
        except Exception:
            pass

        self.handler.set_states(states=self._initial_states, env_ids=env_ids_list)

        # Reset optional query histories to avoid leakage across episodes.
        for query in getattr(self.handler, "optional_queries", {}).values():
            reset_fn = getattr(query, "reset", None)
            if callable(reset_fn):
                reset_fn(env_ids_list)

        ids = torch.tensor(env_ids_list, dtype=torch.long, device=self.device)
        self._episode_steps[ids] = 0

        # Refresh state before command sampling that may read from `env.scene[...]`.
        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)

        # Reset commands (may use termination_manager flags for adaptive sampling).
        cmd_metrics = self.command_manager.reset(ids)
        if cmd_metrics:
            metrics = self.extras.setdefault("metrics", {})
            if isinstance(metrics, dict):
                for k, v in cmd_metrics.items():
                    metrics[f"command/{k}"] = v

        self.command_manager.compute()

        # Reset action buffers for the environments we just reset.
        self._reset_action_buffers(ids)

        # Refresh state after command-side writes (e.g., teleporting robot).
        full_state = self.handler.get_states()
        self.scene.update_from_states(full_state)

        # Reset managers for episodic logging (IsaacLab-style surfaces).
        try:
            rew_metrics = self.reward_manager.reset(ids)
        except Exception:
            rew_metrics = {}
        try:
            term_metrics = self.termination_manager.reset(ids)
        except Exception:
            term_metrics = {}
        try:
            curr_metrics = self.curriculum_manager.reset(ids)
        except Exception:
            curr_metrics = {}

        log = self.extras.get("log")
        if isinstance(log, dict):
            for k, v in rew_metrics.items():
                log[k] = v
            for k, v in term_metrics.items():
                log[k] = v
            for k, v in curr_metrics.items():
                log[k] = v

        # Recorder hook: post-reset
        self.recorder_manager.record_post_reset(env_ids)

    def _reset_action_buffers(self, env_ids: torch.Tensor | None) -> None:
        dim = int(getattr(self.action_manager, "total_action_dim", 0))
        if dim <= 0:
            return

        action = getattr(self.action_manager, "action", None)
        prev = getattr(self.action_manager, "prev_action", None)
        expected_shape = (int(self.num_envs), int(dim))

        if (
            not isinstance(action, torch.Tensor)
            or not isinstance(prev, torch.Tensor)
            or action.shape != expected_shape
            or prev.shape != expected_shape
        ):
            self.action_manager.action = torch.zeros(expected_shape, device=self.device, dtype=torch.float32)
            self.action_manager.prev_action = torch.zeros(expected_shape, device=self.device, dtype=torch.float32)
            action = self.action_manager.action
            prev = self.action_manager.prev_action

        if env_ids is None:
            action.zero_()
            prev.zero_()
            return

        ids = env_ids.to(device=self.device, dtype=torch.long)
        action[ids] = 0.0
        prev[ids] = 0.0
