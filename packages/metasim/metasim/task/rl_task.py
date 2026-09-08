from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torchvision.utils import make_grid

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.types import CompatActionInput, Info, TensorState
from metasim.utils.log import warn_once
from metasim.utils.state import list_state_to_tensor, select_envs


def _terminal_copy(states, env_ids: list[int]):
    """Objects and robots of ``env_ids`` copied out of ``states`` (cameras are not part of a recorded episode)."""
    if not env_ids or states is None:
        return None
    physics_only = TensorState(objects=states.objects, robots=states.robots, cameras={}, extras={})
    return select_envs(physics_only, env_ids)


class RLTaskEnv(BaseTaskEnv):
    """Common utilities for RL tasks."""

    max_episode_steps = 1000

    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize environment."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._observation_space: spaces.Space | None = None
        self._action_space: spaces.Space | None = None

        self.asymmetric_obs = False

        super().__init__(scenario, device)

        self.num_envs = scenario.num_envs
        # ``self.robots`` (list) and ``self.robot`` (first, scalar) are both
        # part of the contract subclasses rely on — keep them in sync here.
        self.robots: list = list(scenario.robots) if scenario.robots else []
        self.robot = self.robots[0] if self.robots else None
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Observation buffers for RSL-RL compatibility
        self._obs_buf = None
        self._priv_obs_buf = None

        # Reward terms default to empty so the base ``_reward`` (which guards on
        # ``len(self.reward_functions) == 0``) is self-consistent before a
        # subclass assigns its terms. Subclasses set these *after*
        # ``super().__init__()``; initialising here also keeps ``reset()`` /
        # reward computation safe if triggered during construction.
        self.reward_functions: list = []
        self.reward_weights: list = []

        # convert list state to tensor state for reset acceleration
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)
        # first reset
        self.reset(env_ids=list(range(self.num_envs)))

        # obs size
        states = self.handler.get_states(mode="tensor")
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]

        # action bounds from joint limits in handler/API tensor order
        self.joint_names_by_robot = {
            robot.name: self.handler.get_joint_names(robot.name, sort=True) for robot in self.robots
        }
        action_low = []
        action_high = []
        for robot in self.robots:
            limits = robot.joint_limits
            joint_names = self.joint_names_by_robot[robot.name]
            action_low.extend(limits[j][0] for j in joint_names)
            action_high.extend(limits[j][1] for j in joint_names)
        self._action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self._action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)
        self.num_actions = self._action_low.shape[0]

        self._build_clamp_bounds()

    # -------------------------------------------------------------------------
    # hooks / spaces
    # -------------------------------------------------------------------------

    def _get_initial_states(self) -> list[dict]:
        """Return per-env initial states (override in subclasses)."""
        return None  # base expects subclass override

    @property
    def observation_space(self) -> spaces.Space:
        """Observation Box(num_obs,)."""
        if self._observation_space is None:
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_obs,),
                dtype=np.float32,
            )
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Action Box(num_actions,) with range [-1, 1]."""
        if self._action_space is None:
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_actions,),
                dtype=np.float32,
            )
        return self._action_space

    # -------------------------------------------------------------------------
    # env api
    # -------------------------------------------------------------------------
    def reset(self, states=None, env_ids=None, seed: int | None = None) -> tuple[torch.Tensor, Info]:
        """Reset selected envs.

        Args:
            env_ids: Indices to reset; None resets all.
            states: Optional external states to set for the selected envs. If None, use initial states.
            seed: Optional reproducibility seed forwarded to ``handler.set_seed`` when
                the backend implements it. See ``TaskBase.reset`` for the warn-if-
                unsupported semantics.

        Returns:
            (obs, info).
        """
        if seed is not None:
            set_seed = getattr(self.handler, "set_seed", None)
            if callable(set_seed):
                set_seed(seed)
            elif not getattr(self, "_seed_unsupported_warned", False):
                from loguru import logger as _log

                _log.warning(
                    f"{type(self).__name__}: handler "
                    f"{type(self.handler).__name__} does not implement set_seed; "
                    f"reset(seed={seed}) is a no-op on the simulator side."
                )
                self._seed_unsupported_warned = True
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0
        raw_states = self._initial_states if states is None else states
        self._run_reset_callbacks(env_ids)  # before the states are materialised, as BaseTaskEnv does
        states_to_set = self._prepare_states(raw_states, env_ids)
        # the auto-reset inside step() fires on nearly every training step; a render refresh there is
        # only worth paying when the observation actually contains a camera frame
        has_cameras = bool(getattr(getattr(self.handler, "scenario", None), "cameras", None))
        self._write_reset_states(states_to_set, env_ids, refresh=has_cameras)

        states = self.handler.get_states(mode="tensor")
        first_obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states)

        # Update observation buffers for RSL-RL compatibility
        self._obs_buf = first_obs
        if isinstance(priv_obs, torch.Tensor):
            self._priv_obs_buf = priv_obs.to(self.device)
        else:
            self._priv_obs_buf = first_obs

        info = {"privileged_observation": priv_obs}
        return first_obs, info

    #: set by a recorder: ``info["terminal_states"]`` then carries the auto-reset envs' pre-reset state
    record_terminal_states: bool = False

    def _note_auto_reset(self, env_ids: list[int], states, info: dict) -> None:
        """Record in ``info`` which envs this step auto-resets and the state their episodes ended in.

        ``info["auto_reset_env_ids"]`` and ``info["terminal_states"]`` (the objects and robots of
        those envs, copied before the reset when ``record_terminal_states`` is set; None otherwise,
        so training runs pay no gather) let a recorder keep the terminal (action, state) pair instead
        of the reset pose. Call it right before resetting: ``RLTaskEnv.step`` does; a task that owns
        its own reset loop must too, or a recorder cannot tell the two apart.
        """
        info["auto_reset_env_ids"] = list(env_ids)
        info["terminal_states"] = (
            _terminal_copy(states, info["auto_reset_env_ids"]) if self.record_terminal_states else None
        )

    def _build_clamp_bounds(self) -> None:
        """Build the joint-angle bounds the dict action path clamps to, for the robots they certainly apply to.

        A robot every backend drives by position takes joint angles in its action slots, so ``joint_limits``
        bounds them, and the tensor path already clips them there. A robot with an effort-driven joint takes
        torques in some or all of those slots, and which ones depends on the backend (MuJoCo routes per
        joint, Isaac Gym and Newton send the whole vector to force), so an angle limit may not describe them
        at all: those robots are left out here and keep the behaviour they have on both paths. See the
        CHANGELOG for the open half. A handler that does not answer the predicate leaves every robot out.

        The bounds are read back from a float32 tensor so the dict path clamps to the numbers the tensor
        path's float32 bounds produce, not to a Python float that rounds elsewhere.
        """
        self._clamp_limits_by_robot: dict[str, dict[str, tuple[float, float]]] = {}
        reports_position = getattr(self.handler, "_robot_reports_position_target", None)
        if reports_position is None:
            return
        for robot in self.robots:
            if not reports_position(robot.name):
                continue
            joint_names = self.joint_names_by_robot[robot.name]
            low = torch.tensor([robot.joint_limits[j][0] for j in joint_names], dtype=torch.float32)
            high = torch.tensor([robot.joint_limits[j][1] for j in joint_names], dtype=torch.float32)
            self._clamp_limits_by_robot[robot.name] = {
                j: (float(low[i]), float(high[i])) for i, j in enumerate(joint_names)
            }

    def _clamp_dict_actions(self, actions):
        """``dof_pos_target`` entries of a dict action clamped to the same bounds the tensor path uses.

        The same command must reach the engine whichever way it was written, or a replayed demo (dict
        actions) drives the robot somewhere a policy rollout of the same numbers (a tensor) cannot.
        Only the robots ``_build_clamp_bounds`` could settle are clamped;
        ``dof_vel_target`` / ``dof_effort_target`` entries are passed through. A clamp that fires is warned
        about once per robot and joint, because a demo recorded outside the config's limits then replays as a
        different trajectory. A value already in bounds is not rewritten and NaN is
        left as it is, so an action that needs no clamping reaches the handler as the caller's own object;
        the caller's dicts are never modified. The scan is per env, robot and joint, on the dict path only.
        """
        # a subclass that builds on BaseTaskEnv.__init__ has no bounds, and neither has a scenario whose
        # robots are all effort-driven: the action goes through as it did before, without the per-env scan
        if not isinstance(actions, list) or not getattr(self, "_clamp_limits_by_robot", None):
            return actions
        clamped = list(actions)
        changed = False
        for env_id, env_action in enumerate(actions):
            if not isinstance(env_action, dict):
                continue
            for robot_name, robot_action in env_action.items():
                limits = self._clamp_limits_by_robot.get(robot_name)
                targets = robot_action.get("dof_pos_target") if isinstance(robot_action, dict) else None
                if not limits or not targets:
                    continue
                # NaN is neither below nor above a bound: it is out of range in no direction, and the tensor
                # path passes it through too
                if not any(
                    value < limits[joint][0] or value > limits[joint][1]
                    for joint, value in targets.items()
                    if joint in limits
                ):
                    continue
                new_targets = {}
                for joint, value in targets.items():
                    low, high = limits.get(joint, (None, None))
                    # a value the limits do not describe, or one that is not a number (a demo may carry None
                    # for an unactuated joint), reaches the backend as it did before
                    if low is None or not isinstance(value, (int, float)) or not (value < low or value > high):
                        new_targets[joint] = value
                        continue
                    new_targets[joint] = min(max(value, low), high)
                    warn_once(
                        ("rl_task.clamped_action", robot_name, joint),
                        f"{type(self).__name__}: joint target {joint!r} of {robot_name!r} was outside the "
                        f"configured joint_limits {(low, high)} and was clamped; a demo recorded outside them "
                        "replays as a different trajectory. Warned once per robot and joint.",
                    )
                if clamped[env_id] is env_action:
                    clamped[env_id] = dict(env_action)
                clamped[env_id][robot_name] = {**robot_action, "dof_pos_target": new_targets}
                changed = True
        return clamped if changed else actions

    def step(
        self,
        actions: CompatActionInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]:
        """One step with joint-space actions.

        A tensor action is clipped to the joint-limit box (``_action_low`` / ``_action_high``). A dict action
        is clipped to the same limits for the robots those limits certainly describe, so the same command
        reaches the engine whichever way it was written; see ``_build_clamp_bounds``.
        """
        self._episode_steps += 1

        # Sanctioned action-transform hook (default identity). Lets tasks apply
        # delta control / unnormalisation here instead of overriding step().
        actions = self._process_action(actions)

        if isinstance(actions, (torch.Tensor, np.ndarray)):
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)

            real_actions = torch.maximum(torch.minimum(actions, self._action_high), self._action_low)
            self.handler.set_dof_targets(real_actions)
        else:
            self.handler.set_dof_targets(self._clamp_dict_actions(actions))
        self.handler.simulate()
        states = self.handler.get_states(mode="tensor")
        obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states)

        # normalised to the step() contract, shared with BaseTaskEnv (see ``_as_step_tensor``)
        reward = self._as_step_tensor(self._reward(states), dtype=torch.float32, hook="_reward")
        terminated = self._as_step_tensor(self._terminated(states), dtype=torch.bool, hook="_terminated")
        time_out = self._as_step_tensor(self._time_out(states), dtype=torch.bool, hook="_time_out")

        # Cache observations for RSL-RL compatibility
        self._obs_buf = obs
        if isinstance(priv_obs, torch.Tensor):
            self._priv_obs_buf = priv_obs.to(self.device)
        else:
            self._priv_obs_buf = obs

        episode_done = terminated | time_out
        info = {
            "privileged_observation": priv_obs,
            "episode_steps": self._episode_steps.clone(),
            # The terminal observation, snapshotted before the auto-reset below overwrites `obs`
            # in place for the done envs. Off-policy learners bootstrap truncated episodes from
            # this (V(s_T) for a time-out is a real value; V(reset state) is not), so it must be
            # the state the episode actually ended in.
            "observations": {"raw": {"obs": obs.clone()}},
        }

        done_ids = episode_done.nonzero(as_tuple=False).squeeze(-1).tolist()
        self._note_auto_reset(done_ids, states, info)
        if done_ids:
            self.reset(env_ids=done_ids)
            states_after = self.handler.get_states(mode="tensor")
            obs_after = self._observation(states_after).to(self.device)
            obs[done_ids] = obs_after[done_ids]

        return obs, reward, terminated, time_out, info

    def render(self) -> np.ndarray:
        """Return an RGB grid image."""
        state = self.handler.get_states(mode="tensor")
        rgb = next(iter(state.cameras.values())).rgb  # (N, H, W, C)
        if make_grid is not None:
            grid = make_grid((rgb.permute(0, 3, 1, 2) / 255.0), nrow=int(max(1, rgb.shape[0] ** 0.5)))
            return (grid.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return rgb[0].cpu().numpy().astype(np.uint8)

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] to joint limits."""
        return (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low

    def _reward(self, env_states) -> torch.Tensor:
        """Weighted sum of reward terms."""
        total_reward = None
        if len(self.reward_functions) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for reward_func, weight in zip(self.reward_functions, self.reward_weights, strict=False):
            val = reward_func(env_states)
            if total_reward is None:
                total_reward = torch.zeros_like(val)
            total_reward += weight * val
        return total_reward

    def _terminated(self, env_states) -> torch.Tensor:
        """Terminal flags (default: none)."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _prepare_states(self, env_states, env_ids) -> torch.Tensor:
        """Prepare for the states before reset(do domain randomization)."""
        return env_states

    # -------------------------------------------------------------------------
    # RSL-RL compatibility properties
    # -------------------------------------------------------------------------

    @property
    def obs_buf(self) -> torch.Tensor:
        """Cached observation buffer for RSL-RL compatibility.

        This property enables RLTaskEnv-based environments to work with
        RSL-RL's OnPolicyRunner without needing a wrapper.
        """
        if self._obs_buf is None:
            # Lazy initialization on first access
            states = self.handler.get_states(mode="tensor")
            self._obs_buf = self._observation(states).to(self.device)
        return self._obs_buf

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Cached privileged observation buffer for RSL-RL compatibility.

        Returns privileged observations if available, otherwise returns
        the same as obs_buf (symmetric actor-critic).
        """
        if self._priv_obs_buf is None:
            # Lazy initialization on first access
            states = self.handler.get_states(mode="tensor")
            priv_obs = self._privileged_observation(states)
            if isinstance(priv_obs, torch.Tensor):
                self._priv_obs_buf = priv_obs.to(self.device)
            else:
                # Fallback to symmetric observations
                self._priv_obs_buf = self.obs_buf
        return self._priv_obs_buf
