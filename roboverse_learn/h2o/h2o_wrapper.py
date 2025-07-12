# h2o_wrapper.py
"""
Minimal rsl_rl-compatible wrapper for the H2O legged robot.
Fill in the TODO blocks as you port features.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List

import torch

from metasim.cfg.scenario import ScenarioCfg

# -- project specific (adapt paths/names) ---------------------------------
from metasim.cfg.tasks.h2o.base_legged_cfg import BaseLeggedTaskCfg
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper
from roboverse_learn.skillblender_rl.utils import (
    get_body_reindexed_indices_from_substring,
    get_joint_reindexed_indices_from_substring,
)

# -------------------------------------------------------------------------


class H2OWrapper(RslRlWrapper):
    """rsl_rl vector-env wrapper for H2O."""

    # ------------------------------------------------------------------ #
    # 1. ctor & indices                                                  #
    # ------------------------------------------------------------------ #
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.up_axis_idx = 2                             # z-up world

        # ---- static indices (used by rewards / observations) ----------
        self._parse_rigid_body_indices(scenario.robots[0])
        self._parse_joint_indices(scenario.robots[0])

        # ---- cfg-level meta -------------------------------------------
        self.dt = scenario.decimation * scenario.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim

        self._prepare_reward_function(scenario.task)
        self._init_buffers()

    # ---------------- rigid-body indices ------------------------------ #
    def _parse_rigid_body_indices(self, robot_cfg):
        """Resolve and cache body indices once at start-up."""
        self.feet_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot_cfg.name, robot_cfg.feet_links, device=self.device
        )
        self.cfg.feet_indices = self.feet_indices

    # ---------------- joint indices ----------------------------------- #
    def _parse_joint_indices(self, robot_cfg):
        """Resolve joint groups (only if needed by reward/obs)."""
        self.cfg.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env.handler, robot_cfg.name, robot_cfg.upper_body_joints, device=self.device
        )

    # ------------------------------------------------------------------ #
    # 2. runtime buffers                                                 #
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        n, obs_dim, act_dim = self.num_envs, self.num_obs, self.num_actions

        self.obs_buf   = torch.zeros(n, obs_dim, device=self.device)
        self.rew_buf   = torch.zeros(n, device=self.device)
        self.reset_buf = torch.ones (n, dtype=torch.bool, device=self.device)

        self.actions   = torch.zeros(n, act_dim, device=self.device)
        self.base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(n, 1)
        self.gravity_vec = torch.tensor([0, 0, -1], device=self.device).repeat(n, 1)

        # History stacks for frame-stacked observations
        self.obs_history = deque(
            [torch.zeros_like(self.obs_buf)] * self.cfg.frame_stack,
            maxlen=self.cfg.frame_stack,
        )

    # ------------------------------------------------------------------ #
    # 3. core RL hooks                                                   #
    # ------------------------------------------------------------------ #
    def clip_actions(self, a: torch.Tensor) -> torch.Tensor:
        lim = self.cfg.normalization.clip_actions
        return torch.clamp(a, -lim, lim)

    def _pre_physics_step(self, a: torch.Tensor) -> torch.Tensor:
        a = self.clip_actions(a)
        self.actions[:] = a
        return a

    def _physics_step(self, actions: torch.Tensor):
        """
        Isaac/MetaSim env.step – must return terminated / timeout flags.
        Only terminated|timeout are required here.
        """
        env_state, _, term, tout, _ = self.env.step(actions)
        self.reset_buf = term | tout
        return env_state

    def _post_physics_step(self, env_state):
        # --- reward (simple placeholder) ------------------------------
        self.rew_buf[:] = 0.0   # TODO: plug actual reward

        # --- observation ---------------------------------------------
        self._compute_observations(env_state)

        # --- book-keeping --------------------------------------------
        self.obs_history.append(self.obs_buf.clone())

        return self.obs_buf, None, self.rew_buf

    # public API
    def step(self, actions):
        acts = self._pre_physics_step(actions)
        st   = self._physics_step(acts)
        obs, priv, rew = self._post_physics_step(st)
        return obs, priv, rew, self.reset_buf, {}

    def reset(self, env_ids: List[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if not env_ids:
            return
        self.env.reset(self.init_states, env_ids)
        self.reset_buf[env_ids] = False
        # clear history
        for h in self.obs_history:
            h[env_ids] = 0

    # ------------------------------------------------------------------ #
    # 4. observation assembly                                            #
    # ------------------------------------------------------------------ #
    def _compute_observations(self, env_state):
        """
        Build actor observation.
        Example below uses only DOF pos & vel;
        Expand with reference-tracking or vision if you need.
        """
        dof_pos = env_state.robots[self.robot.name].dof_pos
        dof_vel = env_state.robots[self.robot.name].dof_vel
        proj_g  = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.obs_buf[:] = torch.cat([dof_pos, dof_vel, proj_g], dim=-1)

    # ------------------------------------------------------------------ #
    # 5. reward functions (skeleton)                                     #
    # ------------------------------------------------------------------ #
    def _prepare_reward_function(self, task: BaseLeggedTaskCfg):
        """Register reward fns according to cfg.scales dict."""
        self.reward_scales: dict[str, float] = task.reward_weights
        self.reward_fns: dict[str, Callable] = {}      # name → callable
        for name, scale in self.reward_scales.items():
            fn_name = f"reward_{name}"
            if hasattr(self, fn_name):
                self.reward_fns[name] = getattr(self, fn_name)
            else:
                print(f"[H2OWrapper] WARNING: reward fn {fn_name} missing")

    # Example reward stub
    def reward_alive(self, env_state, robot, cfg):
        return torch.ones(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # 6. optional curriculum / pushes (leave blank if not needed)        #
    # ------------------------------------------------------------------ #
    def _post_physics_step_callback(self):  # called each sim-step
        pass

    def _push_robots(self):                 # random impulses
        pass
