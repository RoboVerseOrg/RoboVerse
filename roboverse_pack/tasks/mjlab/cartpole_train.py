"""Full RL-trainable port of mjlab cartpole tasks to MetaSim.

Mirrors mjlab/src/mjlab/tasks/cartpole/cartpole_env_cfg.py one-for-one:

  - Action     : 1D effort on `slider` actuator, clipped to [-1, 1]
  - Observation: [joint_pos_rel(2), joint_vel_rel(2), cos(hinge), sin(hinge)] = 6D
  - Reward     : dm_control smooth = upright * centered * small_control * small_velocity
  - Termination: time_out only (max_episode_steps)
  - Reset      : initial pos as in BalanceInit / SwingupInit + small joint-offset noise

The class is intentionally a thin numpy-on-CPU implementation that runs
through MetaSim's MujocoHandler — `physics.data.ctrl` is the only thing we
write; everything else (obs, reward, termination) is derived from the
authoritative MuJoCo state. This is what enables the "training in our
framework matches mjlab native" claim: both stacks step identical physics on
identical XML; only the manager glue differs in code paths, not numerically.

Usage:
    python -m tools.mjlab_integration.policy_replay.train_metasim_cartpole \
        --task mjlab.cartpole_balance_train --total-steps 200_000
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

from ._locator import _CARTPOLE_XML, mjlab_asset

# dm_control tolerance defaults: value_at_margin = 0.1
_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tol_np(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0:
        return (x == 0).astype(np.float32)
    scaled = x / margin * _GAUSSIAN_SCALE
    return np.exp(-0.5 * scaled * scaled)


def _quadratic_tol_np(x: np.ndarray, margin: float) -> np.ndarray:
    if margin == 0:
        return (x == 0).astype(np.float32)
    scaled = x / margin * _QUADRATIC_SCALE
    return np.clip(1.0 - scaled * scaled, a_min=0.0, a_max=None)


def _cartpole_scenario() -> ScenarioCfg:
    return ScenarioCfg(
        scene=SceneCfg(mjcf_path=mjlab_asset(_CARTPOLE_XML)),
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        decimation=5,
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )


# --- public env class ---------------------------------------------------------


class _MjlabCartpoleTrainBase(BaseTaskEnv):
    """Shared training scaffold for balance + swingup."""

    scenario = _cartpole_scenario()
    # mjlab default: episode_length_s = 50s, control_dt = 0.05s (dt 0.01 * decim 5)
    max_episode_steps = 1000
    _hinge_init: float = 0.0  # 0 for balance, math.pi for swingup
    # mjlab cartpole reset_joints_by_offset:
    #   slider: position_range=(-0.1, 0.1), velocity_range=(-0.01, 0.01)  [balance only; swingup uses 0]
    #   hinge:  position_range=(-0.034, 0.034), velocity_range=(-0.01, 0.01)
    _slider_pos_range: float = 0.1
    _hinge_pos_range: float = 0.034
    _vel_range: float = 0.01

    def __init__(self, scenario=None, device=None):
        super().__init__(scenario=scenario or self.scenario, device=device)
        # mjlab cartpole cfg: SimulationCfg(mujoco=MujocoCfg(disableflags=("contact",)))
        # SimParamCfg has no equivalent option, so we patch mj_model.opt directly
        # after handler launch. Without this, the cartpole slider has subtle
        # contact dynamics that mjlab's policy was NOT trained against, so policy
        # transfer fails.
        import mujoco

        model = self.handler.physics.model
        # dm_control's Physics wraps mujoco MjModel; access via .ptr or direct
        ptr = model.ptr if hasattr(model, "ptr") else model
        ptr.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    # ---- spaces ----
    # mjlab cartpole obs: 5D, term order = [cart_pos, pole_angle_cos, pole_angle_sin, cart_vel, pole_vel]
    # See mjlab/src/mjlab/tasks/cartpole/cartpole_env_cfg.py:157-174
    #   "cart_pos":   joint_pos_rel(cart_cfg)   = qpos[slider] - default = qpos[0]   (1D)
    #   "pole_angle": pole_angle_cos_sin(hinge) = [cos(qpos[1]), sin(qpos[1])]       (2D)
    #   "cart_vel":   joint_vel_rel(cart_cfg)   = qvel[slider] = qvel[0]             (1D)
    #   "pole_vel":   joint_vel_rel(hinge_cfg)  = qvel[hinge]  = qvel[1]             (1D)
    def _observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ---- helpers ----
    @property
    def _data(self):
        # Direct read access to the underlying MuJoCo MjData (CPU).
        return self.handler.physics.data

    def _state_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(self._data.qpos[:2], dtype=np.float64).copy()
        qvel = np.asarray(self._data.qvel[:2], dtype=np.float64).copy()
        return qpos, qvel

    def _build_obs(self, qpos: np.ndarray, qvel: np.ndarray) -> torch.Tensor:
        # mjlab obs term order (5D): cart_pos | pole_angle_cos | pole_angle_sin | cart_vel | pole_vel
        # mjlab's joint_pos_rel = qpos - default_init_pos. Slider init = 0 always,
        # so cart_pos_rel = qpos[0]. Note hinge_pos is NOT in obs at all — only
        # its cos/sin via pole_angle_cos_sin, which uses raw qpos[1] (not rel).
        cart_pos = qpos[0]  # slider init = 0
        cart_vel = qvel[0]
        pole_vel = qvel[1]
        pole_cos = math.cos(qpos[1])
        pole_sin = math.sin(qpos[1])
        obs = np.array([cart_pos, pole_cos, pole_sin, cart_vel, pole_vel], dtype=np.float32)
        return torch.from_numpy(obs).to(self.device).unsqueeze(0)

    # ---- core env api ----
    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            a = actions.detach().cpu().numpy()
        else:
            a = np.asarray(actions, dtype=np.float64)
        a = a.flatten()
        # mjlab does NOT clip in Python — passes raw policy action to MJCF ctrl;
        # MuJoCo's actuator ctrlrange=[-1, 1] auto-clamps at the sim level. The
        # reward function however sees the RAW (pre-clip) action — that's what
        # `env.action_manager.action` returns. Preserve this here.
        raw_action = float(a[0])
        self._last_action = raw_action

        # Cartpole has 1 motor on slider; write raw — MuJoCo handles ctrlrange.
        self._data.ctrl[:] = 0.0
        self._data.ctrl[0] = raw_action

        # Step physics. handler.simulate runs `decimation` substeps via
        # self.physics.step(self.decimation) — same as raw mujoco.
        self.handler.simulate()

        qpos, qvel = self._state_arrays()
        obs = self._build_obs(qpos, qvel)
        # IMPORTANT: pass raw_action (not clipped ctrl) to reward — mjlab does this.
        reward = self._reward_from_state(qpos, qvel, raw_action)
        self._episode_steps = self._episode_steps + 1
        timeout = self._episode_steps >= self.max_episode_steps
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info: dict[str, Any] = {}
        return obs, reward, terminated, timeout, info

    def reset(self, states=None, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        # dm_control's Physics has a reset() that calls mj_resetData internally.
        physics = self.handler.physics
        with physics.reset_context():
            physics.data.qpos[0] = 0.0  # slider
            physics.data.qpos[1] = self._hinge_init  # hinge
            # mjlab reset_joints_by_offset (per-joint position + velocity range)
            rng = np.random.default_rng()
            slider_range = self._slider_pos_range if self._hinge_init == 0.0 else 0.0  # swingup uses 0
            physics.data.qpos[0] += rng.uniform(-slider_range, slider_range)
            physics.data.qpos[1] += rng.uniform(-self._hinge_pos_range, self._hinge_pos_range)
            physics.data.qvel[0] = rng.uniform(-self._vel_range, self._vel_range)
            physics.data.qvel[1] = rng.uniform(-self._vel_range, self._vel_range)
        self._last_action = 0.0
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        qpos, qvel = self._state_arrays()
        return self._build_obs(qpos, qvel), {}

    # ---- reward ----
    # mjlab's RewardManager.compute() multiplies each term by control_dt (=0.05 for
    # cartpole: dt 0.01 × decimation 5). Matching that scaling makes our episode
    # returns line up with mjlab's reported numbers (perfect balance → ep_sum ≈ 50).
    _CONTROL_DT = 0.05

    def _reward_from_state(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: float) -> torch.Tensor:
        # Port of mjlab cartpole_smooth_reward (dm_control style) × control_dt.
        hinge_angle = qpos[1]
        cart_pos = qpos[0]
        hinge_vel = qvel[1]

        pole_cos = math.cos(hinge_angle)
        upright = (pole_cos + 1.0) / 2.0
        centered = (1.0 + float(_gaussian_tol_np(np.array([cart_pos]), 2.0)[0])) / 2.0
        small_control = (4.0 + float(_quadratic_tol_np(np.array([ctrl]), 1.0)[0])) / 5.0
        small_velocity = (1.0 + float(_gaussian_tol_np(np.array([hinge_vel]), 5.0)[0])) / 2.0
        raw = upright * centered * small_control * small_velocity
        r = raw * 1.0 * self._CONTROL_DT  # weight=1.0 × dt
        return torch.tensor([r], dtype=torch.float32, device=self.device)


@register_task("mjlab.cartpole_balance_train")
class MjlabCartpoleBalanceTrain(_MjlabCartpoleTrainBase):
    """Balance variant: pole starts upright (hinge=0); RL must keep it there."""

    _hinge_init = 0.0


@register_task("mjlab.cartpole_swingup_train")
class MjlabCartpoleSwingupTrain(_MjlabCartpoleTrainBase):
    """Swing-up variant: pole starts hanging (hinge=π); RL must swing + balance."""

    _hinge_init = math.pi
