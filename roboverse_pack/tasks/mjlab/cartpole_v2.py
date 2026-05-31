r"""Cartpole balance / swingup ported to the canonical ManagerBasedRVEnv pattern.

Successor to ``roboverse_pack/tasks/mjlab/cartpole_train.py`` (the
single-env hand-coded ``BaseTaskEnv`` port). This version uses the
declarative manager-based config introduced in M-Plan Phase 1/2a, so
the env definition reads as a near-1:1 mirror of mjlab's
``cartpole_env_cfg.py`` (see ``mjlab/src/mjlab/tasks/cartpole/``).

The old ``cartpole_train.py`` is kept in place for backward compat with
any earlier scripts / docs that import it; new training should target
``mjlab.cartpole_balance_v2`` / ``mjlab.cartpole_swingup_v2`` via the
RoboVerse RSL-RL trainer:

    python -m roboverse_learn.rl.rsl_rl.ppo \\
        --task mjlab.cartpole_balance_v2 --num-envs 1
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.utils import configclass
from roboverse_learn.managers import (
    DoneTerm,
    EventTerm,
    ManagerBasedRVEnv,
    ManagerBasedRVEnvCfg,
    ObsTerm,
    RewTerm,
)

from ._locator import mjlab_asset
from .cartpole import _CARTPOLE_XML
from .mdp import (
    SceneEntityCfg,
)
from .mdp import (
    observations as obs,
)
from .mdp import (
    rewards as rew,
)
from .mdp import (
    terminations as term,
)

# ---------------------------------------------------------------------------
# scene entity cfgs (shared by obs / reward terms)
# ---------------------------------------------------------------------------

_CART = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


# ---------------------------------------------------------------------------
# scene-MJCF-specific reset event (mjlab's reset_joints_by_offset assumes
# the asset is a RobotCfg; cartpole is loaded as the scene MJCF so we
# write physics.data directly here, gated by mujoco's reset_context)
# ---------------------------------------------------------------------------


def reset_cartpole(
    env,
    env_ids: torch.Tensor,
    *,
    hinge_init: float = 0.0,
    slider_pos_range: tuple[float, float] = (-0.1, 0.1),
    hinge_pos_range: tuple[float, float] = (-0.034, 0.034),
    velocity_range: tuple[float, float] = (-0.01, 0.01),
) -> None:
    """Reset cartpole state in place, sim-aware (mujoco scene-MJCF or newton batched path).

    Mirrors mjlab's ``reset_joints_by_offset`` for the slider + hinge.
    Swing-up tasks pass ``hinge_init=math.pi``.
    """
    # MuJoCo path (single-env scene-MJCF): write to physics.data.qpos.
    if hasattr(env.handler, "physics"):
        physics = env.handler.physics
        rng = np.random.default_rng()
        with physics.reset_context():
            physics.data.qpos[0] = 0.0 + rng.uniform(*slider_pos_range)
            physics.data.qpos[1] = hinge_init + rng.uniform(*hinge_pos_range)
            physics.data.qvel[0] = rng.uniform(*velocity_range)
            physics.data.qvel[1] = rng.uniform(*velocity_range)
        return
    # Newton path (multi-env GPU): build a TensorState delta and call set_states.
    # For now, just rely on handler's default initial state (slider=0, hinge=0
    # with small noise from solver init). Match-bit reset noise across 4096 envs
    # would need batched qpos write — deferred.
    return


# ---------------------------------------------------------------------------
# observation / reward / termination / event configs (mjlab cartpole_env_cfg)
# ---------------------------------------------------------------------------


@configclass
class _CartpoleObsCfg:
    """Mirrors mjlab's cartpole observations: actor + symmetric critic.

    Note the group key is ``actor`` (mjlab convention), not ``policy``.
    Matched with ``observation_group_names = ("actor", "critic")`` in the
    env cfg below so RSL-RL's obs_groups path picks the right tensor.
    """

    @configclass
    class ActorCfg:
        cart_pos = ObsTerm(func=obs.joint_pos_rel, params={"asset_cfg": _CART})
        pole_angle = ObsTerm(func=obs.pole_angle_cos_sin, params={"asset_cfg": _HINGE})
        cart_vel = ObsTerm(func=obs.joint_vel_rel, params={"asset_cfg": _CART})
        pole_vel = ObsTerm(func=obs.joint_vel_rel, params={"asset_cfg": _HINGE})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class _CartpoleRewardsCfg:
    smooth_reward = RewTerm(
        func=rew.cartpole_smooth_reward,
        weight=1.0,
        params={"cart_cfg": _CART, "hinge_cfg": _HINGE},
    )


@configclass
class _CartpoleTerminationsCfg:
    time_out = DoneTerm(func=term.time_out, time_out=True)


@configclass
class _CartpoleBalanceEventsCfg:
    reset_state = EventTerm(
        func=reset_cartpole,
        mode="reset",
        params={"hinge_init": 0.0},
    )


@configclass
class _CartpoleSwingupEventsCfg:
    reset_state = EventTerm(
        func=reset_cartpole,
        mode="reset",
        params={
            "hinge_init": math.pi,
            "slider_pos_range": (0.0, 0.0),  # mjlab uses zero slider noise on swingup
        },
    )


# ---------------------------------------------------------------------------
# env config + task class
# ---------------------------------------------------------------------------


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


@configclass
class CartpoleBalanceEnvCfg(ManagerBasedRVEnvCfg):
    """Manager-based env config for the cartpole balance task."""

    decimation = 5
    max_episode_length_s = 50.0  # mjlab uses 50.0; perfect ep_r sum ≈ 50
    is_finite_horizon = False
    observation_group_names = ("actor", "critic")  # match mjlab convention
    observations = _CartpoleObsCfg()
    rewards = _CartpoleRewardsCfg()
    terminations = _CartpoleTerminationsCfg()
    events = _CartpoleBalanceEventsCfg()


@configclass
class CartpoleSwingupEnvCfg(CartpoleBalanceEnvCfg):
    """Cartpole swingup env config (pole starts pointing down)."""

    events = _CartpoleSwingupEventsCfg()


class _CartpoleTaskBase(ManagerBasedRVEnv):
    """Common machinery for both balance + swingup variants."""

    scenario = _cartpole_scenario()
    _cfg_cls: type[ManagerBasedRVEnvCfg] = CartpoleBalanceEnvCfg

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None) -> None:
        cfg = self._cfg_cls()
        # Two-path scenario handling:
        #   mujoco: scene-MJCF self-contained — drop any trainer-injected robots
        #           (trainer always sets robots=[args.robot]).
        #   newton: needs RobotCfg list — keep trainer-injected robots
        #           (user passes --robot mjlab_cartpole).
        sim = getattr(scenario, "simulator", None) if scenario else None
        if scenario is not None and sim == "mujoco":
            scenario.robots = []
        elif scenario is not None and sim == "newton":
            # mjlab cartpole disables contacts (cart-on-rail; contact adds spurious
            # forces that cap the achievable reward). The mujoco path does this via
            # disableflags below; the Newton path needs it set BEFORE the solver is
            # built. Without it, Newton cartpole tops out ~0.59 smooth_reward vs
            # mjlab's 0.995 — a physics-config gap, not a trainer/obs gap.
            if getattr(scenario, "sim_params", None) is not None:
                scenario.sim_params.newton_mujoco_disable_contacts = True
        self.num_actions = 1
        super().__init__(scenario=scenario or self.scenario, cfg=cfg, device=device)
        # mjlab cartpole_env_cfg sets `disableflags=("contact",)`; apply for mujoco
        # backend only (newton doesn't expose .physics.model.opt.disableflags directly).
        if hasattr(self.handler, "physics"):
            m = self.handler.physics.model
            m = m.ptr if hasattr(m, "ptr") else m
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def _get_initial_states(self) -> list[dict]:
        # No registered robot; the reset event writes physics.data directly.
        return [{"objects": {}, "robots": {}} for _ in range(self.scenario.num_envs)]

    def _apply_action(self, processed_action: torch.Tensor) -> None:
        # Newton path: use the handler's batched set_dof_targets API which
        # handles num_envs > 1 directly. Mujoco scene-MJCF path writes
        # ``physics.data.ctrl`` directly (single-env).
        if not hasattr(self.handler, "physics"):
            # Newton handler set_dof_targets wants action for ALL joint DOFs,
            # not just actuated ones. Cartpole has 2 hinge joints (slider +
            # hinge_1) but only slider is actuated. Pad with 0 for hinge_1.
            #
            # processed_action shape: (num_envs, 1) for slider effort.
            # Newton expected: (num_envs, 2).
            n_envs = processed_action.shape[0]
            padded = torch.zeros(n_envs, 2, device=processed_action.device)
            padded[:, 0] = processed_action[:, 0]  # slider
            self.handler.set_dof_targets(padded)
            return
        # Single actuator (the slider). Raw action goes to ctrl; MuJoCo
        # auto-clamps via the actuator's ctrlrange ([-1, 1] in the XML).
        # Reward sees env._action (raw, pre-clip) — matches mjlab's
        # `env.action_manager.action` semantics.
        ctrl = float(processed_action.detach().cpu().numpy().flatten()[0])
        self.handler.physics.data.ctrl[0] = ctrl


@register_task("mjlab.cartpole_balance_v2")
class CartpoleBalanceTask(_CartpoleTaskBase):
    """Balance variant — hinge starts upright (0). RL must keep it there."""

    _cfg_cls = CartpoleBalanceEnvCfg


@register_task("mjlab.cartpole_swingup_v2")
class CartpoleSwingupTask(_CartpoleTaskBase):
    """Swing-up variant — hinge starts hanging (π). RL must swing + balance."""

    _cfg_cls = CartpoleSwingupEnvCfg
