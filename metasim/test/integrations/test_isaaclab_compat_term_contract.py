from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.integrations.isaaclab.compat.contract import CompatTermError, WarnOnce
from metasim.integrations.isaaclab.compat.event_registry import EventTermRegistry
from metasim.integrations.isaaclab.compat.managers import (
    CompatActionManager,
    CompatCurriculumManager,
    CompatObservationManager,
    CompatRecorderManager,
    CompatRewardManager,
    CompatTerminationManager,
)
from metasim.integrations.isaaclab.compat.scene import CompatScene
from metasim.integrations.isaaclab.shim import ensure_isaaclab_shim
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg


class _FakeHandler:
    def __init__(self, *, num_envs: int, joint_names: list[str], body_names: list[str], env_origins: torch.Tensor):
        self._num_envs = int(num_envs)
        self._joint_names = list(joint_names)
        self._body_names = list(body_names)
        self.scene = SimpleNamespace(env_origins=env_origins)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_joint_names(self, _obj_name: str, sort: bool = True) -> list[str]:
        out = list(self._joint_names)
        return sorted(out) if sort else out

    def get_body_names(self, _obj_name: str, sort: bool = True) -> list[str]:
        out = list(self._body_names)
        return sorted(out) if sort else out


@pytest.mark.general
def test_observation_contract_normalizes_and_converts():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _vec(env):
        return torch.arange(env.num_envs, dtype=torch.float32)

    def _scalar(env):
        return 1.0

    obs_cfg = SimpleNamespace(
        policy=SimpleNamespace(
            enable_corruption=False,
            concatenate_terms=True,
            v=SimpleNamespace(func=_vec, params={}),
            s=SimpleNamespace(func=_scalar, params={}),
        )
    )

    mgr = CompatObservationManager(obs_cfg, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.active_terms == {"policy": ["v", "s"]}
    assert mgr.get_term("v", group="policy").func is _vec
    assert list(mgr.terms["policy"].keys()) == ["v", "s"]
    out = mgr.compute()
    assert "policy" in out
    assert out["policy"].shape == (env.num_envs, 2)


@pytest.mark.general
def test_observation_contract_bad_shape_best_effort_skips():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _bad(env):
        return torch.zeros(3)

    obs_cfg = SimpleNamespace(
        policy=SimpleNamespace(
            enable_corruption=False,
            concatenate_terms=True,
            bad=SimpleNamespace(func=_bad, params={}),
        )
    )

    mgr = CompatObservationManager(obs_cfg, env=env, strict=False, warn_once=WarnOnce())
    out = mgr.compute()
    assert out["policy"].shape == (env.num_envs, 0)


@pytest.mark.general
def test_observation_contract_bad_shape_strict_raises():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _bad(env):
        return torch.zeros(3)

    obs_cfg = SimpleNamespace(
        policy=SimpleNamespace(
            enable_corruption=False,
            concatenate_terms=True,
            bad=SimpleNamespace(func=_bad, params={}),
        )
    )

    mgr = CompatObservationManager(obs_cfg, env=env, strict=True, warn_once=WarnOnce())
    with pytest.raises(CompatTermError):
        mgr.compute()


@pytest.mark.general
def test_reward_contract_broadcasts_and_scales_by_step_dt():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        step_dt = 0.5
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _scalar(env):
        return 1.0

    rew_cfg = SimpleNamespace(
        alive=SimpleNamespace(func=_scalar, params={}, weight=2.0),
    )

    mgr = CompatRewardManager(rew_cfg, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.active_terms == ["alive"]
    assert mgr.get_term("alive").func is _scalar
    rew = mgr.compute()
    assert rew.shape == (env.num_envs,)
    assert torch.allclose(rew, torch.ones(env.num_envs) * 1.0)  # 1.0 * weight(2) * step_dt(0.5)


@pytest.mark.general
def test_termination_contract_casts_to_bool_and_splits_timeouts():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _done(env):
        return torch.tensor([0.0, 1.0], dtype=torch.float32)

    term_cfg = SimpleNamespace(
        crash=SimpleNamespace(func=_done, params={}, time_out=False),
        timeout=SimpleNamespace(func=_done, params={}, time_out=True),
    )

    mgr = CompatTerminationManager(term_cfg, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.active_terms == ["crash", "timeout"]
    terminated, time_outs = mgr.compute()
    assert terminated.dtype == torch.bool
    assert time_outs.dtype == torch.bool
    assert torch.equal(terminated, torch.tensor([False, True]))
    assert torch.equal(time_outs, torch.tensor([False, True]))
    assert torch.equal(mgr.dones, torch.tensor([False, True]))
    assert torch.equal(mgr.get_term("crash"), torch.tensor([False, True]))


@pytest.mark.general
def test_curriculum_manager_stub_executes_terms_and_exposes_active_terms():
    class _Env:
        num_envs = 2
        device = torch.device("cpu")
        scenario = SimpleNamespace(simulator="mujoco")

    env = _Env()

    def _curriculum_term(env, env_ids):
        if isinstance(env_ids, slice):
            count = env.num_envs
        else:
            count = int(torch.as_tensor(env_ids).numel())
        return {"count": torch.tensor(float(count))}

    cfg = SimpleNamespace(level=SimpleNamespace(func=_curriculum_term, params={}))
    mgr = CompatCurriculumManager(cfg, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.active_terms == ["level"]

    mgr.compute(env_ids=torch.tensor([0, 1], dtype=torch.long))
    extras = mgr.reset(env_ids=torch.tensor([0, 1], dtype=torch.long))
    assert "Curriculum/level/count" in extras


@pytest.mark.general
def test_recorder_manager_stub_exposes_hook_surface_and_noops():
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scenario=SimpleNamespace(simulator="mujoco"))
    mgr = CompatRecorderManager(None, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.active_terms == []
    mgr.record_pre_reset(torch.tensor([0], dtype=torch.long))
    mgr.record_post_reset(torch.tensor([0], dtype=torch.long))
    mgr.record_pre_step()
    mgr.record_post_step()
    mgr.record_post_physics_decimation_step()


@pytest.mark.general
def test_action_registry_builds_payload_for_pos_vel_effort():
    ensure_isaaclab_shim()
    from isaaclab.envs.mdp import JointEffortActionCfg, JointPositionActionCfg, JointVelocityActionCfg  # type: ignore

    handler = _FakeHandler(
        num_envs=2,
        joint_names=["j1", "j0"],  # intentionally unsorted
        body_names=["base"],
        env_origins=torch.zeros((2, 3), dtype=torch.float32),
    )
    robot_cfg = RobotCfg(
        name="robot",
        joint_limits={"j0": (-1.0, 1.0), "j1": (-1.0, 1.0)},
        default_joint_positions={".*": 0.0},
    )
    scenario = ScenarioCfg(simulator="newton", num_envs=2, robots=[robot_cfg], objects=[], cameras=[], headless=True)
    scene = CompatScene(handler=handler, scenario=scenario, device=torch.device("cpu"))

    env = SimpleNamespace(
        scenario=scenario,
        num_envs=scenario.num_envs,
        device=torch.device("cpu"),
        scene=scene,
    )

    actions_cfg = SimpleNamespace(
        joint_pos=JointPositionActionCfg(asset_name="robot", joint_names=["j0", "j1"], use_default_offset=False),
        joint_vel=JointVelocityActionCfg(asset_name="robot", joint_names=["j0"]),
        joint_effort=JointEffortActionCfg(asset_name="robot", joint_names=["j1"]),
    )

    mgr = CompatActionManager(actions_cfg, env=env, strict=True, warn_once=WarnOnce())
    assert mgr.total_action_dim == 4
    assert list(mgr.terms.keys()) == ["joint_pos", "joint_vel", "joint_effort"]

    actions = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    payload = mgr.process(actions)
    assert isinstance(payload, list) and len(payload) == 2
    assert "robot" in payload[0]
    assert payload[0]["robot"]["dof_pos_target"]["j0"] == pytest.approx(1.0)
    assert payload[0]["robot"]["dof_pos_target"]["j1"] == pytest.approx(2.0)
    assert payload[0]["robot"]["dof_vel_target"]["j0"] == pytest.approx(3.0)
    assert payload[0]["robot"]["dof_vel_target"]["j1"] == pytest.approx(0.0)
    assert payload[0]["robot"]["dof_effort_target"]["j0"] == pytest.approx(0.0)
    assert payload[0]["robot"]["dof_effort_target"]["j1"] == pytest.approx(4.0)


@pytest.mark.general
def test_event_registry_gates_physx_only_event_on_non_isaacsim():
    ensure_isaaclab_shim()
    from roboverse_pack.tasks.beyondmimic.isaaclab.mdp import events as bm_events

    env = SimpleNamespace(scenario=SimpleNamespace(simulator="mujoco"))
    registry = EventTermRegistry(env=env, strict=False, warn_once=WarnOnce())

    supported = registry.wrap(
        name="randomize_joint_default_pos", term_cfg=SimpleNamespace(func=bm_events.randomize_joint_default_pos)
    )
    assert supported is bm_events.randomize_joint_default_pos

    gated = registry.wrap(
        name="randomize_rigid_body_com", term_cfg=SimpleNamespace(func=bm_events.randomize_rigid_body_com)
    )
    assert gated is not None and gated is not bm_events.randomize_rigid_body_com
    # Must not raise when called in best-effort mode.
    gated()


@pytest.mark.general
def test_event_registry_strict_raises_for_unsupported_event():
    ensure_isaaclab_shim()
    from roboverse_pack.tasks.beyondmimic.isaaclab.mdp import events as bm_events

    env = SimpleNamespace(scenario=SimpleNamespace(simulator="mujoco"))
    registry = EventTermRegistry(env=env, strict=True, warn_once=WarnOnce())
    with pytest.raises(CompatTermError):
        registry.wrap(
            name="randomize_rigid_body_com", term_cfg=SimpleNamespace(func=bm_events.randomize_rigid_body_com)
        )
