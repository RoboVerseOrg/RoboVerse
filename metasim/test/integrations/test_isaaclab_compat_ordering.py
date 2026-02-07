from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import RobotState, TensorState


@dataclass
class _FakeScenario:
    simulator: str = "mujoco"
    num_envs: int = 2
    headless: bool = True
    decimation: int = 1
    env_spacing: float = 1.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    sim_params: SimParamCfg = field(default_factory=lambda: SimParamCfg(dt=0.01))
    robots: list[RobotCfg] = field(default_factory=list)
    objects: list = field(default_factory=list)
    cameras: list = field(default_factory=list)
    lights: list = field(default_factory=list)

    def check_assets(self) -> None:
        # Unit tests should never download assets.
        return None


@pytest.mark.general
def test_manager_call_order_reset_and_step(monkeypatch):
    # Import here to avoid importing IsaacLab compat at module import time.
    from metasim.integrations.isaaclab.compat.env import HandlerBackedManagerBasedRLEnv

    calls: list[str] = []

    class _FakeHandler(BaseSimHandler):
        def __init__(self, scenario: _FakeScenario, optional_queries=None):
            super().__init__(scenario, optional_queries)
            self._device = torch.device("cpu")
            self._root_state = torch.zeros((scenario.num_envs, 13), dtype=torch.float32)
            self._root_state[:, 3] = 1.0
            self._body_state = torch.zeros((scenario.num_envs, 1, 13), dtype=torch.float32)
            self._body_state[:, :, 3] = 1.0
            self._joint_pos = torch.zeros((scenario.num_envs, 2), dtype=torch.float32)
            self._joint_vel = torch.zeros_like(self._joint_pos)

        @property
        def device(self) -> torch.device:
            return self._device

        def launch(self) -> None:
            return None

        def render(self) -> None:
            return None

        def close(self) -> None:
            return None

        def _set_states(self, states, env_ids=None) -> None:
            return None

        def _set_dof_targets(self, actions):
            return None

        def _simulate(self):
            return None

        def _get_states(self, env_ids=None) -> TensorState:
            robot_name = self.scenario.robots[0].name
            rs = RobotState(
                root_state=self._root_state,
                body_names=["base"],
                body_state=self._body_state,
                joint_pos=self._joint_pos,
                joint_vel=self._joint_vel,
                joint_pos_target=torch.zeros_like(self._joint_pos),
                joint_vel_target=torch.zeros_like(self._joint_pos),
                joint_effort_target=torch.zeros_like(self._joint_pos),
            )
            return TensorState(objects={}, robots={robot_name: rs}, cameras={}, extras=self.get_extra())

        def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
            joints = ["j1", "j0"]
            return sorted(joints) if sort else joints

        def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
            bodies = ["base"]
            return sorted(bodies) if sort else bodies

    # Patch compat env handler construction to use the fake handler (avoid real sim / parallel wrapper).
    import metasim.integrations.isaaclab.compat.env as compat_env

    monkeypatch.setattr(compat_env, "get_sim_handler_class", lambda _sim: _FakeHandler)

    # ------------------------------------------------------------------
    # Terms that append to `calls` when invoked
    # ------------------------------------------------------------------
    class _CmdTerm:
        def __init__(self, _cfg, env):
            self.command = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

        def reset(self, env_ids):
            calls.append("cmd.reset")

        def compute(self, dt: float):
            calls.append("cmd.compute")

    class _RecorderTerm:
        def __init__(self, _cfg, _env):
            return None

        def record_pre_reset(self, env_ids):
            calls.append("rec.pre_reset")

        def record_post_reset(self, env_ids):
            calls.append("rec.post_reset")

        def record_pre_step(self):
            calls.append("rec.pre_step")

        def record_post_step(self):
            calls.append("rec.post_step")

        def record_post_physics_decimation_step(self):
            calls.append("rec.post_physics")

    def _startup_event(env):
        calls.append("event.startup")

    def _obs(env):
        calls.append("obs")
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

    def _reward(env):
        calls.append("reward")
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def _done(env):
        calls.append("done")
        return torch.tensor([True, False], device=env.device, dtype=torch.bool)

    def _curriculum(env, env_ids):
        _ = env_ids
        calls.append("curr.compute")
        return None

    cfg = SimpleNamespace(
        # IsaacLab knobs
        decimation=2,
        episode_length_s=1.0,
        sim=SimpleNamespace(dt=0.01),
        # scene stub (used for num_envs/env_spacing patching only)
        scene=SimpleNamespace(num_envs=2, env_spacing=1.0),
        # managers
        actions=SimpleNamespace(),  # no action terms needed for ordering assertions
        observations=SimpleNamespace(
            policy=SimpleNamespace(
                enable_corruption=False,
                concatenate_terms=True,
                o=SimpleNamespace(func=_obs, params={}),
            )
        ),
        rewards=SimpleNamespace(r=SimpleNamespace(func=_reward, params={}, weight=1.0)),
        terminations=SimpleNamespace(done=SimpleNamespace(func=_done, params={}, time_out=False)),
        commands=SimpleNamespace(motion=SimpleNamespace(class_type=_CmdTerm)),
        events=SimpleNamespace(startup_evt=SimpleNamespace(func=_startup_event, mode="startup", params={})),
        curriculum=SimpleNamespace(c=SimpleNamespace(func=_curriculum, params={})),
        recorders=SimpleNamespace(trace=SimpleNamespace(class_type=_RecorderTerm)),
    )

    robot_cfg = RobotCfg(
        name="robot",
        joint_limits={"j0": (-1.0, 1.0), "j1": (-1.0, 1.0)},
        default_joint_positions={".*": 0.0},
    )
    scenario = _FakeScenario(robots=[robot_cfg], num_envs=2, decimation=1, headless=True)

    env = HandlerBackedManagerBasedRLEnv(
        scenario=scenario,
        cfg=cfg,
        args=SimpleNamespace(),
        strict=True,
        reset_in_env_wrapper=True,  # avoid implicit reset during __init__
    )

    # ------------------------------------------------------------------
    # reset() ordering
    # ------------------------------------------------------------------
    calls.clear()
    env.reset()
    assert calls == [
        "rec.pre_reset",
        "cmd.reset",
        "cmd.compute",
        "event.startup",
        "obs",
        "rec.post_reset",
    ]

    # ------------------------------------------------------------------
    # step() ordering (includes auto-reset)
    # ------------------------------------------------------------------
    calls.clear()
    actions = torch.zeros((env.num_envs, 0), device=env.device, dtype=torch.float32)
    env.step(actions)
    assert calls == [
        "rec.pre_step",
        "rec.post_physics",
        "rec.post_physics",
        "reward",
        "done",
        "rec.post_step",
        "rec.pre_reset",
        "curr.compute",
        "cmd.reset",
        "cmd.compute",
        "rec.post_reset",
        "cmd.compute",
        "obs",
    ]

    # ------------------------------------------------------------------
    # episodic logging surface (RSL-RL picks up extras["log"])
    # ------------------------------------------------------------------
    assert "log" in env.extras
    assert "Episode_Reward/r" in env.extras["log"]
    assert "Episode_Termination/done" in env.extras["log"]
