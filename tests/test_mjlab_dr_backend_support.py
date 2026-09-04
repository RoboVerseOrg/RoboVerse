"""Regression: mjlab DR events must not silently no-op on non-MuJoCo backends.

Every DR event in ``roboverse_pack/tasks/mjlab/mdp/events_dr.py`` used to start
with ``if not hasattr(env.handler, "physics"): return`` — and only the MuJoCo
handler sets ``.physics``. Since ``MujocoHandler`` rejects ``num_envs > 1``, any
real (multi-env) RL run of ``mjlab:velocity_go1_v2`` had to use Newton or MJX,
where friction / mass / COM / encoder-bias randomization all vanished without a
word: the Go1 policy trained with zero domain randomization while its cfg said
otherwise. ``push_by_setting_velocity`` additionally ended in
``except Exception: pass``.

These tests pin the fixed contract, using stub handlers (no GPU, no simulator):
  - Newton: the events actually write the Newton model fields.
  - unsupported backends (mjx, …): the events raise, naming event and backend.
  - push_by_setting_velocity: a failing ``set_states`` propagates.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.types import RobotState, TensorState
from roboverse_pack.tasks.mjlab.mdp.events_dr import (
    body_com_offset,
    body_mass,
    encoder_bias,
    geom_friction,
    push_by_setting_velocity,
)
from roboverse_pack.tasks.mjlab.mdp.scene_entity import SceneEntityCfg

NUM_ENVS = 2
BODIES = ("trunk", "FR_calf", "FL_calf")
JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")


class _WpArray:
    """Minimal stand-in for a ``warp.array``: host copy on read, explicit assign."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)

    def numpy(self):
        return self._data.copy()  # warp copies device -> host, so writes need assign()

    def assign(self, host):
        self._data = np.asarray(host, dtype=np.float64).copy()


class _FakeNewtonModel:
    """Newton flattens every env's bodies/shapes into one model; so does this."""

    def __init__(self):
        # body ids: env0 -> 0..2, env1 -> 3..5 ; one collision shape per body.
        self.body_key = [name for _ in range(NUM_ENVS) for name in BODIES]
        self.shape_key = [f"{name}_collision" for _ in range(NUM_ENVS) for name in BODIES]
        self.body_shapes = {i: [i] for i in range(len(self.body_key))}
        n = len(self.body_key)
        self.body_mass = _WpArray([5.0] * n)
        self.body_inv_mass = _WpArray([1.0 / 5.0] * n)
        self.body_inertia = _WpArray([np.eye(3) for _ in range(n)])
        self.body_inv_inertia = _WpArray([np.eye(3) for _ in range(n)])
        self.body_com = _WpArray([[0.0, 0.0, 0.0]] * n)
        self.shape_material_mu = _WpArray([1.0] * n)
        self.shape_material_torsional_friction = _WpArray([0.005] * n)
        self.shape_material_rolling_friction = _WpArray([0.0001] * n)


class _FakeHandler:
    """Handler without ``.physics`` — i.e. anything that is not MuJoCo."""

    def __init__(self, simulator: str, robot_name: str = "go1"):
        self.scenario = SimpleNamespace(
            simulator=simulator,
            num_envs=NUM_ENVS,
            robots=[SimpleNamespace(name=robot_name)],
        )
        self.num_envs = NUM_ENVS
        self.device = torch.device("cpu")
        self._model = _FakeNewtonModel()
        self._robot_name = robot_name
        self.set_states_calls: list[TensorState] = []
        self.set_states_error: Exception | None = None
        self.root_state = torch.zeros((NUM_ENVS, 13))
        self.root_state[:, 3] = 1.0  # unit quat

    def _get_body_indices(self, env_id: int, obj_name: str) -> list[int]:
        if obj_name != self._robot_name:
            return []
        return [env_id * len(BODIES) + i for i in range(len(BODIES))]

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if obj_name != self._robot_name:
            raise KeyError(obj_name)
        return sorted(JOINTS) if sort else list(JOINTS)

    def get_states(self, mode: str = "tensor") -> TensorState:
        return TensorState(
            objects={},
            robots={self._robot_name: RobotState(root_state=self.root_state.clone())},
            cameras={},
        )

    def set_states(self, states: TensorState, env_ids=None) -> None:
        if self.set_states_error is not None:
            raise self.set_states_error
        self.set_states_calls.append(states)


def _make_env(simulator: str):
    handler = _FakeHandler(simulator)
    env = SimpleNamespace(
        handler=handler,
        scenario=handler.scenario,
        num_envs=NUM_ENVS,
        device=torch.device("cpu"),
    )
    return env, handler


def _foot_cfg() -> SceneEntityCfg:
    cfg = SceneEntityCfg("go1")
    cfg.geom_names = ("FR_calf_collision", "FL_calf_collision")
    return cfg


def _trunk_cfg() -> SceneEntityCfg:
    cfg = SceneEntityCfg("go1")
    cfg.body_names = ("trunk",)
    return cfg


# ---------------------------------------------------------------------------
# Newton: the randomization must actually land in the model
# ---------------------------------------------------------------------------


@pytest.mark.general
def test_geom_friction_writes_newton_shape_friction():
    env, handler = _make_env("newton")
    before = handler._model.shape_material_mu.numpy()

    geom_friction(env, asset_cfg=_foot_cfg(), ranges=(0.3, 1.2), operation="abs", axes=(0,), shared_random=True)

    after = handler._model.shape_material_mu.numpy()
    # env0 calf shapes are ids 1,2 ; env1 calf shapes are ids 4,5 ; trunk (0, 3) untouched.
    calf_ids = [1, 2, 4, 5]
    assert not np.allclose(after[calf_ids], before[calf_ids]), "friction DR silently no-opped on Newton"
    assert np.all((after[calf_ids] >= 0.3) & (after[calf_ids] <= 1.2))
    assert np.allclose(after[[0, 3]], before[[0, 3]]), "trunk shapes must not be randomized"
    # shared_random shares one sample within an env, but envs stay independent.
    assert after[1] == after[2] and after[4] == after[5]


@pytest.mark.general
def test_body_mass_writes_newton_body_mass_and_inverse():
    env, handler = _make_env("newton")

    body_mass(env, asset_cfg=_trunk_cfg(), operation="mul", ranges=(1.5, 1.6))

    mass = handler._model.body_mass.numpy()
    inv_mass = handler._model.body_inv_mass.numpy()
    trunk_ids = [0, 3]  # trunk of env0 / env1
    assert np.all(mass[trunk_ids] > 5.0), "mass DR silently no-opped on Newton"
    assert np.all((mass[trunk_ids] >= 5.0 * 1.5) & (mass[trunk_ids] <= 5.0 * 1.6))
    assert np.allclose(inv_mass[trunk_ids], 1.0 / mass[trunk_ids])
    assert np.allclose(mass[[1, 2, 4, 5]], 5.0), "only the selected body may change"
    # inertia is rescaled by the mass ratio so the body stays physical.
    inertia = handler._model.body_inertia.numpy()
    assert np.allclose(inertia[0], np.eye(3) * (mass[0] / 5.0))


@pytest.mark.general
def test_body_com_offset_writes_newton_body_com():
    env, handler = _make_env("newton")

    body_com_offset(env, asset_cfg=_trunk_cfg(), operation="add", ranges={0: (0.02, 0.025), 2: (-0.03, -0.02)})

    com = handler._model.body_com.numpy()
    assert np.all(com[[0, 3], 0] >= 0.02), "COM DR silently no-opped on Newton"
    assert np.all(com[[0, 3], 2] <= -0.02)
    assert np.allclose(com[[0, 3], 1], 0.0), "unlisted axis must stay untouched"
    assert np.allclose(com[[1, 2, 4, 5]], 0.0), "only the selected body may change"


@pytest.mark.general
def test_encoder_bias_populates_env_buffer_without_physics():
    env, _ = _make_env("newton")
    cfg = SceneEntityCfg("go1", joint_names=tuple(sorted(JOINTS)))

    encoder_bias(env, asset_cfg=cfg, bias_range=(-0.015, 0.015))

    bias = getattr(env, "_encoder_bias", None)
    assert bias is not None, "encoder-bias DR silently no-opped on Newton"
    assert bias.shape == (NUM_ENVS, len(JOINTS))
    assert torch.all(bias.abs() <= 0.015)
    assert torch.any(bias != 0.0)


# ---------------------------------------------------------------------------
# Unsupported backends must fail loudly, not quietly
# ---------------------------------------------------------------------------


@pytest.mark.general
@pytest.mark.parametrize(
    ("event", "kwargs"),
    [
        (geom_friction, {"asset_cfg": _foot_cfg()}),
        (body_mass, {"asset_cfg": _trunk_cfg()}),
        (body_com_offset, {"asset_cfg": _trunk_cfg(), "ranges": {0: (-0.02, 0.02)}}),
    ],
)
def test_unsupported_backend_raises_naming_event_and_backend(event, kwargs):
    env, _ = _make_env("mjx")
    with pytest.raises(NotImplementedError) as excinfo:
        event(env, **kwargs)
    message = str(excinfo.value)
    assert event.__name__ in message
    assert "mjx" in message


# ---------------------------------------------------------------------------
# push_by_setting_velocity must not swallow handler failures
# ---------------------------------------------------------------------------


@pytest.mark.general
def test_push_by_setting_velocity_applies_root_velocity():
    env, handler = _make_env("newton")

    push_by_setting_velocity(env, None, velocity_range={"x": (0.4, 0.5), "yaw": (0.7, 0.8)})

    assert len(handler.set_states_calls) == 1, "push silently skipped set_states"
    pushed = handler.set_states_calls[0].robots["go1"].root_state
    assert torch.all(pushed[:, 7] >= 0.4) and torch.all(pushed[:, 7] <= 0.5)  # lin_vel x
    assert torch.all(pushed[:, 12] >= 0.7) and torch.all(pushed[:, 12] <= 0.8)  # ang_vel yaw
    assert torch.all(pushed[:, 8:12] == 0.0)  # untouched components


@pytest.mark.general
def test_push_by_setting_velocity_propagates_set_states_failure():
    env, handler = _make_env("newton")
    handler.set_states_error = RuntimeError("handler cannot write root velocity")

    with pytest.raises(RuntimeError, match="handler cannot write root velocity"):
        push_by_setting_velocity(env, None, velocity_range={"x": (-0.5, 0.5)})


@pytest.mark.general
def test_push_by_setting_velocity_pushes_scene_mjcf_free_joint():
    """MuJoCo Go1 is a scene MJCF with ``scenario.robots = []`` (see velocity_go1_v2).

    The robot is then absent from the TensorState, which used to make the push
    return silently — so the MuJoCo Go1 never got pushed either. It must now
    land on the floating base's free-joint dofs.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='trunk' pos='0 0 1'>"
        "<freejoint/><geom type='box' size='.1 .1 .1'/>"
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    handler = SimpleNamespace(
        physics=SimpleNamespace(model=model, data=data),
        scenario=SimpleNamespace(simulator="mujoco", num_envs=1, robots=[]),
        get_states=lambda mode="tensor": TensorState(objects={}, robots={}, cameras={}),
    )
    env = SimpleNamespace(handler=handler, scenario=handler.scenario, num_envs=1, device=torch.device("cpu"))

    push_by_setting_velocity(env, None, velocity_range={"x": (0.4, 0.5), "yaw": (0.7, 0.8)})

    qvel = np.asarray(data.qvel[:6])
    assert 0.4 <= qvel[0] <= 0.5, "push silently no-opped on the MuJoCo scene-MJCF path"
    assert 0.7 <= qvel[5] <= 0.8
    assert np.allclose(qvel[1:5], 0.0)


def _seed(seed: int) -> None:
    """Seed the way a task's ``reset(seed=)`` does: through the handler contract, not by hand."""
    from metasim.sim.base import BaseSimHandler

    BaseSimHandler.set_seed(None, seed)  # seeds python / numpy / torch; the method uses no handler state


@pytest.mark.general
def test_dr_draws_are_reproducible_through_set_seed():
    """The samplers draw from the RNG ``set_seed`` seeds: same seed, same DR; different seed, different DR."""
    from roboverse_pack.tasks.mjlab.mdp.events_dr import _make_sampler

    _seed(7)
    a = _make_sampler((0.5, 1.5), "uniform")((4,))
    _seed(7)
    assert np.array_equal(a, _make_sampler((0.5, 1.5), "uniform")((4,)))
    _seed(8)
    assert not np.array_equal(a, _make_sampler((0.5, 1.5), "uniform")((4,)))

    def com_after(seed):
        _seed(seed)
        env, handler = _make_env("newton")
        body_com_offset(env, asset_cfg=_trunk_cfg(), operation="add", ranges={0: (0.02, 0.025), 2: (-0.03, -0.02)})
        return handler._model.body_com.numpy().copy()

    def bias_after(seed):
        _seed(seed)
        env, _ = _make_env("newton")
        encoder_bias(
            env, asset_cfg=SceneEntityCfg("go1", joint_names=tuple(sorted(JOINTS))), bias_range=(-0.015, 0.015)
        )
        return env._encoder_bias.clone()

    assert np.array_equal(com_after(3), com_after(3)) and not np.array_equal(com_after(3), com_after(4))
    assert torch.equal(bias_after(3), bias_after(3)) and not torch.equal(bias_after(3), bias_after(4))
