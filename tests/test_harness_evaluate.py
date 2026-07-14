"""Tests for ``roboverse_learn.eval.harness._evaluate`` — the ``evaluate()`` entry point.

Backend-free: a fake task class (registered through a patched ``get_task_class``) stands in for a
simulator, so ``evaluate``/``EvalResult``/``ParityReport``, the camera wiring, the connect-time
negotiation, the ``RLTaskEnv`` rejection, and the ``env.close()`` / ``policy.close()`` lifecycle
are all exercised without mujoco.
"""

from __future__ import annotations

import pytest
import torch

from roboverse_learn.eval.harness import _evaluate as ev
from roboverse_learn.eval.harness.obs import ActionBatch
from roboverse_learn.eval.harness.policy import PolicyCard
from roboverse_learn.eval.harness.spec import ActionSpec, FieldSpec, ObsSpec, Space


class _Act:
    def __init__(self, is_ee=False):
        self.is_ee = is_ee


class _Robot:
    def __init__(self, name, joints, ee_joints=()):
        self.name = name
        self.joint_limits = {j: (0.0, 1.0) for j in joints}
        self.actuators = {j: _Act(is_ee=(j in ee_joints)) for j in joints}
        self.gripper_open_q = None
        self.gripper_joint_name = None
        self.ee_body_name = None


def _franka():
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    grip = ["panda_finger_joint1", "panda_finger_joint2"]
    return _Robot("franka", arm + grip, ee_joints=grip)


class _Cam:
    def __init__(self, rgb):
        self.rgb = rgb
        self.depth = None


class _RS:
    def __init__(self, jp):
        self.joint_pos = jp
        self.joint_pos_target = jp.clone()
        self.body_state = None
        self.body_names = None


class _States:
    def __init__(self, robots, cameras):
        self.robots = robots
        self.cameras = cameras


class _Handler:
    def __init__(self, order, cameras):
        self._order = order
        self._cameras = cameras
        self.jp = torch.zeros(1, len(order["franka"]))

    def get_joint_names(self, robot, sort=True):
        return self._order[robot]

    def get_states(self, mode="tensor"):
        return _States({"franka": _RS(self.jp)}, self._cameras)


class _Scenario:
    """Stand-in for ScenarioCfg: only what _run_one touches."""

    def __init__(self):
        self.robots = [_franka()]
        self.cameras = []
        self.simulator = "mujoco"
        self.num_envs = 1
        self.headless = True

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


class _FakeTaskEnv:
    """Fake BaseTaskEnv: terminates (success) at step 2 on 'mujoco', never on any other sim."""

    scenario = _Scenario()
    max_episode_steps = 10
    closed = 0

    def __init__(self, scenario):
        self.scenario = scenario
        self.num_envs = scenario.num_envs
        self.device = torch.device("cpu")
        cams = {}
        for c in scenario.cameras or []:
            cams[c.name] = _Cam(torch.randint(0, 255, (self.num_envs, c.height, c.width, 3), dtype=torch.uint8))
        order = {"franka": sorted(scenario.robots[0].joint_limits.keys())}
        self.handler = _Handler(order, cams)
        self._step = 0
        self.applied = []

    def step(self, target):
        self.applied.append(target)
        self._step += 1
        term = torch.tensor([self._step == 2 and self.scenario.simulator == "mujoco"] * self.num_envs)
        return (None, None, term, torch.zeros(self.num_envs, dtype=torch.bool), None)

    def reset(self, seed=None, env_ids=None):
        self._step = 0
        return None

    def close(self):
        type(self).closed += 1


@pytest.fixture
def fake_task(monkeypatch):
    _FakeTaskEnv.scenario = _Scenario()  # fresh per test (scenario.update mutates in place)
    _FakeTaskEnv.closed = 0
    monkeypatch.setattr(ev, "get_task_class", lambda name: _FakeTaskEnv)
    return _FakeTaskEnv


class _HoldPolicy:
    """Echoes the joint-space obs back as the target; records whether close() ran."""

    def __init__(self):
        self.aspec = None
        self.ospec = None
        self.closed = 0
        self.seen_task = []

    def describe(self):
        return PolicyCard("hold", ObsSpec(()), ActionSpec((), chunk_len=1))

    def bind(self, obs_spec, action_spec):
        self.ospec = obs_spec
        self.aspec = action_spec

    def reset(self, env_ids):
        pass

    def act(self, obs):
        self.seen_task.append(dict(obs.task))
        t = {f.key: obs.tensors.get(f.key, torch.zeros(obs.batch_size, *f.shape)) for f in self.aspec.fields}
        return ActionBatch(self.aspec, obs.env_ids, t)

    def close(self):
        self.closed += 1


@pytest.mark.general
def test_evaluate_single_sim_returns_evalresult(fake_task):
    policy = _HoldPolicy()
    res = ev.evaluate("fake.task", policy, simulators="mujoco", episodes=2, num_envs=1, max_steps=5)
    assert isinstance(res, ev.EvalResult)
    assert res.task == "fake.task" and res.simulator == "mujoco"
    assert res.episodes == 2 and res.successes == 2  # terminates at step 2 in both waves
    assert res.success_rate == 1.0
    assert res.per_episode_success == (True, True)
    assert res.steps_mean == pytest.approx(2.0)
    assert fake_task.closed == 1  # env.close() ran
    assert policy.closed == 1  # ...and policy.close() (it was leaking a ws socket per run)


@pytest.mark.general
def test_evaluate_multi_sim_returns_parity_report(fake_task):
    rep = ev.evaluate("fake.task", _HoldPolicy(), simulators=["mujoco", "sapien3"], episodes=1, num_envs=1, max_steps=4)
    assert isinstance(rep, ev.ParityReport)
    assert set(rep.results) == {"mujoco", "sapien3"}
    assert rep.results["mujoco"].success_rate == 1.0
    assert rep.results["sapien3"].success_rate == 0.0  # only mujoco's checker fires in the fake
    assert rep.success_rate_spread() == pytest.approx(1.0)
    assert rep.divergent() and not rep.divergent(tol=1.5)
    assert fake_task.closed == 2  # one env per backend, both closed


@pytest.mark.general
def test_parity_report_agreeing_backends_are_not_divergent():
    r = ev.EvalResult(task="t", simulator="a", episodes=4, successes=2)
    r2 = ev.EvalResult(task="t", simulator="b", episodes=4, successes=2)
    rep = ev.ParityReport(task="t", results={"a": r, "b": r2})
    assert rep.success_rate_spread() == 0.0 and not rep.divergent()
    assert ev.ParityReport(task="t").success_rate_spread() == 0.0  # no results -> no spread


@pytest.mark.general
def test_evaluate_rejects_rl_task_env(monkeypatch):
    # RLTaskEnv auto-resets in step(), which silently corrupts the wave-based episode accounting.
    from metasim.task.rl_task import RLTaskEnv

    class _FakeRL(RLTaskEnv):
        scenario = _Scenario()
        closed = 0

        def __init__(self, scenario):  # skip the real (backend-building) __init__
            self.scenario = scenario

        def close(self):
            type(self).closed += 1

    monkeypatch.setattr(ev, "get_task_class", lambda name: _FakeRL)
    with pytest.raises(NotImplementedError, match="RLTaskEnv"):
        ev.evaluate("fake.rl", _HoldPolicy(), episodes=1, max_steps=2)
    assert _FakeRL.closed == 1  # the env is still closed on the error path


@pytest.mark.general
def test_evaluate_camera_policy_end_to_end(fake_task):
    # BLOCKER regression: evaluate() had NO way to supply a camera, so a vision/VLA policy — the
    # harness's headline use case — could never be satisfied: the derived obs spec had no camera
    # field and negotiation failed with "missing required field".
    from metasim.scenario.cameras import PinholeCameraCfg

    class _VisionPolicy(_HoldPolicy):
        def describe(self):
            needs = ObsSpec((FieldSpec("camera0.rgb", Space.RGB, (24, 32, 3), dtype="uint8", frame="camera0"),))
            return PolicyCard("vision", needs, ActionSpec((), chunk_len=1))

        def act(self, obs):
            assert obs.tensors["camera0.rgb"].shape == (obs.batch_size, 24, 32, 3)
            self.saw_pixels = int(obs.tensors["camera0.rgb"].abs().sum())
            return super().act(obs)

    cam = PinholeCameraCfg(name="camera0", data_types=["rgb"], width=32, height=24)
    policy = _VisionPolicy()
    res = ev.evaluate("fake.task", policy, episodes=1, num_envs=1, max_steps=3, cameras=[cam])
    assert res.successes == 1
    assert policy.saw_pixels > 0  # real pixels reached the policy, not a black fallback

    # ...and without cameras= the negotiation fails at connect, naming the missing field
    with pytest.raises(ValueError, match=r"camera0\.rgb"):
        ev.evaluate("fake.task", _VisionPolicy(), episodes=1, num_envs=1, max_steps=3)


@pytest.mark.general
def test_evaluate_rejects_policy_with_incompatible_action_control(fake_task):
    # a policy declaring control="ee_pose" used to be handed a joint_pos spec silently
    class _EePolicy(_HoldPolicy):
        def describe(self):
            produces = ActionSpec(
                (FieldSpec("arm.ee_pose", Space.EE_POSE, (7,), chain="arm"),), control="ee_pose", chunk_len=1
            )
            return PolicyCard("ee", ObsSpec(()), produces)

    with pytest.raises(ValueError, match="ee_pose"):
        ev.evaluate("fake.task", _EePolicy(), episodes=1, num_envs=1, max_steps=2)


@pytest.mark.general
def test_evaluate_language_payload_reaches_policy(fake_task, monkeypatch):
    # ObsBatch.task is the language/goal channel; a task that exposes an instruction must deliver
    # it (and derive_obs_spec must advertise the optional task.language field).
    policy = _HoldPolicy()
    ev.evaluate("fake.task", policy, episodes=1, num_envs=1, max_steps=3)
    assert policy.ospec is not None
    assert "task.language" not in policy.ospec.keys()  # this task exposes no instruction
    assert policy.seen_task == [{}] * len(policy.seen_task)

    monkeypatch.setattr(_FakeTaskEnv, "get_language_instruction", lambda self: "pick up the cube", raising=False)
    policy2 = _HoldPolicy()
    ev.evaluate("fake.task", policy2, episodes=1, num_envs=1, max_steps=3)
    assert "task.language" in policy2.ospec.keys()  # ...now the optional field is advertised
    assert not policy2.ospec.field("task.language").required
    assert policy2.seen_task and all(t == {"language": "pick up the cube"} for t in policy2.seen_task)


@pytest.mark.general
def test_evaluate_rejects_policy_without_bind(fake_task):
    class _NoBind:
        def describe(self):
            return PolicyCard("nobind", ObsSpec(()), ActionSpec((), chunk_len=1))

        def reset(self, env_ids):
            pass

        def act(self, obs):
            raise AssertionError("not reached")

    with pytest.raises(TypeError, match="bind"):
        ev.evaluate("fake.task", _NoBind(), episodes=1, num_envs=1, max_steps=2)


@pytest.mark.general
def test_evaluate_malformed_camera_raises(fake_task):
    from metasim.scenario.cameras import PinholeCameraCfg

    depth_only = PinholeCameraCfg(name="camera0", data_types=["depth"], width=32, height=24)
    with pytest.raises(ValueError, match="rgb"):
        ev.evaluate("fake.task", _HoldPolicy(), episodes=1, num_envs=1, max_steps=2, cameras=[depth_only])
