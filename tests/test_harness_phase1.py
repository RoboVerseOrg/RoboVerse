"""Phase-1 tests: EnvAdapter tensor-action layout + VecEvalRunner wave counting.

Backend-free — a FakeHandler/FakeEnv stands in for a simulator so the adapter's
obs-slicing and the runner's episode bookkeeping are unit-tested without mujoco.
"""

from __future__ import annotations

import pytest
import torch

from roboverse_learn.eval.harness.embodiment import infer_embodiment
from roboverse_learn.eval.harness.env_adapter import EnvAdapter
from roboverse_learn.eval.harness.obs import ActionBatch
from roboverse_learn.eval.harness.runner import VecEvalRunner
from roboverse_learn.eval.harness.spec import derive_action_spec, derive_obs_spec


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
        self.ee_body_name = f"{name}_hand"


def _franka():
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    grip = ["panda_finger_joint1", "panda_finger_joint2"]
    return _Robot("franka", arm + grip, ee_joints=grip)


class _Cam:
    def __init__(self, rgb):
        self.rgb = rgb
        self.depth = None


class _RS:
    def __init__(self, jp, target=None, body_state=None, body_names=None):
        self.joint_pos = jp
        self.joint_pos_target = target
        self.body_state = body_state
        self.body_names = body_names


class _State:
    def __init__(self, robots, cameras=None):
        self.robots = robots
        self.cameras = cameras or {}


class _Handler:
    """Fake handler. ``target`` (joint_pos_target) / ``bodies`` / ``cameras`` are opt-in."""

    def __init__(self, order, jp, target=None, bodies=None, body_state=None, cameras=None):
        self._order = order
        self._jp = jp
        self._target = target or {}
        self._bodies = bodies or {}
        self._body_state = body_state or {}
        self._cameras = cameras or {}

    def get_joint_names(self, robot, sort=True):
        return self._order[robot]

    def get_states(self, mode="tensor"):
        robots = {
            r: _RS(
                self._jp[r].clone(),
                target=self._target.get(r).clone() if r in self._target else None,
                body_state=self._body_state.get(r),
                body_names=self._bodies.get(r),
            )
            for r in self._order
        }
        return _State(robots, self._cameras)

    def get_body_names(self, robot):
        return self._bodies.get(robot)


class _Env:
    def __init__(self, handler, num_envs, device=None):
        self.handler = handler
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.last_target = None

    def step(self, target):
        self.last_target = target
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (None, None, z, z, None)

    def reset(self, **kw):
        return None


def _adapter(num_envs=2, **handler_kw):
    r = _franka()
    emb = infer_embodiment([r])
    # handler joint order = sorted joints; joint_pos initialised to a known per-joint value
    order = {"franka": sorted(r.joint_limits.keys())}
    jp = {"franka": torch.arange(len(order["franka"]), dtype=torch.float32).repeat(num_envs, 1)}
    env = _Env(_Handler(order, jp, **handler_kw), num_envs)
    obs_spec = derive_obs_spec(emb, include_ee_pose=False)
    action_spec = derive_action_spec(emb, control="joint_pos")
    return EnvAdapter(env, emb, obs_spec, action_spec), env, order["franka"]


@pytest.mark.general
def test_env_adapter_action_tensor_layout():
    adapter, env, order = _adapter(num_envs=2)
    # command env 0's arm to all 5.0; gripper to 0.9; leave env 1 to hold
    action = ActionBatch(
        adapter.action_spec,
        torch.tensor([0]),
        {"arm.joint_pos": torch.full((1, 7), 5.0), "gripper.gripper": torch.full((1, 2), 0.9)},
    )
    adapter.apply(action)
    target = env.last_target  # (num_envs, total_dof) in handler sorted-joint order
    assert target.shape == (2, len(order))
    col = {j: i for i, j in enumerate(order)}
    # env 0: arm joints set to 5.0, finger joints to 0.9
    for i in range(1, 8):
        assert target[0, col[f"panda_joint{i}"]] == 5.0
    assert target[0, col["panda_finger_joint1"]] == 0.9 and target[0, col["panda_finger_joint2"]] == 0.9
    # env 1: held at its current joint_pos (= the arange init), NOT the commanded values
    for j, i in col.items():
        assert target[1, i] == float(i)


def _cuda_adapter(dev, *, include_ee_pose=False):
    r = _franka()
    emb = infer_embodiment([r])
    order = {"franka": sorted(r.joint_limits.keys())}
    n = len(order["franka"])
    jp = {"franka": torch.arange(n, dtype=torch.float32, device=dev).repeat(2, 1)}
    bodies = {"franka": ["panda_link0", "franka_hand"]}
    body_state = {"franka": torch.zeros(2, 2, 13, device=dev)}
    handler = _Handler(order, jp, target={"franka": jp["franka"].clone()}, bodies=bodies, body_state=body_state)
    env = _Env(handler, 2, device=dev)
    return EnvAdapter(env, emb, derive_obs_spec(emb, include_ee_pose=include_ee_pose), derive_action_spec(emb)), env


@pytest.mark.general
def test_env_adapter_apply_on_cuda_device():
    # BLOCKER regression: apply() must allocate the target on the sim device so a CUDA-state
    # backend (mjx/newton/isaacgym) does not hit a cross-device index-assign crash.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = torch.device("cuda:0")
    adapter, env = _cuda_adapter(dev)
    action = ActionBatch(
        adapter.action_spec,
        torch.tensor([0, 1]),
        {"arm.joint_pos": torch.full((2, 7), 5.0, device=dev), "gripper.gripper": torch.zeros(2, 2, device=dev)},
    )
    adapter.apply(action)  # must not raise
    assert env.last_target.device.type == "cuda"


@pytest.mark.general
def test_env_adapter_obs_on_cuda_device():
    # follow-up to the apply() BLOCKER: every obs tensor (incl. ee_pose) must land on the sim
    # device, else the ObsBatch is mixed-device on CUDA.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = torch.device("cuda:0")
    adapter, _ = _cuda_adapter(dev, include_ee_pose=True)
    obs = adapter.obs_batch(torch.tensor([0, 1]))
    assert "arm.ee_pose" in obs.tensors
    for key, t in obs.tensors.items():
        assert t.device.type == "cuda", f"{key} on {t.device}"


@pytest.mark.general
def test_env_adapter_holds_inactive_envs_at_joint_target():
    # BLOCKER regression: the "hold" for envs NOT in the ActionBatch used the MEASURED joint_pos,
    # so a position-controlled arm re-targeted its gravity-sagged pose every step and drooped.
    # RobotState.joint_pos_target is the correct source.
    r = _franka()
    order = {"franka": sorted(r.joint_limits.keys())}
    n = len(order["franka"])
    measured = torch.zeros(4, n)  # "sagged" measurement
    commanded = torch.full((4, n), 3.0)  # the standing target the sim was last given
    emb = infer_embodiment([r])
    handler = _Handler(order, {"franka": measured}, target={"franka": commanded})
    env = _Env(handler, 4)
    adapter = EnvAdapter(env, emb, derive_obs_spec(emb, include_ee_pose=False), derive_action_spec(emb))
    action = ActionBatch(
        adapter.action_spec,
        torch.tensor([0]),  # only env 0 is commanded; envs 1-3 must HOLD
        {"arm.joint_pos": torch.full((1, 7), 5.0), "gripper.gripper": torch.full((1, 2), 0.9)},
    )
    adapter.apply(action)
    held = env.last_target[1:]
    assert torch.allclose(held, torch.full_like(held, 3.0)), held  # the target, not the 0.0 measurement


@pytest.mark.general
def test_env_adapter_camera_obs_reaches_policy():
    # regression: EnvAdapter._camera was dead code — evaluate() had no way to add a camera, so a
    # vision policy could never be satisfied. A declared camera field must carry real pixels.
    r = _franka()
    emb = infer_embodiment([r])
    order = {"franka": sorted(r.joint_limits.keys())}
    jp = {"franka": torch.zeros(2, len(order["franka"]))}
    rgb = torch.randint(0, 255, (2, 8, 6, 3), dtype=torch.uint8)
    handler = _Handler(order, jp, cameras={"camera0": _Cam(rgb)})
    env = _Env(handler, 2)
    obs_spec = derive_obs_spec(emb, cameras=[("camera0", (8, 6))], include_ee_pose=False)
    adapter = EnvAdapter(env, emb, obs_spec, derive_action_spec(emb))
    obs = adapter.obs_batch(torch.tensor([0, 1])).validate()
    assert obs.tensors["camera0.rgb"].shape == (2, 8, 6, 3)
    assert torch.equal(obs.tensors["camera0.rgb"], rgb)


@pytest.mark.general
def test_env_adapter_missing_obs_raises_not_zeros():
    # AGENTS: never turn an unsupported path into a quiet no-op. A camera the backend did not
    # render (or per-body state a backend does not expose) must raise, not yield a black image /
    # a constant identity pose.
    r = _franka()
    emb = infer_embodiment([r])
    order = {"franka": sorted(r.joint_limits.keys())}
    jp = {"franka": torch.zeros(1, len(order["franka"]))}
    env = _Env(_Handler(order, jp), 1)  # no cameras, no body_state
    cam_spec = derive_obs_spec(emb, cameras=[("camera0", (8, 6))], include_ee_pose=False)
    with pytest.raises(RuntimeError, match="camera0"):
        EnvAdapter(env, emb, cam_spec, derive_action_spec(emb)).obs_batch(torch.tensor([0]))
    ee_spec = derive_obs_spec(emb, include_ee_pose=True)
    with pytest.raises(RuntimeError, match="body_state"):
        EnvAdapter(env, emb, ee_spec, derive_action_spec(emb)).obs_batch(torch.tensor([0]))


@pytest.mark.general
def test_env_adapter_language_payload():
    # ObsBatch.task is the language/goal channel Space.TASK advertises; it must actually be
    # populated for tasks that expose an instruction (and stay empty for those that don't).
    adapter, env, _ = _adapter(num_envs=1)
    assert adapter.obs_batch(torch.tensor([0])).task == {}
    env.get_language_instruction = lambda: "pick up the cube"
    adapter2 = EnvAdapter(env, adapter.emb, adapter.obs_spec, adapter.action_spec)
    assert adapter2.obs_batch(torch.tensor([0])).task == {"language": "pick up the cube"}


@pytest.mark.general
def test_env_adapter_empty_embodiment_guard():
    from roboverse_learn.eval.harness.spec import ActionSpec, ObsSpec

    r = _Robot("empty", [])
    emb = infer_embodiment([r]) if r.joint_limits else None
    # a robot with no joints -> empty embodiment -> EnvAdapter must reject, not fail deep later
    from roboverse_learn.eval.harness.embodiment import Embodiment

    env = _Env(_Handler({}, {}), 1)
    with pytest.raises(ValueError):
        EnvAdapter(env, Embodiment(("empty",), ()), ObsSpec(()), ActionSpec(()))


@pytest.mark.general
def test_env_adapter_obs_slicing():
    adapter, env, order = _adapter(num_envs=2)
    obs = adapter.obs_batch(torch.tensor([0, 1]))
    col = {j: i for i, j in enumerate(order)}
    # arm.joint_pos must be the 7 panda_joint values in chain order, per env
    arm = obs.tensors["arm.joint_pos"]
    assert arm.shape == (2, 7)
    expected = torch.tensor([float(col[f"panda_joint{i}"]) for i in range(1, 8)])
    assert torch.allclose(arm[0], expected)
    grip = obs.tensors["gripper.gripper"]
    assert grip.shape == (2, 2)
    assert torch.allclose(grip[0], torch.tensor([float(col["panda_finger_joint1"]), float(col["panda_finger_joint2"])]))


# --------------------------------------------------- runner wave counting
class _ScriptAdapter:
    """Minimal EnvAdapter stand-in: terminates env d at step ``term_at[d]`` each wave.

    Records every applied ActionBatch (``self.applied``) so a test can assert on the action the
    runner actually handed the env, not merely that the rollout finished.
    """

    def __init__(self, num_envs, action_spec, term_at, device=None):
        self.num_envs = num_envs
        self.action_spec = action_spec
        self.device = device or torch.device("cpu")
        self.env = type("E", (), {"task_name": "scripted"})()
        self._term_at = term_at
        self._step = torch.zeros(num_envs, dtype=torch.long)
        self.applied = []

    def reset(self, env_ids=None, seed=None):
        self._step[:] = 0

    def obs_batch(self, env_ids):
        from roboverse_learn.eval.harness.obs import ObsBatch
        from roboverse_learn.eval.harness.spec import ObsSpec

        return ObsBatch(ObsSpec(()), env_ids, {})

    def apply(self, action):
        action.validate()
        self.applied.append({k: v.clone() for k, v in action.tensors.items()})
        term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for d in range(self.num_envs):
            if self._term_at[d] is not None and self._step[d] == self._term_at[d]:
                term[d] = True
        self._step += 1
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (None, None, term, z, None)


class _NullPolicy:
    def reset(self, env_ids):
        pass

    def act(self, obs):
        return ActionBatch(obs.spec, obs.env_ids, {})


@pytest.mark.general
def test_evaluate_is_callable_not_submodule():
    # regression: `from harness import evaluate` must yield the FUNCTION, not a submodule
    # (evaluate.py was renamed to _evaluate.py to avoid the name collision). Check in a
    # fresh interpreter, and also after the eval submodule has been imported.
    import subprocess
    import sys

    code = (
        "import roboverse_learn.eval.harness.demo\n"  # imports the eval submodule first
        "from roboverse_learn.eval.harness import evaluate, EvalResult, ParityReport\n"
        "assert callable(evaluate), type(evaluate)\n"
        "assert isinstance(EvalResult, type) and isinstance(ParityReport, type)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-800:]


@pytest.mark.general
def test_scheduler_perstep_action_passes_validate():
    # the runner's scheduler path emits a FLAT (B,*shape) per-step action but tags it with an
    # action_spec whose chunk_len>1; apply() calls actions.validate() first — it must NOT
    # false-reject the flat action (validate accepts flat OR chunked).
    from roboverse_learn.eval.harness.chunking import ActionChunk, ChunkScheduler

    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    sch = ChunkScheduler(action_spec=spec, num_envs=2)
    chunk = ActionChunk({"arm.joint_pos": torch.zeros(2, 4, 7), "gripper.gripper": torch.zeros(2, 4, 2)}, 4)
    sch.push(chunk, torch.tensor([0, 1]))
    per = sch.action_for(torch.tensor([0, 1]))
    ActionBatch(spec, torch.tensor([0, 1]), per).validate()  # must not raise


@pytest.mark.general
def test_runner_ensemble_mode_applies_ensembled_action():
    # regression: chunk='ensemble' must be wired (TemporalEnsembler) through the runner, not
    # rejected. Assert on the ACTION the runner applied — asserting only `episodes == 2` would
    # pass even if the ensembler emitted all-NaN.
    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos", chunk_len=3)
    adapter = _ScriptAdapter(2, aspec, term_at=[1, None])

    class _ChunkPolicy:
        """Predicts a constant 7.0 for every step of its 3-step chunk."""

        def reset(self, env_ids):
            pass

        def act(self, obs):
            t = {f.key: torch.full((obs.batch_size, 3, *f.shape), 7.0) for f in aspec.fields}
            return ActionBatch(aspec, obs.env_ids, t)

    res = VecEvalRunner(adapter, _ChunkPolicy(), chunk="ensemble").run(episodes=2, max_steps=4)
    assert res.episodes == 2
    assert adapter.applied, "runner applied no action"
    for step in adapter.applied:
        arm = step["arm.joint_pos"]
        assert arm.shape == (2, 7)  # flat per-step action, chunk unrolled by the ensembler
        assert torch.isfinite(arm).all()
        # every prediction targeting this step is 7.0, so any weighted average of them is 7.0
        assert torch.allclose(arm, torch.full_like(arm, 7.0)), arm


@pytest.mark.general
def test_runner_rejects_chunked_policy_in_none_mode():
    # regression: chunk='none' with a chunk_len>1 policy used to reach EnvAdapter.apply and die
    # with a raw torch shape error ("(B,4,7) into (B,7)"); the scheduler/ensemble paths had an
    # actionable guard but 'none' did not.
    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    adapter = _ScriptAdapter(1, aspec, term_at=[None])
    with pytest.raises(ValueError, match="chunk_len=4"):
        VecEvalRunner(adapter, _NullPolicy(), chunk="none")


@pytest.mark.general
def test_runner_allocates_bookkeeping_on_sim_device(monkeypatch):
    # BLOCKER regression: succ/done/ep_len/all_ids were allocated with no device= while
    # terminated/timeout come back on the sim device, so `succ | (active & terminated)` raised
    # "Expected all tensors to be on the same device" on EVERY GPU backend. Guarded on CPU CI by
    # asserting the runner never allocates without an explicit device.
    from roboverse_learn.eval.harness import runner as runner_mod

    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos")
    adapter = _ScriptAdapter(2, aspec, term_at=[1, None])
    seen = []
    for fn in ("zeros", "arange"):
        real = getattr(torch, fn)

        def spy(*a, _real=real, _fn=fn, **kw):
            seen.append((_fn, kw.get("device")))
            return _real(*a, **kw)

        monkeypatch.setattr(runner_mod.torch, fn, spy)
    runner = VecEvalRunner(adapter, _NullPolicy(), chunk="none")
    assert runner.device == adapter.device
    runner.run(episodes=2, max_steps=3)
    assert seen, "runner allocated nothing"
    assert all(dev == adapter.device for _, dev in seen), f"allocation without the sim device: {seen}"


@pytest.mark.general
def test_runner_full_rollout_on_cuda():
    # the real thing the above guards: a backend whose step() returns CUDA tensors must roll out.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = torch.device("cuda:0")
    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos", chunk_len=2)
    adapter = _ScriptAdapter(2, aspec, term_at=[1, None], device=dev)

    class _ChunkPolicy:
        def reset(self, env_ids):
            pass

        def act(self, obs):
            t = {f.key: torch.zeros(obs.batch_size, 2, *f.shape, device=dev) for f in aspec.fields}
            return ActionBatch(aspec, obs.env_ids, t)

    res = VecEvalRunner(adapter, _ChunkPolicy(), chunk="ensemble").run(episodes=2, max_steps=4)
    assert res.episodes == 2 and res.successes == 1  # env0 terminates, env1 times out
    assert adapter.applied[0]["arm.joint_pos"].device.type == "cuda"


@pytest.mark.general
def test_runner_rejects_flat_when_chunked():
    # a policy declaring chunk_len>1 but returning a flat action must fail fast, not silently
    # degrade to chunk_len=1 / no ensembling
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    adapter = _ScriptAdapter(1, spec, term_at=[None])

    class _Liar:
        def reset(self, env_ids):
            pass

        def act(self, obs):  # FLAT (B,*), not the declared (B,4,*)
            return ActionBatch(spec, obs.env_ids, {f.key: torch.zeros(obs.batch_size, *f.shape) for f in spec.fields})

    with pytest.raises(ValueError):
        VecEvalRunner(adapter, _Liar(), chunk="ensemble").run(episodes=1, max_steps=3)


@pytest.mark.general
def test_runner_rejects_unknown_chunk_mode():
    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos")
    adapter = _ScriptAdapter(1, aspec, term_at=[0])
    with pytest.raises(ValueError):
        VecEvalRunner(adapter, _NullPolicy(), chunk="bogus")


@pytest.mark.general
def test_runner_wave_counting_and_success():
    emb = infer_embodiment([_franka()])
    aspec = derive_action_spec(emb, control="joint_pos", chunk_len=1)
    # env0 succeeds at step 2, env1 never terminates (times out)
    adapter = _ScriptAdapter(2, aspec, term_at=[2, None])
    runner = VecEvalRunner(adapter, _NullPolicy(), chunk="none")
    res = runner.run(episodes=4, max_steps=5, seed=0)
    assert res.episodes == 4  # 2 waves x 2 envs
    assert res.successes == 2  # env0 succeeds in each of the 2 waves; env1 times out
    assert res.per_episode_success.count(True) == 2
    # env0 terminated at step index 2 -> ep_len 3; env1 timed out -> 5
    assert res.steps_mean == pytest.approx((3 + 5 + 3 + 5) / 4)
