# RoboVerse unified policy-evaluation harness

One typed, embodiment-general, multi-backend harness for evaluating (and deploying) a
policy against any RoboVerse task. See `../DESIGN.md` for the requirements it is built
against and the known gaps.

## Quickstart

```python
from roboverse_learn.eval.harness import evaluate
from roboverse_learn.eval.harness.adapters import ZeroActionPolicy   # or your own Policy

# single backend (num_envs=1 runs anywhere; num_envs>1 on MuJoCo uses OS-process fan-out,
# so run it from a script under `if __name__ == "__main__":`, not a REPL/`python -` — see Scope)
res = evaluate("maniskill.pick_cube", ZeroActionPolicy(), episodes=10, num_envs=1)
print(res.success_rate)

# multi-sim parity: one policy across backends
rep = evaluate("maniskill.pick_cube", ZeroActionPolicy(), simulators=["mujoco", "sapien3"])
print(rep.success_rate_spread(), rep.divergent())
```

## Writing a policy (the whole contract)

Implement one class — the harness handles embodiment inference, typed obs/action
derivation, vectorized rollout, chunk scheduling, and multi-sim parity. No per-policy
eval-loop boilerplate, no per-arm key convention, no manual batching. See
`adapters/template.py`.

Four methods (all four are in the `Policy` protocol), plus an optional `close()`:

```python
class MyPolicy:
    def describe(self):                      # advertise chunk_len (+ the obs/action you declare)
        return PolicyCard("mine", ObsSpec(()), ActionSpec((), chunk_len=1))
    def bind(self, obs_spec, action_spec):   # harness hands you the derived typed specs
        self.aspec = action_spec
    def reset(self, env_ids): ...            # clear per-env state (partial reset)
    def act(self, obs):                      # ObsBatch (B active envs) -> ActionBatch
        # obs.tensors: canonical-key -> (B, *shape) batched tensors on the sim device
        # obs.task:    non-tensor payload (e.g. {"language": ...}) when the task exposes one
        return ActionBatch(self.aspec, obs.env_ids, {...})
    def close(self): ...                     # optional; evaluate() calls it once at the end
```

Whatever you declare in the card is **checked at connect** against the specs the harness derived
— in both directions (the obs fields you need, and the action fields/`control` you produce). A
mismatch is an error before the rollout starts, not a `KeyError` mid-episode. Declare nothing
(empty specs) to accept whatever `bind()` gives you.

## Environment isolation (policy in its own env)

A policy with conflicting dependencies runs in its **own process/env**; the simulator
runs here; they talk over a *typed* WebSocket. The spec travels with the data, so a
schema mismatch is a typed error at connect rather than a `KeyError` mid-rollout.

```bash
# terminal 1 — policy in its own conda env
python -m roboverse_learn.eval.harness.transport.serve --policy zero --port 8799
```
```python
# terminal 2 — the sim process
from roboverse_learn.eval.harness import evaluate
from roboverse_learn.eval.harness.transport.base import PolicyHandle
from roboverse_learn.eval.harness.transport.ws import WsPolicyTransport
handle = PolicyHandle(WsPolicyTransport("ws://localhost:8799"))
evaluate("maniskill.pick_cube", handle, episodes=10, num_envs=4)
```

## What's inside

| Module | Responsibility |
|---|---|
| `embodiment.py` | `infer_embodiment(list[RobotCfg])` → arms/grippers/base/head/torso/legs, no arm-count ceiling |
| `spec.py` | typed `ObsSpec`/`ActionSpec`, canonical `<chain>.<space>` keys, connect-time `SpecMatch` negotiation |
| `obs.py` | `ObsBatch`/`ActionBatch` — torch-native, num_envs-major carriers |
| `chunking.py` | `TemporalEnsembler` (matched to il's temporal-agg; ring-buffered over the chunk horizon) + `ChunkScheduler` |
| `env_adapter.py` | `BaseTaskEnv` ↔ typed carriers, vectorized tensor-action path |
| `runner.py` | `VecEvalRunner` — wave-based vectorized rollout, per-env episodes |
| `_evaluate.py` | `evaluate()` → `EvalResult` / `ParityReport` |
| `adapters/` | baseline policies (zero/hold-pose/random) + `template.py` |
| `transport/` | in-proc (zero-copy) + ws (isolated), typed frames |

## Vision policies (cameras)

Registered tasks ship **no camera**, so a policy that consumes pixels must be given one —
`evaluate(cameras=[...])` takes the repo's own `CameraCfg`s and puts them in the scenario, and
the harness derives a `<cam>.rgb` obs field for each:

```python
from metasim.scenario.cameras import PinholeCameraCfg

cam = PinholeCameraCfg(name="camera0", data_types=["rgb"], width=64, height=64,
                       pos=(1.0, 0.0, 0.75), look_at=(0.0, 0.0, 0.0))
# the policy declares FieldSpec("camera0.rgb", Space.RGB, (64, 64, 3), dtype="uint8") in its card
evaluate("maniskill.pick_cube", VisionPolicy(), episodes=1, cameras=[cam])
```

Headless MuJoCo rendering needs a GL context: run with `MUJOCO_GL=egl`. If the backend renders
nothing, the adapter **raises** (it does not hand the policy a black image).

## Performance (design property, not a measured number)

The action path builds one `(num_envs, total_dof)` target tensor and scatters per-chain slices
into it (`env_adapter.apply`) instead of constructing `num_envs` python `{joint: float}` dicts via
`.tolist()`, so action construction is one tensor op regardless of `num_envs` rather than a python
loop over envs. That is the regime the harness exists for (batched GPU backends, where the sim
step is batched too). **No benchmark is shipped in this tree, so no speedup number is claimed
here** — measure it on your backend before relying on it.

## Scope / limits (honest)

- **No GPU simulator backend has been run through the harness.** The CPU-only device bug is fixed
  (bookkeeping is allocated on the sim device and the step flags are normalized onto it) and is
  tested with real CUDA tensors against a *fake* backend, but mjx/isaacgym/isaacsim/newton have
  not been exercised end-to-end. MuJoCo (CPU) is the backend that is actually run.
- **`episodes=N` may not be N *independent* episodes.** For tasks whose initial state is a
  fixed replayed trajectory, every wave restores the same state, so the episodes are
  duplicates and `success_rate` carries less evidence than the count suggests.
- One negotiation guarantee is still advisory: a dtype `cast` (e.g. `uint8`→`float32` rgb) is
  recorded in `SpecMatch.plan` but **not applied** — the policy receives the producer's dtype.
  Pose `frame` mismatches *are* enforced (derived `ee_pose` fields are `frame="world"`).
- Missing obs is an **error**, not a zero: a backend without `body_state` (pybullet, genesis)
  raises for an `ee_pose` field — pass `include_ee_pose=False` — and an unrendered camera raises.
- `control="joint_pos"` implemented; `control="ee_pose"` derives a spec but `EnvAdapter` raises
  (needs cuRobo IK, GPU); `control="joint_vel"/"effort"` are rejected.
- `ActionSpec` has no `frequency`: the runner steps the policy at the sim rate and applies no
  decimation, so there is no control-Hz field to lie about. Chunked policies use `chunk_len`.
- Targets `BaseTaskEnv` checker tasks; `RLTaskEnv` (auto-reset in step) is rejected — use
  `roboverse_learn.rl` for RL rollouts.
- The remote (`ws`) path needs `pip install websockets msgpack msgpack-numpy`; the pure
  core and in-proc path need neither. `WsPolicyTransport(url, timeout=…)` caps **every**
  round-trip (default 60s) — raise it for a slow remote model whose single `infer`/`bind`
  (checkpoint load) exceeds it: `WsPolicyTransport(url, timeout=300)`. After any
  `WsProtocolError` the socket is closed and the handle is unusable by design — build a new one.
- `serve_policy(policy=…)` serves ONE client (a second connection is rejected — it would re-bind
  the shared instance). Use `serve_policy(factory=…)` to serve several sim processes at once; the
  `transport.serve` CLI already does.
- `num_envs > 1` on MuJoCo uses `ParallelSimWrapper` (OS-process fan-out) — run from a real
  script, not `python -` stdin (multiprocessing can't re-import stdin).
