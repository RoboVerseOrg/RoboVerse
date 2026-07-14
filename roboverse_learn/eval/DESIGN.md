# RoboVerse Unified Policy-Evaluation Harness — Design

RoboVerse's own policy eval/deploy infrastructure: one typed, embodiment-general,
vectorized, multi-backend way to evaluate a policy against any RoboVerse task.

It exists because policy evaluation in this repo had fragmented into six rollout
drivers and a set of near-copy-paste VLA eval scripts, with success/termination
handling reinvented per task family. The harness replaces that with a single contract
built on what RoboVerse already has: typed `RobotCfg`/`ScenarioCfg` configs, a
`BaseTaskEnv` that already reports typed `terminated`/`reward`, and multiple physics
backends behind one handler API.

## Requirements

These are the properties the design commits to. Each is a real constraint that a
policy-eval layer has to satisfy to be useful here.

| # | Requirement | How the design meets it |
|---|---|---|
| 1 | Don't throw away GPU-sim parallelism at the policy boundary | Batched typed carriers → one batched `infer` per step; zero-copy in-process path |
| 2 | A policy must not have to guess string keys or grep camera names | Typed `ObsSpec`/`ActionSpec`, canonical `<chain>.<space>` keys |
| 3 | An env/policy schema mismatch must not be silent | Connect-time `SpecMatch` in **both** directions (obs the policy needs, action it produces) → an actionable error before the rollout. Caveat: the `cast` op in the plan is recorded, not applied (Known gaps) |
| 4 | One key scheme for every embodiment | Canonical keys with no per-arm-count branch |
| 5 | No fixed arm-count ceiling | `Embodiment`/`Chain` graph: k arms, grippers, base, head, torso, legs |
| 6 | Don't re-send static data every step | **Not implemented.** The ws frames re-send the spec with every obs; a static-vs-stream split is future work |
| 7 | A reconnect must not silently corrupt episode/temporal state | Fail-fast: a broken transport is closed and marked unusable (`WsProtocolError`); there is no silent reconnect |
| 8 | One eval loop, not one per policy | One shared `VecEvalRunner`; a policy is one `act()` |
| 9 | Chunking/temporal ensembling belongs in the contract | First-class `ActionSpec.chunk_len` + one `TemporalEnsembler` |
| 10 | Success/termination must come from the task, uniformly | Taken from `BaseTaskEnv.step`'s typed `terminated`/`timeout` (success = the task's own checker fired) |
| 11 | A policy's cross-engine robustness should be measurable, not assumed | `evaluate(simulators=[...])` → `ParityReport` across backends |

## Module layout (`roboverse_learn/eval/`)

```
harness/
  embodiment.py   # Embodiment/Chain, infer_embodiment(RobotCfg) — N-embodiment
  spec.py         # ObsSpec/ActionSpec/FieldSpec, SpecMatch negotiation, derive_*
  obs.py          # ObsBatch/ActionBatch — torch-native, num_envs-major carriers
  chunking.py     # ActionChunk, TemporalEnsembler, ChunkScheduler
  env_adapter.py  # BaseTaskEnv <-> ObsBatch/ActionBatch, vectorized; joint_pos only (ee/IK raises)
  runner.py       # VecEvalRunner: num_envs rollout, running mask, wave-based episodes
  _evaluate.py    # evaluate(...) + EvalResult / ParityReport (multi-sim)
  policy.py       # Policy protocol: describe/bind/reset/act (+ optional close)
  transport/      # base.py (PolicyHandle, in-proc) serialize.py ws.py serve.py — typed ws isolation
  adapters/       # template.py + scripted baselines (Zero/HoldPose/Random)
```

## Core interfaces

- `Embodiment` = tuple of `Chain(name, kind∈{arm,gripper,base,head,torso,leg,other}, joint_names, robot, ee_body_name)`;
  `infer_embodiment(robots)` uses `actuators[j].is_ee` (authoritative) + name tokens, splits by
  side only when genuinely multi-sided. An unrecognized joint becomes `arm` **only** with
  manipulator evidence (`ee_body_name` / `is_ee` / `gripper_joint_name` / an arm-token joint);
  otherwise it lands in `other` and stays joint-space controllable. Verified against the real
  in-repo cfgs (`tests/test_harness_phase0.py::test_embodiment_real_robot_cfgs`):
  franka→(arm7, gripper2); h1→(left_arm4, right_arm4, left_leg5, right_leg5, torso1) — **no head
  chain**; go2→four legs (front_left … rear_right); cartpole→other(2); shadow_hand→other(24);
  allegro_hand→other(16). A robot that declares a gripper it cannot expose (`ur5e_2f85`,
  `kinova_gen3`: `gripper_open_q` set, no gripper joint in `joint_limits`) **raises** rather than
  dropping the gripper and making every pick task unsolvable.
- `ObsSpec`/`ActionSpec` of typed `FieldSpec(key, space, shape, dtype, chain, frame, required)`;
  `obs.compatible_with(needs) -> SpecMatch(ok, plan, errors)`, and the same in the action
  direction (`action_spec.compatible_with(card.produces_action)`), so a policy that declares
  `control="ee_pose"` is not silently handed a joint_pos spec.
- `ObsBatch`/`ActionBatch(spec, env_ids, tensors)` — batched device tensors, `.validate()`, `.index()`.
  `ObsBatch.task` carries the non-tensor language/goal payload, populated for tasks that expose
  `get_language_instruction()`.
- `TemporalEnsembler` (num_envs- and field-vectorized; numerically matched to il's
  `get_temporal_agg_action`; ring-buffered over the chunk horizon, so memory is
  `O(num_envs · chunk_len² · dim)` and independent of episode length), `ChunkScheduler` (per-env
  chunk cache).

## Principles / non-goals (per AGENTS.md)

- Library not framework; small orthogonal modules; typed returns (dataclasses).
- No new MetaSim types up front — `Embodiment` is *inferred* from existing `RobotCfg`; a
  `RobotCfg.chains` field is a possible *later* MetaSim-first change if inference is insufficient.
- Build middle transports (shmem) only on measured need. No distributed scheduler, no
  policy-server framework, no speculative obs modalities, no silent unit inference
  (declare `FieldSpec.frame`).
- Backward-compat is load-bearing: keep IL/VLA/RL entry points working; wrap them as `Policy`
  adapters rather than rewriting them.

## Status

- **Core** (embodiment/spec/carriers/chunking): implemented, unit-tested against the real in-repo
  robot cfgs.
- **Rollout** (env_adapter tensor-action path, `VecEvalRunner`, `evaluate`/`EvalResult`/`ParityReport`):
  implemented; `evaluate` covered by `tests/test_harness_evaluate.py` (fake backend) and run
  end-to-end on MuJoCo (`python -m roboverse_learn.eval.harness.demo`).
- **Vision**: `evaluate(cameras=[PinholeCameraCfg(...)])` — verified end-to-end on MuJoCo
  (`MUJOCO_GL=egl`): a policy declaring `camera0.rgb` receives real rendered uint8 pixels.
- **Baselines**: `adapters/{scripted (Zero/HoldPose/Random), template}`.
- **Transport**: `transport/{serialize,base,ws,serve}` — in-proc + typed ws isolation. Cross-process
  isolation is exercised for real by `tests/test_harness_transport.py::test_ws_cross_process_roundtrip`,
  which runs the policy in a **separate OS process** (the `transport.serve` CLI) and talks to it
  over a socket. `serve_policy(policy=…)` hosts one shared instance and **rejects** a second client
  (it would re-bind the first one's spec); `serve_policy(factory=…)` gives each connection its own.
- **82 tests passing** —
  `python -m pytest tests/test_harness_phase0.py tests/test_harness_phase1.py tests/test_harness_phase3.py tests/test_harness_transport.py tests/test_harness_evaluate.py`.

### Known gaps — do not describe these as done

These are open defects, listed here so the status above is not read as more than it is:

- **No GPU *simulator* backend has been run end-to-end.** The device bug (episode bookkeeping
  allocated on CPU while `terminated`/`timeout` came back on the sim device) is fixed: the runner
  now allocates on the sim device and normalizes the step flags onto it, and this is tested with
  real CUDA tensors (`test_runner_full_rollout_on_cuda`, `test_env_adapter_apply_on_cuda_device`)
  plus a CPU-CI guard (`test_runner_allocates_bookkeeping_on_sim_device`). But those tests use a
  *fake* adapter/handler on a CUDA device — mjx/isaacgym/isaacsim/newton themselves have **not**
  been exercised through the harness. Treat "works on a GPU backend" as untested.
- **`episodes=N` does not give N independent episodes** for tasks whose initial state is a fixed
  replayed trajectory: each wave re-applies the same `_initial_states`, so waves are duplicates
  and the reported `success_rate` implies more evidence than it has.
- **`SpecMatch.plan` is advisory** — the `cast` op is recorded but never applied (see `spec.py`),
  so a policy that asks for `float32` rgb from a `uint8` camera still receives `uint8`.
- `control="ee_pose"` derives a valid spec but `EnvAdapter` raises for it (needs cuRobo IK);
  `control="joint_vel"/"effort"` are rejected outright. il/vla `Policy` wraps and the shmem
  transport are not built.
- **Requirement 6 (static-vs-stream split) is not implemented**: the ws obs frame re-sends the
  spec every step.
