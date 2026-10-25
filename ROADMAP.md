# RoboVerse Roadmap

Where the project is going and in what order. Detailed per-change notes live in `CHANGELOG.md`
(RoboVerse) and `packages/metasim/CHANGELOG.md` (MetaSim); the development and release rules are
in `CONTRIBUTING.md` and `RELEASING.md`.

## Where we are (September 2026)

| Area | Status |
|---|---|
| Repository | Monorepo: MetaSim (`packages/metasim`, published as `roboverse-metasim`) and RoboVerse content / learning code (`roboverse-py`) are released together from one tag. |
| Backends | MuJoCo, MJX, Newton, SuperDex, SAPIEN 2/3, PyBullet, Isaac Sim, Isaac Gym, Genesis, Blender behind one `BaseSimHandler` contract. MuJoCo, Newton and SuperDex are exercised by the record → replay contracts (L0 action replay, L1 state round-trip); the others are not yet. |
| Tasks | ~6,000 registered tasks across LIBERO, ManiSkill, RLBench, mjlab, humanoid, robosuite, RoboTwin and more; `supported_simulators` is declared per task as it is verified (mjlab v2 first). |
| CI | Lint, MetaSim simulator-free suite (3.10 / 3.11), RoboVerse suite, release dry-run on every PR; GPU backends run in the merge-queue workflow. |
| Releases | `1.0.0b0` in both packages; PyPI trusted publishing wired, first lockstep release pending. |

## Roadmap

The order follows the 2026-09 architecture review: first make the contracts hold, then converge
the parallel implementations, then build the benchmark data model on top.

### 1. Contracts (in progress)

- [x] Content packs win over MetaSim's bundled example pack; shadowed config names are reported.
- [x] `get_states(env_ids=...)` returns exactly those envs on every backend.
- [x] `set_states` restores velocities; L0 / L1 replay contracts pinned per backend.
- [x] `BaseTaskEnv.supported_simulators` declared and checked at construction.
- [ ] `_set_states` accepts `TensorState` only; the dict/tensor dual paths go away.
- [ ] Per-backend capability declaration (`BackendCapabilities`) checked against `RobotCfg` and
      `SimParamCfg` at handler creation; `SimParamCfg` split into common fields + per-backend sections.
- [ ] Queries (`metasim.queries`) call handler capability methods instead of backend privates; an
      unsupported query fails at bind time.
- [ ] One asset resolver (`metasim.assets.resolve`) and one conversion cache for every backend.
- [ ] Cross-backend parity fixture (same task, same initial state, two backends, per-step drift) in
      CI for the CPU backends; Newton / MJX / SuperDex suites in the merge queue.
- [ ] Version policy for simulator packages: pinned ranges and feature flags instead of monkeypatches.

### 2. Convergence

- [ ] One environment loop: `RLTaskEnv` hooks replace the manager-based env and the two hand-written
      locomotion loops; term configs (reward / termination / event) move into MetaSim.
- [ ] Table-driven task families (`TASK_SPECS` + registration loop) replace the generated
      ManiSkill task files.
- [ ] `roboverse_learn/common`: one env factory, one checkpoint format, one seeding path.
- [ ] `scripts/` folds into `tools/` and console entry points; the parity probes become
      `python -m roboverse.parity`.
- [ ] Logging owned by `metasim.utils.log` (`METASIM_LOG_LEVEL`); per-backend `set_seed`.

### 3. Benchmark protocol and data model (`metasim.bench`)

The October 2026 review of the evaluation surface found twelve evaluation loops with four
different success rules, no shared policy contract, no fixed evaluation instances and no result
schema; and in the core, success and termination were one signal. The protocol layer fixes that
first; the data model builds on it.

- [x] One success contract: `info["success"]` on every `step`, separate from `terminated`
      (`BaseTaskEnv.success_is_termination`, `_success`); `auto_reset` declared per env class.
- [x] `metasim.bench`: `EvalProtocol` (`first` / `final` / `held` scoring, cap, seed, explicit
      instances), content-addressed `TaskInstance` pool drawn from the task's demos, the `Policy`
      contract with `hold` / `random` / `replay` baselines, the harness-owned `evaluate` loop and
      the versioned `BenchmarkResult` JSON with provenance; `python -m metasim bench`.
- [ ] Every evaluator in `roboverse_learn` (IL runners, ACT, the three VLA evals, FastTD3,
      rsl-rl) becomes a `Policy` adapter and calls `metasim.bench.evaluate`; the per-script
      success counting and text/JSON outputs go away.
- [ ] Task health gate: `bench run --policy replay` on every task that ships demos, on the backend
      the demos were recorded on, published as the task's ceiling; a task below threshold is
      marked, not scored. `success_status ∈ {native, ported, proxy, none}` declared per task and
      refused by the benchmark unless `native` / `ported`.
- [ ] The task bases keep the un-tiled demo states and actions they load (one source of truth);
      `metasim.bench.instance_pool` reads them instead of parsing the demo file a second time.
- [ ] Checkers move out of `metasim.example` into `metasim.task.success` as the public, declarative
      success predicates (position, bbox, joint, hold counters) every task family uses.
- [ ] Frozen evaluation splits: per task a published instance list (ids + hashes) separate from
      the training demos, plus a stability gate before the policy acts (an instance whose objects
      move at rest is discarded and counted, not scored).
- [ ] Capability dimensions and aggregation: tasks tagged by what they test (generalization,
      precision, long-horizon, memory, language); `bench summarize` reports per-dimension means
      and mean ± std over seeds; a leaderboard is a directory of result JSONs plus that table.
- [ ] `EnvSpec` → `TaskInstance` (content-addressed: assets, robot cfg, physics, initial state
      with velocities, goal) → `Trajectory` (actions with control mode, full states, provenance)
      → `TaskCategory` (generator + success checker + metrics) → `Benchmark` (protocol + metrics).
- [ ] `verify_replay` as the ingest gate for every trajectory; migration of the existing v2
      trajectory files with a pass-rate report.
- [ ] Task generators and trajectory collectors (teleoperation, planning, RL policies, upstream
      converters) behind one interface.

### 4. Embodiments

Three robots are the reference embodiments every piece of infrastructure must support and every
benchmark table must report by: **Franka** (`franka`), **YAM** (`yam`) and **Sharpa Wave**
(`sharpa_wave_right` / `sharpa_wave_left`). "Support" means: a verified `RobotCfg` per backend
asset format, demos that replay on the recording backend, the replay / hold baselines published,
IK and teleop profiles, and a row in the task-by-robot-by-simulator matrix.

- [x] `YamCfg` rebuilt on the Menagerie `i2rt_yam` asset (8 joints, asset gains and home pose, passive
      coupled finger); `SharpaWaveRightCfg` / `SharpaWaveLeftCfg` from Menagerie `sharpa_wave` (pinned
      revision, fetched through `register_upstream_mirror` until mirrored into `roboverse_data`).
      Verified on MuJoCo (1 env + parallel wrapper) and Newton: hold, tracking, gripper coupling,
      a `metasim.bench` evaluation (`tests/test_embodiments.py`, `docs/.../benchmark/usage.md`).
- [x] Handler contract for real assets: passive joints and cfg-synthesized actuators on MuJoCo (the
      Newton handler already drove from the config), so one `RobotCfg` means the same thing on both.
- [ ] Upload `robots/sharpa_wave/mjcf` to the `roboverse_data` dataset (the mirror then becomes a
      fallback) and `urdf`/`usd` descriptions of YAM and Sharpa Wave for the Isaac backends and the
      IK helpers (`get_pyroki_model` reads a URDF).
- [ ] Robot capability matrix generated from the configs (asset formats present, control modes,
      `supported_simulators` of the tasks that use it) and published in the docs.
- [ ] `bench run --robot` for every task family that is robot-agnostic (ManiSkill ports, pick-place,
      benchmark scenes), with per-robot instance pools; a first YAM demo set recorded with the
      collection pipeline so `replay` has a ceiling to report.
- [ ] Sharpa Wave: a mounted-hand variant on Franka and YAM, and a first dexterous task family;
      a Newton default for `njmax` that fits a 22-DoF hand (the solver warns `nefc overflow` today).

### 5. Simulator infrastructure (from the Isaac Lab 3.0 comparison)

- [ ] `BackendCapabilities` as one frozen declaration per handler (replacing the five scattered
      flags), checked against the scenario at construction; `doctor` prints the table.
- [ ] Handler writes with index *or* mask granularity (`write_root_state`, `write_joint_state`,
      `set_joint_targets`) so vectorised backends avoid host syncs; `env_ids` accepted as tensors.
- [ ] Backend registration through an entry-point group; `ScenarioCfg.simulator` validated against
      the registry instead of a `Literal`.
- [ ] Canonical joint / body ordering maps per backend, so `TensorState` is identical across them.
- [ ] Per-env seeding in the parallel wrapper and engine RNG seeding where the backend has one.

## How to help

Pick an unchecked item, open an issue describing the change, and follow `CONTRIBUTING.md`.
Bug reports with a minimal scenario and the backend name are always welcome.
