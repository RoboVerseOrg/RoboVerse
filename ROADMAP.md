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
| Releases | `1.0.0` in both packages (this release); PyPI trusted publishing wired, released in lockstep from one tag. |

## Roadmap

The order follows the 2026-09 architecture review: first make the contracts hold, then converge
the parallel implementations, then build the benchmark data model on top.

### 1. Contracts (in progress)

- [x] Content packs win over MetaSim's bundled example pack; shadowed config names are reported.
- [x] `get_states(env_ids=...)` returns exactly those envs on every backend.
- [x] `set_states` restores velocities; L0 / L1 replay contracts pinned per backend.
- [x] `BaseTaskEnv.supported_simulators` declared and checked at construction.
- [x] `metasim.kernel`: the minimum a port builds on (scenario, `TensorState`, handler, task loop,
      protocol), backend-free and API-pinned; content's reach-ins past the handler surface are counted
      per area and only go down (`tests/test_kernel_boundary.py`, 354 at 1.0.0).
- [ ] Handler physical-property interface: per-body mass / inertia and per-shape friction /
      restitution, per env, on every backend. Today the humanoid and beyondmimic packs each carry
      their own Isaac Sim / Isaac Gym / MuJoCo / Newton branches for it, and the core
      `ObjectRandomizer` covers Isaac Sim only.
- [ ] Site and frame poses by name as a handler query, so LIBERO-90's success checks (63 direct
      `physics.named.data.site_*` reads) and the mjlab ports run on more than MuJoCo.
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
      locomotion loops; term configs (reward / termination / event) move into MetaSim. Done so far:
      `BaseTaskEnv` and `RLTaskEnv` share one step sequence with `_apply_action` / `_on_step` /
      `_on_reset`, and seven family bases dropped their `reset` overrides for `_on_reset`. Left: the
      30 `step` overrides (locomotion loops with in-step PD and decimation, delta-control latches in
      MuJoCo Playground and pick-place, native ports), which need a per-substep hook first.
- [ ] MuJoCo Playground tasks run again: `pick` / `open_cabinet` read Menagerie Panda geoms
      (`left_finger_pad`) the bundled Franka MJCF lacks, `handover` has no ALOHA MJCF, and
      `handover.step` reads an `env_states` attribute nothing sets.
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
- [~] Every evaluator in `roboverse_learn` converges on `metasim.bench.evaluate`. Done: FastTD3 checkpoints
      (`roboverse_learn.rl.fast_td3.bench_policy:FastTD3Policy`) and IL runners / diffusion-policy checkpoints
      (`roboverse_learn.il.runners.bench_policy:ILPolicy`) score through `bench run --policy <import path>
      --checkpoint`. Any policy can also run in its own environment behind `bench serve` and be scored
      with `bench run --policy-url` (`metasim.bench.remote`). Open: ACT, the three VLA evals (each as a
      `Policy` served from its own environment) and rsl_rl, then retiring the legacy loops once their
      extras (domain randomisation, videos) live in the harness.
- [x] Task health gate: `python -m metasim bench health --sim <backend>` replays every task's demos and
      classifies it (`ok` / `low` / `undefined` / `unsupported` / `no-demos` / `error`), exit 1 on `low` or
      `error`. First reading on MuJoCo (3 episodes): `stack_cube` 3/3, `pick_cube` 2/3 (**low**: its demos
      do not fully replay on the backend they target; the next task-side item), `reach_origin` ships no
      demos. ManiSkill sweep: `lift_peg_upright` 3/3; `pick_cube` 9/10 (demo 0 never replays: a data
      item); `draw_triangle` 0/3 (checker vs. replay to investigate); `plug_charger` has no MuJoCo asset;
      `pull_cube` / `push_cube` 3/3 once demo folders download; `roll_ball` / `stack_pyramid` / `draw_triangle`
      0/3: replaying demo 0 on MuJoCo the ball never moves (44-step demo), one pyramid cube is lifted but no
      stack forms, the triangle checker never fires - demos recorded on another backend that do not transfer
      closed-loop (re-record on MuJoCo or publish them as that backend's instances); `place_sphere` / `poke_cube` / `pull_cube_tool` MJCFs reference an absolute path
      of the authoring machine (data items: `assets/maniskill/PlaceSphere/bin/mjcf/PlaceSphere_bin.xml`,
      `assets/maniskill/PokeCube/peg/mjcf/stick.xml`, `assets/maniskill/PullCubeTool/tool/mjcf/tool.xml`).
      Open: run the sweep over every task with demos per release and publish the table.
- [x] Multi-task suites (`metasim.bench.suite`, `bench suite run|report`): splits scored task-macro,
      incomplete splits reported as such; demo-length episode budgets; per-episode instructions from a
      task `prompt_bank` with held-out paraphrases (`--prompt-split heldout`). First suite:
      `tabletop-mujoco` (5 tasks, replay task-macro 0.933 at 3 episodes per task). The design follows
      HandsOn's evaluation protocol (MIT; `THIRD_PARTY_NOTICES.md`).
- [ ] A dexterous suite: HandsOn-style tasks (ball in bowl, fruit in bucket, peg insertion, cup
      upright) re-authored on Sharpa Wave with a floating or arm-mounted base, on MuJoCo and Newton.
      Needs a mounted-hand embodiment (§4), objects with settled licenses (YCB via our own asset
      pipeline), and demos so the suite has a replay ceiling. Report it as a re-hosted variant: scores
      do not compare to HandsOn's own platform.
- [ ] Demo budgets from the replayed success step rather than the recorded length (pick_cube and
      stack_cube demos run to the time-out after the task is solved).
- [ ] The task bases keep the un-tiled demo states and actions they load (one source of truth);
      `metasim.bench.instance_pool` reads them instead of parsing the demo file a second time.
- [x] Checkers move out of `metasim.example` into the core (`metasim.task.checkers`; the old path is
      an alias).
- [ ] Declarative success predicates on top of them (position, bbox, joint, all-of / any-of), so a
      task states its success condition as data.
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
- [ ] Sharpa Wave: a mounted-hand variant on Franka and YAM, and a first dexterous task family.
- [x] A Newton default for `njmax` that fits a 22-DoF hand (a closed fist uses 137 rows; the
      initial-state estimate was 64 and the overflow was dropped).

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
