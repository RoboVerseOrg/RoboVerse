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

### 3. Benchmark data model (`metasim.bench`)

- [ ] `EnvSpec` → `TaskInstance` (content-addressed: assets, robot cfg, physics, initial state
      with velocities, goal) → `Trajectory` (actions with control mode, full states, provenance)
      → `TaskCategory` (generator + success checker + metrics) → `Benchmark` (protocol + metrics).
- [ ] `verify_replay` as the ingest gate for every trajectory; migration of the existing v2
      trajectory files with a pass-rate report.
- [ ] Task generators and trajectory collectors (teleoperation, planning, RL policies, upstream
      converters) behind one interface.

## How to help

Pick an unchecked item, open an issue describing the change, and follow `CONTRIBUTING.md`.
Bug reports with a minimal scenario and the backend name are always welcome.
