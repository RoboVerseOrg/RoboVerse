# Changelog

All notable changes to RoboVerse are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

RoboVerse is the downstream content-pack + research layer on top of the
[MetaSim](https://github.com/RoboVerseOrg/MetaSim) core. See
`MetaSim/CHANGELOG.md` for the core-framework changes that ship with this
release.

## [Unreleased]

### Fixed

- `import roboverse_pack.tasks.benchmark` raised a circular-import error when the tasks package was
  imported first (task discovery); `roboverse_pack.benchmark` now resolves its two helpers lazily.
- `roboverse_pack.tasks.mjlab` no longer downloads assets from HuggingFace at import time; the
  handler's `check_assets()` fetches them when a task is used.

### Changed

- MetaSim dependency renamed to `roboverse-metasim @ git+...` (MetaSim's distribution was renamed;
  import name `metasim` unchanged). The pin moves to a MetaSim tag with the next release.

### Added

- `--sim superdex` in every MuJoCo-capable `get_started` tutorial and a `roboverse-py[superdex]`
  extra (needs the MetaSim SuperDex backend, Python >= 3.12).
- `scripts/parity_superdex_tracking.py`: seeded joint-target tracking and rigid-object drop
  comparisons between two backends (record per backend, compare + plot).
- CI (`.github/workflows/tests.yml`): ruff lint job + the whole `tests/` suite on CPU for every PR;
  `tests/conftest.py` markers `requires_optional` / `requires_asset` skip with a reason instead of
  failing when optional deps or `roboverse_data` assets are absent.
- `RELEASING.md`: branches, PR gates, SemVer, release checklist, PyPI publishing, branch
  protection. `pr-title.yml`, `changelog.yml`, `release.yml`, `CODEOWNERS`.

### Fixed

- `g1_feet` robot cfg crashed at import (wrong `BaseActuatorCfg` keywords) and `unitree_dex3_1` had a degenerate thumb joint limit; a config validation test now guards both.
- Eight Tier-1 task `reset` overrides (humanoid, beyondmimic, six SimplerEnv tasks) dropped `states=`, so `env.reset(states=...)` from the IL/VLA eval runners raised `TypeError`; the contract test now checks full signature substitutability.
- FastTD3: the nine stub configs that say `# Inherits from base.yaml` now actually inherit (`load_config` deep-merges over `configs/base.yaml`; previously `float(None)` at startup); `mjx_walk.yaml` `headless: flase` typo.
- IL `DefaultRunner`: validation loss is computed on the deployed policy (EMA model in `eval()` mode) instead of the train-mode raw model.
- `ManagerBasedRVEnv.reset` returned `None` (Gym wrappers expect `(obs, info)`), had no `num_obs`/`observation_space`, and mjlab cartpole reset noise ignored `set_seed`; all three fixed with a gym-reset regression test.
- Native LIBERO `Open`/`Close`/`TurnOn`/`TurnOff` predicates always evaluated False on
  mujoco >= 3.x (`numpy.int32 in (mjtJoint...)` membership).
- `get_started/multiple_cameras.py` passed a removed `ScenarioCfg` kwarg.
- Stale test expectations (docs build layout, `LiberoBaseTask.reset` stubs after the `seed` change).
- Red flags: SSH submodule URL, dangling `release/metasim` symlink, silent `except: pass` around
  passthrough registration, hard-coded personal paths in scripts/tests, untruthful
  `requires-python >= 3.8` (now 3.10), wheels missing LIBERO json/npz bundles, stale ruff ignores.


## [1.0.0-beta] - 2026-05-31

The headline of this release is **cross-platform parity is now testable and
load-bearing**. Every shipped `RobotCfg`, every contracted handler method,
and every benchmark `reset(seed=N)` is exercised across the supported
backends and either passes or is xfail-documented with a specific reason.

The release is forward- and backward-compatible.

### Added

- **`tests/test_roboverse_robot_cfg_validation.py`** — validates every
  `RobotCfg` shipped in `roboverse_pack`: instantiation, non-empty `name`,
  default joint positions inside `joint_limits` ranges. 60 cfgs × 4 checks,
  9 real downstream bugs surfaced and xfail-documented:
  - `AlohaAgilex`, `G1Tracking` — orphan joints in `default_joint_positions`
  - `YamCfg`, `ArxL5Cfg`, `Vega`, `SoArm100`, `Koch`, `Go2`, `AllegroHand` —
    defaults outside `joint_limits` ranges
- **IL + RL fusion bridge** (`roboverse_learn/fusion/`) — RL-trained policy
  → demo collection (rollout → `save_demo` → `data2zarr` unchanged) +
  IL-to-RL BC warmstart. End-to-end pipeline orchestrator
  (`rl-train → collect → to-zarr → il-train`). 6/6 tests; validated against
  a real cartpole checkpoint and real mujoco/EGL launch. Fails fast on
  scene-MJCF tasks where robot is not a `RobotCfg`.
- **mjlab 1:1 obs/reward parity** — extensive byte-level alignment work:
  - `TerrainGridScanSensor` + `height_scan` obs wired into
    `velocity_rough_go1` / `velocity_rough_g1` (1:1 building block).
  - Continuous rough terrain wired into both velocity-rough quadruped
    tasks.
  - `tracking-g1` motion-tracking obs ports `motion_anchor` + `robot_body`
    1:1.
  - `lift_cube_yam` regains the missing `ee_to_cube` + `cube_to_goal` obs.
  - `velocity_rough` height_scan is now correctly scaled by
    `1 / max_distance` (matches mjlab); clip-then-scale order on the obs
    term matches mjlab's `noise → clip → scale`.
  - go1 / g1 `base_lin_vel` obs reads the IMU velocimeter at the offset
    site (adds the `ω × r` cross-term — fixes a real obs gap that diverged
    under turning).
  - cross-sim cartpole eval pipeline repaired (actor build, device,
    episode metric).
- **RoboTwin v2 passthrough** — 50 RoboTwin tasks registered as native gym
  envs (`RoboVerse/RoboTwin-<task>-v0`), 1:1 by construction, mirrors the
  ManiSkill passthrough pattern.
- **`AGENTS.md` / `CLAUDE.md`** — repo-level dev rules for AI agents
  (double quote, line-length 120, py38 target, lazy imports OK,
  parity-is-load-bearing, multi-repo discipline, commit-as-user).

### Fixed

- **`get_started/0_static_scene.py`** — gracefully skips the cameras-empty
  case instead of crashing on first-time setup.
- **mjlab tooling paths** — lint-clean across mjlab/maniskill tasks;
  tooling paths made portable (no hardcoded `/home/ghr/...`).
- **mjlab MJCF asset resolution** — assets pulled from
  `RoboVerseOrg/roboverse_data` on Hugging Face instead of hardcoded local
  paths (fresh checkout no longer breaks). Dual-mode locator
  `_locator.mjlab_asset` = clone-or-HF.
- **mjlab go1 Newton 1:1** — go1 RL now stands on Newton (PD actuators +
  init pose + action order alignment).

### Documentation

- `ROADMAP.md` — P0/P1 cross-platform infra items marked fixed; P2/P3
  items refreshed.
- mjlab / maniskill / robotwin integration pages wired into the docs
  toctree.

### Tests

- 60 RoboVerse `RobotCfg` configs validated × 4 checks = 240 statically
  exercised; 9 real bugs xfail-documented.
- Cross-4-backend sweep against MetaSim core in the `roboverse` conda env:
  mujoco + sapien3 + newton + passthroughs — 323 passed, 0 regressions.

### Migration

No code changes are required for existing users.

If you maintain an out-of-tree `RobotCfg`, the new validation test will
exercise it as soon as it is imported through `roboverse_pack` — fix any
flagged `default_joint_positions` mismatches before they trip in CI.

---

[Unreleased]: https://github.com/RoboVerseOrg/RoboVerse/compare/v1.0.0-beta...HEAD
[1.0.0-beta]: https://github.com/RoboVerseOrg/RoboVerse/compare/v1.0.0-alpha...v1.0.0-beta
