# Changelog

All notable changes to RoboVerse are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

RoboVerse is the downstream content-pack + research layer on top of the
[MetaSim](https://github.com/RoboVerseOrg/MetaSim) core. See
`MetaSim/CHANGELOG.md` for the core-framework changes that ship with this
release.

## [Unreleased]

### Changed
- The wiki build (`scripts/docs/build_roboverse_wiki.sh`, `deploy-docs.yml`) renders the MetaSim docs from `packages/metasim/docs` instead of cloning the standalone repository; `METASIM_REPO` still forces a clone for pre-monorepo versions.
- The three per-policy `requirements.txt` files under `roboverse_learn/il/policies/{dp,vita,fm}` are removed; their runtime dependencies (`av`, `dill`, `moviepy`, `numba`, `termcolor`, `huggingface-hub`) join the `[learn]` extra, which is now the single install path (`pip install -e ".[learn]"`). The files pinned `hydra-core`, `zarr`, `einops` and `dill` to 2022 versions and were the source of the Dependabot alerts.
- `scripts/` triage: 16 scripts that imported symbols removed before 1.0 (`get_sim_env_class`, `metasim.cfg`, `metasim.sim.EnvWrapper`, `metasim.sim.isaaclab`, …) are deleted (`random_action*`, `motion_planning*`, `train_ppo*`, `replay_state`, `replay_real_asset`, `retarget_demo`, `run_bidexbench_cube_reach`, `clean_usd`, `convert_traj_v1_to_v2`, `test_usd`); their documented uses now point at the `examples/` equivalents. The personal experiment runner (`run_multi.*`), hard-coded conda paths in `run_openvla_eval.sh` (now `OPENVLA_PYTHON` / `LIBEROPLUS_PYTHON`) and committed evaluation results under `scripts/policy/ckpt/` are gone.
- Repository layout: the tutorials moved from `get_started/` to `examples/` (same files, same numbering; `python examples/0_static_scene.py …`), the EmbodiedGen asset tooling from `generation/` to `tools/embodiedgen_integration/` (import path `tools.embodiedgen_integration`), and the test dashboard from `dashboard/` to `tools/dashboard/`; the legacy `release/` scaffold and `RELEASE_NOTES_NEXT.md` are removed (the CHANGELOG is the release text). Every in-repo reference was rewritten; the docs site paths (`roboverse.wiki/metasim/get_started/...`) are unchanged.
- The mjlab v2 tasks (go1 / g1 velocity, yam lift-cube) declare `supported_simulators = ("mujoco", "newton")`; requesting another backend now fails at construction with a clear message instead of inside the backend.
- Packaging hygiene: `requires-python = ">=3.10,<3.13"` and ruff `target-version = "py310"` in both packages (old-style typing rewritten by ruff where safe); the CHANGELOG release header is `[1.0.0b0]`, matching the `pyproject` version `release.yml` checks; a `release-dry-run` CI job builds both distributions on every PR, runs `twine check --strict`, and fails on direct-URL dependencies (PyPI rejects them) or mismatched package versions.
- `roboverse_learn` is linted and formatted by ruff like the rest of the tree (it was in `extend-exclude`, so CI never parsed its 26k lines): 97 auto-fixes applied (unused imports, import order, f-strings), `super(__class__, self)` / nested `max` / `stacklevel` / mutable-default findings fixed, an undefined `lighten` removed from `pymunk_override.__all__`; the remaining style families (`FA100`, `UP006`, `UP045`, `E731`, `B006`, `RUF013`, `F811`) are per-file-ignored for that directory as tracked debt.
- MetaSim now lives in this repository under `packages/metasim` (imported with `git subtree`, history preserved) and is released in lockstep as `roboverse-metasim`; `roboverse-py` depends on `roboverse-metasim>=1.0.0b0,<1.1` instead of a git `@main` URL. Install with `pip install -e "packages/metasim[mujoco]" -e ".[mujoco]"` (MetaSim first). CI runs MetaSim's simulator-free suite on 3.10/3.11 alongside the RoboVerse suite; `changelog.yml` guards one CHANGELOG per package; `release.yml` builds and publishes both from one tag.

### Fixed

- `roboverse_pack.tasks.mjlab` failed to import without the asset checkout (hosted CI, registry discovery): the go1 / yam class-level scenarios parsed an MJCF at import time; they are now built on first access (`lazy_scenario`).
- `import roboverse_pack.tasks.benchmark` raised a circular-import error when the tasks package was
  imported first (task discovery); `roboverse_pack.benchmark` now resolves its two helpers lazily.
- `roboverse_pack.tasks.mjlab` no longer downloads assets from HuggingFace at import time; the
  handler's `check_assets()` fetches them when a task is used.

### Changed

- MetaSim dependency renamed to `roboverse-metasim @ git+...` (MetaSim's distribution was renamed;
  import name `metasim` unchanged). The pin moves to a MetaSim tag with the next release.

### Added
- `backend-compat.yml`: weekly workflow that installs the newest release of each CPU-installable simulator backend (MuJoCo, PyBullet, SuperDex), runs `metasim doctor` and the backend's suite, and opens a `backend-compat` issue on failure; `.github/dependabot.yml` groups simulator bumps into weekly PRs.
- Third-party attribution: `NOTICE` and `THIRD_PARTY_NOTICES.md` index every vendored or adapted component (FastTD3, CleanRL, stable-baselines3, diffusion_policy, BeT/minGPT, ACT/DETR, Isaac Lab, RSL-RL, BeyondMimic, gsnet/pointnet2, MuJoCo mesh tools, …) with its license text in-tree; adapted files carry a header naming upstream, license, changes and the license path; `tests/test_attribution.py` fails the build when a header or an index row is missing; wheels ship `NOTICE`, the index and the vendored `LICENSE*` files (`setuptools>=77`, PEP 639 `license-files`).

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
- ManiSkill task leaves declared `ScenarioCfg(objects=[...])` with no robot, so 2,648 registered tasks crashed with `IndexError` on reset; the family now defaults to `franka` (leaf-declared robots win) and raises a clear `ValueError` when no robot resolves.
- CleanRL PPO overwrote `dones[step]` after `envs.step`, shifting the done buffer by one so GAE bootstrapped `V(reset_state)` across episode boundaries; the duplicate write is removed with a numeric GAE regression test.
- Parity/eval harnesses (`eval_liberoplus_policy_consistency`, `parity_simpler_env`, `eval_*_cross_sim`, `spike_metasim_full_parity`, `verify_native_registration`) swallowed checker exceptions into `False`, compared only shared obs keys, and exited 0 regardless of verdict, so two broken sides printed PASS; they now raise on errors, require equal key sets and non-empty rollouts, and carry the verdict in the exit status.
- `maniskill.*_native` specs truncated stack_pyramid/pull_cube_tool/peg_insertion_side/plug_charger/draw_triangle at 50 steps instead of ManiSkill'\''s 250/100/100/200/300 horizons; pinned to upstream with an always-on table test plus registry-derived checks when `mani_skill` is installed.
- mjlab DR events (`geom_friction`, `body_mass`, `body_com_offset`, `push_by_setting_velocity`) silently no-oped on every backend but single-env MuJoCo, so multi-env Go1 training ran with zero domain randomization; they now write the Newton model (verified on a real 2-env Newton Go1) and raise `NotImplementedError` on backends without support (MJX/IsaacSim runs must drop the EventTerms) instead of skipping.
- Native LIBERO `Open`/`Close`/`TurnOn`/`TurnOff` predicates always evaluated False on
  mujoco >= 3.x (`numpy.int32 in (mjtJoint...)` membership).
- `examples/multiple_cameras.py` passed a removed `ScenarioCfg` kwarg.
- Stale test expectations (docs build layout, `LiberoBaseTask.reset` stubs after the `seed` change).
- Red flags: SSH submodule URL, dangling `release/metasim` symlink, silent `except: pass` around
  passthrough registration, hard-coded personal paths in scripts/tests, untruthful
  `requires-python >= 3.8` (now 3.10), wheels missing LIBERO json/npz bundles, stale ruff ignores.

### Security

- `hydra-core` pinned to 1.3.4 in the dp / vita / fm IL policy requirements (Dependabot: `hydra.utils.instantiate` code execution with untrusted configs in <=1.3.3).

## [1.0.0b0] - 2026-05-31

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

- **`examples/0_static_scene.py`** — gracefully skips the cameras-empty
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
[1.0.0b0]: https://github.com/RoboVerseOrg/RoboVerse/compare/v1.0.0-alpha...v1.0.0-beta
