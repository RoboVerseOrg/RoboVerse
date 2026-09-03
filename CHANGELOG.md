# Changelog

All notable changes to MetaSim are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Distribution renamed to `roboverse-metasim`** (`metasim` on PyPI is an unrelated project); the
  import name stays `metasim`. Downstream requirements must say `roboverse-metasim @ git+...`.
  `metasim.__version__` now reads the installed version from package metadata.

### Added

- **SuperDex backend** (`metasim/sim/superdex/`, `simulator="superdex"`, extra `metasim[superdex]`,
  Python >= 3.12): Meta's Mochi engine behind the `BaseSimHandler` contract. URDF robots/objects
  (collision geometry baked to watertight hulls in a content-addressed cache), primitives and mesh
  objects, link-mounted and world cameras (RGB + depth via pyrender/EGL), `ContactForces` through
  the objects a robot touches, `SimParamCfg.superdex_control_mode` (`pd` honours effort clamps,
  `implicit` uses the native controller). Mass/inertia, joint damping and effort clamps fall back to
  the sibling MJCF so dynamics match the MuJoCo backend; `scripts/parity_superdex_tracking.py`
  (RoboVerse) measures the remaining gap.
- `metasim.utils.setup_util.SIM_BACKENDS`: backend dispatch is a table; adding a simulator is one
  `SimType` value plus one entry.
- `metasim/test/sim/test_superdex_floating_base.py`, `metasim/test/test_superdex_assets_general.py`,
  `metasim/test/sim/test_mujoco_arena_memory.py`.
- `RELEASING.md` + `CONTRIBUTING.md`: branching, PR gates, SemVer, release checklist, PyPI
  trusted publishing, branch-protection settings.
- CI on every PR without a GPU: `ci.yml` (ruff + `-k general` on 3.10/3.11), `pr-title.yml`
  (Conventional Commit titles), `changelog.yml` (entry required for library changes),
  `release.yml` (tag → build, GitHub Release, PyPI), `CODEOWNERS`.

### Fixed

- `RLTaskEnv.step` publishes the *terminal* observation in `info["observations"]["raw"]["obs"]` instead of the episode's first one (off-policy truncation bootstraps in clean_rl SAC/TD3 and FastTD3 read it).
- IsaacGym and PyBullet reported `joint_pos_target` in native DoF order while `joint_pos` is in sorted-name order; both now use `get_joint_names(sort=True)` (completes #12).
- `ParallelSimWrapper`: a worker that died during handler construction or `launch` surfaced as a bare `EOFError`/`ConnectionResetError` from the handshake and left the other workers running; the handshake now raises the worker's own traceback, `close()` tolerates dead workers, and a failed constructor tears the pool down.
- `hf_util.check_and_download_single`: a `roboverse_data/...` path evaluated from a working directory that is not the parent of `ROBOVERSE_DATA_DIR` was reported as a path-traversal attempt; the error now names the CWD / `ROBOVERSE_DATA_DIR` mismatch and where the asset already is. `test_check_and_download_single_falls_back_to_private_roboverse_data` no longer depends on the caller's `ROBOVERSE_DATA_DIR`.
- MuJoCo: `<size memory="512M">` is reserved by default; humanoid + mesh scenes no longer die with
  `mj_stackAlloc: out of memory` (get_started/10_mount_camera.py).
- `hf_util`: a symlinked `roboverse_data` is no longer refused as path traversal; concurrent
  processes wait for an in-flight download instead of failing after 5 s (`ParallelSimWrapper`
  workers, get_started/3_parallel_envs.py).
- Tree is `ruff check` / `ruff format` clean at the pre-commit pin (0.14.5).


## [0.2.0] - 2026-05-31

This release focuses on **cross-platform infrastructure hardening**: tightening
the `BaseSimHandler` contract so the same MetaSim code runs identically on
mujoco, sapien3, newton, isaacsim, mjx, isaacgym, and pybullet without the
silent per-backend divergences that previously corrupted downstream
experiments.

The release is forward- and backward-compatible. Every behavior change is
additive (new warnings, new optional kwargs) — no existing call site needs to
be modified.

### Added

- **`BaseSimHandler.set_seed`** default implementation seeding Python `random`,
  `numpy`, `torch` CPU + CUDA — forwarded automatically through
  `ParallelHandler` (broadcast to workers) and `HybridHandler` (physics +
  render). Closes the `gym.reset(seed=N)` reproducibility gap.
- **`reset(seed=N)`** plumbed through gym adapter → task → handler.
  Backward-compatible: tasks whose `reset` does not accept `seed=` are
  unchanged (guarded by `inspect.signature`).
- **`actions_cache` contract** centralized on `BaseSimHandler` and propagated
  through `ParallelHandler` / `HybridHandler` wrappers; per-backend overrides
  removed.
- **`set_states` ⇄ `get_states` round-trip contract test** runs across every
  registered backend.
- **Backend-contract general test** statically asserts that every concrete
  `BaseSimHandler` subclass overrides each documented contract method; known
  gaps are xfail-documented and self-checked against staleness
  (`metasim/test/test_backend_contract_general.py`).
- **`set_states` key-validation test suite** (26 cases, no sim env) covering
  unknown / control-input / read-only / partial-pose keys
  (`metasim/test/test_set_states_key_validation.py`).
- **`set_dof_targets` key-validation test suite** for unknown robot / joint /
  action key.
- **Parallel error-handling test suite** with synchronous queue stub
  (`metasim/test/test_parallel_error_handling_general.py`).
- **Gym seed-forward signature test** (`metasim/test/test_gym_seed_forward_general.py`).
- **Scenario duplicate-name test** + **close-idempotent AST static test**.
- **Multi-agent (bimanual) trajectory loading** in `get_traj` —
  `list[RobotCfg]` is accepted natively; per-agent slices merge into the v3
  format. Single-agent path unchanged.
- **`docs/source/.../developer_guide/architecture_review.md` v1.1** —
  documents historical/new Issues 8/9/10 with FIXED status, warning catalog
  table, and forward/backward compatibility statements.

### Fixed

- **`set_states` silent drop of `dof_pos_target`** — the dict key that
  previously triggered a silent no-op (15 failed downstream BC experiments)
  now emits a one-shot warning recommending `set_dof_targets`. Same guard
  applied to unknown robot/object names, unknown joint names, read-only
  velocity fields (`vel`, `ang_vel`, `dof_vel`), and partial pose dicts.
- **`set_dof_targets` cache invalidation** — `_invalidate_state_caches` now
  runs after each apply; downstream `get_states` returns the updated values.
- **`get_states` intermittent qpos corruption** — mujoco `named.data` view
  aliasing fixed; `qpos.copy()` returned and `physics.forward` runs after
  `set_states`.
- **`contact_forces` returns zeros without settling** — handlers now run a
  warm step before reading the contact buffer.
- **Newton joint-name parity** — `_collect_joint_names` strips the object
  prefix (`box_base/box_joint` → `box_joint`) when unambiguous, matching
  mujoco / sapien3 / isaacgym / isaacsim. Falls back to the full key on
  collision.
- **Newton `_set_dof_targets` silent no-op** — when the joint-target buffer
  is `None` the handler now warns instead of silently accepting + dropping.
- **Newton `_set_dof_targets` made `@abstractmethod`** so future backends
  cannot accidentally inherit a no-op.
- **Actuator override partial-spec silent inheritance** — mujoco + newton now
  warn when only some of `stiffness` / `damping` / `effort_limit_sim` are
  overridden, surfacing the cross-sim PD-gain divergence that produced
  silent zero-impulse failures.
- **mujoco `_set_states` unknown-key warning** + **idempotent `close()`**.
- **sapien3 `_set_states`** — defaults missing `dof_pos` / `pos` / `rot`
  instead of raising `KeyError`, matching mujoco's behavior. Also instantiates
  and reads **all** robots (was only reading the first).
- **isaacsim `_set_states`** — accepts `list` / `tuple` / `numpy.ndarray`
  pos/rot via `torch.as_tensor` (was crashing on lists). Headless mode falls
  back gracefully when `omni.replicator` is unavailable.
- **mjx backend unblocked** — eager `_init_mjx()` + `_ensure_id_cache(ts=None)`
  in `launch()`; JAX `__dlpack__` 0.10+ protocol with legacy fallback;
  `mjx_panda.xml` path repaired; recursive `<include>` parser handles
  `assetdir` / `meshdir` / `texturedir` with cycle detection.
- **ParallelHandler error surfacing** — every public method drains
  `error_queue` + checks `process.is_alive()`; `EOFError` / `BrokenPipe` on
  recv is translated into a `RuntimeError` carrying the real worker
  traceback. OOM-killed workers are surfaced via the dead-process check.
- **`RobotState.body_state` / `joint_pos` / `joint_vel`** made `Optional`
  (mirrors `ObjectState`) so backends without per-link state (pybullet)
  validate cleanly.
- **`step()` reward normalization** — always returned as `(num_envs,)`
  float32 on `self.device`.
- **Blender hybrid backend** — `METASIM_CYCLES_DEVICE` env var honored when
  configuring Cycles device; physics `_get_states` exceptions now surface
  through `set_states` (previously swallowed).
- **`action_input_to_tensor`** — warns when non-position targets
  (`dof_vel_target`, `dof_torque_target`) are silently dropped.

### Documentation

- `ROADMAP.md` updated — P0/P1 cross-platform infra items marked fixed.
- Architecture review v1.1: new warning catalog table; Issues 8/9/10 added.

### Tests

- 386 general tests pass + 21 xfailed across 5-backend sweep
  (general + mujoco + sapien3 + newton + passthroughs).
- Per-backend PD-convergence shortfalls in `test_default_qpos.py` /
  `test_collision.py` / `test_dof_control.py` are xfail-documented with
  specific reasons (mujoco MJCF `forcerange='-40 40'`, sapien3 settle rate,
  mjx integrator step, newton MuJoCo-Warp PD wiring). xfail (not skip) so a
  flake-to-passing flip is visible.
- 9 downstream RoboVerse `RobotCfg` bugs surfaced by the new validation tests
  (AlohaAgilex / G1Tracking / YamCfg / ArxL5Cfg / Vega / SoArm100 / Koch /
  Go2 / AllegroHand) are xfail-documented in the RoboVerse repo for follow-up.

### Migration

No code changes are required for existing users. New warnings (unknown keys,
partial pose, read-only fields, partial actuator specs, dropped non-position
targets, duplicate scenario names) are informational; they fire once per
unique key per process.

If you previously **relied** on the silent-drop behavior of
`set_states({"dof_pos_target": ...})`, switch to
`handler.set_dof_targets([{robot_name: {"dof_pos_target": ...}}])`.

If you previously caught `KeyError` from sapien3 `_set_states` on missing
`dof_pos`, that branch is now dead — sapien3 defaults to zeros, matching
mujoco.

---

[Unreleased]: https://github.com/RoboVerseOrg/MetaSim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RoboVerseOrg/MetaSim/releases/tag/v0.2.0
