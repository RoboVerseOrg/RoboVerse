# RoboVerse Development Roadmap

This document provides a high-level overview of the RoboVerse improvement roadmap. For detailed analysis and implementation guidelines, see the [Architecture Review](docs/source/metasim/developer_guide/architecture_review.md).

## Current Status

| Aspect | Status | Notes |
|--------|--------|-------|
| Core Framework | Stable | Multi-simulator support working |
| Test Coverage | Needs Work | ~15-20% estimated coverage |
| Documentation | Good | Comprehensive tutorials available |
| API Stability | Beta | Breaking changes possible |

## Priority Issues

### P0 - Critical (Immediate)

- [x] **State Cache Consistency** - Fixed: dual tensor/dict caches in `BaseSimHandler`,
  invalidated on `set_states` / `set_dof_targets` / `simulate`. Regression tests:
  `metasim/test/sim/test_state_modes.py` (integration) + `metasim/test/test_set_states_key_validation.py`
  (unit, no sim env).
- [x] **`set_states` silent-drop guard** - Added: unknown / control-input keys now log a
  one-shot warning (e.g. `dof_pos_target` → "use `set_dof_targets`"). Catches the silent
  no-op bug that broke 15 downstream BC experiments.
- [x] **Robot-config validation tests** - Added static `-k general` tests for every shipped
  `RobotCfg`: instantiation, non-empty `name`, default joint positions match `joint_limits`
  keys + lie inside the limit ranges. Surfaces 9 real config bugs (xfail-documented:
  AlohaAgilex/G1Tracking orphan joints, YamCfg/ArxL5Cfg/Vega/SoArm100/Koch/Go2/AllegroHand
  out-of-range defaults). `metasim/test/test_robot_cfg_validation_general.py` +
  `tests/test_roboverse_robot_cfg_validation.py`.
- [x] **Backend contract enforcement** - Static `-k general` test asserts every concrete
  `BaseSimHandler` subclass overrides each documented contract method. Known incomplete
  backends (pyrep, partial pybullet/genesis) are xfail-documented. Self-check guards the
  xfail list against staleness. `metasim/test/test_backend_contract_general.py`.
- [ ] **CI Coverage Reporting** - Integrate pytest-cov with Codecov

### P1 - High (Next Release)

- [x] **Parallel-sim error handling** - Fixed: every public method on `ParallelHandler`
  now drains `error_queue` + checks `process.is_alive` after wire I/O. EOFError/BrokenPipe
  on recv is translated into a `RuntimeError` carrying the real worker traceback instead
  of a cryptic IPC exception. OOM-killed workers (queue stays empty) are surfaced via the
  dead-process check. Tests: `metasim/test/test_parallel_error_handling_general.py`.
- [x] **Newton joint-name parity** - Fixed: Newton's `_collect_joint_names` now strips
  the object prefix (`box_base/box_joint` → `box_joint`) when the bare name is unambiguous,
  matching mujoco / sapien3 / isaacgym / isaacsim. Cross-platform tasks doing
  `dof_pos["box_joint"]` previously silently broke on Newton. Falls back to the full key
  on collision so no information is lost. Verified by `test_dict_state_all_objects[newton-*]`
  flipping from failed to passed.
- [x] **Newton silent-action drop guard** - Added: Newton's `_set_dof_targets` now warns
  once if `model.control()` has no `joint_target_pos` / `joint_target_vel` / `joint_f`
  buffer — previously every action was silently swallowed. Same antipattern that broke
  MuJoCo `set_states`.
- [x] **Actuator partial-spec warning (MuJoCo + Newton)** - Identified as task #6 root
  cause: when an actuator cfg overrides `stiffness` / `damping` but doesn't set
  `effort_limit_sim`, the asset-authored force-range (MJCF `forcerange` / Newton
  `joint_effort_limit`) silently dominates. Same `stiffness=1e5` config produces
  different effective behaviour per backend. Both handlers now warn once per
  (robot, joint) so the asymmetry is visible.
- [x] **Benchmark reproducibility — `env.reset(seed=N)` now actually reseeds** -
  End-to-end wired: `GymEnvAdapter.reset(seed=)` → `RLTaskEnv.reset(seed=)` /
  `TaskBase.reset(seed=)` → `BaseSimHandler.set_seed`. Added a default
  `set_seed` on the base handler that reseeds Python `random` + NumPy +
  Torch (CPU and CUDA), so every backend gets reproducibility for free.
  Backends with extra internal RNG (Newton warp kernels, Sapien physics
  noise) can override and call `super().set_seed(seed)` first. Tests:
  `test_set_seed_is_deterministic_on_numpy_and_torch` and
  `test_set_seed_differs_with_different_seeds` in
  `metasim/test/test_set_states_key_validation.py`. Backward-compat:
  `seed` is a keyword-only parameter — every existing call site is
  unchanged.
- [ ] **Abstract Method Declarations** - Mostly covered by the new backend-contract test;
  still want `@abstractmethod` on `_get_joint_names` / `_get_body_names` once pyrep /
  partial pybullet / genesis catch up.
- [ ] **Unified Environment Factory** - Create `roboverse.make_env()` API
- [ ] **Configuration System** - Document and standardize config approaches

### P2 - Medium (Future)

- [ ] **Code Quality** - Remove commented code, extract magic numbers
- [ ] **Type Annotations** - Complete type hints across codebase
- [ ] **Performance Benchmarks** - Add regression testing for performance

### P3 - Long-term

- [ ] **Plugin Architecture** - Modular simulator registration
- [ ] **Async Simulation** - Support for async environment stepping
- [ ] **Unified Config System** - Single configuration approach across modules

## Contributing

We welcome contributions! Here's how to get started:

1. **Read the detailed analysis**: [Architecture Review](docs/source/metasim/developer_guide/architecture_review.md)
2. **Pick an issue**: Start with P2 items for low-risk contributions
3. **Follow the protocol**: See "Safe Modification Protocol" in the review doc
4. **Submit a PR**: Include tests and update documentation as needed

### Recommended First Contributions

| Task | Effort | File |
|------|--------|------|
| Add @abstractmethod to `_set_dof_targets` | Very Low | `metasim/sim/base.py` |
| Remove commented-out code | Very Low | Multiple files |
| Add robot config validation tests | Medium | New test file |

## Release Timeline

| Version | Target | Focus |
|---------|--------|-------|
| 0.2.x | Current | Bug fixes, stability |
| 0.3.0 | TBD | Test coverage, API stabilization |
| 0.4.0 | TBD | Unified interfaces, deprecation cleanup |
| 1.0.0 | TBD | Stable API, comprehensive tests |

## Questions?

- [GitHub Discussions](https://github.com/RoboVerseOrg/RoboVerse/discussions)
- [Discord](https://discord.gg/6e2CPVnAD3)
- [Documentation](https://roboverse.wiki)
