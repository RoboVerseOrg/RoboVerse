# RoboVerse — next release (paste-ready notes)

**Suggested tag**: `v1.0.0-beta` (follows the `v1.0.0-alpha` lineage; the
"beta" gate is the cross-platform-parity work landing in this release).

**Usage**: rename `[Unreleased]` → `[1.0.0-beta] - 2026-05-31` in
`CHANGELOG.md`, then:

```bash
git tag -a v1.0.0-beta -m "RoboVerse v1.0.0-beta — cross-platform parity is testable"
git push origin v1.0.0-beta
gh release create v1.0.0-beta --prerelease \
  --title "RoboVerse v1.0.0-beta — cross-platform parity is testable" \
  --notes-file RELEASE_NOTES_NEXT.md
```

This release should be cut alongside **MetaSim v0.2.0** (see
`MetaSim/RELEASE_NOTES_NEXT.md`).

---

## Cross-platform parity is now load-bearing

Every shipped `RobotCfg`, every contracted handler method, and every
benchmark `reset(seed=N)` is exercised across the supported backends and
either passes or is xfail-documented with a specific reason. The release is
forward- and backward-compatible.

### Highlights

- **60 `RobotCfg`s validated**. `tests/test_roboverse_robot_cfg_validation.py`
  exercises every shipped robot for instantiation, non-empty name, and
  defaults-inside-limits. 9 real downstream bugs surfaced:
  - `AlohaAgilex`, `G1Tracking` — orphan joints in `default_joint_positions`
  - `YamCfg`, `ArxL5Cfg`, `Vega`, `SoArm100`, `Koch`, `Go2`, `AllegroHand` —
    defaults outside `joint_limits` ranges
- **mjlab 1:1 obs/reward parity** got measurably tighter. `velocity_rough_go1`
  / `velocity_rough_g1` now ship `height_scan` obs (scale `1/max_distance`,
  clip-then-scale order matches mjlab), continuous rough terrain, and
  `tracking-g1` motion-tracking obs. go1/g1 `base_lin_vel` reads the IMU
  velocimeter at the offset site (adds `ω × r` cross-term — closes a real
  obs gap that diverged under turning).
- **IL + RL fusion bridge** (`roboverse_learn/fusion/`) wires RL training →
  demo collection → IL warmstart end-to-end. 6/6 tests, validated against a
  real cartpole checkpoint and real mujoco/EGL launch.
- **RoboTwin v2 passthrough** — 50 native gym envs registered, 1:1 by
  construction (same pattern as the existing ManiSkill passthrough).
- **`get_started/0_static_scene.py`** no longer crashes on first-time setup
  when no cameras are configured.
- **AGENTS.md / CLAUDE.md** lands: repo-level dev rules for AI agents
  (parity is load-bearing, multi-repo discipline, commit-as-user).

### Migration

No required changes. If you maintain an out-of-tree `RobotCfg`, the new
validation test will exercise it as soon as it is imported through
`roboverse_pack` — fix any flagged `default_joint_positions` mismatches
before they trip in CI.

### Verification

**323 tests pass** in the cross-4-backend sweep
(mujoco + sapien3 + newton + passthroughs) in the `roboverse` conda env,
zero regressions.

### Full changelog

See [`CHANGELOG.md`](./CHANGELOG.md) for the complete commit-by-commit list.
