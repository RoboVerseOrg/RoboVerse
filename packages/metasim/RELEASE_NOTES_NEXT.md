# MetaSim — next release (paste-ready notes)

**Suggested tag**: `v0.2.0` (pyproject is `0.1.17`; minor bump for
backward-compatible additions).

**Usage**: rename `[Unreleased]` → `[0.2.0] - 2026-05-31` in `CHANGELOG.md`,
then:

```bash
git tag -a v0.2.0 -m "MetaSim v0.2.0 — cross-platform infrastructure hardening"
git push origin v0.2.0
gh release create v0.2.0 \
  --title "MetaSim v0.2.0 — cross-platform infrastructure hardening" \
  --notes-file RELEASE_NOTES_NEXT.md
```

---

## Cross-platform infrastructure hardening

This release tightens the `BaseSimHandler` contract so the same MetaSim code
runs identically on mujoco, sapien3, newton, isaacsim, mjx, isaacgym, and
pybullet — without the silent per-backend divergences that previously broke
downstream experiments.

**The release is forward- and backward-compatible.** Every behavior change is
additive (new warnings, new optional kwargs); no existing call site needs to
be modified.

### Highlights

- **`gym.reset(seed=N)` actually seeds the simulator now.** `BaseSimHandler`
  ships a default `set_seed` that seeds Python `random`, `numpy`, and `torch`
  CPU+CUDA; the seed is forwarded through `ParallelHandler` (broadcast to
  workers) and `HybridHandler` (physics + render). Closes the benchmark
  reproducibility gap.
- **`set_states` silent-drop is gone.** The dict key `dof_pos_target` (which
  previously silently broke 15 downstream BC experiments) now emits a
  one-shot warning recommending `set_dof_targets`. Same guard for unknown
  robot/object names, unknown joint names, read-only velocity fields, and
  partial pose dicts.
- **Newton joint-name parity.** Newton previously returned
  `box_base/box_joint` while every other backend returned `box_joint` —
  cross-platform tasks indexing `dof_pos["box_joint"]` silently failed. Fixed
  with safe disambiguation: bare name when unambiguous, full key on
  collision.
- **Sapien3 `_set_states`** no longer raises `KeyError` on missing
  `dof_pos`/`pos`/`rot`; it defaults to zeros, matching mujoco's behavior.
  It also instantiates and reads **all** robots (was only reading the first).
- **mjx backend unblocked** — eager init, JAX 0.10+ `__dlpack__` protocol,
  recursive MJCF `<include>` parser with cycle detection.
- **ParallelHandler error surfacing** — `EOFError` / `BrokenPipe` /
  OOM-killed workers now raise a `RuntimeError` carrying the real worker
  traceback instead of hanging silently.
- **Multi-agent (bimanual) trajectory loading** lands natively in `get_traj`
  via `list[RobotCfg]`. Single-agent path is unchanged.

### Migration

No required changes. If you previously **relied** on the silent drop of
`set_states({"dof_pos_target": ...})`, switch to:

```python
handler.set_dof_targets([{robot_name: {"dof_pos_target": ...}}])
```

If you previously caught `KeyError` from sapien3 `_set_states` on missing
`dof_pos`, that branch is now dead — sapien3 defaults to zeros.

### Verification

**386 general tests pass + 21 xfailed** across a 5-backend sweep
(general + mujoco + sapien3 + newton + passthroughs). Per-backend
PD-convergence shortfalls are xfail-documented with specific reasons (not
silenced).

### Full changelog

See [`CHANGELOG.md`](./CHANGELOG.md) for the complete commit-by-commit list.
