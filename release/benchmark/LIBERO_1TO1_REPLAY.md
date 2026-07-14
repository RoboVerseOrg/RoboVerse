# LIBERO 1:1 replay verification

**Date:** 2026-06-29 (numbers reviewed 2026-07-13)
**Goal:** verify that LIBERO tasks migrated into RoboVerse reproduce their *official* demonstrations
1:1 against LIBERO's own engine.

Two tracks, and they are not equally strong:

1. **Native state-replay (authoritative, below).** Load LIBERO's own compiled MJCF onto MetaSim's
   MuJoCo handler, restore each recorded state, compare bitwise. Runs with `libero`/`robosuite`
   **uninstalled**. Driver: `tools.libero_integration.{libero_replay,sweep_all}` (committed).
2. **Passthrough action-replay (secondary).** Re-simulate the recorded OSC_POSE *actions* through
   the passthrough against LIBERO's robosuite stack. Weaker (controller/contact nondeterminism), and
   its numbers are **withdrawn** here because no driver for it is committed — see below.

This is the "1:1 replay" the integration docs promise (`docs/source/dataset_benchmark/integrations/libero.md`),
and it exercises the **native** stack. Replaying the generic config port on a generic MuJoCo handler
instead — mismatched controller and assets — measures the cross-sim training tier, not this one, and
under-reports LIBERO badly. The two tiers are not comparable; see the tier split below.

## Environment (dedicated — required)

LIBERO pins a legacy stack that is mutually incompatible with the modern sapien3/maniskill venv, so
it needs its own conda env (`/venv/libero1to1`, python 3.10):

- `robosuite==1.4.0` (1.5.2 removed `single_arm_env`, breaking LIBERO import)
- `mujoco==2.3.2` (3.x changed the `mj_fullM` signature robosuite 1.4 calls)
- `numpy==1.22.4`, `gym==0.25.2`, `bddl==1.0.1`, `robomimic==0.2.0`
- `metasim` + `roboverse_pack` installed `--no-deps`; run with
  `PYTHONPATH=MetaSim:RoboVerse:LIBERO MUJOCO_GL=egl`

Each upstream benchmark needs its own dedicated env by design (legacy pins); maniskill uses a
separate sapien3 env, simpler_env needs a sapien2 env.

## Method (native state-replay — the authoritative track)

For each task: open the official demo `.hdf5` (`actions` (T,7) OSC_POSE, `states` (T,110) sim
states) and its embedded compiled `model_file` MJCF; load that MJCF both in raw MuJoCo (LIBERO's
own engine) and on MetaSim's MuJoCo handler; step identical controller-free `ctrl` (engine parity),
then set each recorded state on the handler and compare forward-kinematics body world poses
(state replay). 3 demos/task across all 130 tasks.

## Results — NATIVE bitwise 1:1 (authoritative, libero/robosuite uninstalled)

The `libero_native.*` port loads LIBERO's own compiled MJCF onto MetaSim's MuJoCo handler; verified
by `tools.libero_integration.libero_replay` in the standard `/venv/roboverse` env (no `libero`, no
`robosuite`; needs only the vendored `roboverse_data/libero/assets/` tree + the 130 bundles):

| Suite | Tasks | Demos | engine bitwise | state-replay bitwise | recorded success |
|---|---|---|---|---|---|
| libero_object / goal / spatial / 10 | 40 | 120 | 120/120 | 120/120 | 120/120 |
| libero_90 | 90 | 270 | 270/270 | 270/270 | 270/270 |
| **TOTAL (all 130)** | **130** | **390** | **390/390** | **390/390** | **390/390 (100%)** |

`engine max|Δ| = 0.0`, `state-replay FK max|Δ| = 0.0`. This is the strong 1:1 the parity videos use.

## Secondary — passthrough action-replay → success (needs libero installed)

Re-simulating the recorded OSC_POSE *actions* through the passthrough (rather than restoring the
recorded states) is a strictly weaker check: it accumulates robosuite controller/contact
nondeterminism, so long-horizon suites score lower — native LIBERO's own action-replay behaves the
same way, which is why this is not an integration defect.

> **Removed (review, 2026-07-13):** the per-suite action-replay success table and the
> "12/12 native-vs-passthrough" head-to-head previously printed here were produced by scratch
> scripts (`libero_replay.py`, `libero_headtohead.py`) that live in `/tmp` and were **never
> committed**. They cannot be reproduced from this tree, so the numbers are withdrawn rather than
> restated. Commit a driver under `tools/libero_integration/` and re-measure before citing any
> action-replay figure. The **native, state-replay** result above stands: it is produced by the
> committed `tools.libero_integration.{libero_replay,sweep_all}`.

## Reproduce

```bash
# The authoritative native check — standard /venv/roboverse, no libero, no robosuite:
python -m tools.libero_integration.sweep_all --out all130.json     # all 130 tasks, engine + state
python -m tools.libero_integration.libero_replay \
    --hdf5 <path/to/demo.hdf5> --out report.json --n-demos 3       # one task, 3 demos
```

The passthrough (action-replay) track needs the legacy env described above
(`/venv/libero1to1`, `PYTHONPATH=MetaSim:RoboVerse:LIBERO MUJOCO_GL=egl`) **and** a committed
driver, which this tree does not yet have.

## Status of the broader 1:1 effort

- **maniskill** — verified in the sapien3 env: `parity_native --all --steps 30` = **14/14** tasks,
  object-pose Δ ≤ 9.9e-06 m (12/14 exactly 0), robot qpos ~1.2e-4 (not bitwise); PickCube
  official-demo replay 22/25 (see `docs/.../integrations/maniskill.md`).
- **LIBERO** — native state-replay: all 130 tasks / 390 demos, engine and state-replay bitwise
  (this report). The action-replay track has no committed driver (see above).
- **simpler_env** — native track DONE in a dedicated sapien 2.2.2 env: 25/25 tasks run, 25/25
  seed-deterministic (Δ=0). Passthrough==upstream bitwise check pending the upstream clone (see
  `CROSS_BENCHMARK_1TO1.md`).
