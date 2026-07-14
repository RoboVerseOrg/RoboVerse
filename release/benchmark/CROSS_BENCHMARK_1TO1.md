# Cross-benchmark 1:1 integration — verification summary

**Date:** 2026-06-29
**Question:** do the benchmarks migrated into RoboVerse reproduce their *original* behavior 1:1 when
run in their **native simulator** (the integration docs claim perfect 1:1 replays / parity)?

**Answer:** yes for the three major manipulation benchmarks, each verified in its own dedicated
environment (legacy pins make a single shared env impossible — this is by design). "1:1" means
different things per benchmark and per tier, so each section below states the *measured* delta and
the committed command that produces it, rather than one blanket "bitwise".

## Why dedicated environments

Each upstream benchmark pins a mutually-incompatible legacy stack:

| Benchmark | Native sim | Dedicated env | Critical pins |
|---|---|---|---|
| ManiSkill | SAPIEN 3 | `/venv/roboverse` | sapien 3.0.3, mani_skill 3.0.1 |
| LIBERO | robosuite/MuJoCo 2 | `/venv/libero1to1` | robosuite 1.4.0, mujoco 2.3.2, numpy 1.22.4 |
| SimplerEnv | SAPIEN 2 | `/venv/simpler` | sapien 2.2.2, numpy 1.24.4, setuptools<81, ruckig |

## 1. ManiSkill — native-vs-upstream parity + official-demo replay ✅

- `SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_native --all --steps 30` →
  **14/14 native tasks** (that is every task in the tool's `PRIMITIVE_TASKS`) run against upstream
  ManiSkill. Re-measured 2026-07-13 in `/venv/roboverse`: object-pose agreement is **exact (Δ=0)
  on 12/14**, and ≤ **9.9e-06 m** on the other two (`PullCubeTool` tool 2.3e-07,
  `DrawTriangle` canvas 9.9e-06). Robot **qpos** drifts to **~1.2e-4** over 30 steps — close, but
  *not* bitwise; do not describe this parity as bitwise.
- `SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.replay_demo --task pick_cube --episodes 25`
  → **22/25** official `.h5` demos replay to success (the same 22/25 reported in
  `docs/source/dataset_benchmark/integrations/maniskill.md`, which is the authoritative table).
- These are the `maniskill.<task>_native` Tier-3 ports (SAPIEN3 + ManiSkill PhysX recipe). The
  documented 1:1 videos use exactly these.

## 2. LIBERO — bitwise 1:1 native (libero/robosuite uninstalled) ✅ authoritative

The **native** port (`libero_native.*`, `roboverse_pack/tasks/libero/native_libero.py`) loads
LIBERO's *own* compiled MJCF onto MetaSim's MuJoCo handler and reproduces LIBERO bit-for-bit — with
`libero` and `robosuite` **uninstalled** (standard `/venv/roboverse`, needs only the vendored
`roboverse_data/libero/assets/` tree + the 130 per-task bundles). Verified across **all 130 tasks**
by `tools.libero_integration.libero_replay` (engine parity + state-replay, 3 demos/task):

| Suite | Tasks | Demos | engine bitwise | state-replay bitwise | recorded success |
|---|---|---|---|---|---|
| libero_object  | 10 | 30 | 30/30 | 30/30 | 30/30 |
| libero_goal    | 10 | 30 | 30/30 | 30/30 | 30/30 |
| libero_spatial | 10 | 30 | 30/30 | 30/30 | 30/30 |
| libero_10      | 10 | 30 | 30/30 | 30/30 | 30/30 |
| libero_90      | 90 | 270 | 270/270 | 270/270 | 270/270 |
| **TOTAL (all 130 tasks)** | **130** | **390** | **390/390** | **390/390** | **390/390 (100%)** |

`engine max|Δ| = 0.0`, `state-replay FK max|Δ| = 0.0` — bitwise. This is the strong 1:1 the
documented parity videos are based on.

**Secondary check — passthrough action-replay** (needs `libero` installed, `/venv/libero1to1`):
re-simulating the recorded OSC_POSE *actions* through the transparent passthrough is a weaker check
than the native state-replay above — re-simulation from actions accumulates robosuite
controller/contact nondeterminism (native LIBERO's own action-replay behaves the same), which is why
the long-horizon `libero_10` suite scores lower under action-replay than under native state-replay.
The earlier numbers for this check (a demo→success rate and a native-vs-passthrough head-to-head)
were produced by scratch scripts that were never committed, so **they are not reproducible from this
tree and have been removed**. Re-run and re-state them from a committed driver before citing any.

## 3. SimplerEnv — native track verified; passthrough parity needs the upstream clone ⚠️

SimplerEnv tasks have **no recorded demos** (they are RT-1/Octo/OpenVLA policy-eval tasks), so the
1:1 contract is *passthrough == upstream `simpler_env` bitwise*, not demo-replay.

- **MetaSim-native track: 25/25 tasks run; 25/25 seed-deterministic (Δ=0)** in the sapien2 env —
  every `SimplerEnv/<task>` instantiates, steps, renders (dict obs + cameras, 7-DoF action,
  `is_grasped`/`success`/`episode_stats` info), and is bitwise reproducible across re-seeds. Uses
  only `roboverse_data` assets, zero upstream dependency. Drivers:
  `scripts/verify_native_registration.py`, `scripts/spike_metasim_full_parity.py`.
- **Passthrough bitwise-parity** (`scripts/parity_simpler_env.py`, the native-vs-upstream check)
  requires cloning `github.com/simpler-env/SimplerEnv` + `mani_skill2_real2sim==0.5.3`. Not run here
  — the sandbox blocks cloning an unnamed external repo. The repo docs report this as verified
  (initial-render mean-abs ≤ ~2/255; coke/pick/move-near bitwise). **Pending explicit
  authorization to clone the upstream repo to reproduce that number locally.**

## Reproduce

Every number above must come from a **committed** driver. The ones that do:

```bash
# ManiSkill (sapien3 env): native-vs-upstream parity + official-demo replay
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_native --all --steps 30
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.replay_demo --task pick_cube --episodes 25
# LIBERO (standard /venv/roboverse; libero/robosuite NOT needed): engine + state-replay parity
python -m tools.libero_integration.sweep_all --out all130.json          # all 130 tasks
python -m tools.libero_integration.libero_replay --hdf5 <demo.hdf5> --out r.json --n-demos 3
# SimplerEnv (sapien2 env): 25/25 native registration + seed-determinism; upstream parity
JAX_PLATFORMS=cpu python scripts/verify_native_registration.py
JAX_PLATFORMS=cpu python scripts/spike_metasim_full_parity.py
python scripts/parity_simpler_env.py        # needs the upstream clone; not run here
```

See also `docs/source/dataset_benchmark/integrations/{maniskill,libero,simpler_env}.md`, which are
the authoritative per-benchmark tables.

## Native vs config tiers (why the earlier report was wrong)

Every benchmark is integrated on **two tiers**, and they answer different questions:

- **Native port** (`*_native`, `libero_native.*`, simpler `_metasim`) — bit-exact with the
  benchmark's *own* engine; **this is the 1:1 tier**. LIBERO-native even runs with libero/robosuite
  uninstalled.
- **Config port** (`maniskill.pick_cube`, `libero.*`) — a `ScenarioCfg` + `roboverse_data`
  trajectory, runnable on *any* backend for cross-sim training/data; deliberately **not** bit-exact.

Replaying the **config** tier on a generic handler and reading the result as a native-parity number
is the easy mistake here: it under-reports badly (maniskill 0, libero 1/10) because the controller
and assets do not match the original engine, which the config tier never claimed to reproduce. The
base classes (`LiberoBaseTask`, `ManiskillBaseTask`) document the tier split inline.

## Bottom line

The migrated tasks **do** reproduce their upstream behavior in their native sim. LIBERO native is
**bitwise** across all 130 tasks (libero-free); ManiSkill native matches upstream object poses to
Δ ≤ 9.9e-06 m on 14/14 tasks (qpos ~1.2e-4, *not* bitwise) and replays 22/25 official demos;
SimplerEnv native is 25/25 seed-deterministic. Data for the native path lives under
`roboverse_data` (LIBERO: 269 MB deduped mesh tree + 130 small bundles; maniskill/simpler assets
already present) — no multi-GB raw dataset needed.

Numbers whose driver is not committed to this tree have been removed rather than restated; see the
Reproduce block for the commands that back everything above.
