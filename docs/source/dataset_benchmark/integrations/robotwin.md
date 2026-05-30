# RoboTwin → MetaSim / Sapien integration

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is a 50-task
dual-arm tabletop benchmark built on SAPIEN 3.0.0b1 + mplib + curobo.
Tasks live under `envs/<task>.py` and each declares its own scene
setup, success criterion, and scripted-policy data collector via raw
SAPIEN API.

## Status

- **All 50 tasks collect successfully (breadth)**: a full sweep
  (`tools/robotwin_integration/coverage_sweep.py`) ran every registered
  RoboTwin task through the native code path + data bridge — **50/50**
  plan and check successfully and emit a dense bimanual trajectory
  (78–662 frames; some need up to seed 7). This is collection-success
  across the whole suite, not one hand-picked task.
- **Replay parity is measured, not asserted**: the parity harness
  (`tools/robotwin_integration/parity_robotwin.py`) replays the native
  command-target stream on RoboVerse-SAPIEN3 and compares RoboVerse's
  *achieved* joint state against RoboTwin's *achieved* joint state
  (`entity.get_qpos()`, captured by the bridge — not the command target,
  which would be circular). On `beat_block_hammer` the per-joint achieved
  delta converges with replay resolution: **0.44 → 0.088 → 0.027 →
  0.0059 rad max** (mean 0.033 → 0.0008 rad) at settle = 1/4/8/16. The
  residual is open-loop replay under-stepping, not a mapping error — same
  URDF, same backend family.
- **Embodiment loads in RoboVerse**: RoboTwin's ALOHA-AgileX
  (`arx5_description_isaac.urdf`, 38 DoF: dual 6-DoF arms with 2-finger
  mimic grippers + mobile base + sensor mast) loads and steps in
  MetaSim/Sapien3 after one small handler fix
  (`fix/sapien3-passive-joints`).
- **Native passthrough is 1:1 by construction**: with RoboTwin's deps
  installed in a dedicated `robotwin` conda env, `RoboTwin/<task>`
  resolves to the live native task (see `_passthrough.py`) — same sim,
  planner, and `check_success()` as upstream, the way the ManiSkill
  passthrough is identical to native ManiSkill. The two-env split is
  required because RoboTwin pins SAPIEN 3.0.0b1 / mplib 0.2.1 / curobo,
  which conflict with the `roboverse` env's SAPIEN.
- **Genuine limitations (stated plainly)**: the bridge/replay path is
  *open-loop state replay* — it does not prove dynamical equivalence,
  runs no planner/policy in RoboVerse, and does not check task success in
  RoboVerse. The replayed scene draws the manipulated object as a
  primitive proxy (real meshes not yet loaded), so contact-rich fidelity
  and an in-RoboVerse success check remain future work.

## MetaSim fix that enables this

The `Sapien3Handler` used to crash with `KeyError` when an active
URDF joint wasn't enumerated in `RobotCfg.actuators`. That's the rule
for most clean academic robots but it's wrong for any
embodiment that bundles wheels, suspension, or a sensor mast — those
DoFs exist in the URDF but no one wants them in the actuator dict.

The fix (`fix/sapien3-passive-joints`) switches the lookup to
`actuators.get(name)` and skips undriven joints. `default_joint_positions`
gets the same treatment, defaulting to 0.0 for unenumerated joints.
Two-line change in `_build_sapien`, plus a regression test at
`metasim/test/test_sapien3_passive_joints.py`.

## Asset layout

| Bundle                  | Size       | Needed?                                          |
|-------------------------|------------|--------------------------------------------------|
| `embodiments.zip`       | 220 MB     | **Yes** — robot URDFs + meshes for all 5 robots |
| `objects.zip`           | 3.74 GB    | Yes for task scene actors (YCB-style)            |
| `background_texture.zip`| 11 GB      | Domain-randomization training only               |
| Full dataset            | 1.47 TB    | Demo trajectories + RL checkpoints — not needed for sim parity |

Locator (`roboverse_pack/robots/aloha_agilex_cfg.py`) searches
`~/projects/robotwin/assets/` or `$ROBOTWIN_ASSETS`.

## Data bridge

RoboTwin demos are *single-embodiment bimanual*: one articulation whose
14-D action `[L_arm(6), L_grip, R_arm(6), R_grip]` drives both arms.
RoboVerse expresses this as one name-keyed robot entry — the one-robot
case of the same `*_v2` format the multi-agent loader uses (see the
[multi-agent dataset docs](../dataset/multiagent.md)). Because RoboTwin
and RoboVerse both run SAPIEN3, dof-position-target replay reproduces the
recorded motion closely.

The bridge is two halves, one per conda env, hand-off via a plain pickle:

1. **Collect** (`robotwin` env) —
   `tools/robotwin_integration/collect_bridge.py` drives a native RoboTwin
   task (the same `_passthrough` factory), retries seeds until one plans
   *and* checks successfully, and dumps per frame: the command-target
   `vectors`, RoboTwin's *achieved* qpos `real_vectors`
   (`entity.get_qpos()`, injected via a runtime hook on `get_obs` — no
   upstream edit), the achieved end-effector poses `left/right_endpose`,
   and the initial object poses.
2. **Replay** (`roboverse` env) —
   `get_started/10_robotwin_aloha_replay.py` converts that pickle into a
   name-keyed `*_v2` dataset (via the shared
   `roboverse_pack.tasks.robotwin._convert`), loads it through `get_traj`,
   and replays the ALOHA-AgileX embodiment on SAPIEN3 to video.
3. **Measure parity** (`roboverse` env) —
   `tools/robotwin_integration/parity_robotwin.py` runs the same replay
   and reports the per-joint delta between RoboVerse-achieved and
   RoboTwin-achieved qpos (`--settle N` controls replay resolution;
   `--all` sweeps every collected pickle).

```bash
# 1. collect a demonstration natively, with achieved state (robotwin env)
conda run -n robotwin env MUJOCO_GL=egl python \
  tools/robotwin_integration/collect_bridge.py --task beat_block_hammer \
  --out ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl

# 1b. (optional) sweep the whole 50-task suite -> coverage.json
conda run -n robotwin env MUJOCO_GL=egl SAPIEN_HEADLESS=1 python \
  tools/robotwin_integration/coverage_sweep.py --max-seeds 8

# 2. replay it in RoboVerse (roboverse env)
MUJOCO_GL=egl python get_started/10_robotwin_aloha_replay.py \
  --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl --sim sapien3

# 3. measure achieved-vs-achieved parity (roboverse env)
MUJOCO_GL=egl python tools/robotwin_integration/parity_robotwin.py \
  --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl --settle 8
```

## Native passthrough

`roboverse_pack.tasks.robotwin._passthrough` registers all 50 tasks under
`RoboTwin/<name>` with a lazy entry point. Registration never imports
RoboTwin (safe in any env); making the env imports the native task. Two
runtime quirks are handled in `_make_robotwin_env`: it `chdir`s to the
checkout (RoboTwin reads `./assets/...` relatively at import) and aliases
`warp.torch.*` to the `warp` top level (curobo 0.7.8 expects the old
namespace that warp-lang ≥ 1.5 dropped). This only runs in an env where
RoboTwin's deps (incl. a curobo built against an sm-matching CUDA nvcc)
are installed.

## Setup (RoboTwin env + assets)

```bash
mkdir -p ~/projects && cd ~/projects
git clone --depth 1 https://github.com/RoboTwin-Platform/RoboTwin.git robotwin
cd robotwin && bash script/_install.sh        # deps + curobo (needs nvcc)
cd assets && python _download.py && unzip -q '*.zip'   # embodiments + objects
```

Note: on recent GPUs (e.g. sm_120 / RTX 50-series) curobo must be built
with a matching CUDA nvcc (≥ 12.8); install `cuda-nvcc` of that version in
the env before `pip install -e curobo`. The embodiment locator
(`roboverse_pack/robots/aloha_agilex_cfg.py`) searches
`~/projects/robotwin/assets/` or `$ROBOTWIN_ASSETS`. To just confirm the
embodiment loads (no RoboTwin deps needed), run
`python -m tools.robotwin_integration.aloha_demo`.
