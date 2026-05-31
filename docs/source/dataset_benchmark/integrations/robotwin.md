# RoboTwin → MetaSim / Sapien integration

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is a 50-task
dual-arm tabletop benchmark built on SAPIEN 3.0.0b1 + mplib + curobo.
Tasks live under `envs/<task>.py` and each declares its own scene
setup, success criterion, and scripted-policy data collector via raw
SAPIEN API.

## Status

- **Full policy-reproduction pipeline (collect → train → eval)**: a RoboVerse
  user can collect RoboTwin expert demos, train a Diffusion Policy with
  RoboVerse's own `roboverse_learn/il`, and evaluate it **closed-loop in the
  native RoboTwin env** — the same three-step experience as RoboTwin, with a
  directly comparable success rate. See [Policy reproduction](#policy-reproduction-same-experience-as-robotwin).
  The data path runs through the *unmodified* `data2zarr_dp.py`; the eval rolls
  the policy through RoboTwin's own `take_action` interface.
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
- **Mesh-faithful, 1:1-verified replay**: the replay
  (`tools/robotwin_integration/mesh_replay_robotwin.py`) loads the *real*
  RoboTwin object meshes (rigid GLB/OBJ; URDF articulations baked to
  textured GLB or driven as articulations when they move — doors/lids
  open), with ray-traced rendering (`--rt`) matching RoboTwin. A
  native-vs-RoboVerse side-by-side (`sidebyside.py`, ground truth from
  `native_render.py --replay-bridge`) confirms robot pose + object
  positions + motion + camera + RT lighting are **frame-for-frame 1:1**
  (the native side replays the *same* bridge trajectory, so it is the
  identical episode, not a coincidental match).
- **Genuine limitations (stated plainly)**: the bridge/replay path is
  *open-loop state replay* — a tight delta proves trajectory fidelity, not
  dynamical equivalence, and runs no planner/policy in RoboVerse. The
  separate *physics* object-parity (objects move by contact, not
  teleported) reaches ≤5 cm for ~26/46 tasks and diverges for complex
  contact (the open-loop limit). Pixel-level render parity is bounded by
  the engines (RT vs. RoboTwin's exact lights); a *moving* URDF object
  renders untextured (sapien3's articulation loader drops `.mtl`).

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

## Policy reproduction (same experience as RoboTwin)

A RoboVerse user can reproduce a RoboTwin policy result end to end — collect
expert demos, train an imitation policy, evaluate it closed-loop — using
RoboVerse's own imitation-learning stack (`roboverse_learn/il`), the same
three-step `collect → train → eval` flow a RoboTwin user runs. The trained
policy is evaluated **closed-loop in the native RoboTwin environment** (via the
passthrough), so its success rate is directly comparable to RoboTwin's expert.

```bash
# 1. COLLECT — expert demos with head-camera RGB (robotwin env).
#    One seed per subprocess under a timeout, so a headless-RT hang costs one
#    seed, not the batch; gathers N distinct successful episodes.
bash tools/robotwin_integration/collect_demos_robust.sh \
  --task beat_block_hammer \
  --out-dir ~/projects/robotwin/data/_rv_bridge/bbh_train \
  --want 40 --camera head_camera

# 2. TRAIN — RoboVerse Diffusion Policy on the RoboTwin demos (roboverse env).
#    Converts bridge pkls -> demo dirs -> zarr (the *unmodified* data2zarr_dp.py)
#    -> DP training, with the bimanual 14-D / 240x320 shape overrides.
bash tools/robotwin_integration/train_dp_robotwin.sh \
  --task beat_block_hammer \
  --bridge-dir ~/projects/robotwin/data/_rv_bridge/bbh_train \
  --num 40 --epochs 300 --policy ddpm_unet

# 3. EVAL — closed-loop in native RoboTwin, one command (starts the policy
#    server in the roboverse env, runs the env in the robotwin env, reports the
#    success rate, tears the server down).
bash tools/robotwin_integration/eval_dp_robotwin.sh \
  --task beat_block_hammer \
  --ckpt il_outputs/ddpm_unet/beat_block_hammer/checkpoints/300.ckpt \
  --num-eval 20 --start-seed 100
```

The state/action are **non-circular**: the policy's state observation is
RoboTwin's *achieved* joint qpos (`real_vector`), and the action it learns is the
command target (`vector`) — the same two signals the parity harness uses. The
eval rolls the policy out through `env.take_action(action, 'qpos')`, the exact
closed-loop interface RoboTwin's own `script/eval_policy.py` uses (TOPP-
interpolates the 14-D waypoint, steps physics, fires `eval_success` on
`check_success()`).

Two implementation notes that make this work across the env split:

- **Env-decoupled eval.** The DP model + its deps run in the `roboverse` env, but
  the only closed-loop RoboTwin env runs in the `robotwin` env (conflicting
  SAPIEN/torch). `dp_policy_server.py` (roboverse env) serves inference over a
  socket and `eval_robotwin_policy.py`'s `DPPolicy` (robotwin env) is a thin
  client — mirroring RoboTwin's own policy server/client split. `eval_dp_robotwin.sh`
  hides this behind one command.
- **numba is optional.** The IL image dataset jit-compiles its sampler with numba,
  which fails to import on numpy ≥ 2.0; it now falls back to a pure-numpy path so
  training runs on a modern-numpy `roboverse` env.

An **open-loop action-replay** baseline is built into the same eval harness
(`--policy replay --bridge <pkl>`): it feeds RoboTwin's recorded action stream
back through `take_action` (TOPP, not the original curobo plan). On
`beat_block_hammer` it reproduces success 4/5 closed-loop — a sharp datapoint that
also motivates the reactive DP policy (open-loop replay has no feedback
correction; a learned policy does).

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
   the **per-frame world pose of every scene object** `object_traj`
   (rigid actors *and* URDF articulations via `get_all_articulations()`),
   the **articulation joint qpos** `object_joint_traj` (so opening doors
   replay), and each object's real mesh/URDF path `object_meshes`.
2. **Replay** (`roboverse` env) —
   `tools/robotwin_integration/mesh_replay_robotwin.py` converts the
   trajectory to `*_v2` (shared `roboverse_pack.tasks.robotwin._convert`)
   and replays the ALOHA-AgileX embodiment **with the real object meshes**
   on SAPIEN3 to video. `--mode kinematic` is faithful playback (robot +
   objects teleported to the recorded state each frame); `--mode physics`
   drives the robot by command targets and lets objects move by contact
   (for object-pose parity). `--rt` ray-traces to match RoboTwin;
   `--observer-cam --cam-pos/--cam-lookat/--fovy` set a matched camera.
   (`get_started/10_robotwin_aloha_replay.py` is the minimal get-started
   version with a primitive object proxy.)
3. **Measure parity** (`roboverse` env) —
   `tools/robotwin_integration/parity_robotwin.py` reports the per-joint
   delta between RoboVerse-achieved and RoboTwin-achieved qpos (`--settle N`
   replay resolution; `--all` sweeps every pickle).
4. **Verify 1:1** (both envs) — `tools/robotwin_integration/sidebyside.py`
   builds a native-vs-RoboVerse proof video for any task: it renders the
   RoboTwin ground truth (`native_render.py --replay-bridge`, which drives
   the native env from the *same* bridge trajectory instead of re-planning)
   and the RoboVerse replay from an identical camera, and composites them
   frame-for-frame.

```bash
# 1. collect a demonstration natively, with achieved state + objects (robotwin env)
conda run -n robotwin env MUJOCO_GL=egl python \
  tools/robotwin_integration/collect_bridge.py --task move_can_pot \
  --out ~/projects/robotwin/data/_rv_bridge/move_can_pot.pkl

# 1b. (optional) sweep the whole 50-task suite -> coverage.json
conda run -n robotwin env MUJOCO_GL=egl SAPIEN_HEADLESS=1 python \
  tools/robotwin_integration/coverage_sweep.py --max-seeds 8

# 2. mesh-faithful, ray-traced replay in RoboVerse (roboverse env)
MUJOCO_GL=egl python tools/robotwin_integration/mesh_replay_robotwin.py \
  --bridge ~/projects/robotwin/data/_rv_bridge/move_can_pot.pkl --mode kinematic --video --rt

# 3. measure achieved-vs-achieved joint parity (roboverse env)
MUJOCO_GL=egl python tools/robotwin_integration/parity_robotwin.py \
  --bridge ~/projects/robotwin/data/_rv_bridge/move_can_pot.pkl --settle 8

# 4. one-command native-vs-RoboVerse 1:1 side-by-side (roboverse env)
conda run -n roboverse python tools/robotwin_integration/sidebyside.py --task move_can_pot
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
