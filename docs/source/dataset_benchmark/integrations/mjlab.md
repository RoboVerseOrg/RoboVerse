# mjlab integration (MuJoCo backend)

This page documents the integration of [mjlab](https://github.com/mujocolab/mjlab)'s
registered task suite into RoboVerse's MetaSim/MuJoCo backend.

mjlab pairs the Isaac-Lab manager-based RL API with MuJoCo-Warp's GPU
simulator. RoboVerse's MetaSim already speaks MuJoCo (CPU), so the goal
of this integration is *physics-step parity*: for a given task MJCF and
identical control inputs, the trajectory MetaSim produces should match
what mjlab's underlying `mj_step` produces. Once physics parity holds,
reward functions and observation manager ports become straightforward
because they read from identical state.

## Coverage

All 12 mjlab-registered task IDs are wrapped under
`roboverse_pack.tasks.mjlab` and run end-to-end on MetaSim. Final
parity (200 control steps, raw mujoco vs MetaSim/MuJoCo):

| Task                                          | nq  | qpos_max\|Δ\| | qvel_max\|Δ\| | verdict          |
|-----------------------------------------------|-----|---------------|---------------|------------------|
| `mjlab.cartpole_balance`                      | 2   | 0.00e+00      | 0.00e+00      | bitwise          |
| `mjlab.cartpole_swingup`                      | 2   | 0.00e+00      | 0.00e+00      | bitwise          |
| `mjlab.velocity_flat_g1`                      | 36  | 1.59e-13      | 1.46e-13      | machine epsilon  |
| `mjlab.velocity_rough_g1`                     | 36  | 1.59e-13      | 1.46e-13      | machine epsilon  |
| `mjlab.tracking_flat_g1`                      | 36  | 1.59e-13      | 1.46e-13      | machine epsilon  |
| `mjlab.tracking_flat_g1_no_state_est`         | 36  | 1.59e-13      | 1.46e-13      | machine epsilon  |
| `mjlab.velocity_flat_go1`                     | 19  | 1.58e-01      | 1.71e-01      | contact residual |
| `mjlab.velocity_rough_go1`                    | 19  | 1.58e-01      | 1.71e-01      | contact residual |
| `mjlab.lift_cube_yam`                         | 8   | 1.61e+00      | 1.20e+01      | contact residual |
| `mjlab.lift_cube_yam_rgb`                     | 8   | 1.61e+00      | 1.20e+01      | contact residual |
| `mjlab.lift_cube_yam_depth`                   | 8   | 1.61e+00      | 1.20e+01      | contact residual |
| `mjlab.multi_cube_seg_yam`                    | 8   | 1.61e+00      | 1.20e+01      | contact residual |

The cartpole reward (mjlab's `cartpole_smooth_reward`) is also bitwise
identical (`Σ reward = 114.32` exact on balance, `0.084` exact on
swingup).

## Asset layout

MJCF assets resolve in two ways (see `roboverse_pack/tasks/mjlab/_locator.py`):

1. **Local mjlab clone** — `~/projects/mjlab`, or set `MJLAB_REPO=/path/to/mjlab`.
   Used for development so the canonical mjlab bytes drive parity checks.
2. **HuggingFace fallback** — with no clone present, the referenced XML and its
   meshes are downloaded on demand from `RoboVerseOrg/roboverse_data` under
   `robots/mjlab/…`. This is what lets a fresh checkout run without cloning mjlab.

MJCFs we currently consume:

- `tasks/cartpole/cartpole.xml` — the cartpole *scene* (own floor,
  rails, cameras, plus cart+pole).
- `asset_zoo/robots/unitree_g1/xmls/g1.xml` — Unitree G1 humanoid
  (29 DoF, internal `<freejoint name="floating_base_joint"/>`).
- `asset_zoo/robots/unitree_go1/xmls/go1.xml` — Unitree Go1 quadruped
  (12 DoF, same internal freejoint pattern).
- `asset_zoo/robots/i2rt_yam/xmls/yam.xml` — i2rt YAM 7-DoF arm.

## Loading model: scene vs robot

mjlab assets fall into two shapes that need different MetaSim load
paths:

1. **Complete scene MJCF** (e.g. cartpole.xml). The MJCF declares its
   own floor, rails, lights, cameras, plus the moving body. Loading it
   as `RobotCfg(mjcf_path=…)` triggers namespace-wrapping that
   re-parents the worldbody decorations into a wrapper body, breaking
   the authored world-frame placement. Instead, pass it as the *scene*:

   ```python
   ScenarioCfg(
       scene=SceneCfg(mjcf_path=mjlab_asset("tasks/cartpole/cartpole.xml")),
       robots=[],
       simulator="mujoco",
       add_default_ground=False,  # cartpole.xml already declares a floor
   )
   ```

2. **Bare robot MJCF** (g1, go1, yam). The MJCF only declares the
   robot; no world floor, no cameras. Load as `RobotCfg`:

   ```python
   ScenarioCfg(
       robots=[RobotCfg(
           name="g1",
           num_joints=29,
           fix_base_link=False,
           mjcf_path=mjlab_asset("asset_zoo/robots/unitree_g1/xmls/g1.xml"),
           enabled_self_collisions=False,
       )],
       simulator="mujoco",
       add_default_ground=False,  # raw rollout doesn't add a ground either
   )
   ```

The `add_default_ground=False` is critical when comparing against raw
`MjModel.from_xml_path` — the raw load doesn't auto-attach a ground
plane, so MetaSim mustn't either.

## Fixes that landed in MetaSim

Five small, backward-compatible changes on branch
`fix/mujoco-mjlab-task-parity`:

1. **`ScenarioCfg.add_default_ground`** (default `True`). New opt-out
   field so callers can suppress MetaSim's auto-floor when their scene
   MJCF already provides one. Default keeps the legacy behavior.
2. **Missing-inertial fallback** in `_add_robots_to_model`. Robots
   whose root body has no `<inertial>` (URDF norm, MuJoCo computes
   inertia from geom mass) no longer crash with `AttributeError`.
3. **MJCF root-offset composition** in `_add_robots_to_model`. When the
   re-rooting branch hoists a non-origin root body onto the attachment
   wrapper, the user-supplied `default_position` now composes with the
   MJCF offset instead of overwriting it. Backward-compatible for
   URDFs whose root body sits at origin.
4. **Inner-freejoint promotion** in `_add_robots_to_model`. Any
   author-declared `<freejoint>` inside the robot MJCF is removed
   before attach and re-added on the wrapper (which is top-level by
   construction), so mjlab's go1 / g1 no longer trip MuJoCo's "free
   joint can only be used on top level" compile-time check.
5. **`BaseTaskEnv` class-attr scenario fallback**. The docstring
   already promised "If None, it will use the class variable
   'scenario'"; the implementation now honours it. Required for
   class-based `@register_task("mjlab.*")` wrappers to instantiate
   via `cls()`.

Regression tests live at
`metasim/test/test_mujoco_scene_and_attach.py` (mocked / minimal XML,
no heavy runtime).

## Known divergence sources

The two open residuals (go1 / yam) trace to contact filtering
semantics:

- **raw mujoco default**: `option.flag.filterparent = enable`
  (parent-child contacts filtered); no `<exclude>` pairs.
- **MetaSim with `enabled_self_collisions = False`**: adds `<exclude>`
  for *all* body pairs inside the robot. Over-restrictive.
- **MetaSim with `enabled_self_collisions = True`**: sets
  `filterparent = disable` globally. Over-permissive.

Neither MetaSim option matches mujoco's default. A follow-up MetaSim
PR will add a `self_collisions = "mujoco_default"` sentinel.

## How to reproduce

```bash
conda activate roboverse
cd "$ROBOVERSE"  # repo root

PYTHONPATH="$ROBOVERSE:$METASIM" \
  python -m tools.mjlab_integration.runner --no-render --n-steps 200          # loader-only

PYTHONPATH="$ROBOVERSE:$METASIM" \
  MUJOCO_GL=egl \
  python -m tools.mjlab_integration.full_runner --n-steps 200                 # full pipeline

PYTHONPATH="$ROBOVERSE:$METASIM" \
  MUJOCO_GL=egl \
  python -m tools.mjlab_integration.render_sweep --frame-stride 3             # side-by-side mp4

PYTHONPATH="$ROBOVERSE:$METASIM" \
  python -m tools.mjlab_integration.reward_sweep                              # reward parity

cd "$METASIM"
MUJOCO_GL=egl pytest metasim/test/test_mujoco_scene_and_attach.py -v          # regression tests
```

Artefacts land at `reports/mjlab_integration/`
(`summary_full.json`, `reward_summary.json`, `render_summary.json`, plus
`runs/<task>/qpos_compare_full.png` / `side_by_side_full.mp4` /
`reward_compare.png`).

## Files added

- **Harness** — `RoboVerse/tools/mjlab_integration/`
  - `inventory.py` — task catalog (12 entries)
  - `raw_rollout.py` — pure mujoco ground truth + `RolloutResult` (now
    carries `jnt_qposadr`/`jnt_dofadr`/`jnt_type`)
  - `metasim_rollout.py` — loader-only and full-pipeline drivers
  - `full_runner.py` / `runner.py` — sweep orchestration
  - `render_sweep.py` — side-by-side mp4 (`side_by_side_video` in
    `render.py`)
  - `reward_sweep.py` + `rewards.py` — port of mjlab's reward kernels
  - `diff.py` / `plot.py` — l2 / max-abs / bitwise metrics + overlay
    plots
  - `diagnose_*.py` — step-by-step root-cause scripts kept for
    posterity

- **Task wrappers** — `RoboVerse/roboverse_pack/tasks/mjlab/`
  - `cartpole.py` (Balance / Swingup, scene-mode)
  - `floating_base.py` (G1 / Go1 velocity flat+rough, G1 tracking ×2)
  - `lift_cube.py` (Yam base + RGB/depth/seg variants)
  - `_locator.py` — `mjlab_asset(relpath)` shared by all wrappers
