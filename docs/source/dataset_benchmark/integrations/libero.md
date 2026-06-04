# LIBERO + LIBERO-plus → RoboVerse 1:1 integration

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (*Lifelong Robot
Learning*, Liu et al. 2023) is a 130-task manipulation benchmark of robosuite /
MuJoCo environments across five suites (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`, `libero_90`).
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) (*In-depth Robustness
Analysis of VLA Models*, 2025) is a drop-in superset that expands every suite
with thousands of **perturbation** variants — **10,120 tasks** across seven
dimensions (object layout, camera viewpoint, robot init state, language rewrite,
lighting, background texture, sensor noise).

RoboVerse integrates both as **passthrough** task packs: registration is lazy
and import-safe, and the env is the *native* LIBERO(-plus) `OffScreenRenderEnv`,
so observations / dynamics / 128×128 camera bytes are **bitwise-identical** to
upstream. On top of that we verify a **pure-MetaSim native reproduction** (load
the scene into MetaSim's own MuJoCo handler) and a **bitwise OSC_POSE controller
port** so LIBERO EE-delta policies can run *inside* MetaSim.

Beyond passthrough (which still imports the upstream library), the
`roboverse_pack.tasks.libero_native` pack runs **all 130 base LIBERO tasks with
zero `libero` / `robosuite` import** — the scene (a portable MJCF + deduplicated
shared assets vendored to HuggingFace `RoboVerseOrg/roboverse_data`), the BDDL
success checker, the OSC_POSE controller and the renderer all run on MuJoCo +
NumPy alone. A fresh machine with the upstream packages **deleted** resolves each
task's MJCF and meshes from HF on demand and runs it; verified bitwise on all 130
(see [Native MetaSim tasks](#native-metasim-tasks-run-and-delete-libero)).

## Status

| Capability | Result | Where |
|---|---|---|
| LIBERO passthrough | **130 / 130** tasks, obs+step bitwise (Δ=0) | `roboverse_pack/tasks/libero/` |
| LIBERO-plus passthrough | **10,120 / 10,120** tasks, state/reward/done bitwise (Δ=0) | `roboverse_pack/tasks/libero_plus/` |
| demo-replay parity | passthrough == native, Δ=0 (380 demos) | `scripts/parity_liberoplus_passthrough.py` |
| asset audit | 7/7 perturbation dims genuinely applied, **0 silent fallback** | `scripts/audit_liberoplus_assets.py` |
| MetaSim MuJoCo migration | 6/6 dims: state-set Δ=0, **engine Δ=0** | `scripts/migrate_liberoplus_metasim.py` |
| OSC_POSE port | **bitwise** — per-state joint-torque Δ = 5.55e-15 N·m | `scripts/osc/` |
| BC policy (closed-loop) | clean 100 % / light 0 % / camera 50 % / noise 75 %; passthrough==native Δ=0 | `scripts/policy/` |
| **Native tasks (no libero/robosuite)** | **130 / 130** base tasks: ported BDDL checker bitwise (0 mismatch vs `env._check_success`), render bitwise (state-replay) | `roboverse_pack/tasks/libero_native/` |
| **Vendored assets (library-deletable)** | 130 portable MJCF + **193 deduped assets / 265 MB** on HF; fresh-machine model fields **130/130** exact (max\|Δ\|=0), checker **130/130** | `RoboVerseOrg/roboverse_data` `libero/` |

MetaSim core changes: **0** (the passthrough reuses the unmodified upstream env;
the MetaSim handler loads the scene MJCF verbatim).

## Environment setup

LIBERO pins `numpy<1.24` / `robosuite==1.4.0` / `bddl==1.0.1` / `mujoco==3.2.3`,
which conflict with the default RoboVerse env — so install each in a **dedicated**
conda env. The passthrough is a safe no-op in any env where LIBERO is not
importable (registration registers nothing; the factory raises a clear error).

```bash
# base LIBERO (130 tasks)
conda create -n libero1to1 python=3.8 -y && conda activate libero1to1
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && pip install -e LIBERO

# LIBERO-plus (10,120 tasks) — a separate env; it is a drop-in replacement of LIBERO
conda create -n liberoplus python=3.8 -y && conda activate liberoplus
git clone https://github.com/sylvestf/LIBERO-plus.git && pip install -e LIBERO-plus
# + extract the LIBERO-plus asset bundle (6 GB) into libero/libero/assets
```

The LIBERO-plus passthrough **self-bootstraps** its config: it writes
`~/.libero_plus/config.yaml` pointing at the installed LIBERO-plus package's
bddl / asset / init roots and sets `LIBERO_CONFIG_PATH` (never clobbering a
user-set value or the base `~/.libero`). No manual config editing needed.

## Usage

```python
import roboverse_pack.tasks.libero          # auto-registers Libero/<suite>__<task>
import roboverse_pack.tasks.libero_plus     # auto-registers LiberoPlus/<suite>__<task>
from roboverse_pack.tasks.libero_plus import make_liberoplus_env

# build any of the 10,120 perturbation tasks by (suite, task index)
env = make_liberoplus_env("libero_object", 7, seed=0)
obs, reward, done, info = env.step([0.0] * 7)   # native legacy-gym 4-tuple (kept for fidelity)
```

## Native MetaSim tasks — run (and delete) LIBERO

The passthrough is bitwise but still *imports* `libero` / `robosuite`. The
`roboverse_pack.tasks.libero_native` pack removes that dependency: a LIBERO task's
**scene, success checker, controller and renderer** all run on `mujoco` + `numpy`.

```python
from roboverse_pack.tasks.libero_native.native_task import LiberoNativeTask

# Resolves the portable MJCF + every mesh/texture from a local roboverse_data
# clone, or downloads them from HF on demand. No `libero` / `robosuite` import.
task = LiberoNativeTask.from_vendored("libero_object",
                                      "pick_up_the_alphabet_soup_and_place_it_in_the_basket")
task.reset(state)            # set a flat [time, qpos, qvel] state
task.success()               # ported BDDL goal checker (bitwise vs env._check_success)
task.render()                # agentview RGB (geom-group matched to robosuite)
task.step(action, grip)      # one OSC_POSE policy step (inlined controller)
```

What was ported, and how it stays 1:1:

- **BDDL success checker** (`checker.py`) — every goal predicate reduced to a
  MuJoCo-state test: `In`=oriented-box containment, `On`(object)=`check_ontop`
  with exact `contact_geoms`, `On`(region-site)=`SiteObject.under` + parent
  contact, `Open/Close/TurnOn/TurnOff`=articulated-joint thresholds (any/all).
  Thresholds are recovered at export by probing the live object's own method, so
  they are exactly faithful. **0 mismatch vs `env._check_success` on all 130.**
- **OSC_POSE** (`osc.py`) — robosuite's operational-space control math inlined as
  pure NumPy; per-step joint torque Δ = 1.8e-14 vs robosuite.
- **Scene** — each task ships as a portable MJCF whose `file=` paths are vendored
  asset relpaths; the loader re-applies robosuite's runtime fixture placement
  (`body_pos/quat`, captured with `seed=0`) so the compiled model is
  **field-for-field identical** (`max|Δ| = 0`).
- **Render** — `mujoco.Renderer`'s geom-group mask is matched to robosuite's
  offscreen `vopt` so a state-replay render is pixel-identical.

### Vendored assets on HuggingFace (library-deletable)

`scripts/native/migrate_libero_assets.py` exports all 130 tasks to a deduplicated
tree (193 unique files / 265 MB — the Franka/arena meshes are shared by every
task) published to the
[`RoboVerseOrg/roboverse_data`](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/main/libero)
dataset under `libero/`:

```
libero/
  manifest.json
  assets/robosuite/<...>           # shared Franka / gripper / arena meshes
  assets/libero/<...>              # per-scene objects, textures, meshes
  tasks/<suite>/<task>.xml         # portable MJCF (file= are asset relpaths)
  tasks/<suite>/<task>.goal.json   # resolved BDDL goal + OSC cfg + body_pos/quat
```

`_locator.py` resolves these from a local `roboverse_data` clone or HuggingFace on
demand — so on a machine where `libero` and `robosuite` are **uninstalled**, the
native task still loads and runs. Verified across all 130: model fields exact
**130/130** (`max|Δ| = 0`), ported checker **130/130** (0 mismatch). Use
`ROBOVERSE_DATA_DIR` to point at a shared cache.

## Reproduce — run commands

All commands assume `MUJOCO_GL=egl` (headless) and the dedicated env. The
LIBERO-plus runs additionally take `LIBERO_CONFIG_PATH=$HOME/.libero_plus`.

```bash
# --- passthrough bitwise tests (per env) ---
MUJOCO_GL=egl python -m pytest tests/test_libero_passthrough.py -v        # in libero1to1
MUJOCO_GL=egl python -m pytest tests/test_liberoplus_passthrough.py -v    # in liberoplus

# --- LIBERO-plus passthrough == native, all 7 perturbation dimensions ---
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  python -m scripts.parity_liberoplus_passthrough --per-dim 1 --steps 8

# --- asset audit: prove every perturbation actually changes the render ---
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  python -m scripts.audit_liberoplus_assets --suites libero_spatial libero_object libero_goal libero_10

# --- migrate the scene into MetaSim's own MuJoCo backend (state + engine 1:1) ---
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  python -m scripts.migrate_liberoplus_metasim --suite libero_spatial

# --- OSC_POSE controller bitwise parity vs robosuite (for in-MetaSim control) ---
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  python -m scripts.osc.parity_osc_vs_robosuite --steps 130 --precontact 40

# --- closed-loop BC policy: train (GPU env) then eval through the passthrough (CPU) ---
python -m scripts.policy.train_bc_libero \
  --demos third_party/libero_datasets/libero_object/<task>_demo.hdf5 --epochs 100 \
  --out scripts/policy/ckpt/bc.pt
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  python -m scripts.policy.eval_bc_liberoplus --ckpt scripts/policy/ckpt/bc.pt \
  --base <task> --suite libero_object --episodes 8

# --- native tasks (no libero/robosuite): vendor assets -> verify -> side-by-side ---
ROBOVERSE_DATA_DIR=/path/to/roboverse_data MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
  python -m scripts.native.migrate_libero_assets      # 130 tasks -> roboverse_data/libero
ROBOVERSE_DATA_DIR=/path/to/roboverse_data MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
  python -m scripts.native.verify_vendored            # all 130: model fields + checker 1:1
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
  python -m scripts.native.sweep_all_native           # per-task side-by-side + checker + OSC
# unit tests (assert the native task imports/runs with NO libero/robosuite):
python -m pytest tests/test_libero_native.py -v       # in liberoplus
```

## Side-by-side: native LIBERO-plus vs MetaSim

We load each task's own combined MJCF (Franka + objects + arena + camera) into
**MetaSim's MuJoCo handler** and re-render every state of a demo rollout —
proving the MetaSim backend reproduces the *perturbed* LIBERO-plus scene 1:1.
In each clip, **left = native LIBERO-plus `agentview`, right = MetaSim render**;
both are upright and the demo arm motion is identical.

Generated with `scripts/gen_libero_sidebyside.py` (current code, real demo
motion). The animated GIFs below play inline everywhere; the full-quality mp4 is
linked next to each.

**Background-texture perturbation** (`libero_object … _table_5`) — MetaSim
reproduces the swapped scene texture; per-frame native-vs-MetaSim pixel
**MAE = 0.24 / 255** (sub-pixel — renderer config only; state is exact).
[full-quality mp4](../../_static/integrations/libero/sb_liberoplus_texture.mp4)

![texture side-by-side: native LIBERO-plus (left) vs MetaSim (right)](../../_static/integrations/libero/sb_liberoplus_texture.gif)

**Camera-viewpoint perturbation** (`libero_object … _view_…`) — MetaSim
reproduces the shifted camera; per-frame **MAE = 0.16 / 255**.
[full-quality mp4](../../_static/integrations/libero/sb_liberoplus_camera.mp4)

![camera side-by-side: native LIBERO-plus (left) vs MetaSim (right)](../../_static/integrations/libero/sb_liberoplus_camera.gif)

Per-frame `max|qpos − recorded| = 0.0` (state exact); the MetaSim engine step
matches reference MuJoCo to `max|Δ| = 1.6e-4` (`migrate_liberoplus_metasim`).

### Demo-replay parity (all 5 suites)

![replay parity](../../_static/integrations/libero/chart_replay_parity.png)
![per-suite](../../_static/integrations/libero/chart_suites.png)

| suite | tasks | demos | pt-vs-native max\|Δ\| | success (pt / native) |
|---|---|---|---|---|
| libero_spatial | 10 | 50 | **0** | 41/50 · 41/50 |
| libero_object | 10 | 50 | **0** | 43/50 · 43/50 |
| libero_goal | 10 | 50 | **0** | 40/50 · 40/50 |
| libero_10 | 10 | 50 | **0** | 37/50 · 37/50 |
| libero_90 | 90 | 180 | **0** | 154/180 · 154/180 |

The load-bearing number is **passthrough-vs-native = 0** (bitwise across all
380 demos). The ~65 non-successful replays diverge *identically* on both backends
(LIBERO's intrinsic open-loop OSC_POSE replay non-determinism, not a RoboVerse gap).

### OSC_POSE controller parity

`scripts/osc/parity_osc_vs_robosuite.py` reports:

```
(A) per-step torque parity at every real state incl. contact (130 states):
      joint-torque max|Δ| = 5.55e-15 N·m     <- bitwise control-law faithfulness
(B) pre-contact open-loop rollout on MJB-exact model (40 steps):
      arm-qpos max|Δ|     = 1.25e-05 rad
```

The ported `MujocoOSCPose` reuses robosuite's `control_utils` / `transform_utils`
math verbatim, reimplementing only the sim-data access (`mj_fullM`, `mj_jacSite`,
EE pose/vel, `qfrc_bias`) on MetaSim's dm_control `Physics`. It emits joint
torques into the existing `dof_torque` path → additive, opt-in, zero blast radius.

### Real policy robustness (closed-loop, via the passthrough)

An ACT-style BC policy trained on **clean** `libero_object` demos, evaluated
closed-loop on LIBERO-plus perturbations — exactly the robustness signal the
benchmark measures:

| variant | clean | light | sensor-noise | camera |
|---|---|---|---|---|
| success | **8/8 = 1.00** | 0/8 | 6/8 | 4/8 |

`passthrough == native` under the policy: full-rollout state max\|Δ\| = **0**.

## Details and gotchas

- **Config bootstrap.** LIBERO-plus reads bddl/asset/init roots from
  `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero`). Because base LIBERO and
  LIBERO-plus share the package name, the passthrough self-writes a dedicated
  `~/.libero_plus` config from the *installed* LIBERO-plus package so the 10k
  perturbation tasks resolve, without touching the base config.
- **Perturbation encoding.** File-based dims (background texture `_table_N`,
  lighting `_light_N`, layout `_add_N`) are real BDDL files; parametric dims
  (camera `_view_…`, robot init `_initstate_…`, language `_language_N`, noise
  `_noise_N`) are synthetic descriptors that the env wrapper parses to apply the
  perturbation — so the passthrough must **not** `os.path.isfile` them; it uses
  the benchmark's authoritative `get_task_bddl_file_path`.
- **Global EGL render context.** robosuite/MuJoCo share a process-global EGL
  context — two `OffScreenRenderEnv` alive at once clobber each other's GL
  textures, so a texture-only perturbation reads as "0 effect". **Render one env
  at a time** (build → render → close). This is required for any side-by-side or
  batched rendering, and is why the asset audit / parity harnesses are sequential.
- **Sensor-noise is upstream-stochastic.** Noise (motion/gaussian/fog/glass blur)
  is added to `agentview_image` *after* render with an unseeded `np.random` — the
  physics/state/reward/done are unaffected (bitwise); only the corrupted image
  differs across interleaved runs. It reproduces when each env runs in isolation.
- **Lossless model transfer.** `env.sim.model.get_xml()` → reload is lossy on
  mesh inertias; the binary `mujoco.mj_saveModel` / `from_binary_path` (MJB) path
  is lossless (inertials Δ=0) — use MJB when an exact MetaSim model is required.
- **OSC sticky orientation goal.** robosuite only updates `goal_ori` when the
  orientation delta is non-zero (a *sticky* goal); the port must match this or it
  drifts under near-zero wrist deltas. With the fix the control law is bitwise.
- **GPU split (sm_120).** The LIBERO sim env pins py3.8 / torch 2.4.1, which can't
  use an sm_120 GPU (RTX 5090). Train policies in an sm_120-capable env (torch
  ≥2.7 / cu128) and run closed-loop eval with CPU inference in the sim env, or via
  a small sim↔policy socket bridge (`scripts/policy/bridge_*`).

## Scope notes (honest)

- The passthrough is **MuJoCo-only by design**: LIBERO is a robosuite/MuJoCo
  benchmark; porting its MJCF to SAPIEN/Newton needs asset re-authoring + a
  different contact model = an approximate cross-sim port, not 1:1.
- The BC policy is a compact single-task demonstration (pipeline + robustness +
  passthrough==native), not a SOTA reproduction. The official
  `Sylvest/openvla-7b-oft-finetuned-libero-plus` checkpoint runs through the same
  bridge for absolute benchmark numbers.
- The native pack's **success checking and physics are bitwise** (checker 0
  mismatch, model fields `max|Δ| = 0`). Its **render** is recompiled from the
  vendored MJCF, so meshes are re-triangulated and shading differs by ~2–5 / 255
  vs robosuite (visually identical); the export path's compiled `model.mjb`
  reproduces the agentview pixel-exactly when that is needed.
- The native pack covers the **130 base tasks**. The 10,120 LIBERO-plus
  perturbations remain passthrough-only (their value is the upstream
  perturbation pipeline; vendoring all of them is future work).
