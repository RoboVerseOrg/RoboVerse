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
```

## Side-by-side: native LIBERO vs MetaSim

Beyond "passthrough == native by construction", we load each task's own combined
MJCF (Franka + objects + arena + camera) into **MetaSim's MuJoCo handler** and
reproduce the demo — proving the MetaSim backend simulates the LIBERO scene 1:1.

In each frame, **left = native LIBERO `agentview_rgb`, right = MetaSim render**.

**Stage 1 — kinematics** (per-frame `set_states`, geometry/pose 1:1):

![Stage 1 side-by-side: native LIBERO (left) vs MetaSim (right)](../../_static/integrations/libero/sidebyside_kinematic_frame.png)

<video controls width="640" preload="metadata" poster="../../_static/integrations/libero/sidebyside_kinematic_frame.png">
  <source src="../../_static/integrations/libero/sidebyside_kinematic_spatial_task0.mp4" type="video/mp4">
</video>

▶ [play / download the kinematic side-by-side video](../../_static/integrations/libero/sidebyside_kinematic_spatial_task0.mp4)

**Stage 2 — dynamics** (MetaSim's engine steps under the captured ctrl):

![Stage 2 side-by-side: native LIBERO (left) vs MetaSim engine (right)](../../_static/integrations/libero/sidebyside_dynamics_frame.png)

<video controls width="640" preload="metadata" poster="../../_static/integrations/libero/sidebyside_dynamics_frame.png">
  <source src="../../_static/integrations/libero/sidebyside_dynamics_spatial_task0.mp4" type="video/mp4">
</video>

▶ [play / download the dynamics side-by-side video](../../_static/integrations/libero/sidebyside_dynamics_spatial_task0.mp4) Per-frame
`max|qpos − recorded| = 0.0` (exact); the MetaSim engine step matches reference
MuJoCo to `max|Δ| = 1.6e-4` (float accumulation). Residual pixel MAE ≈ 2.5–5/255
is renderer config (lighting / anti-aliasing), not physics.

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
