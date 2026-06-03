# SimplerEnv → RoboVerse 1:1 integration

[SimplerEnv](https://github.com/simpler-env/SimplerEnv) (*Evaluating Real-World
Robot Manipulation Policies in Simulation*, Li et al. 2024) is the standard
real-to-sim evaluation suite for generalist manipulation policies (RT-1, RT-1-X,
RT-2-X, Octo, OpenVLA …). It ships **25 tasks** built on
[ManiSkill2-real2sim](https://github.com/simpler-env/ManiSkill2_real2sim) +
SAPIEN across two embodiments — **Google Robot** (21 tasks: pick coke can ×4,
pick object, move near ×3, open/close drawer ×8, place-in-drawer ×5) and
**WidowX / Bridge** (4 tasks: spoon-on-towel, carrot-on-plate, stack cube,
eggplant-in-basket) — with a *real-image greenscreen overlay* so the rendered
observation matches the real-robot eval distribution.

RoboVerse integrates SimplerEnv on **two tracks**:

1. **MetaSim-native** (primary) — every one of the 25 tasks is rebuilt **entirely
   through the MetaSim API**: each asset (robot, articulated cabinet, mesh / convex
   objects, ground, mounted cameras) is declared in a `ScenarioCfg`, stepped
   through the SAPIEN 2 handler, wrapped in `BaseTaskEnv`, and registered with
   `@register_task`. The SimplerEnv control / grasp / overlay **logic** is vendored
   under `_native/` with **zero** import of the upstream `simpler_env` /
   `mani_skill2_real2sim` packages — verified by a meta-path block *and* a
   zero-import grep test, so **the upstream clone is deletable**.
2. **Passthrough** (optional) — a transparent `gymnasium.make` forward to
   `simpler_env.make` when the clone *is* installed; **bitwise 1:1 by construction**.

## Status

| Capability | Result | Where |
|---|---|---|
| MetaSim-native tasks | **25 / 25** built via `ScenarioCfg` + handler + `@register_task` | `roboverse_pack/tasks/simpler_env/_metasim/` |
| Native vs upstream parity | **22 / 25 bitwise (Δ=0)**, 3 WidowX sub-pixel (worst **0.0018 / 255**), success **25 / 25** | `scripts/spike_metasim_full_parity.py` |
| Zero-upstream / deletable | meta-path block + grep test green; runs with the clone absent | `scripts/verify_native_registration.py` |
| Passthrough | bitwise 1:1 by construction (forwards `reset`/`step` verbatim) | `roboverse_pack/tasks/simpler_env/_passthrough.py` |
| Registration | 25 gym ids (`SimplerEnv/<task>`) + 25 MetaSim ids (`simpler.<task>`) | `roboverse_pack/tasks/simpler_env/_metasim/registry.py` |

MetaSim core changes: **4 backward-compatible SAPIEN-2 handler extensions** — all
opt-in (mesh `RigidObjCfg` loading, mounted-camera intrinsics, PhysX `SceneConfig`
overrides, primitive `fix_base_link` / `collision_enabled`); existing scenarios are
untouched (the new code paths only activate on the new optional fields).

> The 3 non-zero WidowX deltas (spoon 0.0018, carrot 0.0003, eggplant 0.0005, all
> in mean-abs over `[0,255]`) are sub-pixel and come from SAPIEN's contact-solver
> nondeterminism (~1.8e-6) plus GPU edge anti-aliasing — literal bitwise-to-upstream
> is not attainable on these, the same bar we report for the mjlab and menagerie
> integrations. State, reward, termination and `info["success"]` match exactly.

## Environment setup

SimplerEnv pins `SAPIEN==2.2.2`, `numpy==1.24.4`, `mani_skill2_real2sim==0.5.3`,
which conflict with the default RoboVerse env — install in a **dedicated** conda
env. The MetaSim-native track needs only SAPIEN 2 + the migrated `roboverse_data`
assets (no upstream package); the passthrough track additionally needs the
upstream clone.

```bash
conda create -n simpler python=3.10 -y && conda activate simpler
pip install sapien==2.2.2 numpy==1.24.4
# native track: + the SimplerEnv assets under roboverse_data/assets/simpler_env/
# passthrough track (optional):
git clone https://github.com/simpler-env/SimplerEnv.git && pip install -e SimplerEnv
```

Verified on an **RTX 5090 (sm_120)** with the NVIDIA Vulkan ICD — SAPIEN 2
rendering works, no sm_120 wall.

## Usage

```python
import roboverse_pack.tasks.simpler_env          # auto-registers SimplerEnv/<task> + simpler.<task>

# (1) MetaSim-native via gym
import gymnasium as gym
env = gym.make("SimplerEnv/google_robot_pick_coke_can")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# (2) MetaSim-native via the MetaSim task registry
from metasim.task.registry import get_task_class
task = get_task_class("simpler.widowx_stack_cube")()

# (3) optional upstream passthrough (requires the SimplerEnv clone)
from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough
register_simpler_env_passthrough(prefix="SimplerEnvPassthrough/")
env = gym.make("SimplerEnvPassthrough/google_robot_pick_coke_can")
```

## Reproduce — run commands

All commands assume the dedicated `simpler` env and CPU JAX (`JAX_PLATFORMS=cpu`,
to keep the GPU for SAPIEN rendering only).

```bash
# --- native registration + 25/25 make/reset/step with the upstream clone DELETED ---
JAX_PLATFORMS=cpu python scripts/verify_native_registration.py

# --- exhaustive MetaSim-native vs upstream-equivalent parity over all 25 tasks ---
#     (per-task subprocess-isolated; writes /tmp/metasim_full_parity.json)
JAX_PLATFORMS=cpu python scripts/spike_metasim_full_parity.py

# --- 25 side-by-side 1:1 galleries [native | reference | diff x30] ---
JAX_PLATFORMS=cpu python scripts/render_metasim_1to1_gallery.py

# --- tests ---
python -m pytest tests/test_simpler_env_native.py -v        # registry(25) + zero-import + smoke
python -m pytest tests/test_simpler_env_passthrough.py -v   # upstream forward (needs the clone)
```

## Side-by-side: MetaSim-native vs reference (all 25 tasks)

For every task we run the **MetaSim-native** env (`ScenarioCfg` + handler) and the
verified **`_native` reference** (== upstream) from the *same seed* and the *same*
scripted reach-close-lift motion, in separate subprocesses (SAPIEN keeps a
process-global renderer). Each clip is **`[ MetaSim-native | reference | abs-diff ×30 ]`**
— the right panel is (near-)black, i.e. the two are pixel-identical.

Regenerate with `scripts/render_metasim_1to1_gallery.py`.

### Google Robot — pick coke can (4)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_pick_coke_can.mp4" type="video/mp4"></video>
<p><code>google_robot_pick_coke_can</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_pick_horizontal_coke_can.mp4" type="video/mp4"></video>
<p><code>google_robot_pick_horizontal_coke_can</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_pick_vertical_coke_can.mp4" type="video/mp4"></video>
<p><code>google_robot_pick_vertical_coke_can</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_pick_standing_coke_can.mp4" type="video/mp4"></video>
<p><code>google_robot_pick_standing_coke_can</code> · Δ=0.0</p></div>
</div>

### Google Robot — pick object & move near (4)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_pick_object.mp4" type="video/mp4"></video>
<p><code>google_robot_pick_object</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_move_near.mp4" type="video/mp4"></video>
<p><code>google_robot_move_near</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_move_near_v0.mp4" type="video/mp4"></video>
<p><code>google_robot_move_near_v0</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_move_near_v1.mp4" type="video/mp4"></video>
<p><code>google_robot_move_near_v1</code> · Δ=0.0</p></div>
</div>

### Google Robot — open drawer (4)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_open_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_open_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_open_top_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_open_top_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_open_middle_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_open_middle_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_open_bottom_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_open_bottom_drawer</code> · Δ=0.0</p></div>
</div>

### Google Robot — close drawer (4)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_close_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_close_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_close_top_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_close_top_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_close_middle_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_close_middle_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_close_bottom_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_close_bottom_drawer</code> · Δ=0.0</p></div>
</div>

### Google Robot — place in closed drawer (5)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_place_in_closed_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_place_in_closed_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_place_in_closed_top_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_place_in_closed_top_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_place_in_closed_middle_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_place_in_closed_middle_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_place_in_closed_bottom_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_place_in_closed_bottom_drawer</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/google_robot_place_apple_in_closed_top_drawer.mp4" type="video/mp4"></video>
<p><code>google_robot_place_apple_in_closed_top_drawer</code> · Δ=0.0</p></div>
</div>

### WidowX / Bridge — put-on (4)

<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/widowx_spoon_on_towel.mp4" type="video/mp4"></video>
<p><code>widowx_spoon_on_towel</code> · Δ=0.0018</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/widowx_carrot_on_plate.mp4" type="video/mp4"></video>
<p><code>widowx_carrot_on_plate</code> · Δ=0.0003</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/widowx_stack_cube.mp4" type="video/mp4"></video>
<p><code>widowx_stack_cube</code> · Δ=0.0</p></div>
<div style="flex:1 1 360px;max-width:520px;text-align:center">
<video width="100%" autoplay loop muted playsinline style="border-radius:4px"><source src="/roboverse/_static/integrations/simpler_env/widowx_put_eggplant_in_basket.mp4" type="video/mp4"></video>
<p><code>widowx_put_eggplant_in_basket</code> · Δ=0.0005</p></div>
</div>

## Design notes & honest caveats

- **Per-episode random objects.** Several families (pick-object, move-near,
  place-in-drawer) sample objects per episode. To keep the active physics solve
  bitwise, the full candidate set is declared once in `ScenarioCfg`; each episode
  *activates* its subset and *parks* the rest at `HIDDEN_POS=(0,0,-100)` with motion
  locked and collision groups disabled, so inactive actors cannot perturb the solve.
- **One env per process.** SAPIEN keeps a process-global renderer/engine, so (exactly
  as upstream) only one env may be alive per process. The parity harness isolates each
  task in its own subprocess; this is a property of the underlying simulator, not the
  integration.
- **Open-loop forwarding, not policy success.** We verify the rendering / state /
  reward / success *contract* (open-loop, scripted or seeded actions). Closed-loop
  policy success is a property of the policy and is out of scope for the integration
  claim.
- **Assets.** The native track reads from `roboverse_data/assets/simpler_env/`
  (Google Robot + WidowX URDFs, the `mk_station` cabinet, scene GLBs, model DB, and
  the real-image overlays). These must be present locally; an HF mirror upload is
  pending.
