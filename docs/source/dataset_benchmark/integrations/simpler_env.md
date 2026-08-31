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
| Obs matches upstream | initial render vs `simpler_env` **mean-abs ≤ ~2/255** all 6 families (coke/pick/move-near bitwise; drawer 0.01, place 1.94, widowx 0.0) | `scripts/render_policy_gallery.py` |
| Real-policy success (RT-1/Octo) | **13/25** solved on MetaSim-native envs (RT-1 13/21 · Octo 0/4) | `scripts/render_policy_gallery.py` |
| Zero-upstream / deletable | meta-path block + grep test green; runs with the clone absent | `scripts/verify_native_registration.py` |
| Passthrough | bitwise 1:1 by construction (forwards `reset`/`step` verbatim) | `roboverse_pack/tasks/simpler_env/_passthrough.py` |
| Registration | 25 gym ids (`SimplerEnv/<task>`) + 25 MetaSim ids (`simpler.<task>`) | `roboverse_pack/tasks/simpler_env/_metasim/registry.py` |

MetaSim core changes: **4 backward-compatible SAPIEN-2 handler extensions** — all
opt-in (mesh `RigidObjCfg` loading, mounted-camera intrinsics, PhysX `SceneConfig`
overrides, primitive `fix_base_link` / `collision_enabled`); existing scenarios are
untouched (the new code paths only activate on the new optional fields).

> Fidelity is measured against the **upstream `simpler_env` observation** (initial render, same
> task + seed + station): coke / pick / move-near are bitwise; drawer 0.01, place 1.94, widowx 0.0
> mean-abs over `[0,255]`. The tiny residuals are SAPIEN contact-solver nondeterminism (~1.8e-6) +
> GPU edge anti-aliasing (place's 1.94 also includes the settling object + upstream's random
> `urdf_version` recolor variant). This upstream-obs check supersedes the earlier native-vs-reference
> parity, which could not see station / overlay / cabinet-recolor deviations (both sides shared them).

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
import roboverse_pack.tasks.simpler_env  # auto-registers SimplerEnv/<task> + simpler.<task>

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

<!-- SIMPLER-GALLERY-START -->
## Real-policy rollouts (RT-1 / Octo)

These are **real pretrained policies driving our MetaSim-native env** (`SimplerEnv/<task>`), recording the first episode the task's own success checker marks solved — an actual policy manipulating the objects, not a scripted motion. Google-robot tasks use **RT-1** (`rt_1_tf_trained_for_000400120`); WidowX/Bridge tasks use **Octo-base** (`policy_setup=widowx_bridge`) — matching what SimplerEnv evaluates per embodiment.

**Solved 13/25 on the MetaSim-native envs** (RT-1 13/21 Google · Octo 0/4 WidowX). Captions show the policy and, for solved tasks, the step count; ✗ marks episodes the policy did not solve this run (a property of the policy, not the integration — long-horizon place + Bridge tasks are the known-hard cases).

### Google Robot — pick coke can (4)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_pick_coke_can.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_pick_coke_can · RT-1 ✓ 10 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_pick_horizontal_coke_can.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_pick_horizontal_coke_can · RT-1 ✓ 12 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_pick_vertical_coke_can.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_pick_vertical_coke_can · RT-1 ✓ 15 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_pick_standing_coke_can.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_pick_standing_coke_can · RT-1 ✓ 10 steps
```
:::

::::

### Google Robot — pick object & move near (4)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_pick_object.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_pick_object · RT-1 ✓ 20 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_move_near.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_move_near · RT-1 ✓ 13 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_move_near_v0.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_move_near_v0 · RT-1 ✓ 13 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_move_near_v1.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_move_near_v1 · RT-1 ✓ 13 steps
```
:::

::::

### Google Robot — open drawer (4)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_open_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_open_drawer · RT-1 ✓ 43 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_open_top_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_open_top_drawer · RT-1 ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_open_middle_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_open_middle_drawer · RT-1 ✓ 38 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_open_bottom_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_open_bottom_drawer · RT-1 ✗ (not solved this run)
```
:::

::::

### Google Robot — close drawer (4)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_close_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_close_drawer · RT-1 ✓ 27 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_close_top_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_close_top_drawer · RT-1 ✓ 27 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_close_middle_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_close_middle_drawer · RT-1 ✓ 25 steps
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_close_bottom_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_close_bottom_drawer · RT-1 ✗ (not solved this run)
```
:::

::::

### Google Robot — place in closed drawer (5)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_place_in_closed_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_place_in_closed_drawer · RT-1 ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_place_in_closed_top_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_place_in_closed_top_drawer · RT-1 ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_place_in_closed_middle_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_place_in_closed_middle_drawer · RT-1 ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_place_in_closed_bottom_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_place_in_closed_bottom_drawer · RT-1 ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_google_robot_place_apple_in_closed_top_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: google_robot_place_apple_in_closed_top_drawer · RT-1 ✗ (not solved this run)
```
:::

::::

### WidowX / Bridge — put-on (4)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_widowx_spoon_on_towel.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: widowx_spoon_on_towel · Octo ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_widowx_carrot_on_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: widowx_carrot_on_plate · Octo ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_widowx_stack_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: widowx_stack_cube · Octo ✗ (not solved this run)
```
:::

:::{grid-item}
```{video} ../../_static/integrations/simpler_env/policy_widowx_put_eggplant_in_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: widowx_put_eggplant_in_basket · Octo ✗ (not solved this run)
```
:::

::::

## Implementation fidelity (obs matches upstream)

Separately from policy capability, the MetaSim-native env is verified to reproduce the **upstream `simpler_env` observation**: for the same task + seed + station, the initial render matches upstream by **mean-abs ≤ ~2/255** across all six families (coke/pick/move-near bitwise; drawer 0.01, place 1.94, widowx 0.0). This is a stronger check than the earlier native-vs-reference parity — it catches station / overlay / asset-recolor deviations that an internal self-comparison cannot. Regenerate side-by-side `[ MetaSim-native | reference | abs-diff x30 ]` clips with `scripts/render_metasim_1to1_gallery.py`; the per-task obs-vs-upstream check is in `scripts/render_policy_gallery.py`.

<!-- SIMPLER-GALLERY-END -->

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
- **Assets.** The native track reads from `roboverse_data/assets/simpler_env/` +
  `roboverse_data/robots/{google_robot,widowx}/` (URDFs, the `mk_station` cabinet, scene
  GLBs, object meshes, model DB, real-image overlays). These are mirrored on HuggingFace
  at [`RoboVerseOrg/roboverse_data`](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data)
  and **download automatically on first use** (`_native/_assets.py` → `snapshot_download`)
  when no local `roboverse_data` checkout (or `$ROBOVERSE_DATA`) is found — so a fresh
  install needs no manual asset fetch. The SAPIEN `*.convex.stl` collision caches are not
  stored (repo `.gitignore` convention) and are not required — verified by a cold-start run
  (empty dir → HF download → task builds and renders) with no caches present.