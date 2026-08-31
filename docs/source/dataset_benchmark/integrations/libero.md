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
`roboverse_pack.tasks.libero.native_libero` pack runs **all 130 base LIBERO tasks
with zero `libero` / `robosuite` import at runtime** — each task loads its own
compiled scene MJCF onto MetaSim's MuJoCo handler, with a robosuite-free OSC
controller and a native BDDL success checker. Mesh assets are resolved from the
deduplicated tree on HuggingFace `RoboVerseOrg/roboverse_data`, so a machine with the
upstream packages **uninstalled** still loads and runs them; the checker is verified
bitwise (0 mismatch vs `env._check_success`) on all 130
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
| **Native tasks (no libero/robosuite)** | **130 / 130** base tasks: ported BDDL checker bitwise (0 mismatch vs `env._check_success` over state-replay); `BaseTaskEnv` on MetaSim's handler, discoverable via `get_task_class("libero_native.<task>")` | `roboverse_pack/tasks/libero/native_libero.py` |
| **Vendored assets (library-deletable)** | deduped meshes (~265 MB, shared Franka/arena) on HF; native task loads with `libero`/`robosuite` uninstalled | `RoboVerseOrg/roboverse_data` `libero/assets/` |

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
import roboverse_pack.tasks.libero  # auto-registers Libero/<suite>__<task>
import roboverse_pack.tasks.libero_plus  # auto-registers LiberoPlus/<suite>__<task>
from roboverse_pack.tasks.libero_plus import make_liberoplus_env

# build any of the 10,120 perturbation tasks by (suite, task index)
env = make_liberoplus_env("libero_object", 7, seed=0)
obs, reward, done, info = env.step([0.0] * 7)  # native legacy-gym 4-tuple (kept for fidelity)
```

## Native MetaSim tasks — run (and delete) LIBERO

The passthrough is bitwise but still *imports* `libero` / `robosuite`. The
`roboverse_pack.tasks.libero.native_libero` pack removes that dependency: each task
loads its **own** compiled scene MJCF onto MetaSim's groundless `MujocoHandler`,
drives it with a robosuite-free OSC controller, and judges success natively from the
BDDL goal — with **no `import libero` / `import robosuite` at runtime**. All 130 base
tasks are registered, so they're discoverable through the task registry like any
other RoboVerse task and run with `libero`/`robosuite` uninstalled:

```python
from metasim.task.registry import get_task_class

Task = get_task_class("libero_native.10__KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it")
env = Task()  # loads the vendored scene; no libero
obs, info = env.reset()
obs, reward, terminated, time_out, info = env.step(action)  # 7-D OSC delta + gripper
```

Each task is a first-class `BaseTaskEnv` (`NativeLiberoEnv`); a static bundle under
`roboverse_pack/tasks/libero/native_bundles/<task>/` holds the libero-free inputs —
`model.xml` (the demo's embedded MJCF), `goal.json` (parsed BDDL `:goal`), `init.npz`
(a demo's initial qpos/qvel). What makes it 1:1:

- **Scene / physics** — the demo's own robosuite-compiled MJCF runs on MetaSim's
  dm_control handler (the same MuJoCo substrate LIBERO uses); engine step is
  bit-for-bit identical and state-replay reproduces the recorded success.
- **BDDL success checker** (`_native_util.check_bddl_success`) — every goal predicate
  reduced to a MuJoCo-state test: `In` = `SiteObject.in_box` (with LIBERO's
  `lb[2]-=0.01` floor slack); `On`(object) = above + contact + xy-aligned;
  `On`(region-site) = `SiteObject.under` **and** contact with the region's parent
  *object* (climbing to the object root; arena-table regions are parentless →
  containment only); `Open`/`Close`/`TurnOn`/`TurnOff` use each object's own
  threshold ranges (open and close are **not** complements — there is a dead zone —
  and close/turn-off require **all** joints). **Verified 0 mismatch vs
  `env._check_success` over demo state-replay on all 130 base tasks.**
- **Control** — `roboverse_pack.tasks.robosuite._osc.NativeOSC`, robosuite's OSC_POSE
  math reimplemented in NumPy (arm + parallel-jaw gripper).

### Vendored assets (library-deletable)

A bundle's `model.xml` references mesh/texture files by path; those are resolved from
the deduplicated tree published to the
[`RoboVerseOrg/roboverse_data`](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/main/libero)
dataset under `libero/assets/{robosuite,libero}/` (the Franka/arena meshes are shared
by every task — ~265 MB total, not per-task copies). `remap_libero_model` rebases the
roots onto that tree and downloads any missing file from HuggingFace on demand, so on
a machine with `libero`/`robosuite` **uninstalled** the task still loads. Build the
tree with `tools/libero_integration/vendor_shared_assets.py`; override the roots with
`ROBOVERSE_DATA_DIR`, or `LIBERO_ASSETS` / `ROBOSUITE_ASSETS` to point at local
installs.

### 1:1 side-by-side — all 130 base tasks

Every base LIBERO task, rendered by state-replaying the same demo states into both
native LIBERO (robosuite, left) and the MetaSim-native task (`NativeLiberoEnv`, right).
The BDDL success checker matches `env._check_success` **bitwise on all 130**
(0 mismatch over the full replay); the side-by-side render is visually aligned
(mean MAE 1.85/255 — re-triangulated meshes + sub-pixel shading). Expand a suite to view.

:::::{dropdown} OBJECT · 10 tasks · checker 1:1
:open:

::::{grid} 2
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the alphabet soup and place it in the basket · MAE 0.183
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_bbq_sauce_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the bbq sauce and place it in the basket · MAE 0.179
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_butter_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the butter and place it in the basket · MAE 0.187
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_chocolate_pudding_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the chocolate pudding and place it in the basket · MAE 0.187
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_cream_cheese_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the cream cheese and place it in the basket · MAE 0.185
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_ketchup_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the ketchup and place it in the basket · MAE 0.192
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_milk_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the milk and place it in the basket · MAE 0.177
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_orange_juice_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the orange juice and place it in the basket · MAE 0.178
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_salad_dressing_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the salad dressing and place it in the basket · MAE 0.187
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_object__pick_up_the_tomato_sauce_and_place_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the tomato sauce and place it in the basket · MAE 0.181
```
:::
::::
:::::

:::::{dropdown} GOAL · 10 tasks · checker 1:1
:open:

::::{grid} 2
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__open_the_middle_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: open the middle drawer of the cabinet · MAE 3.239
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__open_the_top_drawer_and_put_the_bowl_inside.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: open the top drawer and put the bowl inside · MAE 3.34
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__push_the_plate_to_the_front_of_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: push the plate to the front of the stove · MAE 4.172
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_bowl_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the bowl on the plate · MAE 4.147
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_bowl_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the bowl on the stove · MAE 4.086
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the bowl on top of the cabinet · MAE 3.211
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_cream_cheese_in_the_bowl.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the cream cheese in the bowl · MAE 4.857
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_wine_bottle_on_the_rack.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the wine bottle on the rack · MAE 3.188
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__put_the_wine_bottle_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put the wine bottle on top of the cabinet · MAE 4.409
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_goal__turn_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: turn on the stove · MAE 5.111
```
:::
::::
:::::

:::::{dropdown} SPATIAL · 10 tasks · checker 1:1
:open:

::::{grid} 2
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl between the plate and the ramekin and · MAE 2.108
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl from table center and place it on the · MAE 2.92
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl in the top drawer of the wooden cabin · MAE 3.34
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl next to the cookie box and place it o · MAE 2.86
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl next to the plate and place it on the · MAE 3.204
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl next to the ramekin and place it on t · MAE 2.302
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl on the cookie box and place it on the · MAE 1.309
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl on the ramekin and place it on the pl · MAE 2.767
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl on the stove and place it on the plat · MAE 2.423
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_spatial__pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick up the black bowl on the wooden cabinet and place it on · MAE 2.149
```
:::
::::
:::::

:::::{dropdown} LONG-10 · 10 tasks · checker 1:1
:open:

::::{grid} 2
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 3 turn on the stove and put the moka pot on it · MAE 2.623
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 put the black bowl in the bottom drawer of the cab · MAE 5.98
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 6 put the yellow and white mug in the microwave and  · MAE 2.975
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 8 put both moka pots on the stove · MAE 1.878
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 1 put both the alphabet soup and the cream cheese box · MAE 0.228
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 put both the alphabet soup and the tomato sauce in  · MAE 0.245
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 put both the cream cheese box and the butter in the · MAE 0.244
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 5 put the white mug on the left plate and put the yel · MAE 0.162
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 6 put the white mug on the plate and put the chocolat · MAE 0.185
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_10__STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 1 pick up the book and place it in the back compartmen · MAE 0.795
```
:::
::::
:::::

:::::{dropdown} 90 · 90 tasks · checker 1:1
:open:

::::{grid} 2
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 close the top drawer of the cabinet · MAE 2.031
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 close the top drawer of the cabinet and put the b · MAE 0.887
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 put the black bowl in the top drawer of the cabin · MAE 2.787
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 put the butter at the back in the top drawer of t · MAE 2.058
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 put the butter at the front in the top drawer of  · MAE 1.198
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 10 put the chocolate pudding in the top drawer of th · MAE 0.814
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 1 open the bottom drawer of the cabinet · MAE 2.302
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 1 open the top drawer of the cabinet · MAE 1.827
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 1 open the top drawer of the cabinet and put the bow · MAE 1.996
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 1 put the black bowl on the plate · MAE 0.243
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 1 put the black bowl on top of the cabinet · MAE 2.186
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 open the top drawer of the cabinet · MAE 2.271
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 put the black bowl at the back on the plate · MAE 1.878
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 put the black bowl at the front on the plate · MAE 2.68
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 put the middle black bowl on the plate · MAE 1.653
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 put the middle black bowl on top of the cabinet · MAE 1.636
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 stack the black bowl at the front on the black bow · MAE 2.706
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 2 stack the middle black bowl on the back black bowl · MAE 0.531
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 3 put the frying pan on the stove · MAE 1.36
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 3 put the moka pot on the stove · MAE 2.165
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE3_turn_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 3 turn on the stove · MAE 2.265
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 3 turn on the stove and put the frying pan on it · MAE 2.368
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 close the bottom drawer of the cabinet · MAE 6.14
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 close the bottom drawer of the cabinet and open th · MAE 5.067
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 put the black bowl in the bottom drawer of the cab · MAE 5.936
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 put the black bowl on top of the cabinet · MAE 6.936
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 put the wine bottle in the bottom drawer of the ca · MAE 6.266
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 4 put the wine bottle on the wine rack · MAE 6.079
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 5 close the top drawer of the cabinet · MAE 3.543
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 5 put the black bowl in the top drawer of the cabine · MAE 3.02
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 5 put the black bowl on the plate · MAE 4.746
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 5 put the black bowl on top of the cabinet · MAE 1.428
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 5 put the ketchup in the top drawer of the cabinet · MAE 1.303
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE6_close_the_microwave.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 6 close the microwave · MAE 2.831
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 6 put the yellow and white mug to the front of the w · MAE 4.343
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE7_open_the_microwave.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 7 open the microwave · MAE 3.065
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 7 put the white bowl on the plate · MAE 0.669
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 7 put the white bowl to the right of the plate · MAE 2.79
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 8 put the right moka pot on the stove · MAE 1.224
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE8_turn_off_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 8 turn off the stove · MAE 2.618
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 put the frying pan on the cabinet shelf · MAE 2.353
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 put the frying pan on top of the cabinet · MAE 3.292
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 put the frying pan under the cabinet shelf · MAE 2.627
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 put the white bowl on top of the cabinet · MAE 2.034
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_turn_on_the_stove.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 turn on the stove · MAE 2.767
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: KITCHEN 9 turn on the stove and put the frying pan on it · MAE 2.738
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 1 pick up the alphabet soup and put it in the basket · MAE 0.236
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 1 pick up the cream cheese box and put it in the bask · MAE 0.24
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 1 pick up the ketchup and put it in the basket · MAE 0.223
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 1 pick up the tomato sauce and put it in the basket · MAE 0.23
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 pick up the alphabet soup and put it in the basket · MAE 0.252
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 pick up the butter and put it in the basket · MAE 14.423
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 pick up the milk and put it in the basket · MAE 0.25
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 pick up the orange juice and put it in the basket · MAE 0.239
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 2 pick up the tomato sauce and put it in the basket · MAE 0.24
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 3 pick up the alphabet soup and put it in the tray · MAE 0.163
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 3 pick up the butter and put it in the tray · MAE 0.164
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 3 pick up the cream cheese and put it in the tray · MAE 0.172
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 3 pick up the ketchup and put it in the tray · MAE 0.159
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 3 pick up the tomato sauce and put it in the tray · MAE 0.16
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 4 pick up the black bowl on the left and put it in th · MAE 0.548
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 4 pick up the chocolate pudding and put it in the tra · MAE 0.463
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 4 pick up the salad dressing and put it in the tray · MAE 0.169
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 4 stack the left bowl on the right bowl and place the · MAE 0.474
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 4 stack the right bowl on the left bowl and place the · MAE 0.437
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 5 put the red mug on the left plate · MAE 0.17
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 5 put the red mug on the right plate · MAE 0.16
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 5 put the white mug on the left plate · MAE 0.175
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 5 put the yellow and white mug on the right plate · MAE 0.171
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 6 put the chocolate pudding to the left of the plate · MAE 0.177
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 6 put the chocolate pudding to the right of the plate · MAE 0.18
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 6 put the red mug on the plate · MAE 0.177
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: LIVING 6 put the white mug on the plate · MAE 0.185
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 1 pick up the book and place it in the front compartme · MAE 0.498
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 1 pick up the book and place it in the left compartmen · MAE 1.423
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 1 pick up the book and place it in the right compartme · MAE 0.996
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 1 pick up the yellow and white mug and place it to the · MAE 1.019
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 2 pick up the book and place it in the back compartmen · MAE 0.233
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 2 pick up the book and place it in the front compartme · MAE 0.189
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 2 pick up the book and place it in the left compartmen · MAE 1.122
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 2 pick up the book and place it in the right compartme · MAE 1.075
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 3 pick up the book and place it in the front compartme · MAE 0.949
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 3 pick up the book and place it in the left compartmen · MAE 0.868
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 3 pick up the book and place it in the right compartme · MAE 0.39
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 3 pick up the red mug and place it to the right of the · MAE 1.126
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 3 pick up the white mug and place it to the right of t · MAE 1.222
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 4 pick up the book in the middle and place it on the c · MAE 0.603
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 4 pick up the book on the left and place it on top of  · MAE 0.766
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 4 pick up the book on the right and place it on the ca · MAE 0.271
```
:::
:::{grid-item}
```{video} ../../_static/integrations/libero/native_sb/libero_90__STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: STUDY 4 pick up the book on the right and place it under the · MAE 1.145
```
:::
::::
:::::


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

# --- native tasks (no libero/robosuite): vendor shared assets + build bundles ---
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
  python -m tools.libero_integration.vendor_shared_assets  # dedup meshes -> roboverse_data/libero/assets
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
  python -m tools.libero_integration.vendor_native_libero --hdf5 <demo.hdf5> --bddl <task.bddl> --name <task>
# unit tests (checker semantics + a native task runs with NO libero/robosuite):
ROBOVERSE_DATA_DIR=/path/to/roboverse_data MUJOCO_GL=egl \
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
