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
- **Object fidelity is exact, suite-wide**: the bridge records each manipulated
  object's *real* RoboTwin asset so the replay loads the same mesh/URDF — and it
  gets the exact instance, not just the category. For mesh objects it captures the
  `model_id`; for URDF objects (pot/cabinet/laptop/microwave) it hooks
  `rand_create_sapien_urdf_obj` to record the precise instance directory (RoboTwin
  picks a random `modelid` per episode, *excluding* the `visual/` dir) plus the
  `model_data.json` scale. Objects created multiple times under the same name
  (e.g. two `001_bottle`, three blocks, three bottles in `put_bottles_dustbin`) are
  disambiguated per-instance by creation order, so every object is kept — a
  name-keyed capture silently dropped all but one. A full re-collection found
  **13 of 50 tasks** create same-named duplicates; all now replay every object.
- **Rendering is 1:1 in geometry, with a documented engine residual**: the
  side-by-side (`sidebyside.py`) puts the native RoboTwin render next to the
  RoboVerse replay from an identical camera and the same bridge trajectory. Robot
  pose, object instances/positions/motion, table, and ground match frame-for-frame.
  Both render ray-traced with matched settings (32 samples, path depth 8); the only
  residual is a background colour tint, because the two SAPIEN builds
  (RoboTwin's 3.0.0b1 vs MetaSim's) use different *default* RT environment maps and
  neither sets one explicitly. This is an engine-build difference, not a
  reproduction error.
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

## 1:1 visualization — all 50 tasks

Every task rendered **native RoboTwin (left) vs RoboVerse replay (right)**, same observer pose, frame-for-frame: the RoboVerse replay is driven by the *same* recorded bridge trajectory (`native_render.py --replay-bridge`), so robot pose + every object (mesh, instance, pose) line up 1:1 — only cross-engine texture shading differs.

Regenerate **any** of the 50 clips with one command (swap `--task <name>` for any task below):

```bash
# native RoboTwin (robotwin env) + RoboVerse replay (roboverse env), composited side-by-side
conda run -n roboverse python tools/robotwin_integration/sidebyside.py --task move_can_pot
#   -> outputs/robotwin_coverage/sidebyside_move_can_pot.mp4
```

<details><summary><b>All 50 task names + regenerate the whole gallery</b></summary>

`adjust_bottle` · `beat_block_hammer` · `blocks_ranking_rgb` · `blocks_ranking_size` · `click_alarmclock`  
`click_bell` · `dump_bin_bigbin` · `grab_roller` · `handover_block` · `handover_mic`  
`hanging_mug` · `lift_pot` · `move_can_pot` · `move_pillbottle_pad` · `move_playingcard_away`  
`move_stapler_pad` · `open_laptop` · `open_microwave` · `pick_diverse_bottles` · `pick_dual_bottles`  
`place_a2b_left` · `place_a2b_right` · `place_bread_basket` · `place_bread_skillet` · `place_burger_fries`  
`place_can_basket` · `place_cans_plasticbox` · `place_container_plate` · `place_dual_shoes` · `place_empty_cup`  
`place_fan` · `place_mouse_pad` · `place_object_basket` · `place_object_scale` · `place_object_stand`  
`place_phone_stand` · `place_shoe` · `press_stapler` · `put_bottles_dustbin` · `put_object_cabinet`  
`rotate_qrcode` · `scan_object` · `shake_bottle` · `shake_bottle_horizontally` · `stack_blocks_three`  
`stack_blocks_two` · `stack_bowls_three` · `stack_bowls_two` · `stamp_seal` · `turn_switch`  

```bash
for t in \
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size \
    click_alarmclock click_bell dump_bin_bigbin grab_roller \
    handover_block handover_mic hanging_mug lift_pot \
    move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad \
    open_laptop open_microwave pick_diverse_bottles pick_dual_bottles \
    place_a2b_left place_a2b_right place_bread_basket place_bread_skillet \
    place_burger_fries place_can_basket place_cans_plasticbox place_container_plate \
    place_dual_shoes place_empty_cup place_fan place_mouse_pad \
    place_object_basket place_object_scale place_object_stand place_phone_stand \
    place_shoe press_stapler put_bottles_dustbin put_object_cabinet \
    rotate_qrcode scan_object shake_bottle shake_bottle_horizontally \
    stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two \
    stamp_seal turn_switch ; do
  conda run -n roboverse python tools/robotwin_integration/sidebyside.py --task $t
done
```
</details>

### Grasp · tool · press (11)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_beat_block_hammer.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: beat_block_hammer
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_click_bell.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: click_bell
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_click_alarmclock.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: click_alarmclock
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_press_stapler.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: press_stapler
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_grab_roller.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: grab_roller
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_stamp_seal.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stamp_seal
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_rotate_qrcode.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: rotate_qrcode
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_turn_switch.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: turn_switch
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_handover_block.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: handover_block
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_handover_mic.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: handover_mic
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_move_playingcard_away.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: move_playingcard_away
```
:::
::::

### Place onto target (20)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_move_can_pot.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: move_can_pot
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_move_pillbottle_pad.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: move_pillbottle_pad
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_move_stapler_pad.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: move_stapler_pad
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_a2b_left.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_a2b_left
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_a2b_right.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_a2b_right
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_bread_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_bread_basket
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_bread_skillet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_bread_skillet
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_burger_fries.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_burger_fries
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_can_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_can_basket
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_cans_plasticbox.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_cans_plasticbox
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_container_plate.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_container_plate
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_dual_shoes.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_dual_shoes
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_empty_cup.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_empty_cup
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_fan.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_fan
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_mouse_pad.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_mouse_pad
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_object_basket.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_object_basket
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_object_scale.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_object_scale
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_object_stand.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_object_stand
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_phone_stand.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_phone_stand
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_place_shoe.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_shoe
```
:::
::::

### Bottles · pick · shake (6)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_pick_diverse_bottles.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick_diverse_bottles
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_pick_dual_bottles.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick_dual_bottles
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_shake_bottle.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: shake_bottle
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_shake_bottle_horizontally.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: shake_bottle_horizontally
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_adjust_bottle.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: adjust_bottle
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_put_bottles_dustbin.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put_bottles_dustbin
```
:::
::::

### Stack · rank (6)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_stack_blocks_two.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_blocks_two
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_stack_blocks_three.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_blocks_three
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_stack_bowls_two.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_bowls_two
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_stack_bowls_three.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_bowls_three
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_blocks_ranking_rgb.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: blocks_ranking_rgb
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_blocks_ranking_size.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: blocks_ranking_size
```
:::
::::

### Articulated · container (URDF joints) (7)

::::{grid} 2
:gutter: 2

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_open_laptop.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: open_laptop
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_open_microwave.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: open_microwave
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_lift_pot.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: lift_pot
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_put_object_cabinet.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: put_object_cabinet
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_dump_bin_bigbin.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: dump_bin_bigbin
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_hanging_mug.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: hanging_mug
```
:::

:::{grid-item}
```{video} ../../_static/integrations/robotwin/sbs_all_scan_object.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: scan_object
```
:::
::::

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

## Self-contained replay (RoboTwin is deletable)

The replay / side-by-side / object-parity pipeline does **not** need the upstream RoboTwin
checkout at runtime. Every asset a bridge references — object visual/collision meshes, URDF
instances, and the ALOHA-AgileX embodiment — is addressed by its *RoboTwin-internal relpath*
and resolved through one locator,
[`roboverse_pack/tasks/robotwin/_locator.py`](../../../../roboverse_pack/tasks/robotwin/_locator.py):

1. a local RoboTwin clone — `$ROBOTWIN_ASSETS` or `~/projects/robotwin` (dev / fresh collection);
2. otherwise the vendored mirror `roboverse_data/robotwin/` (HuggingFace `RoboVerseOrg/roboverse_data`),
   downloaded on demand — exactly like the mjlab / menagerie locators.

`$ROBOTWIN_ASSETS` is **authoritative**: set it to a non-existent path to force the mirror (this is
how the deletability test runs).

**Vendor the referenced subset once** (only what the 50 bridges use — ~1.65 GB objects + 0.78 GB
embodiment + slim RGB-stripped trajectories, *not* the 1.47 TB full dataset):

```bash
# Against a RoboTwin clone, copy the referenced subset into roboverse_data/robotwin/
python tools/robotwin_integration/migrate_assets.py        # writes manifest.json

# Replay with the clone "deleted" — resolves everything from the mirror:
ROBOTWIN_ASSETS=/nonexistent MUJOCO_GL=egl python \
  tools/robotwin_integration/mesh_replay_robotwin.py \
  --bridge roboverse_data/robotwin/bridges/move_can_pot.pkl --mode kinematic --video
```

To make a *fresh, clone-less* machine work, upload the populated mirror to the HF dataset
(`roboverse_data/` is git-ignored; it is the HF-backed store, not committed):

```bash
huggingface-cli upload RoboVerseOrg/roboverse_data roboverse_data/robotwin robotwin --repo-type dataset
```

The embodiment cfg (`roboverse_pack/robots/aloha_agilex_cfg.py`) resolves through the same locator.

## Policy reproduction (same experience as RoboTwin)

A RoboVerse user can reproduce a RoboTwin policy result end to end — collect
expert demos, train an imitation policy, evaluate it closed-loop — using
RoboVerse's own imitation-learning stack (`roboverse_learn/il`), the same
three-step `collect → train → eval` flow a RoboTwin user runs. The trained
policy is evaluated **closed-loop in the native RoboTwin environment** (via the
passthrough), so its success rate is directly comparable to RoboTwin's own
learned-policy baseline (not the ~100% scripted expert planner).

Cross-task results (closed-loop, native RoboTwin, 20 held-out seeds, 400-step
budget): `beat_block_hammer` **42%** (precision strike, validated over two runs
45%+40%), `move_can_pot` **30%** (pick-place), `click_bell` **50%** (simple
single-arm press). All land at RoboTwin's own DP baseline level — simplest task
highest, as expected. Successful episodes trigger `check_success` early, so the
policy genuinely completes the task rather than replaying. (A 40-demo / 300-epoch
run overfits to 15%; data volume + RoboTwin-matched `n_action_steps=6` close the
gap.) The policy is stochastic (DDPM sampling), so each 20-episode rate has
run-to-run variance (±10–20%).

**Eval robustness (read before trusting a number).** The DP eval renders the
head-camera every step with RoboTwin's RT shader (required for train/eval obs
parity). That RT render path **intermittently deadlocks** headless in upstream
sapien — an episode then hangs to its `--per-ep-timeout` and is counted a failure.
Two safeguards in `eval_dp_robotwin.sh` keep this from corrupting a result: it
waits for any DP-training process to release the GPU before loading the policy
server (train→eval contention makes the first inference hang), and it **aborts
after 3 consecutive no-result hangs** with the server log rather than burning
`N × timeout` and reporting a misleading `0/N`. A genuine all-`0/N` should always
be investigated as a harness/hang issue, never reported as a policy result; the
rates above are from clean runs where all 20 episodes returned a real
success/failure.

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
   (`examples/10_robotwin_aloha_replay.py` is the minimal get-started
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
