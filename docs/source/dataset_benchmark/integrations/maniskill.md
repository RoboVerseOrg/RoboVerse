# ManiSkill ↔ MetaSim / SAPIEN3 integration (native 1:1)

[ManiSkill3](https://github.com/haosulab/ManiSkill) is a SAPIEN-backed manipulation
benchmark. RoboVerse ships **MetaSim-native** ManiSkill tasks that reproduce the
native ManiSkill (`physx_cpu`) rollout **1:1** through the standard `BaseTaskEnv` +
SAPIEN3 handler path — no runtime `mani_skill` import, so the clone is deletable.

This page is self-contained: the reproduction recipe, the per-task measured parity, and
the run/verify commands are all below.

## What "1:1" means here

Native ManiSkill on `sim_backend="physx_cpu"` is bit-deterministic, so a clean SAPIEN3
scene can reproduce it exactly. The reproduction recipe is a set of **opt-in**
`SimParamCfg` knobs on the SAPIEN3 handler (all default-off, so existing tasks are
byte-identical):

1. **Disable gravity on every robot link** (`sapien_disable_robot_gravity`) — how
   ManiSkill holds the arm; the single biggest dynamics factor.
2. **Full PhysX config set globally** before scene creation (`sapien_apply_global_physx`):
   solver iters, PCM/TGS, contact/rest offset, sleep/bounce thresholds, default
   material — mirrors `BaseEnv._set_scene_config`.
3. **Drive** with `force_limit` + `mode="force"` (`sapien_drive_force_mode`).
4. **Table** as a kinematic box with the ground plane far below (`sapien_ground_altitude`);
   `PrimitiveCubeCfg` / `PrimitiveMultiBoxCfg` honor `fix_base_link` (kinematic).
5. **Controller** = ManiSkill `pd_joint_delta_pos` (`_native/control.py`,
   parametric over arm DOF + optional mimic gripper), decimation `sim_freq // control_freq`.

## Shipped tasks (15)

`import roboverse_pack.tasks.maniskill` registers them as `maniskill.<name>_native`:

| Task | robot | object pose Δ vs native | dense reward | success |
| --- | --- | --- | --- | --- |
| `pick_cube` | panda | 4.7e-6 | bitwise (5.96e-8) | bitwise |
| `push_cube` | panda | 2.3e-7 | bitwise | bitwise |
| `pull_cube` | panda | 2.3e-7 | bitwise | bitwise |
| `stack_cube` | panda | 1.5e-6 | bitwise | bitwise |
| `poke_cube` | panda | 2.9e-7 | bitwise | bitwise |
| `lift_peg_upright` | panda | 3.0e-7 | bitwise | bitwise |
| `roll_ball` | panda | 1.2e-7 | bitwise | bitwise |
| `place_sphere` | panda | 1.2e-7 | bitwise | bitwise |
| `stack_pyramid` | panda | 8.4e-7 | (no native dense) | bitwise |
| `pull_cube_tool` | panda | 2.8e-7 | bitwise | bitwise |
| `peg_insertion_side` | panda | 4.8e-5 | bitwise | bitwise |
| `plug_charger` | panda | 5.4e-7 | (no native dense) | bitwise |
| `push_t` | panda_stick | Tee Δ=0 | — | proxy |
| `draw_triangle` | panda_stick | 9.9e-6 | — | proxy |
| `two_robot_pick_cube` | 2× panda_wristcam | cube 3.3e-7 | — | proxy |

- **Dynamics** track native to PhysX float32 roundoff (object pose 1.2e-7–4.8e-5 over
  aggressive random steps; ~1e-6 under demo-like motion).
- **Action-level 1:1 for every ManiSkill robot layout**: panda (7 arm + 1 mimic gripper,
  8-dim), panda_stick (7-dim arm-only, PushT/DrawTriangle), and multi-agent
  (TwoRobotPickCube, 2 × 8-dim split per robot by `ManiSkillMultiRobotTask`).
- **Dense rewards**: all 10 tasks with a native dense reward match `compute_dense_reward`
  to float32 epsilon (5.96e-8–1.19e-7).
- **Success**: all 12 tabletop predicates ported bitwise (including peg-in-hole and
  charger `_compute_distance`).
- **`is_grasped`** matches `Panda.is_grasping` (18/18, contact forces ~0.01 N) via the new
  sapien3 `get_pairwise_contact_force`.
- **Reset distribution** matches ManiSkill's per-episode spawn/goal sampling (a persistent
  RNG advances across resets; an explicit seed is reproducible).

## Side-by-side 1:1 videos

Each clip is three panels: **left** = native ManiSkill (`physx_cpu`), **middle** = the shipped
`maniskill.<name>_native` task driven through the SAPIEN3 handler, **right** = the amplified pixel
difference. Both panels are rendered in ManiSkill's own scene (identical assets/lighting/camera), so
the only variable is the physics state — the diff panel is near-black (mean pixel diff 0.003–0.4 / 255
over 60 steps), which *is* the picture of 1:1.

```{video} ../../_static/integrations/maniskill/pick_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pick_cube — native | shipped maniskill.pick_cube_native | diff×8 (pixel diff 0.0035/255)
```

```{video} ../../_static/integrations/maniskill/push_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: push_cube — native | shipped | diff×8 (0.0034/255)
```

```{video} ../../_static/integrations/maniskill/pull_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pull_cube — native | shipped | diff×8 (0.0034/255)
```

```{video} ../../_static/integrations/maniskill/stack_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_cube — native | shipped | diff×8
```

```{video} ../../_static/integrations/maniskill/poke_cube.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: poke_cube — native | shipped | diff×8
```

```{video} ../../_static/integrations/maniskill/lift_peg_upright.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: lift_peg_upright — native | shipped | diff×8 (0.0034/255)
```

```{video} ../../_static/integrations/maniskill/roll_ball.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: roll_ball — native | shipped | diff×8 (0.0030/255)
```

```{video} ../../_static/integrations/maniskill/place_sphere.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: place_sphere — native | shipped | diff×8 (0.0060/255)
```

```{video} ../../_static/integrations/maniskill/peg_insertion_side.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: peg_insertion_side — native | shipped | diff×8
```

```{video} ../../_static/integrations/maniskill/stack_pyramid.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: stack_pyramid — native | shipped | diff×8
```

```{video} ../../_static/integrations/maniskill/pull_cube_tool.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: pull_cube_tool — native | shipped | diff×8 (0.0035/255)
```

```{video} ../../_static/integrations/maniskill/plug_charger.mp4
:autoplay:
:loop:
:muted:
:playsinline:
:width: 100%
:caption: plug_charger — native | shipped | diff×8
```

Regenerate any clip with
`python -m tools.maniskill_integration.render_parity --task PickCube-v1 --shipped pick_cube`.

## Demo replay

Official ManiSkill `pd_joint_delta_pos` demos replay through the shipped tasks
(`tools/maniskill_integration/replay_demo.py`): seeding each episode's initial state +
goal and replaying the recorded actions reproduces the demonstrated success — **21/25
PickCube demos** (native `physx_cpu` itself reproduces ~24/25 open-loop; the gap is the
~1e-4 CPU-handler residual near the 0.025 m boundary). Action replay is deterministic: an
identical grasp+lift sequence keeps the cube within ~1e-4 m of native through the
contact-rich phase.

## Run

```bash
conda activate maniskill1to1   # roboverse env + mani_skill + sapien3

# instantiate a shipped native task (standard registry path)
python -c "
import roboverse_pack.tasks.maniskill, copy, torch
from metasim.task.registry import get_task_class
cls = get_task_class('maniskill.pick_cube_native')
sc = copy.deepcopy(cls.scenario); sc.simulator='sapien3'; sc.num_envs=1; sc.headless=True; sc.cameras=[]
env = cls(sc); env.reset(seed=0)
for _ in range(50): obs, rew, term, trunc, info = env.step(torch.zeros((1,8)))
"

# measure native<->recipe parity (single-agent: 14 tasks; multi-agent: separate tool)
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_native --all --steps 30
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_multi_agent --task TwoRobotPickCube-v1

# side-by-side 1:1 video (native | shipped task | diff)
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.render_parity --task PickCube-v1 --shipped pick_cube

# replay official ManiSkill demos through the shipped task
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.replay_demo --task pick_cube --episodes 25

# regression tests
python -m pytest tests/test_maniskill_native_task.py tests/test_maniskill_reward_grasp.py \
    tests/test_maniskill_success.py tests/test_maniskill_action_levels.py \
    tests/test_maniskill_reset.py tests/test_maniskill_demo_replay.py
```

## Assets

The ManiSkill panda assets (`panda_v2.urdf` for the gripper arm, `panda_stick.urdf`,
`panda_v3.urdf` for the wrist-cam arm, + `franka_description` / realsense meshes) are
vendored under `roboverse_data/robots/maniskill_panda/` and published to the
HuggingFace-backed `RoboVerseOrg/roboverse_data` dataset. The locators
(`_native/recipe.panda_urdf_path` etc.) prefer the local copy, else token-free
`snapshot_download` from HF, else fall back to an installed `mani_skill` package — so the
clone loads its robots **without** a `mani_skill` install.

## Backward compatibility

All MetaSim-side changes are opt-in (`SimParamCfg` knobs default to `None`/`False`; the new
`PrimitiveMultiBoxCfg`, `get_pairwise_contact_force`, and `PrimitiveCubeCfg`
`fix_base_link` handling are additive) — existing SAPIEN3 tasks are byte-identical
(417 sapien3 + general MetaSim tests pass). The RoboVerse side is purely additive
(`_native/` package + tasks + tools + tests). Both MetaSim and RoboVerse changes are merged
to their respective public `main` branches.

## Remaining

`peg_insertion_side` and `plug_charger` randomize their **internal geometry** per episode
(the box-hole position / charger prong layout). Their success/reward formulas are exact
and their part *poses* randomize per episode, but reproducing the per-episode internal
geometry would require rebuilding the multi-box collision actor each reset (framework-
atypical); it's left as a follow-up to preserve backward compatibility.
