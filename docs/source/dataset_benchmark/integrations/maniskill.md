# ManiSkill ↔ MetaSim / SAPIEN3 integration (native 1:1)

[ManiSkill3](https://github.com/haosulab/ManiSkill) is a SAPIEN-backed
manipulation benchmark. RoboVerse ships **MetaSim-native** ManiSkill
tabletop tasks that reproduce the native ManiSkill (`physx_cpu`) rollout
**1:1** through the standard `BaseTaskEnv` + SAPIEN3 handler path — no
runtime `mani_skill` import, so the clone is deletable.

Live report (videos + measured parity):
<http://localhost:8000/#roboverse/maniskill_integration>.

## What "1:1" means here

Native ManiSkill on `sim_backend="physx_cpu"` is bit-deterministic, so a
clean SAPIEN3 scene can reproduce it exactly. The reproduction recipe
(all opt-in `SimParamCfg` knobs on the SAPIEN3 handler, default-off so
existing tasks are byte-identical):

1. **Disable gravity on every robot link** (`sapien_disable_robot_gravity`)
   — how ManiSkill holds the arm; the single biggest dynamics factor.
2. **Full PhysX config set globally** before scene creation
   (`sapien_apply_global_physx`): solver iters, PCM/TGS, contact/rest
   offset, sleep/bounce thresholds, default material — mirrors
   `BaseEnv._set_scene_config`.
3. **Drive** with `force_limit` + `mode="force"` (`sapien_drive_force_mode`).
4. **Table** as a kinematic box with the ground plane far below
   (`sapien_ground_altitude`); `PrimitiveCubeCfg`/`PrimitiveMultiBoxCfg`
   honor `fix_base_link` (kinematic).
5. **Controller** = ManiSkill `pd_joint_delta_pos` + gripper mimic
   (vendored in `_native/control.py`), decimation `sim_freq // control_freq`.

## Shipped tasks

`import roboverse_pack.tasks.maniskill` registers 12 tabletop tasks as
`maniskill.<name>_native` (also `<name>_native`):

| Task | object pose Δ vs native | dense reward | success |
| --- | --- | --- | --- |
| `pick_cube` | 4.7e-6 | bitwise (5.96e-8) | ✓ |
| `push_cube` | 2.3e-7 | bitwise | ✓ |
| `pull_cube` | 2.3e-7 | bitwise | ✓ |
| `stack_cube` | 1.5e-6 | bitwise | ✓ |
| `poke_cube` | 2.9e-7 | — | ✓ |
| `lift_peg_upright` | 3.0e-7 | bitwise | ✓ |
| `roll_ball` | 1.2e-7 | bitwise | ✓ |
| `place_sphere` | 1.2e-7 | bitwise | ✓ |
| `stack_pyramid` | 8.4e-7 | (no native dense) | ✓ |
| `pull_cube_tool` | 2.8e-7 | bitwise | ✓ |
| `peg_insertion_side` | 4.8e-5 | (todo) | proxy |
| `plug_charger` | 5.4e-7 | (no native dense) | proxy |

Object pose tracks native to PhysX float32 roundoff (1.2e-7–4.8e-5 over
aggressive random steps; ~1e-6 under demo-like motion). Dense rewards
match native `compute_dense_reward` to float32 epsilon (8/10 tasks with a
native dense reward done); `is_grasped` matches `Panda.is_grasping`
(18/18, contact forces ~0.01 N) via the new sapien3
`get_pairwise_contact_force`. Reset uses ManiSkill's spawn + goal
distribution.

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

# measure native<->recipe parity for any task
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_native --all --steps 30

# render a side-by-side 1:1 video (native | shipped task | diff)
SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.render_parity \
    --task PickCube-v1 --shipped pick_cube --steps 60

# regression tests
python -m pytest tests/test_maniskill_native_task.py \
    tests/test_maniskill_reward_grasp.py tests/test_maniskill_success.py
```

## Assets

The ManiSkill Panda (`panda_v2.urdf` + `franka_description` meshes) is
vendored under `roboverse_data/robots/maniskill_panda/` (HuggingFace-backed,
like the other integrations); the locator (`_native/recipe.panda_urdf_path`)
prefers the vendored copy and falls back to the installed `mani_skill`
package.

## Backward compatibility

All MetaSim-side changes are opt-in (`SimParamCfg` knobs default to
`None`/`False`; the new `PrimitiveMultiBoxCfg` and
`get_pairwise_contact_force` are additive) — existing SAPIEN3 tasks are
byte-identical (417 sapien3 + general MetaSim tests pass). The RoboVerse
side is purely additive (new `_native/` package + tasks + tools + tests).

## Remaining

`peg_insertion_side` / `plug_charger` success use ManiSkill internal pose
helpers (peg-in-hole / charger-to-goal) and are not yet ported (geometric
proxy in place); their dense rewards (and demo-`.h5` replay, non-gripper
robots like PushT/DrawTriangle, and multi-agent TwoRobot tasks) are
tracked as follow-ups.
