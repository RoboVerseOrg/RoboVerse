# Task inventory

Generated from the task registry by `generate_task_docs.py`; do not edit by hand. Every name below is
registered in this checkout (`python -m metasim bench list` prints the same list); the release gate
(`bench health --floor --repeat 2`, nightly on MuJoCo and Isaac Sim 6) decides whether a task stays, and
`roboverse_pack/tasks/RETIRED_TASKS.md` records the ones it removed and the commit to restore them from.
How to add one: [Add a task](../../start/add_a_task.md).

110 registered task names in 11 packs.

## Suites

- `tabletop-mujoco` / `core`: `pick_cube`, `stack_cube`, `push_cube`, `pull_cube`, `lift_peg_upright`

## Registered tasks by pack

### embodiedgen (2)

| task | registered in |
|---|---|
| `embodiedgen.put_mug` | `roboverse_pack.tasks.embodiedgen.put_mug` |
| `put_mug` | `roboverse_pack.tasks.embodiedgen.put_mug` |

### humanoid (6)

| task | registered in |
|---|---|
| `g1.walk_g1_dof12` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof12` |
| `g1.walk_g1_dof29` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof29` |
| `unitree_rl.walk_g1_dof12` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof12` |
| `unitree_rl.walk_g1_dof29` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof29` |
| `walk_g1_dof12` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof12` |
| `walk_g1_dof29` | `roboverse_pack.tasks.humanoid.locomotion.walk_g1_dof29` |

### humanoid_bench (12)

| task | registered in |
|---|---|
| `crawl` | `roboverse_pack.tasks.humanoid_bench.crawl` |
| `h1.crawl` | `roboverse_pack.tasks.humanoid_bench.crawl` |
| `h1.sit` | `roboverse_pack.tasks.humanoid_bench.sit` |
| `h1.slide` | `roboverse_pack.tasks.humanoid_bench.slide` |
| `h1.stair` | `roboverse_pack.tasks.humanoid_bench.stair` |
| `humanoid.crawl` | `roboverse_pack.tasks.humanoid_bench.crawl` |
| `humanoid.sit` | `roboverse_pack.tasks.humanoid_bench.sit` |
| `humanoid.slide` | `roboverse_pack.tasks.humanoid_bench.slide` |
| `humanoid.stair` | `roboverse_pack.tasks.humanoid_bench.stair` |
| `sit` | `roboverse_pack.tasks.humanoid_bench.sit` |
| `slide` | `roboverse_pack.tasks.humanoid_bench.slide` |
| `stair` | `roboverse_pack.tasks.humanoid_bench.stair` |

### libero (2)

| task | registered in |
|---|---|
| `libero.pick_butter` | `roboverse_pack.tasks.libero.libero_pick_butter` |
| `pick_butter` | `roboverse_pack.tasks.libero.libero_pick_butter` |

### libero_90 (2)

| task | registered in |
|---|---|
| `kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` | `roboverse_pack.tasks.libero_90.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` |
| `libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` | `roboverse_pack.tasks.libero_90.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` |

### maniskill (28)

| task | registered in |
|---|---|
| `lift_peg_upright` | `roboverse_pack.tasks.maniskill.lift_peg_upright` |
| `lift_peg_upright_dense` | `roboverse_pack.tasks.maniskill.lift_peg_upright` |
| `maniskill.lift_peg_upright` | `roboverse_pack.tasks.maniskill.lift_peg_upright` |
| `maniskill.lift_peg_upright_dense` | `roboverse_pack.tasks.maniskill.lift_peg_upright` |
| `maniskill.pick_cube` | `roboverse_pack.tasks.maniskill.pick_cube` |
| `maniskill.pick_cube_v1_dense` | `roboverse_pack.tasks.maniskill.pick_cube_v1` |
| `maniskill.pick_single_ycb_dense` | `roboverse_pack.tasks.maniskill.pick_single_ycb` |
| `maniskill.place_sphere_dense` | `roboverse_pack.tasks.maniskill.place_sphere` |
| `maniskill.plug_charger` | `roboverse_pack.tasks.maniskill.plug_charger` |
| `maniskill.poke_cube_dense` | `roboverse_pack.tasks.maniskill.poke_cube` |
| `maniskill.pull_cube` | `roboverse_pack.tasks.maniskill.pull_cube` |
| `maniskill.pull_cube_dense` | `roboverse_pack.tasks.maniskill.pull_cube` |
| `maniskill.push_cube` | `roboverse_pack.tasks.maniskill.push_cube` |
| `maniskill.push_cube_dense` | `roboverse_pack.tasks.maniskill.push_cube` |
| `maniskill.roll_ball_dense` | `roboverse_pack.tasks.maniskill.roll_ball` |
| `maniskill.stack_cube_dense` | `roboverse_pack.tasks.maniskill.stack_cube` |
| `pick_cube` | `roboverse_pack.tasks.maniskill.pick_cube` |
| `pick_cube_v1_dense` | `roboverse_pack.tasks.maniskill.pick_cube_v1` |
| `pick_single_ycb_dense` | `roboverse_pack.tasks.maniskill.pick_single_ycb` |
| `place_sphere_dense` | `roboverse_pack.tasks.maniskill.place_sphere` |
| `plug_charger` | `roboverse_pack.tasks.maniskill.plug_charger` |
| `poke_cube_dense` | `roboverse_pack.tasks.maniskill.poke_cube` |
| `pull_cube` | `roboverse_pack.tasks.maniskill.pull_cube` |
| `pull_cube_dense` | `roboverse_pack.tasks.maniskill.pull_cube` |
| `push_cube` | `roboverse_pack.tasks.maniskill.push_cube` |
| `push_cube_dense` | `roboverse_pack.tasks.maniskill.push_cube` |
| `roll_ball_dense` | `roboverse_pack.tasks.maniskill.roll_ball` |
| `stack_cube_dense` | `roboverse_pack.tasks.maniskill.stack_cube` |

### metasim example pack (21)

| task | registered in |
|---|---|
| `franka.obj_env` | `metasim.example.example_pack.tasks.obj_env` |
| `franka.reach_far` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `franka.reach_origin` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `h1.run` | `metasim.example.example_pack.tasks.humanoid_env` |
| `h1.stand` | `metasim.example.example_pack.tasks.humanoid_env` |
| `h1.walk` | `metasim.example.example_pack.tasks.humanoid_env` |
| `humanoid.run` | `metasim.example.example_pack.tasks.humanoid_env` |
| `humanoid.stand` | `metasim.example.example_pack.tasks.humanoid_env` |
| `humanoid.walk` | `metasim.example.example_pack.tasks.humanoid_env` |
| `maniskill.stack_cube` | `metasim.example.example_pack.tasks.stack_cube` |
| `obj_env` | `metasim.example.example_pack.tasks.obj_env` |
| `reach.far` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `reach.origin` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `reach_far` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `reach_origin` | `metasim.example.example_pack.tasks.reach_wrappers` |
| `run` | `metasim.example.example_pack.tasks.humanoid_env` |
| `stack_cube` | `metasim.example.example_pack.tasks.stack_cube` |
| `stand` | `metasim.example.example_pack.tasks.humanoid_env` |
| `walk` | `metasim.example.example_pack.tasks.humanoid_env` |
| `xhand.xhand_env` | `metasim.example.example_pack.tasks.dexhand_env` |
| `xhand_env` | `metasim.example.example_pack.tasks.dexhand_env` |

### mjlab (24)

| task | registered in |
|---|---|
| `mjlab.cartpole_balance_train` | `roboverse_pack.tasks.mjlab.cartpole_train` |
| `mjlab.cartpole_balance_v2` | `roboverse_pack.tasks.mjlab.cartpole_v2` |
| `mjlab.cartpole_swingup_train` | `roboverse_pack.tasks.mjlab.cartpole_train` |
| `mjlab.cartpole_swingup_v2` | `roboverse_pack.tasks.mjlab.cartpole_v2` |
| `mjlab.lift_cube_yam` | `roboverse_pack.tasks.mjlab.lift_cube` |
| `mjlab.lift_cube_yam_depth` | `roboverse_pack.tasks.mjlab.lift_cube` |
| `mjlab.lift_cube_yam_depth_v2` | `roboverse_pack.tasks.mjlab.lift_cube_yam_v2` |
| `mjlab.lift_cube_yam_rgb` | `roboverse_pack.tasks.mjlab.lift_cube` |
| `mjlab.lift_cube_yam_rgb_v2` | `roboverse_pack.tasks.mjlab.lift_cube_yam_v2` |
| `mjlab.lift_cube_yam_v2` | `roboverse_pack.tasks.mjlab.lift_cube_yam_v2` |
| `mjlab.multi_cube_seg_yam` | `roboverse_pack.tasks.mjlab.lift_cube` |
| `mjlab.multi_cube_seg_yam_v2` | `roboverse_pack.tasks.mjlab.lift_cube_yam_v2` |
| `mjlab.tracking_flat_g1` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.tracking_flat_g1_no_state_est` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.tracking_flat_g1_no_state_est_v2` | `roboverse_pack.tasks.mjlab.velocity_g1_v2` |
| `mjlab.tracking_flat_g1_v2` | `roboverse_pack.tasks.mjlab.velocity_g1_v2` |
| `mjlab.velocity_flat_g1` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.velocity_flat_g1_v2` | `roboverse_pack.tasks.mjlab.velocity_g1_v2` |
| `mjlab.velocity_flat_go1` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.velocity_flat_go1_v2` | `roboverse_pack.tasks.mjlab.velocity_go1_v2` |
| `mjlab.velocity_rough_g1` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.velocity_rough_g1_v2` | `roboverse_pack.tasks.mjlab.velocity_g1_v2` |
| `mjlab.velocity_rough_go1` | `roboverse_pack.tasks.mjlab.floating_base` |
| `mjlab.velocity_rough_go1_v2` | `roboverse_pack.tasks.mjlab.velocity_go1_v2` |

### pick_place (10)

| task | registered in |
|---|---|
| `pick_place.approach_grasp_simple` | `roboverse_pack.tasks.pick_place.approach_grasp` |
| `pick_place.base` | `roboverse_pack.tasks.pick_place.base` |
| `pick_place.obstacle` | `roboverse_pack.tasks.pick_place.obstacle` |
| `pick_place.pour_water` | `roboverse_pack.tasks.pick_place.pour_water` |
| `pick_place.wipe_window` | `roboverse_pack.tasks.pick_place.wipe_window` |
| `pick_place_approach_grasp_simple` | `roboverse_pack.tasks.pick_place.approach_grasp` |
| `pick_place_base` | `roboverse_pack.tasks.pick_place.base` |
| `pick_place_obstacle` | `roboverse_pack.tasks.pick_place.obstacle` |
| `pick_place_pour_water` | `roboverse_pack.tasks.pick_place.pour_water` |
| `wipe_window` | `roboverse_pack.tasks.pick_place.wipe_window` |

### rlbench (2)

| task | registered in |
|---|---|
| `close_box` | `roboverse_pack.tasks.rlbench.close_box` |
| `rlbench.close_box` | `roboverse_pack.tasks.rlbench.close_box` |

### task_template (1)

| task | registered in |
|---|---|
| `template.minimal` | `roboverse_pack.tasks.task_template` |

## Retired

441 retired tasks (`roboverse_pack/tasks/retired_tasks.csv`, with the commit to restore each from; `RETIRED_TASKS.md` tells the story), by family: beyondmimic (3), bimanual (1), box_task (1), calvin (5), embodiedgen (1), humanoid (1), libero (178), libero_90 (64), maniskill (81), mjlab (2), mujoco_playground (3), pick_place (14), rlbench (78), robosuite (5), simpler_env (4).
