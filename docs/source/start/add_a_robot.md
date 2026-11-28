# Add a robot

A robot is a `RobotCfg` under `roboverse_pack/robots/` plus the asset files it names. It lands when
one gated task uses it. The generated table in [Robots](../dataset_benchmark/dataset/robots.md) lists
what exists; the configs whose assets did not exist were retired on 2026-11-28.

## 1. The config

Copy `roboverse_pack/robots/robot_template.py`. The fields that matter to the harness:

| field | what it is | who reads it |
|---|---|---|
| `name` | the key every task and result uses | everything |
| `mjcf_path`, `usd_path`, `urdf_path` | one per backend you ship; declare only files that exist | the handlers |
| `actuators` | joint -> stiffness, damping, limits; `is_ee=True` on gripper joints | every backend's PD |
| `joint_limits`, `default_joint_positions` | the action box and the "hold" action | the evaluator, `random`, `hold` |
| `ee_body_name`, `ee_joint_names` | the tool body and the gripper joints | the skills, the kinematics self-test |
| `control_type` | position or effort per joint | the handlers, the action-clip rule |

Joint names must match the asset exactly; the handler's action order is the sorted joint-name list.

## 2. Assets

Put the files under `roboverse_data/robots/<robot>/{mjcf,usd,urdf}/` on the Hub with a licence row in
`THIRD_PARTY_NOTICES.md`. `tools/assets/check_manifest.py --hub-only` fails the pull request for a path
that is not there. MuJoCo and Isaac Sim 6 are the verified backends; a USD that Isaac Lab 3 cannot
build as an articulation is not support, it is a red gate row.

## 3. Self-tests before any task

```bash
python -m metasim robot check <name>                 # MuJoCo: config, assets, kinematics, limits, hold
python -m metasim robot check <name> --sim isaacsim  # the same on Isaac Sim 6
```

Five rows, each a pass or a named failure: the config has limits and defaults for every actuator and the
defaults lie inside the limits; the asset files the config names for that backend exist (locally or on the
Hub, the whole asset folder is fetched); the MJCF loads in MuJoCo's model, the end-effector body is found and
forward kinematics of the default pose is finite (the skill library and the IL adapters depend on this; a robot
without `ee_body_name` is not judged here); the config's joint limits agree with the ranges the MJCF declares;
and holding the default pose for 200 steps leaves every joint within 0.05 rad and a fixed base within 1 cm (a
floating base is reported, not judged). `tests/test_roboverse_robot_cfg_validation.py`
runs the config part on every pull request. On 2026-11-28 Franka, YAM and the Sharpa hand pass all four on
MuJoCo, and Franka passes them on Isaac Sim 6 (YAM has no USD, which the assets row says).

### The sweep of 2026-11-28 (MuJoCo, 43 configs)

| verdict | robots |
|---|---|
| pass (17) | franka, yam, sharpa_wave_left, sharpa_wave_right, z1, so_arm100, ur5e, ur10e, lite6, gen3, iiwa14, g1, g1_dof12, g1_dof29, go1, h1, h1_wrist |
| no MJCF, Isaac/URDF only (13) | aloha, aloha_agilex, fetch, franka_with_gripper_extension, google_robot_static, h1_2_without_hand, iiwa, kinova_gen3_robotiq_2f85, sawyer, unitree_dex3_1_left, ur5e_2f85, vega, xhand_right; most also lack defaults or limits for some actuators, vega's `torso_j1` default is outside its limits |
| asset not on the Hub (3) | g1_tracking (`unitree_description`), openarm_wuji, openarm_bimanual_wuji |
| placeholder config (3) | g1_dof23, g1_dof27, g1_dof29_dex3 (asset paths are `MISSING`) |
| asset defect (4) | shadow_hand (MJCF uses the removed `apirate` option), h1_hand (no actuator for `lh_WRJ2`, no hand limits), arx_l5 (MJCF actuator `gripper` is not named after its joint `joint7`), cartpole (limits ±2.0 vs MJCF ±1.8; standalone build overflows the broadphase buffer) |
| config defect (2) | h1_simple_hand (placeholder limits ±6, missing defaults), koch_1 (sags 0.22 rad at `pitch` under its gains) |
| not a robot (1) | robot_template |

Six of the passes were failures that morning: so_arm100, koch_1, ur5e, ur10e, lite6 and arx_l5 named a tool body
(`tool0`) their MJCF does not have, and so_arm100, koch_1, gen3 and iiwa14 carried another arm's joint limits (the
`limits` row found them; so_arm100's wrist default was 0.7 rad outside the reachable range). Those configs now match
their MJCFs. The remaining rows are the robot backlog, in the order above.

## 4. One gated task

Pin the robot on a tabletop task (`--robot <name>` on `bench record` and `bench health`), record demos
with the skills, pass the gate. Where the asset exists for both engines, run the gate on Isaac Sim 6
too: a robot that tracks the same joint trajectory differently on the two engines is reported as a
parity gap in its config docstring, not hidden.

## What is coming

`metasim robot add`, which generates the config from the asset and then runs `robot check`, and a MuJoCo vs
Isaac parity test (the same joint trajectory tracked on both engines). Until then the steps above are the
contract; the order of planned robots is Franka (reference), YAM, Sharpa, the Wuji hand,
UR5e, xArm7, SO-101, G1, bimanual.
