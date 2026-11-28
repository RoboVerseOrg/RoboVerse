# Add a robot

A robot is a `RobotCfg` under `roboverse_pack/robots/` plus the asset files it names. It lands when
one gated task uses it. The generated table in [Robots](../dataset_benchmark/dataset/robots.md) lists
what exists; the configs whose assets did not exist were retired on 2026-11-28.

## 1. The config

Generate it from the MJCF, then review it:

```bash
python -m metasim robot add --mjcf roboverse_data/robots/<robot>/mjcf/<robot>.xml --name <robot> \
    [--usd ...] [--default joint2=1.047 ...] [--ee-body <body>] [--gripper-joints <joint> ...]
```

`robot add` reads the joints and their ranges, the servo gains of the MJCF actuators (`kp`, `kv`, the force range as
`effort_limit_sim`; a motor becomes an effort-controlled joint), the `home` keyframe or `--default` values for the
default pose, the gripper joints (slide joints and finger / jaw / gripper names, `is_ee=True`), the tool body (the
gripper joints' common parent) and a grasp or tool-centre site on it (`curobo_tcp_rel_pos`). A joint an `<equality>`
couples to a driven one is passive: kept in the limits and defaults, left out of `actuators`, its default mirrored.
The file goes to `roboverse_pack/robots/<name>_cfg.py`; everything the generator guessed is a `TODO` comment and a
line under "Generator notes", and the self-check below runs on the file immediately. On 2026-11-28 the generated
Franka and YAM configs reproduce the hand-written limits, gains, tool bodies and the YAM grasp site exactly; the
default pose is the one thing a Menagerie file without a keyframe cannot provide (`--default` fills it).

`roboverse_pack/robots/robot_template.py` is the hand-written route. The fields that matter to the harness:

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
python -m metasim robot check <name>                 # MuJoCo: config, assets, kinematics, limits, actuators, hold
python -m metasim robot check <name> --sim isaacsim  # the same on Isaac Sim 6
```

Six rows, each a pass or a named failure: the config has limits and defaults for every actuator and the
defaults lie inside the limits; the asset files the config names for that backend exist (locally or on the
Hub, the whole asset folder is fetched); the MJCF loads in MuJoCo's model, the end-effector body is found and
forward kinematics of the default pose is finite (the skill library and the IL adapters depend on this; a robot
without `ee_body_name` is not judged here); the config's joint limits agree with the ranges the MJCF declares;
every MJCF actuator that drives a config joint is named after that joint (the MuJoCo handler finds actuators by
joint name and would add a second servo otherwise, which is the ARX L5 defect below) and every joint the config
drives exists in the MJCF; and holding the default pose for 200 steps leaves every joint within 0.05 rad and a
fixed base within 1 cm (a floating base is reported, not judged). `tests/test_roboverse_robot_cfg_validation.py`
runs the config part on every pull request. On 2026-11-28 Franka, YAM and the Sharpa hand pass all six on
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

## 4. Cross-engine parity

```bash
python -m metasim robot parity <name> --sims mujoco isaacsim            # one process, both engines
python -m metasim robot parity <name> --sims mujoco --save mj.json      # or one engine per environment:
python -m metasim robot parity <name> --load mj.json --sims isaacsim    # ... compared in the second one
```

The same canned joint trajectory on both engines: 50 control steps settling at the default pose, then 400 steps
(6 s at the pinned 1 ms x 15 control step) of a sinusoid per joint at a quarter of its half range, phase-shifted
joint by joint, clipped to the limits. The report is one row per joint: the largest and the RMS difference between
the two engines' measured positions, each engine's own tracking error against the command (reported, not judged:
it depends on the gains), and a verdict against `--tol` (0.05 rad for a hinge joint) or `--tol-slide` (5 mm for a
slide joint). A robot that passes `robot check` on both engines and fails parity is a config whose gains or limits
the two engines read differently; the row names the joint. Where the asset exists for both engines, a robot lands
only with this green.

### The sweep of 2026-11-28 (MuJoCo vs Isaac Sim 6, every config with an MJCF and a USD)

| verdict | robots |
|---|---|
| pass | franka (worst 0.038 rad, joints 2 and 4), ur5e (0.024), ur10e (0.026) |
| FAIL | z1 (joints 3, 4, 6: MuJoCo follows 44 to 58 steps late, 0.33 rad apart, Isaac 6 steps: the MJCF's wrist gains); gen3 (joint_7 0.89 rad: MuJoCo tracking error 1.13 vs 0.28; joints 2 to 4 0.06 to 0.13) |
| reported, not judged | h1, h1_wrist (floating base: the knees end 1.0 to 1.1 rad apart as the two engines let the robot fall) |
| no comparison | iiwa14 (Isaac Lab 3 cannot build the USD: no rigid bodies under the prim), so_arm100, lite6, koch_1 (no USD) |

The regime is the verdict's fine print. The first run used a quarter of each joint's half range in 3 s (1.5 rad/s on
Franka's joint 1, near its velocity limit) and every arm failed, Franka by 0.17 rad on joint 1: at that speed the
position gains saturate the torque limits, and the limits differ between engines (the Menagerie MJCF authors
+-40 N m, the USD 87 N m on joints 1 to 4 and 12 N m on 5 to 7, because `FrankaCfg` sets no `effort_limit_sim`).
Pinning the datasheet limits halves the remaining gap on joints 2 and 4 but changes the MuJoCo dynamics the shipped
demos were recorded under: `bench health` then replays stack_cube at 0.6 and pick_cube at 0.8. The limits therefore
stay unpinned, the gap is written in the config's docstring, and the default amplitude is capped at 0.3 rad (0.63
rad/s), the speed of a tabletop demo. `--amplitude-max 0` restores the full-range probe.

## 5. One gated task

Pin the robot on a tabletop task (`--robot <name>` on `bench record` and `bench health`), record demos
with the skills, pass the gate. Where the asset exists for both engines, run the gate on Isaac Sim 6 too.

## What is coming

`robot add` from a URDF or a USD (today the MJCF is the source; the other paths are declared, not read), the Z1
and Gen3 gain fixes the parity sweep asks for, and re-recording the tabletop demos under pinned Franka effort limits
as a protocol bump. The order of planned robots is Franka (reference), YAM, Sharpa, the Wuji hand, UR5e, xArm7,
SO-101, G1, bimanual.
