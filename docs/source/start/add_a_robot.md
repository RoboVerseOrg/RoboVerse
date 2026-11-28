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

Six rows, each a pass or a named failure: the config has limits and defaults for every actuator, the
defaults lie inside the limits and each asset field names a file of its format (`usd_path` a USD, `mjcf_path`
an MJCF, `urdf_path` a URDF; four mjlab configs pointed `usd_path` at their MJCF, which Isaac Sim spawned as an
empty prim); the asset files the config names for that backend exist (locally or on the Hub, the whole asset
folder is fetched) and, on a USD backend, the USD's payloads and references resolve (the iiwa14 USD is a shell
over `configuration/*.usd`); the MJCF loads in MuJoCo's model, the end-effector body is found and
forward kinematics of the default pose is finite (the skill library and the IL adapters depend on this; a robot
without `ee_body_name` is not judged here); the config's joint limits agree with the ranges the MJCF declares;
every MJCF actuator that drives a config joint is named after that joint (the MuJoCo handler finds actuators by
joint name and would add a second servo otherwise, which is the ARX L5 defect below) and every joint the config
drives exists in the MJCF; and holding the default pose for 200 steps leaves every joint within 0.05 rad and a
fixed base within 1 cm (a floating base is reported, not judged). `tests/test_roboverse_robot_cfg_validation.py`
runs the config part on every pull request. On 2026-11-28 Franka, YAM and the Sharpa hand pass all six on
MuJoCo, and Franka passes them on Isaac Sim 6 (YAM has no USD, which the assets row says).

```bash
python -m metasim robot list                                # every config the robot packages define
python -m metasim robot check --all --out-dir gate/robots   # the sweep: one row per robot, robots.json / robots.md
python -m metasim robot check --all --sim isaacsim --out-dir gate/robots_isaac   # one child process per robot
```

The nightly gate runs the sweep on both engines after the task gate and publishes the tables; the Robots page
renders the newest run per engine from `results/robots/`.

On Isaac Sim 6 (2026-11-28, one child process per robot) 9 of the 42 pass: franka, g1_dof12, g1_dof29, gen3, h1,
h1_wrist, ur10e, ur5e, z1; 10 are `n/a` (no USD, nothing else wrong); 23 fail, most of them the config row before any
boot, and three USDs (g1, go1, iiwa14) do not build on Isaac Lab 3: the handler's contact sensor finds no rigid bodies under the robot
prim (`ValueError: No contact sensors added to the prim`), so those three are an asset or handler question, not a
config one. The generated page carries each robot's detail.

### The sweep of 2026-11-28 (MuJoCo, 42 configs)

The table itself is generated: [Robots, self-check](../dataset_benchmark/dataset/robots.md#self-check) renders the
newest `results/robots/<date>-mujoco-check/robots.json` (17 pass, 21 fail, 4 n/a). A robot with no MJCF is
`n/a` on MuJoCo unless a backend-independent row (config) fails it, which 13 of the Isaac / URDF only configs do. By
cause: 13 configs are Isaac / URDF only and have no MJCF (aloha, aloha_agilex, fetch, franka_with_gripper_extension, google_robot_static,
h1_2_without_hand, iiwa, kinova_gen3_robotiq_2f85, sawyer, unitree_dex3_1_left, ur5e_2f85, vega, xhand_right; most
also lack defaults or limits for some actuators, vega's `torso_j1` default is outside its limits); 3 have assets
missing on the Hub (g1_tracking, openarm_wuji, openarm_bimanual_wuji); 3 are placeholders (g1_dof23, g1_dof27,
g1_dof29_dex3); 4 have asset defects (shadow_hand: MJCF uses the removed `apirate` option; h1_hand: no actuator for
`lh_WRJ2`; arx_l5: MJCF actuator `gripper` is not named after its joint `joint7`; cartpole: limits ±2.0 vs MJCF
±1.8 and a standalone build overflows the broadphase buffer); 2 have config defects (h1_simple_hand: placeholder
limits ±6 and missing defaults; koch_1: sags 0.22 rad at `pitch` under its gains).

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

The table is generated: [Robots, cross-engine parity](../dataset_benchmark/dataset/robots.md#cross-engine-parity)
renders `results/robots/2026-11-28-parity-mujoco-isaacsim/`. Gen3 (0.0004 rad), Z1 (0.0011), Franka (0.0043), UR5e
(0.028) and UR10e (0.036) pass at 0.63 rad/s peak; H1 and H1-wrist are reported, not judged (floating base: the knees
end 1.1 to 1.7 rad apart as the two engines let the robot fall); iiwa14 has no comparison because Isaac Lab 3 cannot
build its USD (no rigid bodies under the prim), and so_arm100, lite6 and koch_1 have no USD.

How the trajectory is shaped is the verdict's fine print, and the sweep had to learn it. Its first version began
each phase-shifted joint with a step from the default to the first sample; that step saturated the position servos
(Franka's joint 1 drifted 0.17 rad apart between engines, Z1's 30 N m wrist servos lagged half a second on MuJoCo,
Gen3's unlimited wrist joint swung a radian) and every arm failed. With the amplitude ramped in over the first period
the table above is what remains, and Franka also agrees within 0.004 rad at the full-range probe (`--amplitude-max 0`,
1.5 rad/s peak, near its velocity limit). Two facts from the detour are kept: `FrankaCfg` sets no `effort_limit_sim`,
so MuJoCo clamps at the Menagerie MJCF's +-40 N m while the USD carries 87 N m (joints 1 to 4) and 12 N m (5 to 7),
a difference these trajectories do not expose; and pinning the datasheet limits changes the MuJoCo dynamics the
shipped demos were recorded under (`bench health` then replays stack_cube at 0.6 and pick_cube at 0.8), so the limits
stay unpinned until the demos are re-recorded for a protocol bump. The default amplitude cap of 0.3 rad (0.63 rad/s,
the speed of a tabletop demo) keeps the verdict in the regime the benchmark runs in.

## 5. One gated task

Pin the robot on a tabletop task (`--robot <name>` on `bench record` and `bench health`), record demos
with the skills, pass the gate. Where the asset exists for both engines, run the gate on Isaac Sim 6 too.

## What is coming

`robot add` from a URDF or a USD (today the MJCF is the source; the other paths are declared, not read), a USD for
iiwa14 that Isaac Lab 3 can build, and re-recording the tabletop demos under pinned Franka effort limits as a protocol
bump. The order of planned robots is Franka (reference), YAM, Sharpa, the Wuji hand, UR5e, xArm7,
SO-101, G1, bimanual.
