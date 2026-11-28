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
python -m metasim robot check <name>                 # MuJoCo: config, assets, kinematics, hold
python -m metasim robot check <name> --sim isaacsim  # the same on Isaac Sim 6
```

Four rows, each a pass or a named failure: the config has limits and defaults for every actuator and the
defaults lie inside the limits; the asset files the config names for that backend exist (locally or on the
Hub); the MJCF loads in MuJoCo's model, the end-effector body is found and forward kinematics of the default
pose is finite (the skill library and the IL adapters depend on this); and holding the default pose for 200
steps leaves every joint within 0.05 rad and a fixed base within 1 cm. `tests/test_roboverse_robot_cfg_validation.py`
runs the config part on every pull request. On 2026-11-28 Franka, YAM and the Sharpa hand pass all four on
MuJoCo, and Franka passes them on Isaac Sim 6 (YAM has no USD, which the assets row says).

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
