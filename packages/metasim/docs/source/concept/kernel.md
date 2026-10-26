# The MetaSim kernel

RoboVerse ports more than twenty benchmark families onto MetaSim: ManiSkill, LIBERO and LIBERO-90,
RLBench, CALVIN, mjlab, MuJoCo Playground, HumanoidBench, SimplerEnv, RoboTwin, Meta-World,
robosuite, MyoSuite, the dm_control suite and others. The kernel is what those ports have in common:
the smallest set of concepts a port needs, measured across all of them rather than designed up front.
`metasim.kernel` re-exports exactly that set, and importing it loads no simulator backend.

```python
from metasim.kernel import ScenarioCfg, RobotCfg, RigidObjCfg, TensorState, RLTaskEnv, register_task
```

## Five concepts

| Concept | Types | Role |
|---|---|---|
| Scenario | `ScenarioCfg`, `RobotCfg`, `RigidObjCfg`, `ArticulationObjCfg`, primitives, `PinholeCameraCfg`, `SimParamCfg` | What exists. Every entity a task reads is declared here. |
| State | `TensorState` (`RobotState`, `ObjectState`, `CameraState`, `extras`) | The one data type between a simulator and content. |
| Handler | `BaseSimHandler`, `BaseQueryType` | A backend in five verbs: `launch`, `set_states(states, env_ids)`, `set_dof_targets(actions)`, `simulate()`, `get_states(env_ids)`; plus joint/body names, capability flags and queries for what a state does not carry. |
| Task | `BaseTaskEnv`, `RLTaskEnv`, the checkers, `register_task` / `get_task_class` | One loop with hooks; success as a predicate over the state; addressable by name. |
| Protocol | `EvalProtocol`, `evaluate`, `Policy`, `BenchmarkSuite` | How a task is scored, kept out of the task (the RoboVerse docs' *Benchmark usage* page covers the CLI). |

Everything else in MetaSim (domain randomization, renderers, viewers, teleoperation, IK, asset
conversion) is built on these five and is optional to a port.

## The task loop

`BaseTaskEnv.step` and `RLTaskEnv.step` run one sequence; a task changes it through hooks:

```text
step(actions)
  _episode_steps += 1
  actions = _process_action(actions)      # delta control, unnormalisation, IK
  pre_physics_step_callback(actions)
  _apply_action(actions)                  # default handler.set_dof_targets; RLTaskEnv clips to joint limits
  handler.simulate()
  states = handler.get_states()
  post_physics_step_callback(states)
  _on_step(states)                        # commands, latches, grasp state, history
  _reward / _terminated / _time_out / _success
  [RLTaskEnv] reset(done env ids)

reset(states=None, env_ids=None, seed=None)
  handler.set_seed(seed); _episode_steps[env_ids] = 0; reset_callback(env_ids)
  handler.set_states(states or initial states, env_ids)
  _on_reset(env_ids)                      # checkers, latches, per-env buffers
```

| Hook | Override it to | Default |
|---|---|---|
| `_process_action(actions)` | map a policy action to a command (deltas, `[-1, 1]` to joint range, IK) | identity |
| `_apply_action(actions)` | write the command some other way (effort, a subset of actuators) | `set_dof_targets` (clipped under `RLTaskEnv`) |
| `_on_step(states)` | keep per-step state the reward or termination reads | nothing |
| `_on_reset(env_ids)` | reset checkers and per-env buffers after the new states are written | nothing |
| `_reward`, `_terminated`, `_success`, `_time_out`, `_observation` | define the task | zeros / state |

Override these instead of `step` or `reset`. A copied loop has drifted from the base before: `RLTaskEnv`
once ran no step callbacks at all, and a checker reset placed before `super().reset()` placed its
detection region from the *previous* episode's object pose. `tests/test_task_reset_seed_contract.py`
lists the `step` overrides that remain (native ports and controllers that run inside the step) and
blocks new ones.

## The handler boundary

Content (tasks, randomizers, queries in `roboverse_pack`, learning code in `roboverse_learn`) reads
the simulator through `BaseSimHandler`'s public surface. Reading past it (`handler.physics` on
MuJoCo, `handler.gym` / `_envs` on Isaac Gym, `handler.scene` on Isaac Sim, `object_ids` on SAPIEN)
ties a task to one backend, which is how a task that declares two simulators ends up working on one.

`tests/test_kernel_boundary.py` counts those reach-ins per content area and holds each area to a
budget that only goes down. As of 1.0.0 (LIBERO-90's 63 site reads are gone; see below):

| Area | Reach-ins | Mostly |
|---|---:|---|
| `tasks/mjlab` | 59 | `physics` (MuJoCo-native ports for mjlab parity) |
| `tasks/beyondmimic` | 55 | `scene`, `gym`, `_envs` (per-backend randomization) |
| `roboverse_pack/randomization` | 49 | `scene`, `gym`, `physics` (mass and friction per backend) |
| `roboverse_pack/queries` | 44 | backend branches inside queries (expected: a query is where backend code belongs) |
| `tasks/maniskill` | 35 | `object_ids` (SAPIEN friction in the native ports) |
| `tasks/libero` | 21 | `physics` in the native replay port |
| others | 28 | |

The largest recurring gap still open is a per-body **physical property** interface (mass, friction,
restitution per env), which today is written four times over, once per backend, in two content packs.

## Sites: frames derived from the state

LIBERO's success predicates are defined on MJCF *sites*: a basket's `contain_region`, a drawer's
`top_region`, a stove's `cook_region`. A site is fixed to one body, so its world pose is that body's
pose composed with a constant offset. `metasim.utils.site_util.site_pose(states, obj, site)` computes it
from the `TensorState` and the offset parsed once from the object's MJCF: the root pose for a rigid
object, the pose of the site's nearest named body (from `body_names` / `body_state`) for an
articulation, with the object's `scale` applied as the MuJoCo handler applies it.

```python
from metasim.utils.site_util import site_pose

region = site_pose(states, basket_cfg, "contain_region")  # SitePose: pos (N, 3), quat (N, 4), rot (N, 3, 3)
local = (region.rot.transpose(1, 2) @ (obj_pos - region.pos).unsqueeze(-1)).squeeze(-1)
```

It needs no handler query, so a site check runs on every backend that reports the body. Against
MuJoCo's own `site_xpos` / `site_xmat` it agrees to 1e-16 on the LIBERO assets (float64 offsets), and
to 1.4e-8 (position) / 2.3e-7 (rotation) on 1149 LIBERO-90 demo states, the float32 rounding of the
state; the LIBERO-90 checkers give the same verdict on all of those states as the `physics` reads they
replaced. `include`, `frame`, default classes and `xyaxes` / `zaxis` frames are rejected with an error
rather than misread.

## What a port needs, measured

Counting the MetaSim imports across 31 content areas (the task families in `roboverse_pack/tasks`, the other `roboverse_pack` subpackages and `roboverse_learn`):

| Used by | Names |
|---|---|
| 12 to 18 families | `ScenarioCfg`, `register_task`, `BaseTaskEnv`, `RigidObjCfg`, `PhysicStateType` |
| 6 to 10 families | `PrimitiveCubeCfg`, `ArticulationObjCfg`, `TensorState`, `SimParamCfg`, `RLTaskEnv`, `configclass` |
| 4 to 5 families | `PinholeCameraCfg`, `RobotCfg`, `BaseActuatorCfg`, the checkers (`DetectedChecker`, `RelativeBboxDetector`) |

The checkers were the one kernel concept that lived outside the core (in the example pack, imported
by 33 task files); they are now `metasim.task.checkers`, and the old path is an alias of the same
module.
