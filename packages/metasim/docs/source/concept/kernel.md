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
budget that only goes down. As of 1.0.0:

| Area | Reach-ins | Mostly |
|---|---:|---|
| `tasks/libero_90` | 63 | `physics.named.data.site_xpos` / `site_xmat` in success checks |
| `tasks/mjlab` | 59 | `physics` (MuJoCo-native ports for mjlab parity) |
| `tasks/beyondmimic` | 55 | `scene`, `gym`, `_envs` (per-backend randomization) |
| `roboverse_pack/randomization` | 49 | `scene`, `gym`, `physics` (mass and friction per backend) |
| `roboverse_pack/queries` | 44 | backend branches inside queries (expected: a query is where backend code belongs) |
| `tasks/maniskill` | 35 | `object_ids` (SAPIEN friction in the native ports) |
| `tasks/libero` | 21 | `physics` in the native replay port |
| others | 28 | |

The two largest recurring gaps are kernel capabilities still to add: a per-body **physical property**
interface (mass, friction, restitution per env) that today is written four times over, once per
backend, in two content packs; and **site and frame poses** by name, which LIBERO-90 reads from MuJoCo
directly. Each, once on the handler, removes a family's reach-ins and makes its tasks portable.

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
