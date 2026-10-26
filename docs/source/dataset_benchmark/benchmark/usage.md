# Benchmark Usage

RoboVerse scores a policy with **one evaluation loop, one scoring protocol and one result
format**, whichever task, robot or simulator is involved. The pieces live in
`metasim.bench` (`packages/metasim/metasim/bench/`) and are driven by `python -m metasim bench`.

## The contract

| Piece | Where | What it fixes |
|---|---|---|
| Task success | `info["success"]` from every `env.step` (`BaseTaskEnv.success_is_termination`, `_success`) | A task reports whether its success predicate holds *now*. For manipulation tasks that is the same signal as `terminated`; an RL task whose early termination means failure reports success separately. |
| Instances | `metasim.bench.instance_pool` / `TaskInstance` | The initial conditions a run is scored on: the task's demo file (`traj_filepath`) when it has one, else its own initial states, else the default reset. Each instance has a content-addressed id (`index:seed:hash`). |
| Protocol | `metasim.bench.EvalProtocol` | Episode count, scoring rule (`first` / `final` / `held` with `hold_steps`), episode cap, seed, explicit instance list. Written into every result. |
| Policy | `metasim.bench.Policy` (`reset(env_ids, instances, ctx)`, `act(obs, ctx)`) | The harness owns the loop; a policy maps observations to actions for all env slots. `hold`, `random` and `replay` ship as baselines. |
| Result | `metasim.bench.BenchmarkResult` (`roboverse.bench.result`, schema version 1) | One JSON per run: task, robots, simulator, protocol, policy (with checkpoint hash), one row per episode, provenance (backend versions, time base, asset hashes, git commit), summary with a 95% Wilson interval. |

## Command line

```bash
# registered tasks (no simulator is imported)
python -m metasim bench list --filter pick

# score the demo-replay baseline on one task, one robot, one backend
python -m metasim bench run --task stack_cube --sim mujoco --robot franka \
    --policy replay --episodes 50 --success-rule first --out results/stack_cube_replay.json

# a policy of your own: an import path to a class or factory satisfying metasim.bench.Policy
python -m metasim bench run --task stack_cube --sim mujoco --policy my_pkg.eval:MyPolicy \
    --checkpoint ckpt/policy.pt --policy-arg device=cuda --episodes 50 --seed 0

# pool result files into a leaderboard table; check files against the schema
python -m metasim bench summarize results/
python -m metasim bench validate results/*.json
```

`bench run` builds the task from its registry name, applies `--sim`, `--robot`, `--num-envs`
and headless mode to a copy of the task's scenario, plans `--episodes` instances round-robin from
the task's instance pool, and refuses nothing silently: a task without a class-level scenario, an
unknown policy spec, an instance index outside the pool or a robot the demo file has no entry for
is an error. Episode `i` is planned with seed `--seed + i`; because a simulator's RNG is one per
handler, the episodes that start in the same reset call share the lowest of their seeds, and every
`EpisodeResult.seed` records the seed that reset was actually given (with `--num-envs 1` that is
exactly `seed + i`). Results from different seeds or machines are comparable when task, robots,
simulator and protocol agree; `summarize` pools exactly those and keeps hand-picked `--instance`
runs apart.

## Trained policies

The learning stacks ship adapters that satisfy `metasim.bench.Policy`, so a checkpoint is scored by
the same harness, protocol and result schema as the baselines:

```bash
# a FastTD3 actor (roboverse_learn/rl/fast_td3): the checkpoint carries its training config
python -m metasim bench run --task <rl_task> --sim mujoco --num-envs 16 \
    --policy roboverse_learn.rl.fast_td3.bench_policy:FastTD3Policy --checkpoint runs/<task>/final.pt

# a diffusion-policy checkpoint (roboverse_learn/il): the runner is built from the config stored in it
python -m metasim bench run --task stack_cube --sim mujoco --robot franka \
    --policy roboverse_learn.il.runners.bench_policy:ILPolicy --checkpoint outputs/<run>/latest.ckpt \
    --policy-arg camera=camera0
```

`--checkpoint` reaches the adapter as its `checkpoint` argument and its SHA-256 lands in the result;
`--policy-arg key=value` passes any other constructor argument. The adapters wrap the evaluators'
own code paths (`actor(obs_normalizer(obs))`; `BaseEvalRunner.get_action` on `{rgb, joint_qpos}`),
so a number here is the number the stack's evaluator would produce under the same protocol. The
`roboverse_learn.il` and VLA evaluation scripts keep their own loops for now (domain randomisation,
videos); they are the next to converge on the harness.

## Scoring rules

- `first` (default): the episode counts as solved if the predicate held at any step.
- `final`: solved if it held at the episode's last step.
- `held`: solved if it held for `--hold-steps` consecutive steps at some point (the rule
  SimplerEnv-style tasks use with five steps).

A time-out, whether the task's `max_episode_steps` or the evaluator's `--max-steps` cap, is never a
success by itself; the result records `terminated`, `time_out`, `steps` and the first step the
predicate held, so a reader can re-score under another rule.

## Baselines every task should pass

- `replay` replays the task's recorded demos from their own initial states. On the backend the
  demos were recorded on this is the task's ceiling and a health check: a low replay success rate
  means the task, its assets or its checker are broken, not the policy. On another backend it
  measures how far the demos transfer.
- `hold` commands the default joint positions and must score near zero.
- `random` draws uniform joint targets inside the limits (seeded from the protocol).

The health gate runs the replay baseline over every task that ships demos and classifies each one:

```bash
python -m metasim bench health --sim mujoco --episodes 5 --threshold 0.8      # every task with demos
python -m metasim bench health --sim mujoco --task stack_cube --task pick_cube  # a few tasks
```

| status | meaning |
|---|---|
| `ok` / `low` | replay success at / below `--threshold` |
| `undefined` | the task declares no success signal |
| `unsupported` | the task does not run on `--sim` (`supported_simulators`) or has no class-level scenario |
| `no-demos` | no demo file (skipped in the default sweep, reported for an explicit `--task`) |
| `error` | construction or evaluation raised; the first line of the error is the detail |

It writes the ordinary result files, `health.json` and `health.md` to `--out-dir` and exits 1 when any
task is `low` or `error`, so it can guard a release.

## Suites

A suite fixes a task list, in splits, under one protocol, and scores it **task-macro**: the mean of
the per-task success rates, so a task with many episodes does not drown out the others. A split
with a task that has no score, because it crashed or was not run, is reported as `INCOMPLETE` rather
than averaged over the tasks that happened to run.

```bash
python -m metasim bench suite list
python -m metasim bench suite run --suite tabletop-mujoco --policy replay            # every split
python -m metasim bench suite run --suite tabletop-mujoco --policy-url http://127.0.0.1:8000 --split core
python -m metasim bench suite report --suite tabletop-mujoco bench_results/suites/tabletop-mujoco/mujoco
```

`suite run` writes one result file per task plus `suite.json` / `suite.md`, and exits 1 unless every
task scored. Suites are content. RoboVerse registers them in `roboverse_pack/suites/`, and another
pack does the same in its own `<root>.suites` package, through the `metasim.suites` entry point
group or `METASIM_SUITE_PACKAGES`:

```python
from metasim.bench import BenchmarkSuite, EvalProtocol, SuiteTask, register_suite

register_suite(BenchmarkSuite(
    name="my-suite",
    simulator="mujoco",
    splits={"core": ("stack_cube", "pick_cube"), "heldout": (SuiteTask(task="push_cube", robots=("yam",)),)},
    protocol=EvalProtocol(episodes=50, demo_budget_factor=3.0),
))
```

| suite | backend | splits | tasks | replay baseline (3 episodes per task, 2026-10-26) |
|---|---|---|---|---|
| `tabletop-mujoco` | MuJoCo | `core` | pick_cube, stack_cube, push_cube, pull_cube, lift_peg_upright | task-macro 0.933 (pick_cube 2/3: its demo 0 fails on MuJoCo) |

Only tasks whose demos replay on the suite's backend belong in a suite; `bench health` is the gate.

## Episode budgets and instructions

- **Demo budget.** `--demo-budget 3` (`EvalProtocol.demo_budget_factor`) caps each episode at 3x the
  75th-percentile demo length of the task (`--demo-budget-quantile`). Long tasks get the time a
  demonstrator needed, and short ones are not padded. The budget never lengthens a task's own time-out:
  when it is longer, the task's cap applies, and the result's `notes.episode_cap_source` says which one
  ended the episodes. A task without demos keeps its own cap.
- **Instructions.** A task may declare a `prompt_bank`
  (`metasim.bench.PromptBank(canonical=..., train=(...), heldout=(...))`). `--prompt-split` picks one
  of three modes. `canonical` (the default) uses the task's own text. `train` uses the canonical text
  20% of the time and a training paraphrase otherwise. `heldout` uses a paraphrase that `train` never
  produces. The draw is a hash of the task, episode and seed, so it is reproducible and independent
  of the simulator. A policy reads it as `instance.prompt` in `reset`, and the result records
  `prompt` / `prompt_variant` per episode. A task with only `task_desc` supports `canonical` and
  `train`, and refuses `heldout`.

## Served policies

A policy whose dependencies do not install next to the simulator (a VLA pinned to its own torch or
jax) runs in its own environment behind `bench serve`, and the evaluator scores it over HTTP:

```bash
# in the policy's environment
python -m metasim bench serve --policy my_pkg.policy:MyPolicy --checkpoint ckpt.pt --port 8000
# in the simulator's environment
python -m metasim bench run --task stack_cube --sim mujoco --policy-url http://127.0.0.1:8000 \
    --require-metadata checkpoint=ckpt.pt --obs-key robots --obs-key cameras/front/rgb
```

The wire protocol (`roboverse.bench.remote/1`) is plain HTTP with NumPy NPZ packets, loaded without
pickle. `GET /metadata` is a handshake whose fields go into the result's policy record.
`--require-metadata` fails the run before the first episode when they differ. `POST /reset` carries
the `PolicyContext` and the starting instances, with their prompts. `POST /act` sends every array leaf
of the observation and gets back actions of shape `(num_envs, action_dim)`, or
`(num_envs, horizon, action_dim)` for an action chunk, which the client executes step by step. The
server binds `127.0.0.1` unless `--host` says otherwise. It answers one request at a time, and the
policy sees observations as nested mappings of tensors with attribute access. The simulator's
scenario does not cross the wire, so `ctx.extras` is empty on the server.

The suite and split structure, the demo-length budget and the held-out instruction split follow the
evaluation design of [HandsOn](https://github.com/TsinghuaC3I/HandsOn) (MIT). They are reimplemented
on RoboVerse's own protocol; see `THIRD_PARTY_NOTICES.md`.

## Embodiments

Results are reported per robot. Three embodiments are the reference set every task family and
every backend is measured against (`ROADMAP.md` §4):

| `--robot` | Robot | Asset | Joints | Notes |
|---|---|---|---|---|
| `franka` | Franka Emika Panda | `roboverse_data/robots/franka` (MJCF, URDF, USD) | 9 | two finger joints, both driven |
| `yam` | I2RT YAM arm + parallel-jaw gripper | `roboverse_data/robots/Yet_Another_Manipulator_YAM/mjcf` (Menagerie `i2rt_yam`, MIT) | 8 | `left_finger` driven, `right_finger` passive (equality-coupled) |
| `sharpa_wave_right` / `sharpa_wave_left` | Sharpa Wave tactile hand | `roboverse_data/robots/sharpa_wave/mjcf` (Menagerie `sharpa_wave`, Apache-2.0) | 22 | one servo per joint; fetched from the pinned Menagerie revision until mirrored |

What a robot config promises, and what the MuJoCo and Newton handlers enforce:

- **Joints.** `joint_limits` and `default_joint_positions` list every joint of the asset, in any
  order; the handler's joint vector is the asset's joints sorted by name, and that is the layout of
  `joint_pos`, `joint_pos_target` and the action tensor.
- **Actuators are named after their joint.** `RobotCfg.actuators` is keyed by joint name. A joint the
  MJCF actuates keeps the asset's servo unless the config sets `stiffness` / `damping`; a joint the
  MJCF does not actuate gets a servo synthesized from the config (so Menagerie and mjlab files that
  ship no `<actuator>` block work as-is). A joint in neither place is **passive**: it is observed, it
  owns a column of the action tensor that is ignored, and its target reports its position.
- **Gripper.** `gripper_open_q` / `gripper_close_q` give the driven finger joint(s) their open and
  closed targets; `ee_body_name` (+ `curobo_tcp_rel_pos`) names the grasp frame for the IK helpers.
- **Assets.** Paths are `roboverse_data/...`; missing files are fetched from the HuggingFace data
  repos, or from an upstream a pack registered with `metasim.utils.hf_util.register_upstream_mirror`.

`tests/test_embodiments.py` runs this contract for the three robots on MuJoCo (single env and the
parallel wrapper) and evaluates the `hold` baseline on YAM through `metasim.bench.evaluate`.

## From Python

```python
from metasim.bench import EvalProtocol, evaluate, load_policy
from metasim.task.registry import get_task_class

cls = get_task_class("stack_cube")
env = cls(cls.scenario.replace(simulator="mujoco", num_envs=1, headless=True), "cpu")
result = evaluate(env, load_policy("replay"), EvalProtocol(episodes=10, seed=0), task_name="stack_cube")
print(result.summary)          # {'episodes': 10, 'successes': ..., 'success_rate_ci95': [...], ...}
result.save("results/stack_cube_replay.json")
```

## Domain randomization

Generalization levels are configured through `metasim.randomization.dr_manager.DRConfig`
(`level`: 0 none, 1 scene and material, 2 adds light, 3 adds camera; `scene_mode`;
`randomization_seed`) and applied by `DomainRandomizationManager`; the imitation-learning runners
take them as `--dr_level` / `--dr_scene_mode` / `--dr_seed`. Record the level you evaluated at in
the result's `notes` until the protocol carries it as a field.
