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
