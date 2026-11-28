# Evaluate a policy in ten minutes

RoboVerse is an evaluation entry: one evaluator, one result schema, explicit information-access tiers,
and a suite whose every task passed the release gate. This page takes a policy from nothing to a
scored result file.

## 1. Install

```bash
python -m pip install -e "packages/metasim[mujoco]" -e ".[mujoco]"   # from a checkout; MuJoCo is the default engine
python -m metasim doctor                                               # which backends this machine can run
python -m metasim bench list                                           # the registered tasks
```

Assets and demos download from the Hub on first use. Isaac Sim 6 is the second verified engine:
`tools/install/isaacsim6.sh`.

## 2. Run the baselines

```bash
python -m metasim bench run --task pick_cube --sim mujoco --policy replay      # the task's own demos: the ceiling
python -m metasim bench run --task pick_cube --sim mujoco --policy hold        # doing nothing: the floor
python -m metasim bench run --task pick_cube --sim mujoco --policy metasim.bench.skills:ScriptedPolicy
```

Each run writes `bench_results/<task>__<robot>__<sim>__<policy>__<time>.json`: every episode's success,
steps, seed and instance, plus provenance (simulator versions, asset hashes). `bench summarize <files>`
pools them into a table with Wilson intervals.

## 3. Plug in your policy

A policy is three things: a `name`, `reset(env_ids, instances, ctx)` and `act(obs, ctx)` returning one
action per env (a `(num_envs, action_dim)` tensor in the handler's joint order, or per-env
`{robot: {"dof_pos_target": {joint: value}}}` dicts). `ctx` carries the joint names, limits, default
action and instruction; `obs` is the observation the tier allows. Declare `family = "vla"` (or `il`,
`rl`, `agent`) and the result records it.

```python
class MyPolicy:
    name = "my-policy"
    family = "il"

    def reset(self, env_ids, instances, ctx): ...
    def act(self, obs, ctx): ...
```

```bash
python -m metasim bench run --task pick_cube --sim mujoco --policy my_pkg.my_module:MyPolicy --checkpoint ckpt.pt
```

A model that lives in another environment (its own torch, its own CUDA) is served instead; the
harness never imports it:

```bash
python -m metasim bench serve --policy my_pkg.my_module:MyPolicy --checkpoint ckpt.pt --port 8000   # in the model's env
python -m metasim bench conformance --policy-url http://127.0.0.1:8000 --robot franka             # before any episode
python -m metasim bench run --task pick_cube --sim mujoco --policy-url http://127.0.0.1:8000
```

The wire protocol is `roboverse.bench.remote/1` (three routes, JSON and NPZ, no pickle; the module docstring of
`metasim.bench.remote` is the specification), so a server can be written in any language. `bench conformance`
exercises it the way the evaluator will, with no simulator: metadata speaks the protocol, reset accepts the
context, act returns finite actions of the right shape (an action chunk is reported with its horizon), a reset of
one env still answers for every env, a malformed body is refused with a JSON error. It also reports, without
judging, whether the policy is deterministic, whether its actions leave the action box, and the `/act` latency.
`--robot franka` checks against Franka's real action box instead of a synthetic one.

## 4. Say what the policy may see

Two numbers are comparable only when the policies were given the same information. The tier is part
of the protocol and is recorded in every result:

```bash
python -m metasim bench access                                                  # the presets
python -m metasim bench run --task pick_cube --sim mujoco --policy-url ... --access visual --camera front
```

`visual` strips object poses and leaves the camera image and the robot's own joints; `--camera front`
adds a preset camera to a task that configures none. The `agent-*` tiers are for language-model agents
that call tools and record their model calls and tokens.

## 5. Score a whole suite and compare methods

```bash
python -m metasim bench suite run --suite tabletop-mujoco --policy-url http://127.0.0.1:8000 --access visual --camera front
python -m metasim bench matrix bench_results/suites/tabletop-mujoco/*/       # methods x tiers
```

The suite scores task-macro success under one protocol (demo-derived episode caps, seed 0) and refuses
to pool results that were run under different tiers, protocols or simulators. The matrix in
[benchmark usage](../dataset_benchmark/benchmark/usage.md) shows what the rows look like today.

## What to do with the result

Commit the result file with your model's checkpoint hash; it is the unit the leaderboard accepts.
`bench validate <file>` checks it against the schema.
