# Unified harness — cross-backend parity sweep

**Date:** 2026-07-11 (re-verified 2026-07-13 after the harness defect fixes)
**What:** the `roboverse_learn.eval.harness` `ParityReport` path run over maniskill tasks on
**mujoco + sapien3**, to validate that the harness executes a task *consistently across physics
engines*.

Policy: `ZeroActionPolicy` (baseline — the goal here is infra cross-backend *consistency*, not
policy success; meaningful success-parity needs a trained policy, which needs checkpoints not
available on this box).

## Result

| Task | mujoco | sapien3 | note |
|---|---|---|---|
| maniskill.pick_cube | ran | ran | ✅ both backends |
| maniskill.stack_cube | ran | ran | ✅ both backends |
| maniskill.lift_peg_upright | ran | ran | ✅ both backends |
| maniskill.pull_cube_tool | fail | fail | missing `roboverse_data` traj (pre-existing, both) |
| maniskill.push_cube | fail | fail | missing `roboverse_data` traj (pre-existing, both) |
| maniskill.pick_single_ycb | fail | fail | not a registered task id (sweep name error) |

**Every valid, asset-available task ran identically on both backends (3/3).** The failures are
consistent across both engines and are *not* harness/backend issues: two are missing
`roboverse_data` trajectories (the task's `__init__` downloads a traj even though the harness
doesn't use it), one was a bad task id in the sweep.

Re-run 2026-07-13 (after the harness defect fixes) — the table above reproduces exactly: the three
tasks run on both backends, `pull_cube_tool`/`push_cube` fail identically on both with
`File roboverse_data/trajs/maniskill/<task>/v2 neither exists…`, and `pick_single_ycb` is not a
registered task id.

## Interpretation

- The harness's headline **multi-sim** capability works: one policy, one API, executed
  consistently on MuJoCo and SAPIEN3.
- `ParityReport.success_rate_spread()` / `.divergent()` give a one-number cross-engine
  robustness signal once a real policy is plugged in.
- To turn this into a *success* parity leaderboard, plug a trained policy (or the il/vla
  adapters) into `evaluate(task, policy, simulators=[...])`.

> **Correction (review):** an earlier version of this note claimed the SAPIEN3 run
> "exercises the GPU device path on a real CUDA backend". That claim does not hold — whatever
> this sweep exercised, it was not a CUDA-returning step path. At the time the runner allocated
> its episode bookkeeping (`succ`/`done`/`ep_len`) on CPU while `terminated`/`timeout` came back
> on the sim device, so a backend returning CUDA tensors raised
> `Expected all tensors to be on the same device`.
>
> **Update (2026-07-13):** that defect is **fixed** — the runner now allocates its bookkeeping on
> the sim device and normalizes the step flags onto it, guarded by
> `tests/test_harness_phase1.py::{test_runner_allocates_bookkeeping_on_sim_device,
> test_runner_full_rollout_on_cuda}`. But those tests drive a *fake* backend on a CUDA device: **no
> GPU simulator (mjx/isaacgym/isaacsim/newton) has been run through the harness**, so this sweep
> still says nothing about them. See "Known gaps" in `roboverse_learn/eval/DESIGN.md`.

## Reproduce

Re-run on 2026-07-13 after the harness fixes — `maniskill.pick_cube` still runs on both backends
(`spread = 0.0`, both `success_rate = 0.0`, as expected for a zero-action baseline):

```python
from roboverse_learn.eval.harness import evaluate
from roboverse_learn.eval.harness.adapters import ZeroActionPolicy
rep = evaluate("maniskill.pick_cube", ZeroActionPolicy(), simulators=["mujoco", "sapien3"],
               episodes=1, num_envs=1, max_steps=4)
print(rep.results, rep.success_rate_spread())
```

Run it with `MUJOCO_GL=egl SAPIEN_HEADLESS=1` on a headless box.
