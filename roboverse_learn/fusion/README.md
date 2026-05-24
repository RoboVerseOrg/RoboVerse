# roboverse_learn.fusion — IL + RL workflow fusion

A thin, additive bridge that lets RoboVerse's reinforcement-learning
(`roboverse_learn/rl`) and imitation-learning (`roboverse_learn/il`) halves
share **one task/env** and **one demonstration format**, closing the loop in
both directions. Nothing in `rl/` or `il/` is modified; this package only adds
glue and reuses the existing entry points.

```
                ┌──────────────────────── RL ────────────────────────┐
   task + reward│   roboverse_learn/rl/rsl_rl/ppo.py  →  policy.pt     │
                └──────────────────────────┬──────────────────────────┘
                                           │  fusion.rl_to_demo
                                           ▼  (rollout + save_demo)
                         roboverse_demo/<task>/success/demo_XXXX/
                              {metadata.json, rgb.mp4, ...}     ← canonical demo
                                           │  il/data2zarr_dp.py (unchanged)
                                           ▼
                              data_policy/<task>_<N>.zarr
                                           │  il/train.py (unchanged)
                                           ▼
                ┌──────────────────────── IL ────────────────────────┐
                │   diffusion / ACT / BC policy                       │
                └──────────────────────────┬──────────────────────────┘
                                           │  fusion.bc_warmstart
                                           ▼  (copy MLP actor weights)
                         rsl-rl ActorCritic actor  →  RL fine-tune
```

## RL → IL: collect demos from a trained policy

`rl_to_demo.collect_demos_from_policy` rolls out a trained rsl-rl policy in a
RoboVerse task (built **with a camera**, since RL trains camera-less) and writes
each successful episode in the *exact* layout the IL pipeline already ingests,
via `metasim.utils.save_util.save_demo`. No new dataset format is introduced.

```bash
python -m roboverse_learn.fusion.collect \
    --task mjlab.lift_cube_yam_v2 --robot yam --sim mujoco \
    --checkpoint outputs/rsl_rl_ppo/.../model_499.pt \
    --out_dir roboverse_demo/lift_cube_yam --num_demos 100
# then the standard IL flow, unchanged:
python roboverse_learn/il/data2zarr_dp.py --task_name lift_cube_yam \
    --metadata_dir roboverse_demo/lift_cube_yam/success --expert_data_num 100
python roboverse_learn/il/train.py ...
```

Supported checkpoint formats (auto-detected): TorchScript `policy.pt`, a raw
`actor_state_dict` / bare `mlp.*` dict, or an rsl-rl `model_*.pt` runner
checkpoint. For continuous locomotion (no success checker) pass `--keep_all`.

**Task scope.** Demo collection targets *standard manipulation tasks* whose
state exposes the robot as a `RobotCfg` plus objects and a camera (the same
tasks `il/data2zarr_dp.py` consumes, e.g. `StackCube_franka`, `lift_*`). The
mjlab manager tasks (cartpole / velocity g1/go1) load their articulation as a
**scene MJCF**, so `handler.get_states().robots` is empty and `save_demo` has no
robot to serialise — the collector detects this and raises a clear error rather
than producing a broken demo. Those locomotion policies are evaluated with the
RL eval/parity tooling instead.

## One command: the whole loop

`pipeline.py` chains the existing entry points (it shells out to each; it does
not reimplement any trainer). `--dry_run` prints the exact commands without
running them.

```bash
# already have an RL checkpoint -> collect + to-zarr + il-train:
python -m roboverse_learn.fusion.pipeline --task mjlab.lift_cube_yam_v2 \
    --name lift --checkpoint outputs/lift/policy.pt --num_demos 100
# or train the RL policy first too:
python -m roboverse_learn.fusion.pipeline --task mjlab.lift_cube_yam_v2 --name lift \
    --stages rl-train,collect,to-zarr,il-train --rl_iterations 1500
```

## IL → RL: warm-start an RL actor from a BC policy

`bc_warmstart.load_bc_into_actor_critic` copies a behaviour-cloning MLP's Linear
weights into an rsl-rl `ActorCritic` actor (matched by shape, in order; the
critic is left for RL to learn). `extract_actor_mlp_state_dict` does the reverse
(seed BC from a trained RL actor). State-MLP policies only — image/diffusion/ACT
policies have no weight-compatible trunk and should use the demo bridge instead.

```python
import torch
from rsl_rl.modules import ActorCritic
from roboverse_learn.fusion import load_bc_into_actor_critic

ac = ActorCritic(...)                       # fresh RL actor-critic
bc_sd = torch.load("bc_policy.pt")          # a state-MLP BC policy
n = load_bc_into_actor_critic(ac, bc_sd)    # actor now == BC; fine-tune with PPO
```

## Tests

* `tests/test_fusion_demo_contract.py` — synthesises a rollout, runs the real
  `data2zarr_dp.py`, and asserts the zarr is IL-consumable (no simulator).
* `tests/test_fusion_bc_warmstart.py` — IL↔RL weight copy reproduces outputs
  bit-for-bit and fails fast on shape/depth mismatch.

## Scope

This is the *workflow* fusion (shared env + shared demo format + bidirectional
warm-start). It deliberately does **not** unify the two config systems
(RL=tyro `@configclass`, IL=Hydra) or merge the trainers — duplication there is
cheaper than the wrong abstraction (see `AGENTS.md`).
