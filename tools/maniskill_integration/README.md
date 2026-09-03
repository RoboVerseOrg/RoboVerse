# ManiSkill integration

ManiSkill3 tasks brought into MetaSim on SAPIEN, with a parity harness that compares observations,
rewards and rollouts against upstream.

| Module | Purpose |
|---|---|
| `recipe.py`, `inventory.py` | the ManiSkill sim/control recipe (`sim_freq`, `control_freq`, decimation) and the task inventory used by `roboverse_pack/tasks/maniskill/_native` |
| `parity_native.py`, `parity_multi_agent.py` | run a task on ManiSkill and on MetaSim from the same seed and report the per-step deltas |
| `maniskill_rollout.py`, `metasim_rollout.py`, `replay_demo.py`, `render_demo_replay.py` | rollouts and demonstration replay on either side |
| `render_compare.py`, `render_parity.py`, `render_passthrough.py`, `run_sweep.py` | rendered side-by-side galleries and sweeps |

Run from the repository root, e.g. `python -m tools.maniskill_integration.parity_native --task pick_cube`.
Requires `mani_skill` and `sapien` (extra `sapien3`) for the upstream side.
