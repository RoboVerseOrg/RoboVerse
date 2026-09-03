# robosuite integration

robosuite tasks reproduced in MetaSim with a robosuite-free replay path and a parity harness.

| Module | Purpose |
|---|---|
| `vendor_assets.py`, `inventory.py`, `common.py` | vendor the upstream MJCF assets and enumerate tasks |
| `robosuite_rollout.py`, `metasim_rollout.py`, `replay_parity.py`, `benchmark_replay.py` | rollouts on both sides and replay parity |
| `verify_native.py`, `verify_native_controller.py`, `verify_robosuite_free.py` | verify the native task registration, the ported OSC controller and the robosuite-free path |
| `coverage_sweep.py`, `runner.py`, `policies.py`, `render.py`, `plots.py`, `diff.py` | sweeps, scripted policies, rendering and reports |

Run from the repository root, e.g. `python -m tools.robosuite_integration.verify_native`.
Requires `robosuite` only for the upstream side of a parity run.
