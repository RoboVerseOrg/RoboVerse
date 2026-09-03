# mjlab integration

mjlab (MuJoCo locomotion / manipulation tasks) reproduced in MetaSim as the `mjlab.*` task family,
with rollout, reward and rendering comparisons against upstream.

| Module | Purpose |
|---|---|
| `runner.py`, `full_runner.py` | run an mjlab task on MetaSim (and, with upstream installed, on mjlab) and compare |
| `metasim_rollout.py`, `raw_rollout.py`, `stand_demo.py` | rollouts of the ported tasks; `g1_walk/` holds the G1 walking reference |
| `reward_sweep.py`, `rewards.py`, `plot.py` | reward-term parity sweeps and plots |
| `render.py`, `render_sweep.py` | rendered comparisons |
| `diagnose_*.py`, `diff.py`, `dump_metasim_xml.py`, `inventory.py` | diagnostics for MJCF patching, geometry and attachment positions |

Run from the repository root, e.g. `python -m tools.mjlab_integration.runner --task mjlab.cartpole_v2`.
The mjlab assets are downloaded from Hugging Face on first use (`roboverse_pack/tasks/mjlab/_locator.py`).
