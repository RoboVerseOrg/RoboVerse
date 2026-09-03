# RoboTwin integration

RoboTwin (50 bimanual ALOHA tasks) brought into MetaSim on SAPIEN: demonstration conversion, replay,
rendering and policy evaluation.

| Module | Purpose |
|---|---|
| `migrate_assets.py`, `inventory.py` | migrate the RoboTwin assets into `roboverse_data` and enumerate tasks |
| `robotwin_to_demo.py`, `collect_bridge.py`, `collect_demos_robust.sh` | convert / collect demonstrations |
| `aloha_demo.py`, `aloha_render.py`, `mesh_replay_robotwin.py`, `native_render.py`, `sidebyside.py` | replay and rendered comparisons |
| `parity_robotwin.py`, `coverage_sweep.py` | parity harness and coverage sweep |
| `dp_policy_server.py`, `eval_robotwin_policy.py` | serve a diffusion policy and evaluate it on the ported tasks |

Run from the repository root, e.g. `python -m tools.robotwin_integration.aloha_demo`.
Requires `sapien` (extra `sapien3`); the RoboTwin checkout is only needed for migration.
