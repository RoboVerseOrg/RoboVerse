# LIBERO integration

LIBERO / LIBERO+ tasks reproduced 1:1 in MetaSim (bitwise MuJoCo parity) and replayed without a
robosuite dependency.

| Module | Purpose |
|---|---|
| `vendor_native_libero.py`, `vendor_shared_assets.py`, `vendor_all_native.py` | copy the upstream LIBERO MJCF bundles and shared assets into `roboverse_pack/tasks/libero/native_bundles/` |
| `gen_liberoplus.py` | generate the LIBERO+ task variants (`liberoplus_tasks.json`, `liberoplus_cheap_tasks.json`) |
| `libero_replay.py`, `libero_success.py` | replay upstream demonstrations in MetaSim and evaluate the LIBERO success predicates |
| `liberoplus_native_smoke.py`, `verify_liberoplus.py`, `sweep_all.py` | smoke / verification sweeps across the suite |

Run from the repository root, e.g. `python -m tools.libero_integration.vendor_native_libero`.
Requires the `roboverse_data` checkout; upstream LIBERO is only needed for vendoring.
