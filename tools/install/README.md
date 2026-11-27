# tools/install

Scripted, verified install sequences for simulator stacks that cannot be expressed as a single
`pip install` (packages outside PyPI, forced CUDA builds, ordering constraints). Each script is the
documentation: if the docs and the script disagree, the script is what was tested.

| script | stack | Python | verified |
|---|---|---|---|
| `isaacsim6.sh` | Isaac Sim 6.0.1 + Isaac Lab release/3.0.0, torch 2.11.0+cu130 | 3.12 | 2026-11-27, RTX 5090 (driver 595) |

Every script ends with `python -m metasim doctor --backend <sim>`; a green doctor is the definition
of "installed". The weekly `backend-compat` workflow reruns the pip-only stacks; the Isaac stacks are
rerun by hand when `metasim/sim/_versions.py` moves.
