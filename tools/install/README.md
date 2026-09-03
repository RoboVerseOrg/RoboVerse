# tools/install

Scripted, verified install sequences for simulator stacks that cannot be expressed as a single
`pip install` (packages outside PyPI, forced CUDA builds, ordering constraints). Each script is the
documentation: if the docs and the script disagree, the script is what was tested.

| script | stack | Python | verified |
|---|---|---|---|
| `isaacsim5.sh` | Isaac Sim 5.0.0 + Isaac Lab v2.2.1, torch 2.7.0+cu128 | 3.11 | 2026-09-03, RTX 5090 |

Every script ends with `python -m metasim doctor --backend <sim>`; a green doctor is the definition
of "installed". The weekly `backend-compat` workflow reruns the pip-only stacks; the Isaac stacks are
rerun by hand when `metasim/sim/_versions.py` moves.
