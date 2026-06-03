"""Smoke-test native LIBERO-plus tasks through the MetaSim registry — libero-free.

Blocks ``import libero`` and ``import robosuite``, then for a representative
``libero_plus_native.*`` task per perturbation dimension: resolve it via
``get_task_class``, instantiate (loads the perturbed MJCF on the groundless
MujocoHandler), reset, step the native OSC controller, and confirm the native
BDDL success checker evaluates — proving the perturbed scenes are first-class,
runnable MetaSim tasks with LIBERO + robosuite uninstalled.

    MUJOCO_GL=egl python -m tools.libero_integration.liberoplus_native_smoke
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# hard-block the upstream libraries: the native tasks must not need them
sys.modules["libero"] = None
sys.modules["robosuite"] = None

import numpy as np
import torch

_BUNDLES = Path("roboverse_pack/tasks/libero/native_bundles_plus")


def main() -> None:
    import roboverse_pack.tasks.libero  # noqa: F401  (registers libero_plus_native.*)
    from metasim.task.registry import get_task_class

    # one representative task per dimension
    per_dim: OrderedDict[str, str] = OrderedDict()
    for bdir in sorted(_BUNDLES.iterdir()):
        if not (bdir / "model.xml").exists():
            continue
        dim = bdir.name.split("__", 1)[0]
        per_dim.setdefault(dim, bdir.name)

    results, ok = [], 0
    for dim, name in per_dim.items():
        task_id = f"libero_plus_native.{name}"
        try:
            env = get_task_class(task_id)()
            obs, _ = env.reset()
            for t in range(8):
                a = torch.zeros(7)
                a[0] = 0.1 * np.sin(t / 3.0)
                a[6] = 1.0 if t >= 4 else -1.0
                obs, r, term, timeout, extra = env.step(a)
            succ = bool(extra["success"])
            env.close()
            ok += 1
            rec = {"dim": dim, "task": task_id, "obs_dim": int(obs.shape[-1]), "stepped": True, "success_checker": succ}
            print(f"  [{dim:9s}] {task_id[:60]:60s} obs={int(obs.shape[-1])} step=OK checker={succ}", flush=True)
        except Exception as e:
            rec = {"dim": dim, "task": task_id, "stepped": False, "error": str(e)[:160]}
            print(f"  [{dim:9s}] {task_id[:60]:60s} ERROR {str(e)[:70]}", flush=True)
        results.append(rec)

    n = len(per_dim)
    print(f"\nLIBERO-plus NATIVE smoke (libero+robosuite blocked): {ok}/{n} dims instantiate+step")
    print("registered libero_plus_native tasks:", sum(1 for b in _BUNDLES.iterdir() if (b / "model.xml").exists()))
    out = Path("research_report/projects/roboverse/libero_1to1/metasim_1to1/liberoplus_native.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dims_ok": f"{ok}/{n}", "tasks": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
