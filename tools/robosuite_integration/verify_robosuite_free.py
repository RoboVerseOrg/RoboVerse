"""Prove the robosuite tasks run with robosuite UNINSTALLED.

Blocks ``import robosuite`` (simulating the package being deleted), then loads the
fully-native task envs (``roboverse_pack.tasks.robosuite.native``) — vendored MJCF
+ native OSC controller + native reward/obs/success — and:

* Lift / PickPlaceCan / NutAssemblySquare: replays the OFFICIAL robomimic benchmark
  demo *actions* through the native controller from each demo's initial state, and
  reports native task-success (these have official datasets).
* Lift also: a scripted closed-loop grasp (no dataset needed).
* Stack / Door: smoke-run (env loads, steps, native success computes) — no public
  benchmark dataset exists for these, so closed-loop success isn't asserted here.

Vendor the bundles first (while robosuite is installed):
    for t in Lift Stack Door PickPlaceCan NutAssemblySquare; do
      python -m tools.robosuite_integration.vendor_assets --env $t --out .robosuite_bundle/${t,,}; done
Then:
    python -m tools.robosuite_integration.verify_robosuite_free --n-demos 10
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.modules["robosuite"] = None  # simulate robosuite deleted

import h5py
import numpy as np

_DATASETS = {
    "Lift": "datasets/lift_ph_low_dim_v141.hdf5",
    "PickPlaceCan": "datasets/can_ph_low_dim_v141.hdf5",
    "NutAssemblySquare": "datasets/square_ph_low_dim_v141.hdf5",
}


def _lift_grasp_policy(env, t):
    eef, cube = env._eef_pos(), env._body_pos(env._cube)
    delta = cube - eef
    a = np.zeros(7)
    over = np.linalg.norm(delta[:2]) < 0.015
    a[6] = 1.0 if (over and delta[2] > -0.02) or t > 45 else -1.0
    if not over:
        a[:2] = np.clip(delta[:2] * 20, -1, 1)
        a[2] = np.clip(delta[2] * 20, -1, 1) * 0.3
    elif t < 45:
        a[:2] = np.clip(delta[:2] * 20, -1, 1)
        a[2] = np.clip(delta[2] * 20, -1, 1)
    else:
        a[2] = 1.0
    return a


def replay_benchmark(env_name: str, n_demos: int) -> dict:
    from roboverse_pack.tasks.robosuite.native import NATIVE_ENVS

    f = h5py.File(_DATASETS[env_name], "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[:n_demos]
    n_succ = 0
    for dm in demos:
        states = f["data"][dm]["states"][()]
        actions = f["data"][dm]["actions"][()]
        env = NATIVE_ENVS[env_name](seed=0)
        nq, nv = len(env._arm_qpos) and env.m.nq, env.m.nv
        env.set_full_state(states[0][1 : 1 + env.m.nq], states[0][1 + env.m.nq : 1 + env.m.nq + env.m.nv])
        succ = False
        for a in actions:
            _, _, _, info = env.step(a)
            succ = succ or info["success"]
        n_succ += int(env._check_success())
        env.close()
    f.close()
    return {"task": env_name, "n": len(demos), "success": n_succ, "rate": n_succ / len(demos)}


def smoke(env_name: str) -> dict:
    from roboverse_pack.tasks.robosuite.native import NATIVE_ENVS

    env = NATIVE_ENVS[env_name](seed=0)
    env.reset()
    ok = True
    for _ in range(20):
        try:
            env.step(np.zeros(7))
        except Exception as e:
            ok = False
            print(f"   {env_name} step error: {e}")
            break
    s = env._check_success()
    env.close()
    return {"task": env_name, "loads_and_steps": ok, "success_callable": isinstance(s, bool)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=10)
    args = ap.parse_args()
    try:
        import robosuite  # noqa: F401

        print("WARNING: robosuite importable — block failed")
    except Exception as e:
        print(f"robosuite import blocked ({type(e).__name__}); running fully native.\n")

    print("== official benchmark action-replay through NATIVE controller (robosuite uninstalled) ==")
    for t in ("Lift", "PickPlaceCan", "NutAssemblySquare"):
        r = replay_benchmark(t, args.n_demos)
        print(f"  {t:18s}: native success {r['success']}/{r['n']} ({r['rate']:.0%})")

    print("\n== scripted closed-loop (no dataset needed) ==")
    from roboverse_pack.tasks.robosuite.native import NativeLiftEnv

    ns = 0
    for ep in range(args.n_demos):
        env = NativeLiftEnv(seed=ep)
        env.reset()
        for t in range(90):
            env.step(_lift_grasp_policy(env, t))
        ns += int(env._check_success())
        env.close()
    print(f"  Lift scripted grasp: {ns}/{args.n_demos} success")

    print("\n== smoke (no public dataset) ==")
    for t in ("Stack", "Door"):
        r = smoke(t)
        print(f"  {t:18s}: loads+steps={r['loads_and_steps']} success_callable={r['success_callable']}")


if __name__ == "__main__":
    main()
