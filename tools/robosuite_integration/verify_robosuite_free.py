"""Prove the Lift task runs with robosuite UNINSTALLED.

Blocks ``import robosuite`` (simulating the package being deleted), then loads the
fully-native :class:`roboverse_pack.tasks.robosuite.native_lift.NativeLiftEnv`
(vendored MJCF + native OSC controller + native reward/obs/success) and runs a
scripted grasp policy. Reports task success — with zero robosuite in the loop.

Run the one-time vendoring first (while robosuite is still installed):
    python -m tools.robosuite_integration.vendor_assets --env Lift --out .robosuite_bundle/lift
Then:
    python -m tools.robosuite_integration.verify_robosuite_free --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

# --- simulate robosuite being deleted ---
sys.modules["robosuite"] = None

import numpy as np  # noqa: E402


def _grasp_policy(env, t: int) -> np.ndarray:
    """Closed-loop OSC grasp: center over cube, descend, close, lift."""
    eef = env._eef_pos()
    cube = env._cube_pos()
    delta = cube - eef
    horiz = float(np.linalg.norm(delta[:2]))
    a = np.zeros(7)
    over = horiz < 0.015
    a[6] = 1.0 if (over and delta[2] > -0.02) or t > 45 else -1.0  # close once over the cube / late
    if not over:
        a[:2] = np.clip(delta[:2] * 20.0, -1, 1)
        a[2] = np.clip(delta[2] * 20.0, -1, 1) * 0.3
    elif t < 45:
        a[:2] = np.clip(delta[:2] * 20.0, -1, 1)
        a[2] = np.clip(delta[2] * 20.0, -1, 1)
    else:
        a[2] = 1.0  # lift
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=90)
    args = ap.parse_args()

    # confirm robosuite really is unavailable
    try:
        import robosuite  # noqa: F401

        print("WARNING: robosuite is importable — the block did not take effect")
    except Exception as e:
        print(f"robosuite import blocked ({type(e).__name__}); running fully native.")

    from roboverse_pack.tasks.robosuite.native_lift import NativeLiftEnv

    n_succ = 0
    returns = []
    for ep in range(args.episodes):
        env = NativeLiftEnv(seed=ep)
        env.reset()
        ret = 0.0
        success = False
        for t in range(args.steps):
            _, r, done, info = env.step(_grasp_policy(env, t))
            ret += r
            success = success or info["success"]
        n_succ += int(env._check_success())
        returns.append(ret)
        print(f"  episode {ep}: success={env._check_success()} return={ret:.2f}")
        env.close()

    print(
        f"\nNATIVE Lift (robosuite uninstalled): {n_succ}/{args.episodes} success "
        f"({n_succ / args.episodes:.0%}) | mean return {np.mean(returns):.2f}"
    )


if __name__ == "__main__":
    main()
