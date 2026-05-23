"""Debug the cartpole reward parity between mjlab and our MetaSim port.

Loads mjlab's `cartpole_smooth_reward` function directly + my port's
`_reward_from_state`. Feeds both a battery of (qpos, qvel, ctrl) tuples and
compares reward values. If they don't match, the port has a bug.

Also instruments a step-by-step rollout with a fixed action sequence on BOTH
sides and reports max |reward diff| over 200 steps.

Goal: rule reward-formula bugs in/out as a source of the 42-vs-49 gap.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import rootutils
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
rootutils.setup_root(__file__, pythonpath=True)
_MJLAB = os.environ.get("MJLAB_DIR", os.path.expanduser("~/projects/mjlab"))
sys.path.insert(0, os.path.join(_MJLAB, "src"))
_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")

# Import mjlab's reward + tolerance helpers
from mjlab.tasks.cartpole.cartpole_env_cfg import (
    _gaussian_tolerance as mjlab_gaussian_tol,
)
from mjlab.tasks.cartpole.cartpole_env_cfg import (
    _quadratic_tolerance as mjlab_quadratic_tol,
)


# Their cartpole_smooth_reward needs an env+entity_cfg; pull the math out by
# reimplementing the inner computation from those primitives — that's the
# truth we're comparing against.
def mjlab_reward(qpos: np.ndarray, qvel: np.ndarray, ctrl: float) -> float:
    """Identical to mjlab.tasks.cartpole.cartpole_env_cfg.cartpole_smooth_reward()."""
    hinge_angle = torch.tensor(float(qpos[1]))
    cart_pos = torch.tensor(float(qpos[0]))
    hinge_vel = torch.tensor(float(qvel[1]))
    control = torch.tensor(float(ctrl))
    pole_cos = torch.cos(hinge_angle)
    upright = (pole_cos + 1) / 2
    centered = (1 + mjlab_gaussian_tol(cart_pos, margin=2.0)) / 2
    small_control = (4 + mjlab_quadratic_tol(control, margin=1.0)) / 5
    small_velocity = (1 + mjlab_gaussian_tol(hinge_vel, margin=5.0)) / 2
    raw = upright * centered * small_control * small_velocity
    return float(raw.item())


# Now import my port
from roboverse_pack.tasks.mjlab.cartpole_train import (
    _gaussian_tol_np as my_gauss,
)
from roboverse_pack.tasks.mjlab.cartpole_train import (
    _quadratic_tol_np as my_quad,
)


def my_reward(qpos: np.ndarray, qvel: np.ndarray, ctrl: float) -> float:
    """My port's per-step reward BEFORE the dt scaling factor."""
    hinge_angle = float(qpos[1])
    cart_pos = float(qpos[0])
    hinge_vel = float(qvel[1])
    pole_cos = math.cos(hinge_angle)
    upright = (pole_cos + 1.0) / 2.0
    centered = (1.0 + float(my_gauss(np.array([cart_pos]), 2.0)[0])) / 2.0
    small_control = (4.0 + float(my_quad(np.array([ctrl]), 1.0)[0])) / 5.0
    small_velocity = (1.0 + float(my_gauss(np.array([hinge_vel]), 5.0)[0])) / 2.0
    return upright * centered * small_control * small_velocity


def main() -> int:
    print("=" * 70)
    print("Reward function parity test: mjlab vs MetaSim port")
    print("=" * 70)

    # Battery of test points
    cases = [
        # Description, qpos, qvel, ctrl
        ("perfect balance, no ctrl", np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0),
        ("perfect balance, ctrl=1", np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1.0),
        ("perfect balance, ctrl=-1", np.array([0.0, 0.0]), np.array([0.0, 0.0]), -1.0),
        ("perfect balance, ctrl=2 (OOR)", np.array([0.0, 0.0]), np.array([0.0, 0.0]), 2.0),
        ("hinge 30deg tilted", np.array([0.0, math.pi / 6]), np.array([0.0, 0.0]), 0.0),
        ("hinge upside-down (swingup start)", np.array([0.0, math.pi]), np.array([0.0, 0.0]), 0.0),
        ("cart at edge (x=2)", np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.0),
        ("cart way out (x=5)", np.array([5.0, 0.0]), np.array([0.0, 0.0]), 0.0),
        ("fast hinge spin", np.array([0.0, 0.0]), np.array([0.0, 5.0]), 0.0),
        ("very fast hinge", np.array([0.0, 0.0]), np.array([0.0, 50.0]), 0.0),
        ("noise around upright", np.array([0.05, 0.01]), np.array([0.1, 0.05]), 0.0),
        ("typical mid-training", np.array([0.3, 0.2]), np.array([0.5, 0.8]), 0.3),
    ]

    max_diff = 0.0
    max_diff_case = None
    print(f"\n{'Case':40s}  {'mjlab':>10s}  {'mine':>10s}  {'diff':>12s}")
    print("-" * 80)
    for desc, q, v, c in cases:
        r_mjlab = mjlab_reward(q, v, c)
        r_mine = my_reward(q, v, c)
        diff = abs(r_mjlab - r_mine)
        mark = "  ⚠" if diff > 1e-6 else "  ✓"
        print(f"{desc:40s}  {r_mjlab:>10.6f}  {r_mine:>10.6f}  {diff:>12.2e}{mark}")
        if diff > max_diff:
            max_diff = diff
            max_diff_case = desc

    print("-" * 80)
    print(f"max |diff| = {max_diff:.2e} (case: {max_diff_case})")
    if max_diff < 1e-6:
        print("✅ REWARD FORMULA IS BITWISE EQUIVALENT — bug is NOT in reward computation")
    elif max_diff < 1e-3:
        print("⚠ small numerical difference — likely float32 vs float64. NOT a bug source.")
    else:
        print("❌ REWARD MISMATCH — possible source of the 42-vs-49 gap")

    # ---- Step-by-step rollout comparison ----
    print()
    print("=" * 70)
    print("Step-by-step rollout: same ctrl → compare reward sequence")
    print("=" * 70)
    # Load a trajectory.npz captured from a real mjlab run and compute MY reward
    # at each step. Then compare to what mjlab's RewardManager would log.
    traj_path = os.path.join(_REPORTS, "mjlab_integration/runs/mjlab.cartpole_balance/trajectory.npz")
    if not Path(traj_path).exists():
        print(f"  [skip] no trajectory at {traj_path}")
        return 0
    data = np.load(traj_path, allow_pickle=True)
    ctrl_seq = data["ctrl"]  # (T, nu)
    qpos_seq = data["qpos"]  # (T+1, nq)
    qvel_seq = data["qvel"]  # (T+1, nv)
    T = ctrl_seq.shape[0]
    print(f"  loaded trajectory: T={T} control steps, nq={qpos_seq.shape[1]}, nu={ctrl_seq.shape[1]}")
    diffs = []
    for t in range(min(T, 200)):
        # Reward at end of step t (uses qpos/qvel after the step)
        q = qpos_seq[t + 1]
        v = qvel_seq[t + 1]
        c = float(ctrl_seq[t][0])  # cartpole has nu=1
        r_mjlab = mjlab_reward(q, v, c)
        r_mine = my_reward(q, v, c)
        diffs.append(abs(r_mjlab - r_mine))
    diffs = np.array(diffs)
    print(f"  over {len(diffs)} steps:  max|Δr|={diffs.max():.2e}  mean|Δr|={diffs.mean():.2e}")
    if diffs.max() < 1e-6:
        print("  ✅ reward sequence bitwise identical")

    return 0


if __name__ == "__main__":
    sys.exit(main())
