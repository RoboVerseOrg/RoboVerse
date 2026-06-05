"""Verify the native MetaSim port behaves identically to robosuite.

Steps a fresh native robosuite env and the registered ``robosuite.<task>``
MetaSim env through the *same* deterministic action sequence from the *same*
seed, then compares per-step reward, success and final qpos. If the engine and
the controller/reward/obs reuse are faithful, every quantity matches to machine
precision — which means any policy scores identically on both.

Usage:
    python -m tools.robosuite_integration.verify_native \
        --out <report>/runs --tasks Lift Stack Door PickPlaceCan NutAssemblySquare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import numpy as np

from .inventory import ALL_TASKS, get


def _scripted_policy(action_dim: int):
    """Deterministic, non-trivial probe: press down + close gripper, then wiggle.

    Produces changing reaching/grasping reward and (sometimes) success, so the
    parity check exercises non-zero task signals rather than all-zeros.
    """

    def policy(t: int) -> np.ndarray:
        a = np.zeros(action_dim)
        a[2] = -1.0 if t < 25 else 1.0  # z: descend then lift
        if action_dim >= 7:
            a[6] = 1.0 if t >= 20 else -1.0  # gripper: open then close
        a[0] = 0.15 * np.sin(2 * np.pi * t / 30)  # gentle x wiggle
        return a

    return policy


def verify_task(env_name: str, *, n_steps: int) -> dict:
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    import roboverse_pack.tasks.robosuite  # noqa: F401  (ensure tasks are registered)
    from metasim.task.registry import get_task_class

    task = get(env_name)

    # --- native robosuite reference ---
    cfg = load_composite_controller_config(controller=task.controller, robot=task.robot)
    ref = robosuite.make(
        task.env_name,
        robots=task.robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=task.control_freq,
        horizon=task.horizon,
        ignore_done=True,
        hard_reset=False,
    )
    np.random.seed(0)
    ref.reset()
    action_dim = ref.action_dim
    policy = _scripted_policy(action_dim)

    ref_rew, ref_succ = [], []
    for t in range(n_steps):
        ref.step(policy(t))
        ref_rew.append(float(ref.reward(None)))
        ref_succ.append(bool(ref._check_success()))
    ref_qpos = ref.sim.data.qpos.copy()
    ref.close()

    # --- MetaSim native port ---
    env = get_task_class(task.metasim_name)(seed=0)
    env.reset()
    ms_rew, ms_succ = [], []
    for t in range(n_steps):
        _, r, term, timeout, extra = env.step(policy(t))
        ms_rew.append(float(r.item()))
        ms_succ.append(bool(extra["success"]))
    ms_qpos = np.asarray(env._hd.qpos).copy()
    env.close()

    ref_rew, ms_rew = np.array(ref_rew), np.array(ms_rew)
    rew_max = float(np.abs(ref_rew - ms_rew).max())
    qpos_max = float(np.abs(ref_qpos - ms_qpos).max())
    succ_match = bool((np.array(ref_succ) == np.array(ms_succ)).all())
    out = {
        "task": task.metasim_name,
        "env_name": task.env_name,
        "n_steps": n_steps,
        "reward_max_abs_diff": rew_max,
        "qpos_final_max_abs_diff": qpos_max,
        "success_sequence_match": succ_match,
        "reward_bitwise_equal": bool((ref_rew == ms_rew).all()),
        "ref_total_reward": float(ref_rew.sum()),
        "metasim_total_reward": float(ms_rew.sum()),
        "ref_any_success": bool(np.any(ref_succ)),
        "metasim_any_success": bool(np.any(ms_succ)),
    }
    print(
        f"  [{task.env_name:18s}] reward_max_diff={rew_max:.2e} qpos_final_max_diff={qpos_max:.2e} "
        f"succ_match={succ_match} ret(ref={out['ref_total_reward']:.3f}/ms={out['metasim_total_reward']:.3f})"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--tasks", nargs="*", default=[t.env_name for t in ALL_TASKS])
    args = ap.parse_args()

    print(f"native-port parity: {len(args.tasks)} tasks, {args.n_steps} steps")
    results = []
    for name in args.tasks:
        try:
            res = verify_task(name, n_steps=args.n_steps)
        except Exception as e:
            import traceback

            traceback.print_exc()
            res = {"task": name, "error": str(e)}
        results.append(res)
        run_dir = args.out / get(name).slug
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "native_parity.json").write_text(json.dumps(res, indent=2))

    (args.out.parent / "native_parity.json").write_text(json.dumps({"tasks": results}, indent=2))
    print(f"wrote {args.out.parent / 'native_parity.json'}")


if __name__ == "__main__":
    main()
