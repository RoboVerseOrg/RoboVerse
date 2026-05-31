"""Closed-loop policy evaluation in native RoboTwin (the policy-reproduction harness).

Runs in the ``robotwin`` conda env. The native passthrough env is the *only*
closed-loop RoboTwin environment (RoboVerse has no RoboTwin handler env), so this
is where a learned policy must be evaluated to claim it reproduces RoboTwin's
results. A policy maps RoboTwin's per-step observation (head-camera RGB + joint
state) to a 14-D joint action ``[L_arm(6), L_grip, R_arm(6), R_grip]``, which is
rolled out through ``env.take_action(action, 'qpos')`` -- the *exact* closed-loop
interface RoboTwin's own ``script/eval_policy.py`` uses (TOPP-interpolates the
waypoint, steps physics, sets ``eval_success`` when ``check_success()`` fires).
Success rate over N seeds is directly comparable to RoboTwin's expert.

Policies (``--policy``):

- ``replay`` -- open-loop action replay: emit the bridge-recorded command stream
  ``vectors[t]`` one waypoint per ``take_action``. Needs no model/GPU and answers
  a sharp 1:1 question: does RoboTwin's recorded bimanual action trajectory, driven
  through ``take_action``'s TOPP (not the original curobo plan that produced it),
  still complete the task? It is also the harness self-test the DP path reuses.
- ``dp`` -- a RoboVerse-trained Diffusion Policy / ACT checkpoint
  (``roboverse_learn/il``). Loaded in-process if importable in this env, else point
  ``--server`` at a ``policy_model_server`` running in the ``roboverse`` env.

Run (robotwin env)::

    # open-loop action-replay baseline (the recorded trajectory, via take_action)
    conda run -n robotwin env MUJOCO_GL=egl SAPIEN_HEADLESS=1 python \\
        tools/robotwin_integration/eval_robotwin_policy.py --task beat_block_hammer \\
        --policy replay --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PASSTHROUGH = os.path.join(_HERE, "..", "..", "roboverse_pack", "tasks", "robotwin", "_passthrough.py")


def _load_passthrough():
    spec = importlib.util.spec_from_file_location("rt_passthrough", _PASSTHROUGH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Policy:
    """Maps a RoboTwin observation to a 14-D joint action (or None to stop)."""

    def reset(self) -> None: ...

    def predict(self, obs: dict) -> np.ndarray | None:
        raise NotImplementedError


class ReplayPolicy(Policy):
    """Emit the bridge-recorded command targets one per step (open-loop replay)."""

    def __init__(self, bridge: dict):
        self.vectors = [np.asarray(v, dtype=float) for v in bridge["vectors"]]
        self.seed = int(bridge.get("seed", 0))
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def predict(self, obs: dict) -> np.ndarray | None:
        if self._t >= len(self.vectors):
            return None  # trajectory exhausted -> let the episode end
        a = self.vectors[self._t]
        self._t += 1
        return a


class DPPolicy(Policy):
    """A RoboVerse-trained DP/ACT checkpoint (roboverse_learn/il).

    Kept thin: the IL model + its obs preprocessing live in roboverse_learn; this
    only adapts RoboTwin's obs dict (head_camera RGB + joint state) to the model's
    expected input and the 14-D action back out. Filled in once a checkpoint exists
    (#19); until then it raises an actionable error rather than silently no-op'ing.
    """

    def __init__(self, ckpt: str, camera: str = "head_camera"):
        self.ckpt = ckpt
        self.camera = camera
        raise NotImplementedError(
            "DP policy eval needs a trained checkpoint (task #19). Train with "
            "roboverse_learn/il/train.py on a RoboTwin zarr, then wire the DP "
            "inference here (or use --server with policy_model_server in roboverse env)."
        )


def _build_policy(args) -> Policy:
    if args.policy == "replay":
        if not args.bridge:
            raise SystemExit("--policy replay requires --bridge <pkl>")
        with open(os.path.expanduser(args.bridge), "rb") as f:
            return ReplayPolicy(pickle.load(f))
    if args.policy == "dp":
        if not args.ckpt:
            raise SystemExit("--policy dp requires --ckpt <checkpoint>")
        return DPPolicy(args.ckpt, args.camera)
    raise SystemExit(f"unknown policy {args.policy!r}")


def _eval_one(pt, task: str, task_config: str, seed: int, policy: Policy, rgb: bool) -> tuple[bool, int]:
    """Run one closed-loop episode; return (success, steps_taken)."""
    # is_test=True makes setup_demo load the per-task step_lim and enable eval mode.
    data_type = {k: False for k in ("rgb", "third_view", "depth", "pointcloud", "observer", "endpose", "qpos")}
    data_type["rgb"] = rgb  # the policy may consume head_camera RGB
    env = pt._make_robotwin_env(
        task_name=task, task_config=task_config, seed=seed, is_test=True, data_type=data_type, render_freq=0,
    )  # fmt: skip
    policy.reset()
    try:
        step_lim = env.step_lim or 1000
        while env.take_action_cnt < step_lim:
            obs = env.get_obs()
            action = policy.predict(obs)
            if action is None:
                break
            env.take_action(np.asarray(action, dtype=float), action_type="qpos")
            if env.eval_success:
                break
        succ = bool(env.eval_success)
        steps = int(env.take_action_cnt)
    finally:
        env.close_env()
    return succ, steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="RoboTwin task name (envs/<task>.py)")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--policy", choices=["replay", "dp"], default="replay")
    ap.add_argument("--bridge", default=None, help="bridge pkl (for --policy replay; uses its recorded seed)")
    ap.add_argument("--ckpt", default=None, help="checkpoint (for --policy dp)")
    ap.add_argument("--camera", default="head_camera", help="image obs camera for --policy dp")
    ap.add_argument("--num-eval", type=int, default=1, help="episodes to evaluate")
    ap.add_argument("--start-seed", type=int, default=None, help="first seed (default: bridge seed for replay, else 0)")
    args = ap.parse_args(argv)

    pt = _load_passthrough()
    policy = _build_policy(args)
    needs_rgb = args.policy == "dp"

    # Replay is only meaningful at the seed the trajectory was recorded for (object
    # placement is seed-dependent); default to the bridge's seed.
    if args.start_seed is not None:
        base_seed = args.start_seed
    elif isinstance(policy, ReplayPolicy):
        base_seed = policy.seed
    else:
        base_seed = 0

    successes = 0
    for i in range(args.num_eval):
        seed = base_seed + i
        succ, steps = _eval_one(pt, args.task, args.task_config, seed, policy, needs_rgb)
        successes += int(succ)
        print(f"[{args.task} seed {seed}] {'SUCCESS' if succ else 'FAIL'} ({steps} steps)")

    rate = successes / max(1, args.num_eval)
    print(f"RESULT {args.task} | policy={args.policy} | success {successes}/{args.num_eval} = {rate * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
