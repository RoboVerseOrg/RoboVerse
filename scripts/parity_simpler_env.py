"""1:1 parity check for the SimplerEnv passthrough.

For each task, constructs the *same* SimplerEnv env two ways — natively via
``simpler_env.make(task_name)`` and through the RoboVerse passthrough via
``gymnasium.make("SimplerEnv/<task_name>")`` — drives both with an identical
seeded action sequence, and compares the full per-step signature: the rendered
camera image, the reward, the termination flags, and ``info["success"]`` /
``info["episode_stats"]``.

Because the passthrough forwards ``reset`` / ``step`` verbatim, a correct
integration is bitwise identical. This harness turns that into a measured number.

Isolation note (important): SimplerEnv is built on ManiSkill2-real2sim + SAPIEN,
which keeps **process-global renderer/engine state**. Two SAPIEN scenes alive in
the same process at the same time perturb each other's success evaluation (the
same class of issue as a shared EGL/GL context). So each task is compared by
running the native and the wrapped env **one at a time, each in its own
subprocess** — the way SimplerEnv itself guarantees determinism. Within a worker
the native env is fully rolled out and closed before the wrapped env is created.

Usage::

    python scripts/parity_simpler_env.py --tasks google_robot_pick_coke_can --steps 8
    python scripts/parity_simpler_env.py --all --steps 8 --json /tmp/simpler_parity.json
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys

# ---------------------------------------------------------------------------
# Worker: runs inside a fresh subprocess for a single task. Prints one
# "RESULT_B64:<...>" line with a base64-encoded JSON of {native, wrapped, ...}.
# ---------------------------------------------------------------------------

_WORKER = r"""
import base64, json, sys, traceback
import numpy as np

TASK, SEED, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def _camera(env):
    uid = getattr(getattr(env, "unwrapped", env), "robot_uid", "") or ""
    if "google_robot" in uid:
        return "overhead_camera"
    if "widowx" in uid:
        return "3rd_view_camera"
    img = None
    try:
        img = next(iter(env.reset(seed=SEED)[0]["image"].keys()))
    except Exception:
        pass
    return img


def _sig(obs, rew, term, trunc, info, cam):
    img = np.asarray(obs["image"][cam]["rgb"]).astype(np.int64)
    return {
        "img_sum": int(img.sum()),
        "img_sq": int((img * img).sum()),
        "img_shape": list(np.asarray(obs["image"][cam]["rgb"]).shape),
        "rew": float(rew),
        "term": bool(term), "trunc": bool(trunc),
        "success": bool(info.get("success", False)),
    }


def _make_acts(space):
    rng = np.random.RandomState(SEED)
    return [rng.uniform(space.low, space.high).astype(space.dtype) for _ in range(STEPS)]


def _rollout(env, acts, cam):
    obs, info = env.reset(seed=SEED)
    frames = [_sig(obs, 0.0, False, False, info, cam)]
    for a in acts:
        o, rw, tm, tr, inf = env.step(a)
        frames.append(_sig(o, rw, tm, tr, inf, cam))
    env.close()
    return frames


def main():
    res = {"task": TASK}
    try:
        import simpler_env
        nat = simpler_env.make(TASK)
        cam = "overhead_camera" if "google_robot" in (nat.robot_uid or "") else "3rd_view_camera"
        acts = _make_acts(nat.action_space)
        res["native"] = _rollout(nat, acts, cam)
        del nat  # native fully closed before the wrapped scene is created

        import gymnasium as gym
        from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough
        register_simpler_env_passthrough()
        wrp = gym.make(f"SimplerEnv/{TASK}")
        res["wrapped"] = _rollout(wrp, acts, cam)

        diffs = []
        for i, (a, b) in enumerate(zip(res["native"], res["wrapped"])):
            for k in a:
                if a[k] != b[k]:
                    diffs.append({"step": i, "key": k, "native": a[k], "wrapped": b[k]})
        res["diffs"] = diffs
        res["parity_1to1"] = len(diffs) == 0
    except Exception:
        res["error"] = traceback.format_exc()
        res["parity_1to1"] = False
    print("RESULT_B64:" + base64.b64encode(json.dumps(res).encode()).decode())


if __name__ == "__main__":
    main()
"""


def check_task(task_name: str, *, steps: int, seed: int) -> dict:
    """Run one task's native-vs-wrapped comparison in a fresh subprocess."""
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, task_name, str(seed), str(steps)],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_B64:"):
            return json.loads(base64.b64decode(line[len("RESULT_B64:") :]).decode())
    return {"task": task_name, "parity_1to1": False, "error": proc.stderr[-2000:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=["google_robot_pick_coke_can"])
    parser.add_argument("--all", action="store_true", help="run every installed SimplerEnv task")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default=None, help="write results to this JSON path")
    args = parser.parse_args()

    import simpler_env

    tasks = list(simpler_env.ENVIRONMENTS) if args.all else args.tasks

    results = []
    for t in tasks:
        r = check_task(t, steps=args.steps, seed=args.seed)
        results.append(r)
        n_diff = len(r.get("diffs", [])) if "diffs" in r else "n/a"
        status = "OK 1:1" if r.get("parity_1to1") else "DIFF/ERR"
        line = f"[{status:8}] {t:48s} diffs={n_diff}"
        if "error" in r:
            line += f" err={r['error'].strip().splitlines()[-1][:80]}"
        print(line)

    n_ok = sum(1 for r in results if r.get("parity_1to1"))
    # strip bulky per-frame arrays from the saved JSON; keep diffs + verdict
    slim = [{k: v for k, v in r.items() if k not in ("native", "wrapped")} for r in results]
    print(f"\n{n_ok}/{len(results)} tasks bitwise 1:1")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"results": slim, "n_ok": n_ok, "n_total": len(results), "steps": args.steps, "seed": args.seed},
                f,
                indent=2,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
