"""Record / replay / verify SimplerEnv trajectories through the RoboVerse passthrough.

SimplerEnv ships no reference trajectories — a trajectory is produced by stepping a
policy (here: a deterministic seeded action sequence) through the env. This tool
makes that data *reproducible* and proves it, with these operations:

* ``record``  — roll a task out through ``gymnasium.make("SimplerEnv/<task>")`` and
  save the full trajectory to a ``.npz`` (the seed-driving action sequence + every
  per-step signal: rendered-image content hash, reward, terminated, truncated,
  ``info["success"]``, and the language instruction). With ``--full`` the raw RGB
  frames and agent qpos are saved too, so the literal observation data lands on disk.
* ``replay``  — reload a saved ``.npz``, step the *recorded* action sequence from the
  recorded ``(task, seed)`` in a fresh process, and assert the regenerated
  trajectory matches the saved one bit-for-bit. This is the reproducibility
  contract: a trajectory is fully reproducible from ``(task, seed, actions)``.
* ``determinism`` — record the same ``(task, seed)`` twice in two fresh processes
  and assert the two trajectories are identical (the simulator is deterministic per
  seed).
* ``verify``  — record -> replay -> determinism for a task (or ``--all``) and emit a
  single reproducibility verdict (+ optional JSON).

Each env is built in its own subprocess: SimplerEnv/SAPIEN keep process-global
renderer state, so only one env may live per process (same constraint the parity
harness handles).

Usage::

    python tools/parity/trajectory_simpler_env.py verify --task google_robot_pick_coke_can --steps 20
    python tools/parity/trajectory_simpler_env.py record --all --steps 20 --out-dir /tmp/simpler_trajs
    python tools/parity/trajectory_simpler_env.py record --task widowx_stack_cube --full --out-dir /tmp/simpler_trajs
    python tools/parity/trajectory_simpler_env.py replay --path /tmp/simpler_trajs/google_robot_pick_coke_can.npz
    python tools/parity/trajectory_simpler_env.py verify --all --steps 12 --json /tmp/simpler_traj_repro.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Subprocess worker: builds ONE env, rolls out a seeded (or supplied) action
# sequence, and emits the trajectory as a base64 .npz blob on a single stdout
# line. Running in a fresh process guarantees clean SAPIEN renderer state.
# ---------------------------------------------------------------------------

_WORKER = r"""
import base64, io, sys, traceback
import numpy as np

TASK, SEED, STEPS, FULL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
ACTS_B64 = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None


def _camera(env):
    uid = getattr(getattr(env, "unwrapped", env), "robot_uid", "") or ""
    if "google_robot" in uid:
        return "overhead_camera"
    if "widowx" in uid:
        return "3rd_view_camera"
    return None


def _img(obs, cam):
    return np.asarray(obs["image"][cam]["rgb"])


def main():
    try:
        import gymnasium as gym
        from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough
        register_simpler_env_passthrough()
        env = gym.make(f"SimplerEnv/{TASK}")
        cam = _camera(env)

        if ACTS_B64:
            actions = np.load(io.BytesIO(base64.b64decode(ACTS_B64)))
        else:
            rng = np.random.RandomState(SEED)
            sp = env.action_space
            actions = np.stack([rng.uniform(sp.low, sp.high).astype(sp.dtype) for _ in range(STEPS)])

        obs, info = env.reset(seed=SEED)
        img_sum, img_sq, rew, term, trunc, succ, instr = [], [], [], [], [], [], []
        rgbs, qpos = [], []

        def _log(o, r, t, tr, inf):
            im = _img(o, cam).astype(np.int64)
            img_sum.append(int(im.sum()))
            img_sq.append(int((im * im).sum()))
            rew.append(float(r)); term.append(bool(t)); trunc.append(bool(tr))
            succ.append(bool(inf.get("success", False)))
            try:
                instr.append(str(env.get_language_instruction()))
            except Exception:
                instr.append("")
            if FULL:
                rgbs.append(_img(o, cam).astype(np.uint8))
                try:
                    qpos.append(np.asarray(o["agent"]["qpos"], dtype=np.float32))
                except Exception:
                    qpos.append(np.zeros(0, dtype=np.float32))

        _log(obs, 0.0, False, False, info)
        for a in actions:
            o, r, t, tr, inf = env.step(a)
            _log(o, r, t, tr, inf)
        env.close()

        arrs = dict(
            task=np.array(TASK), seed=np.array(SEED), steps=np.array(STEPS),
            camera=np.array(cam or ""),
            actions=actions.astype(np.float32),
            img_sum=np.array(img_sum, dtype=np.int64),
            img_sq=np.array(img_sq, dtype=np.int64),
            reward=np.array(rew, dtype=np.float64),
            terminated=np.array(term, dtype=bool),
            truncated=np.array(trunc, dtype=bool),
            success=np.array(succ, dtype=bool),
            instruction=np.array(instr),
        )
        if FULL:
            arrs["rgb"] = np.stack(rgbs)
            if qpos and qpos[0].size:
                arrs["qpos"] = np.stack(qpos)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrs)
        print("NPZ_B64:" + base64.b64encode(buf.getvalue()).decode())
    except Exception:
        print("ERR_B64:" + base64.b64encode(traceback.format_exc().encode()).decode())


if __name__ == "__main__":
    main()
"""


def _run_worker(task, seed, steps, full=False, actions_b64="-"):
    """Run the worker subprocess; return (npz_dict | None, error_str | None)."""
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, task, str(seed), str(steps), "1" if full else "0", actions_b64],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("NPZ_B64:"):
            data = np.load(io.BytesIO(base64.b64decode(line[len("NPZ_B64:") :])), allow_pickle=False)
            return {k: data[k] for k in data.files}, None
        if line.startswith("ERR_B64:"):
            return None, base64.b64decode(line[len("ERR_B64:") :]).decode()
    return None, (proc.stderr[-2000:] or "no NPZ emitted")


_SIG_KEYS = ["img_sum", "img_sq", "reward", "terminated", "truncated", "success", "instruction"]


def _traj_diffs(a, b) -> list[str]:
    diffs = []
    keys = list(_SIG_KEYS)
    if "rgb" in a and "rgb" in b:
        keys += ["rgb"]
    if "qpos" in a and "qpos" in b:
        keys += ["qpos"]
    for k in keys:
        if k not in a or k not in b:
            diffs.append(f"{k}:missing")
        elif not np.array_equal(a[k], b[k]):
            diffs.append(k)
    return diffs


def _npbytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _save_npz(traj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **traj)


def op_record(task, seed, steps, full, out_dir):
    traj, err = _run_worker(task, seed, steps, full=full)
    if err:
        return {"task": task, "ok": False, "error": err.strip().splitlines()[-1][:160]}
    path = os.path.join(out_dir, f"{task}.npz")
    _save_npz(traj, path)
    return {
        "task": task,
        "ok": True,
        "path": path,
        "steps": int(traj["steps"]),
        "frames": len(traj["reward"]),
        "full": full,
        "bytes": os.path.getsize(path),
    }


def op_replay(path):
    data = np.load(path, allow_pickle=False)
    saved = {k: data[k] for k in data.files}
    task = str(saved["task"])
    seed = int(saved["seed"])
    full = "rgb" in saved
    acts_b64 = base64.b64encode(_npbytes(saved["actions"])).decode()
    rep, err = _run_worker(task, seed, int(saved["steps"]), full=full, actions_b64=acts_b64)
    if err:
        return {"task": task, "ok": False, "error": err.strip().splitlines()[-1][:160]}
    diffs = _traj_diffs(saved, rep)
    return {"task": task, "ok": len(diffs) == 0, "diffs": diffs, "frames": len(saved["reward"])}


def op_determinism(task, seed, steps, runs=2):
    trajs = []
    for _ in range(runs):
        t, err = _run_worker(task, seed, steps, full=False)
        if err:
            return {"task": task, "ok": False, "error": err.strip().splitlines()[-1][:160]}
        trajs.append(t)
    diffs = []
    for i in range(1, len(trajs)):
        diffs += [f"run0-vs-run{i}:{d}" for d in _traj_diffs(trajs[0], trajs[i])]
    return {"task": task, "ok": len(diffs) == 0, "runs": runs, "diffs": diffs}


def op_verify(task, seed, steps, out_dir):
    rec = op_record(task, seed, steps, full=False, out_dir=out_dir)
    if not rec["ok"]:
        return {"task": task, "reproducible": False, "stage": "record", **rec}
    rep = op_replay(rec["path"])
    det = op_determinism(task, seed, steps, runs=2)
    return {
        "task": task,
        "reproducible": bool(rep["ok"] and det["ok"]),
        "replay_diffs": rep.get("diffs", rep.get("error")),
        "determinism_diffs": det.get("diffs", det.get("error")),
        "frames": rec["frames"],
        "path": rec["path"],
    }


def _tasks(args):
    if args.all:
        import simpler_env

        return list(simpler_env.ENVIRONMENTS)
    return args.task if isinstance(args.task, list) else [args.task]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", nargs="*", default=["google_robot_pick_coke_can"])
        sp.add_argument("--all", action="store_true")
        sp.add_argument("--steps", type=int, default=20)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out-dir", default="/tmp/simpler_trajs")
        sp.add_argument("--json", default=None)

    rec = sub.add_parser("record")
    common(rec)
    rec.add_argument("--full", action="store_true")
    rep = sub.add_parser("replay")
    rep.add_argument("--path", required=True)
    common(sub.add_parser("determinism"))
    common(sub.add_parser("verify"))

    args = p.parse_args()

    if args.cmd == "replay":
        res = [op_replay(args.path)]
    else:
        res = []
        for t in _tasks(args):
            if args.cmd == "record":
                res.append(op_record(t, args.seed, args.steps, args.full, args.out_dir))
            elif args.cmd == "determinism":
                res.append(op_determinism(t, args.seed, args.steps))
            else:
                res.append(op_verify(t, args.seed, args.steps, args.out_dir))

    okkey = {"record": "ok", "replay": "ok", "determinism": "ok", "verify": "reproducible"}[args.cmd]
    for r_ in res:
        tag = "OK" if r_.get(okkey) else "FAIL"
        if "replay_diffs" in r_:
            extra = f" replay={r_['replay_diffs'] or 0} det={r_['determinism_diffs'] or 0}"
        elif r_.get("diffs"):
            extra = f" diffs={r_['diffs']}"
        elif "bytes" in r_:
            extra = f" -> {r_.get('path')} ({r_['bytes']} B, {r_.get('frames')} frames)"
        else:
            extra = ""
        if "error" in r_:
            extra += f" err={r_['error']}"
        label = r_.get("task", getattr(args, "path", ""))
        print(f"[{tag:4}] {label:48s}{extra}")

    n_ok = sum(1 for r_ in res if r_.get(okkey))
    print(f"\n{n_ok}/{len(res)} {args.cmd} OK")

    if getattr(args, "json", None):
        with open(args.json, "w") as f:
            json.dump({"cmd": args.cmd, "results": res, "n_ok": n_ok, "n_total": len(res)}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
