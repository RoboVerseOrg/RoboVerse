"""LIBERO-plus sim client (runs in the `liberoplus` env, py3.8 + CPU torch).

The sim half of the bridge: builds LIBERO-plus perturbation envs through the
RoboVerse passthrough and runs the official openVLA-OFT eval loop, delegating
action prediction to the resident policy server (see bridge_policy_server.py).
Replicates openvla-oft's run_libero_eval loop (num_steps_wait dummy steps, then
execute the full action chunk before requerying), so the only thing that changes
vs. the upstream eval is that the policy lives in another process/env.

Run (after starting the server):
    LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \\
    python -m scripts.policy.bridge_sim_client --suite libero_object \\
        --base pick_up_the_alphabet_soup_and_place_it_in_the_basket --episodes 10 --port 5555
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
from collections import deque

import h5py
import numpy as np

from roboverse_pack.tasks.libero_plus import _passthrough as pt

DEMO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "libero_datasets"
)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _recvn(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            raise ConnectionError("closed")
        b += d
    return b


def _recv(c):
    (n,) = struct.unpack(">I", _recvn(c, 4))
    return pickle.loads(_recvn(c, n))


def _send(c, o):
    d = pickle.dumps(o, protocol=4)
    c.sendall(struct.pack(">I", len(d)) + d)


def _obs_for_policy(obs):
    return {
        k: np.asarray(obs[k])
        for k in (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
    }


def _check_success(env):
    """Task success from the env's own checker.

    An error (or a missing checker) is **not** a failed episode: swallowing it into
    ``False`` silently reports a 0% success rate for a policy that may be fine. Fail loudly.
    """
    for owner in (getattr(env, "env", None), env):
        for attr in ("_check_success", "check_success"):
            fn = getattr(owner, attr, None) if owner is not None else None
            if fn:
                return bool(fn())
    raise RuntimeError(
        f"{type(env).__name__} exposes no _check_success()/check_success(); success cannot be evaluated."
    )


def _variants(suite, base):
    names = [n for n in pt.list_liberoplus_tasks(suite) if n.startswith(base)]

    def first(pred, default=None):
        return next((n for n in names if pred(n)), default)

    out = [("clean", first(lambda n: n.endswith("_view_0_0_100_0_0_initstate_0"), names[0]))]
    for label, pred in (
        ("light", lambda n: "_light_" in n),
        ("sensor-noise", lambda n: "_view_0_0_100_0_0_initstate_0_noise_" in n),
        ("camera", lambda n: "_view_" in n and "_view_0_0_100_0_0_" not in n and n.endswith("_initstate_0")),
    ):
        m = first(pred)
        if m:
            out.append((label, m))
    return out


def _episode(env, conn, init_state, task_label, unnorm_key, chunk_exec, num_wait, dummy, max_steps):
    try:
        obs = env.set_init_state(init_state)
    except Exception:
        obs = env.env._get_observations()
    for _ in range(num_wait):
        obs, _, _, _ = env.step(dummy)
    queue = deque()
    for _ in range(max_steps):
        if not queue:
            _send(conn, {"cmd": "act", "obs": _obs_for_policy(obs), "task": task_label, "unnorm_key": unnorm_key})
            queue.extend(_recv(conn)["actions"][:chunk_exec])
        obs, _, done, _ = env.step(np.asarray(queue.popleft()).tolist())
        if done or _check_success(env):
            return True
    return False


def run(suite, base, episodes, port):
    conn = socket.create_connection(("127.0.0.1", port))
    _send(conn, {"cmd": "init"})
    init = _recv(conn)
    chunk_exec, num_wait, dummy = init["chunk"], init["num_steps_wait"], np.asarray(init["dummy_action"])
    max_steps = MAX_STEPS[suite]
    print(f"# openVLA-OFT (official LIBERO-plus ckpt) closed-loop via bridge — chunk={chunk_exec}, wait={num_wait}\n")

    demo = os.path.join(DEMO_ROOT, suite, f"{base}_demo.hdf5")
    with h5py.File(demo, "r") as h:
        inits = [np.asarray(h["data"][f"demo_{i}"]["states"])[0] for i in range(episodes)]

    print(f"{'variant':14s} {'success':>10s}  task")
    print("-" * 76)
    results = {}
    for label, name in _variants(suite, base):
        tid = pt.list_liberoplus_tasks(suite).index(name)
        n_succ = 0
        for i in range(episodes):
            env = pt.make_liberoplus_env(suite, tid, seed=i, camera_heights=128, camera_widths=128)
            task_label = getattr(env, "language_instruction", None) or base.replace("_", " ")
            ok = _episode(env, conn, inits[i % len(inits)], task_label, suite, chunk_exec, num_wait, dummy, max_steps)
            env.close()
            n_succ += int(ok)
        results[label] = n_succ / episodes
        print(f"{label:14s} {n_succ}/{episodes} = {n_succ / episodes:.2f}  {name[:44]}")
    print("-" * 76)
    print("\nresults:", {k: round(v, 3) for k, v in results.items()})
    _send(conn, {"cmd": "bye"})
    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--base", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    return run(args.suite, args.base, args.episodes, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
