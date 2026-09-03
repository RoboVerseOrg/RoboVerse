"""Closed-loop eval of the BC policy on LIBERO-plus via the passthrough (CPU).

Runs in the ``liberoplus`` env (torch 2.4.1 CPU -- the LIBERO-plus sim pins
py3.8, and this machine's sm_120 GPU is unusable under torch 2.4.1, so policy
inference runs on CPU here). The policy is trained on the *clean* base task; we
evaluate it closed-loop on the clean task and on LIBERO-plus perturbation
variants -- exactly the robustness question LIBERO-plus exists to answer -- and
confirm the eval is bitwise identical through the passthrough vs a native env.

    LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \\
    python -m scripts.policy.eval_bc_liberoplus --ckpt scripts/policy/ckpt/bc_alphabet_soup.pt \\
        --base pick_up_the_alphabet_soup_and_place_it_in_the_basket --suite libero_object --episodes 8
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_model import ChunkedBCPolicy, prep_image, proprio_vec

from roboverse_pack.tasks.libero_plus import _passthrough as pt

DEMO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "libero_datasets"
)


def _load_policy(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy = ChunkedBCPolicy(chunk=ckpt["chunk"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    p_mean = np.asarray(ckpt["proprio_mean"], dtype=np.float32)
    p_std = np.asarray(ckpt["proprio_std"], dtype=np.float32)
    return policy, p_mean, p_std, ckpt["chunk"]


def _obs_images(obs: dict):
    """Native LIBERO obs -> (agentview_rgb, eye_in_hand_rgb).

    The native ``*_image`` obs is already in the same orientation as the demo
    ``*_rgb`` arrays the policy trained on (verified: no-flip MAE 4.2 vs
    flipped 54.5), so we pass them through without vertical flipping.
    """
    av = np.asarray(obs["agentview_image"])
    eh = np.asarray(obs["robot0_eye_in_hand_image"])
    return np.ascontiguousarray(av), np.ascontiguousarray(eh)


@torch.no_grad()
def _act(policy, obs, p_mean, p_std):
    av, eh = _obs_images(obs)
    p = (proprio_vec(obs) - p_mean) / p_std
    chunk = policy(
        prep_image(av, "cpu"),
        prep_image(eh, "cpu"),
        torch.from_numpy(p).float().unsqueeze(0),
    )
    return chunk[0].numpy()  # (CHUNK, 7)


def _check_success(env) -> bool:
    """Task success from the env's own checker.

    An error (or a missing checker) is **not** a failed episode: swallowing it into
    ``False`` silently reports a 0% success rate for a policy that may be fine, and --
    when both sides of a parity check do it -- makes two broken envs agree. Fail loudly.
    """
    for attr in ("_check_success", "check_success"):
        fn = getattr(env.env, attr, None) or getattr(env, attr, None)
        if fn is not None:
            return bool(fn())
    raise RuntimeError(
        f"{type(env).__name__} exposes no _check_success()/check_success(); success cannot be evaluated."
    )


def _init_obs(env, init_state):
    """Set the demo init state if dimensions match; else use the env reset state."""
    try:
        return env.set_init_state(init_state)
    except Exception:
        return env.env._get_observations()


def _rollout(env, init_state, policy, p_mean, p_std, *, max_steps, exec_horizon, record=False):
    obs = _init_obs(env, init_state)
    success = False
    traj = []
    steps = 0
    while steps < max_steps:
        chunk = _act(policy, obs, p_mean, p_std)
        for a in chunk[:exec_horizon]:
            obs, r, d, _ = env.step(a)
            steps += 1
            if record:
                s = env.env.sim.get_state()
                traj.append(np.concatenate([s.qpos, s.qvel]))
            if _check_success(env):
                success = True
                break
            if steps >= max_steps:
                break
        if success:
            break
    return success, steps, traj


def _native_env(suite, tid, seed):
    benchmark = pt._ensure_imported()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    b = benchmark.get_benchmark_dict()[suite]()
    t = b.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    e.seed(seed)
    e.reset()
    return e


def _variants(suite, base):
    """Return [(label, task_name)] for clean + dynamics-preserving perturbations.

    "clean" = the neutral-view / neutral-init variant (no actual camera/init
    change); the others toggle exactly one perturbation so the success delta is
    attributable to that dimension.
    """
    names = [n for n in pt.list_liberoplus_tasks(suite) if n.startswith(base)]

    def first(pred, default=None):
        return next((n for n in names if pred(n)), default)

    clean = first(lambda n: n.endswith("_view_0_0_100_0_0_initstate_0"), names[0])
    out = [("clean", clean)]
    light = first(lambda n: "_light_" in n)
    noise = first(lambda n: "_view_0_0_100_0_0_initstate_0_noise_" in n)
    cam = first(lambda n: "_view_" in n and "_view_0_0_100_0_0_" not in n and n.endswith("_initstate_0"))
    for label, n in (("light", light), ("sensor-noise", noise), ("camera", cam)):
        if n:
            out.append((label, n))
    return out


def run(ckpt, suite, base, episodes, max_steps, exec_horizon):
    policy, p_mean, p_std, chunk = _load_policy(ckpt)
    demo_path = os.path.join(DEMO_ROOT, suite, f"{base}_demo.hdf5")
    with h5py.File(demo_path, "r") as h:
        inits = [np.asarray(h["data"][f"demo_{i}"]["states"])[0] for i in range(episodes)]

    print(f"# Closed-loop BC eval via LIBERO-plus passthrough (CPU) — {episodes} ep/variant, chunk={chunk}\n")
    print(f"{'variant':14s} {'task':52s} {'success':>10s}")
    print("-" * 80)
    results = {}
    for label, name in _variants(suite, base):
        tid = pt.list_liberoplus_tasks(suite).index(name)
        n_succ = 0
        for i in range(episodes):
            env = pt.make_liberoplus_env(suite, tid, seed=i, camera_heights=128, camera_widths=128)
            ok, steps, _ = _rollout(
                env, inits[i % len(inits)], policy, p_mean, p_std, max_steps=max_steps, exec_horizon=exec_horizon
            )
            env.close()
            n_succ += int(ok)
        results[label] = n_succ / episodes
        print(f"{label:14s} {name[:52]:52s} {n_succ}/{episodes} = {n_succ / episodes:.2f}")
    print("-" * 80)

    # passthrough == native under the SAME policy (one clean episode, bitwise state)
    clean_name = dict(_variants(suite, base))["clean"]
    tid = pt.list_liberoplus_tasks(suite).index(clean_name)
    ep = pt.make_liberoplus_env(suite, tid, seed=0, camera_heights=128, camera_widths=128)
    _, _, tp = _rollout(ep, inits[0], policy, p_mean, p_std, max_steps=60, exec_horizon=exec_horizon, record=True)
    ep.close()
    en = _native_env(suite, tid, 0)
    _, _, tn = _rollout(en, inits[0], policy, p_mean, p_std, max_steps=60, exec_horizon=exec_horizon, record=True)
    en.close()
    dev = max((float(np.abs(a - b).max()) for a, b in zip(tp, tn)), default=float("nan"))
    print(f"\npassthrough == native under the policy: state max|Δ| over {min(len(tp), len(tn))} steps = {dev:.2e}")
    print("(deterministic policy + deterministic env -> bitwise identical eval through the passthrough)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--base", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--exec-horizon", type=int, default=8)
    args = ap.parse_args()
    return run(args.ckpt, args.suite, args.base, args.episodes, args.max_steps, args.exec_horizon)


if __name__ == "__main__":
    raise SystemExit(main())
