"""LIBERO-plus policy-eval consistency — passthrough == native under a policy.

A real VLA checkpoint (e.g. ``openvla-7b-oft-finetuned-libero-plus``) needs a
GPU; this machine's RTX 5090 is sm_120 and the LIBERO-plus env pins torch 2.4.1
(cu121, max sm_90), so neural-policy inference fails with
``no kernel image is available for execution on the device``. To still test the
*eval path* (not just env construction), we use a **deterministic open-loop
policy**: the recorded expert demo's action sequence -- exactly the methodology
of the base-LIBERO policy-eval evidence.

For a dynamics-preserving perturbation task (lighting / sensor-noise / neutral
camera leave qpos/qvel layout unchanged) we set the demo's init state, replay
the demo actions through the **passthrough** env and an independently-built
**native** env, and compare the full rollout: raw MuJoCo sim-state, every
non-image obs key, reward, done, and task success, step by step. The claim:
*given the same policy, the passthrough adds zero error to the eval result.*

Run (env ``liberoplus``)::

    MUJOCO_GL=egl python -m scripts.eval_liberoplus_policy_consistency
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from roboverse_pack.tasks.libero_plus import _passthrough as pt

_DEMO_ROOT = os.environ.get(
    "LIBERO_DEMO_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "libero_datasets"),
)

# (suite, demo-task-stem, perturbation-task-name, label). Perturbations chosen to
# preserve dynamics so a base demo's actions are a valid open-loop policy.
CASES = [
    (
        "libero_object",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_1",
        "Light Conditions",
    ),
    (
        "libero_object",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_3",
        "Sensor Noise",
    ),
]


def _load_demo(suite: str, stem: str, demo_idx: int = 0):
    path = os.path.join(_DEMO_ROOT, suite, f"{stem}_demo.hdf5")
    with h5py.File(path, "r") as h:
        g = h["data"][f"demo_{demo_idx}"]
        return np.asarray(g["actions"]), np.asarray(g["states"])[0]


def _raw_state(env) -> np.ndarray:
    s = env.env.sim.get_state()
    return np.concatenate([[s.time], np.asarray(s.qpos), np.asarray(s.qvel)])


def _success(env) -> bool:
    try:
        return bool(env.env._check_success())
    except Exception:
        return False


def _rollout(env, init_state, actions, max_steps):
    env.set_init_state(init_state)
    states, obss, rews, dones, succ = [], [], [], [], []
    for a in actions[:max_steps]:
        o, r, d, _ = env.step(a)
        states.append(_raw_state(env))
        obss.append({k: np.asarray(v) for k, v in o.items()})
        rews.append(float(r))
        dones.append(bool(d))
        succ.append(_success(env))
    env.close()
    return states, obss, rews, dones, succ


def _task_id(suite: str, name: str) -> int:
    names = pt.list_liberoplus_tasks(suite)
    return names.index(name)


def _native_env(suite: str, tid: int, seed: int):
    """Independent native construction (raw OffScreenRenderEnv), no passthrough."""
    benchmark = pt._ensure_imported()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(seed)
    env.reset()
    return env


def run(max_steps: int = 120) -> int:
    print("# LIBERO-plus policy-eval consistency (deterministic open-loop demo policy)")
    print("# real VLA = sm_120-blocked; this verifies passthrough == native on the EVAL path\n")
    header = f"{'perturbation':18s} {'state|Δ|':>9s} {'obs|Δ|':>9s} {'rew|Δ|':>9s} {'done':>5s} {'success(pt/nv)':>14s}"
    print(header)
    print("-" * len(header))
    worst = 0.0
    n_ok = 0
    for suite, stem, pert_name, label in CASES:
        actions, init_state = _load_demo(suite, stem)
        tid = _task_id(suite, pert_name)
        # one env at a time (shared global EGL context)
        # passthrough rollout, then an independently-constructed native rollout.
        sp, op, rp, dp, fp = _rollout(pt.make_liberoplus_env(suite, tid, seed=0), init_state, actions, max_steps)
        sn, on, rn, dn, fn = _rollout(_native_env(suite, tid, seed=0), init_state, actions, max_steps)
        ds = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(sp, sn))
        do = 0.0
        for pa, pb in zip(op, on):
            for k in set(pa) & set(pb):
                if any(hh in k.lower() for hh in ("image", "rgb", "depth")):
                    continue
                do = max(do, float(np.abs(pa[k].astype(np.float64) - pb[k].astype(np.float64)).max()))
        dr = max(abs(x - y) for x, y in zip(rp, rn))
        dd = all(x == y for x, y in zip(dp, dn))
        succ_match = fp[-1] == fn[-1]
        worst = max(worst, ds, do, dr)
        ok = ds == 0.0 and do == 0.0 and dr == 0.0 and dd and succ_match
        n_ok += int(ok)
        print(f"{label:18s} {ds:9.2e} {do:9.2e} {dr:9.2e} {dd!s:>5s} {str(fp[-1]) + '/' + str(fn[-1]):>14s}")
    print("-" * len(header))
    print(f"\n{n_ok}/{len(CASES)} perturbation tasks: passthrough == native across the full demo-policy rollout")
    print(f"worst state/obs/reward max|Δ| = {worst:.3e}")
    if n_ok == len(CASES) and worst == 0.0:
        print("RESULT: PASS — under a (deterministic) policy, passthrough eval == native eval (Δ=0).")
        return 0
    print("RESULT: FAIL — non-zero deviation on the eval path.")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
