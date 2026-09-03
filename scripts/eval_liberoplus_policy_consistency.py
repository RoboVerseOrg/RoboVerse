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
    import h5py  # optional dependency: only the demo loader needs it

    with h5py.File(path, "r") as h:
        g = h["data"][f"demo_{demo_idx}"]
        return np.asarray(g["actions"]), np.asarray(g["states"])[0]


def _raw_state(env) -> np.ndarray:
    s = env.env.sim.get_state()
    return np.concatenate([[s.time], np.asarray(s.qpos), np.asarray(s.qvel)])


def _success(env) -> bool:
    """Task success straight from the env's own checker.

    Deliberately **not** wrapped in ``try/except``: swallowing the error into ``False``
    makes two equally-broken sides compare ``False == False`` and score perfect success
    parity. An error here means we cannot evaluate this side, which is an ERROR, never a
    match -- see AGENTS.md, "Parity Is Load-Bearing".
    """
    check = getattr(env.env, "_check_success", None)
    if check is None:
        raise RuntimeError(
            f"{type(env.env).__name__} has no _check_success(); the harness cannot evaluate task "
            "success, so it cannot claim success parity."
        )
    return bool(check())


_IMAGE_HINTS = ("image", "rgb", "depth")


def _state_keys(obs: dict) -> set:
    """The non-image obs keys -- the ones this harness is able to compare bitwise."""
    return {k for k in obs if not any(h in k.lower() for h in _IMAGE_HINTS)}


def _obs_diff(obss_a, obss_b) -> float:
    """max|Δ| over every non-image obs key, requiring **identical key sets**.

    Diffing over ``set(a) & set(b)`` would let a side that returns ``{}`` (or that
    silently drops a key) score a perfect obs parity over zero compared keys. A missing
    key is a failure, not an unexamined key.
    """
    if len(obss_a) != len(obss_b):
        raise RuntimeError(f"rollout length mismatch: {len(obss_a)} vs {len(obss_b)} steps")
    if not obss_a:
        raise RuntimeError("empty rollout: no observations were compared")
    diff = 0.0
    for i, (pa, pb) in enumerate(zip(obss_a, obss_b)):
        ka, kb = _state_keys(pa), _state_keys(pb)
        if ka != kb:
            raise RuntimeError(
                f"step {i}: observation key sets differ -- passthrough-only={sorted(ka - kb)}, "
                f"native-only={sorted(kb - ka)}"
            )
        if not ka:
            raise RuntimeError(f"step {i}: no non-image observation keys to compare -- nothing would be verified")
        for k in sorted(ka):
            va, vb = pa[k], pb[k]
            if va.shape != vb.shape:
                raise RuntimeError(f"step {i}: obs[{k!r}] shape {va.shape} != {vb.shape}")
            diff = max(diff, float(np.abs(va.astype(np.float64) - vb.astype(np.float64)).max()))
    return diff


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
    if not states:
        raise RuntimeError("rollout produced zero steps; there is nothing to compare")
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
    """Compare passthrough vs native under the demo policy. Errors propagate: a side we
    cannot evaluate is an ERROR, and must never be able to reach the PASS branch.
    """
    if not CASES:
        raise RuntimeError("no cases to run; an empty run proves nothing and is not a PASS")
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
        do = _obs_diff(op, on)
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
