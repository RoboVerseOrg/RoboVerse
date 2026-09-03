"""LIBERO-plus passthrough 1:1 parity harness.

Proves that the RoboVerse LIBERO-plus passthrough
(``roboverse_pack.tasks.libero_plus.make_liberoplus_env``) builds a
**bitwise-identical** environment to a natively-constructed LIBERO-plus
``OffScreenRenderEnv`` -- i.e. the passthrough adds *zero* error -- across all
seven perturbation dimensions (object layout, camera viewpoint, robot init
state, language rewrite, lighting, background texture, sensor noise).

For each sampled perturbation task we build the env twice (passthrough vs an
inline native construction mirroring ``libero/lifelong/evaluate.py``), apply the
same seed + stored init state + a deterministic action sequence, and compare the
full observation dict (including the 128x128 camera images, byte for byte),
reward and done at every step. The load-bearing number is
``passthrough-vs-native max|Δ| == 0``.

Run (in the dedicated ``liberoplus`` conda env, headless EGL)::

    LIBERO_PLUS_MAX_TASKS_PER_SUITE=1 \\
    MUJOCO_GL=egl python -m scripts.parity_liberoplus_passthrough --steps 10

(The ``LIBERO_PLUS_MAX_TASKS_PER_SUITE`` cap only speeds the import-time gym
registration; the harness builds envs directly via the factory and ignores it.)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from roboverse_pack.tasks.libero_plus import _passthrough as pt

# Perturbation dimensions defined by the LIBERO-plus paper / task_classification.
PERTURBATION_DIMS = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)


def _classification_path() -> str:
    pkg = pt._liberoplus_pkg_dir()
    if pkg is None:
        raise RuntimeError("LIBERO-plus is not importable in this env.")
    return os.path.join(pkg, "benchmark", "task_classification.json")


def _sample_tasks(per_dim: int = 1) -> list[tuple[str, int, str, str]]:
    """Pick ``per_dim`` tasks for each (suite, perturbation-dimension) pair.

    Returns ``[(suite, task_id, task_name, category), ...]``. ``task_id`` is the
    index inside the live benchmark suite (matched by task *name*, never by the
    classification ``id``, so it stays correct regardless of ordering).
    """
    with open(_classification_path()) as f:
        classification = json.load(f)

    samples: list[tuple[str, int, str, str]] = []
    for suite, items in classification.items():
        names = pt.list_liberoplus_tasks(suite)
        name_to_id = {n: i for i, n in enumerate(names)}
        by_dim: dict[str, list[tuple[int, str]]] = {d: [] for d in PERTURBATION_DIMS}
        for it in items:
            cat, name = it.get("category"), it.get("name")
            if cat in by_dim and name in name_to_id:
                by_dim[cat].append((name_to_id[name], name))
        for dim in PERTURBATION_DIMS:
            for tid, name in by_dim[dim][:per_dim]:
                samples.append((suite, tid, name, dim))
    # libero_90 carries the un-perturbed base scenes (sanity anchor).
    base_names = pt.list_liberoplus_tasks("libero_90")
    if base_names:
        samples.append(("libero_90", 0, base_names[0], "Base (libero_90)"))
    return samples


def _action_sequence(n_steps: int, action_dim: int = 7, seed: int = 0) -> np.ndarray:
    """A fixed, deterministic action sequence: 5 zero warmup steps then a small
    sinusoid (mirrors the LIBERO eval warmup, kept identical across both envs)."""
    rng = np.random.RandomState(seed)
    base = 0.05 * np.sin(np.linspace(0, 3.14, n_steps))[:, None] * np.ones((1, action_dim))
    jitter = 0.01 * rng.randn(n_steps, action_dim)
    acts = base + jitter
    acts[:5] = 0.0
    return acts.astype(np.float64)


def _native_env(suite: str, task_id: int, seed: int):
    """Construct the env exactly like upstream LIBERO-plus eval, independently."""
    benchmark = pt._ensure_imported()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(seed)
    env.reset()
    try:  # mirror the passthrough's handling of the upstream get_task_init_states bug
        env.set_init_state(bench.get_task_init_states(task_id)[0])
    except (UnboundLocalError, FileNotFoundError):
        pass
    return env


_IMAGE_HINT = ("image", "rgb", "depth")


def _is_image_key(k: str) -> bool:
    return any(h in k.lower() for h in _IMAGE_HINT)


def _record(env, acts: np.ndarray) -> list[dict]:
    """Step ``env`` through ``acts`` and record (obs, reward, done) per step."""
    traj = []
    for a in acts:
        obs, rew, done, _ = env.step(a)
        traj.append({"obs": {k: np.asarray(v) for k, v in obs.items()}, "rew": float(rew), "done": bool(done)})
    return traj


def _compare(traj_a: list[dict], traj_b: list[dict]) -> tuple[float, float, float, bool]:
    """Return (state_obs|Δ|, image_obs|Δ|, reward|Δ|, done_match) across the traj.

    ``state_obs`` = every non-image obs key (proprio / object poses / etc.).
    Images are bucketed separately because the sensor-noise perturbation corrupts
    ``agentview_image`` with *unseeded* upstream RNG (stochastic per run), and
    headless EGL has rare ±1 framebuffer rounding -- neither is a passthrough or
    physics difference.

    The obs key sets must be **equal**. Comparing only their intersection would let a
    passthrough that returns ``{}`` -- or that silently drops a key -- score a perfect
    bitwise parity over zero compared keys. A missing key is a failure, not an
    unexamined key, so any structural mismatch raises rather than scoring Δ=0.
    """
    if len(traj_a) != len(traj_b):
        raise RuntimeError(f"rollout length mismatch: {len(traj_a)} vs {len(traj_b)} steps")
    if not traj_a:
        raise RuntimeError("empty rollout: nothing was compared, so this is not parity")
    dev_state = dev_img = dev_rew = 0.0
    done_match = True
    for i, (sa, sb) in enumerate(zip(traj_a, traj_b)):
        ka, kb = set(sa["obs"]), set(sb["obs"])
        if ka != kb:
            raise RuntimeError(
                f"step {i}: observation key sets differ -- passthrough-only={sorted(ka - kb)}, "
                f"native-only={sorted(kb - ka)}"
            )
        if not any(not _is_image_key(k) for k in ka):
            raise RuntimeError(f"step {i}: no non-image observation keys to compare -- nothing would be verified")
        for k in sorted(ka):
            va, vb = sa["obs"][k], sb["obs"][k]
            if va.shape != vb.shape:
                raise RuntimeError(f"step {i}: obs[{k!r}] shape {va.shape} != {vb.shape}")
            d = float(np.abs(va.astype(np.float64) - vb.astype(np.float64)).max())
            if _is_image_key(k):
                dev_img = max(dev_img, d)
            else:
                dev_state = max(dev_state, d)
        dev_rew = max(dev_rew, abs(sa["rew"] - sb["rew"]))
        done_match = done_match and (sa["done"] == sb["done"])
    return dev_state, dev_img, dev_rew, done_match


def run(per_dim: int, steps: int, seed: int) -> int:
    samples = _sample_tasks(per_dim)
    if not samples:
        raise RuntimeError("no tasks sampled: a run that checks 0 tasks proves nothing and is not a PASS")
    if steps <= 0:
        raise RuntimeError("steps must be >= 1: a run that takes 0 steps proves nothing and is not a PASS")
    acts = _action_sequence(steps, seed=seed)
    print(f"# LIBERO-plus passthrough 1:1 parity — {len(samples)} sampled tasks, {steps} steps each")
    print("# state/reward/done must be bitwise (Δ=0); image bucketed (noise=stochastic upstream)\n")
    header = f"{'suite':14s} {'dim':22s} {'state|Δ|':>9s} {'img|Δ|':>8s} {'rew|Δ|':>9s} {'done':>5s}  task"
    print(header)
    print("-" * len(header))

    worst_state = worst_rew = 0.0
    n_ok = 0
    for suite, tid, name, dim in samples:
        # Render one env at a time (global EGL context is shared -> two live envs
        # clobber each other's textures). Record passthrough fully, close, then native.
        env_pt = pt.make_liberoplus_env(suite, tid, seed=seed, init_state_index=0)
        traj_pt = _record(env_pt, acts)
        env_pt.close()

        env_nv = _native_env(suite, tid, seed)
        traj_nv = _record(env_nv, acts)
        env_nv.close()

        dev_state, dev_img, dev_rew, done_match = _compare(traj_pt, traj_nv)
        worst_state = max(worst_state, dev_state)
        worst_rew = max(worst_rew, dev_rew)
        # 1:1 contract: physics (state) + reward + done bitwise. Image must also be
        # bitwise EXCEPT sensor-noise (upstream stochastic image corruption).
        img_ok = dev_img == 0.0 or dim == "Sensor Noise"
        ok = dev_state == 0.0 and dev_rew == 0.0 and done_match and img_ok
        n_ok += int(ok)
        flag = "" if ok else "  <-- MISMATCH"
        print(
            f"{suite:14s} {dim:22s} {dev_state:9.2e} {dev_img:8.1f} {dev_rew:9.2e} "
            f"{done_match!s:>5s}  {name[:40]}{flag}"
        )

    print("-" * len(header))
    print(f"\n{n_ok}/{len(samples)} tasks pass the 1:1 contract (state+reward+done bitwise, image ok)")
    print(f"worst passthrough-vs-native state max|Δ| = {worst_state:.3e}; reward max|Δ| = {worst_rew:.3e}")
    if n_ok == len(samples) and worst_state == 0.0 and worst_rew == 0.0:
        print("RESULT: PASS — passthrough adds zero error to physics/reward/done (1:1 by construction).")
        return 0
    print("RESULT: FAIL — non-zero deviation detected.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-dim", type=int, default=1, help="tasks sampled per (suite, perturbation dim)")
    ap.add_argument("--steps", type=int, default=10, help="env steps compared per task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return run(args.per_dim, args.steps, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
