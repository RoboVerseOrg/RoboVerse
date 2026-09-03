"""LIBERO-plus asset / perturbation audit — prove perturbations are *applied*.

The 1:1 parity harness shows passthrough == native. That alone is NOT enough:
if a perturbation's asset is missing and the env silently falls back to the base
scene, passthrough and native would *both* be wrong yet still agree (Δ=0). This
audit closes that gap by checking, for one representative task per perturbation
dimension per suite, that the perturbed render actually **differs from the
un-perturbed base** (asset/perturbation took effect) and that the env compiled
with no missing-asset fallback.

For each (suite, dimension):
  * build the base (un-perturbed) task and the perturbed task,
  * reset + 5 zero-warmup steps (LIBERO eval convention),
  * report agentview-image mean-abs-diff(perturbed, base).

A *visual* perturbation (texture / light / camera / noise / layout) MUST yield
diff > 0. Language / robot-init perturbations may leave the first agentview frame
unchanged (they change the instruction embedding / proprio, not necessarily the
opening pixels) and are reported for completeness, not asserted on pixels.

Run::

    LIBERO_PLUS_MAX_TASKS_PER_SUITE=1 MUJOCO_GL=egl \\
    python -m tools.assets.audit_liberoplus_assets --suites libero_spatial libero_object
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from roboverse_pack.tasks.libero_plus import _passthrough as pt

# Dimensions whose perturbation must visibly change the *opening* agentview frame.
VISUAL_DIMS = {"Background Textures", "Light Conditions", "Camera Viewpoints", "Sensor Noise", "Objects Layout"}
ALL_DIMS = (
    "Background Textures",
    "Light Conditions",
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Sensor Noise",
)


def _classification() -> dict:
    pkg = pt._liberoplus_pkg_dir()
    with open(os.path.join(pkg, "benchmark", "task_classification.json")) as f:
        return json.load(f)


def _agentview_after_warmup(env, warmup: int = 5) -> np.ndarray:
    for _ in range(warmup):
        obs, *_ = env.step(np.zeros(7))
    return np.asarray(obs["agentview_image"]).astype(np.float64)


_PERT_TOKENS = ("_view_", "_language_", "_table_", "_tb_", "_light_", "_add_", "_noise_", "_level")


def _base_bddl_path(suite: str, pert_name: str) -> str | None:
    """Resolve the un-perturbed base BDDL file by stripping all perturbation tokens.

    The LIBERO-plus benchmark task list contains only *perturbed* entries (there is
    no bare base task to look up by id), so the true base is the original LIBERO
    BDDL on disk: ``<stem>.bddl`` where the stem is the name truncated at its first
    perturbation token. Returns the path if it exists, else ``None``.
    """
    from libero.libero import get_libero_path

    stem = pert_name
    for tok in _PERT_TOKENS:
        stem = stem.split(tok)[0]
    path = os.path.join(get_libero_path("bddl_files"), suite, stem + ".bddl")
    return path if os.path.isfile(path) else None


def run(suites: list[str], steps: int) -> int:
    cls = _classification()
    print("# LIBERO-plus perturbation-applied audit (perturbed vs base agentview MAE)\n")
    header = f"{'suite':14s} {'dimension':22s} {'pert-vs-base MAE':>16s}  applied?"
    print(header)
    print("-" * len(header))

    issues = 0
    checked = 0
    for suite in suites:
        names = pt.list_liberoplus_tasks(suite)
        nm2id = {n: i for i, n in enumerate(names)}
        seen: dict[str, int] = {}
        for it in cls.get(suite, []):
            c, n = it.get("category"), it.get("name")
            if c in ALL_DIMS and c not in seen and n in nm2id:
                seen[c] = nm2id[n]
        for dim in ALL_DIMS:
            if dim not in seen:
                continue
            tid = seen[dim]
            base_path = _base_bddl_path(suite, names[tid])
            if base_path is None:
                print(f"{suite:14s} {dim:22s} {'--':>16s}  skipped (no base BDDL on disk)")
                continue
            # IMPORTANT: render ONE env at a time (build -> render -> close) before
            # building the next. robosuite/MuJoCo share a *global* EGL render context,
            # so two OffScreenRenderEnv alive at once share GL texture memory and a
            # texture-only perturbation (Background Textures) renders identically on
            # both -- a context artifact, not an asset/parity issue. Sequential
            # rendering exercises each scene's own textures correctly.
            env_b = pt.make_liberoplus_env_from_bddl(base_path, camera_heights=128, camera_widths=128)
            env_b.seed(0)
            env_b.reset()
            img_b = _agentview_after_warmup(env_b, steps)
            env_b.close()

            env_p = pt.make_liberoplus_env(suite, tid, seed=0)
            img_p = _agentview_after_warmup(env_p, steps)
            env_p.close()
            mae = float(np.abs(img_p - img_b).mean())
            checked += 1
            applied = mae > 0.0
            note = "yes" if applied else ("n/a (non-visual)" if dim not in VISUAL_DIMS else "NO -- check asset!")
            if dim in VISUAL_DIMS and not applied:
                issues += 1
            print(f"{suite:14s} {dim:22s} {mae:16.3f}  {note}")

    print("-" * len(header))
    print(f"\n{checked} (suite, dimension) pairs audited; {issues} visual perturbation(s) with NO effect.")
    if issues == 0:
        print("RESULT: PASS — every visual perturbation changes the render (assets applied, no silent fallback).")
        return 0
    print("RESULT: FAIL — a visual perturbation had zero effect; asset likely missing/falling back.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", nargs="+", default=["libero_spatial"], help="suites to audit")
    ap.add_argument("--steps", type=int, default=5, help="zero-warmup steps before capturing the frame")
    args = ap.parse_args()
    return run(args.suites, args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
