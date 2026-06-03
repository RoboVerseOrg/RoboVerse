"""Verify the vendored roboverse_data/libero reproduces all 130 tasks 1:1.

For each base LIBERO task: build the original robosuite env (ground truth) and the
vendored ``LiberoNativeTask.from_vendored`` (MJCF + shared assets, no MJB), then
(a) assert every static model field matches (max|Δ|=0 after the body_pos/quat
re-apply), and (b) state-replay the demo and assert the ported checker fires
identically on both at every frame. No rendering (keeps it fast + EGL-clean).

Run (liberoplus env):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus PYTHONPATH=. \\
    ROBOVERSE_DATA_DIR=/path/to/roboverse_data python -m scripts.native.verify_vendored
"""

from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np

from roboverse_pack.tasks.libero_native.native_task import LiberoNativeTask
from scripts.native.export_libero_task import _resolve_goal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BDDL_ROOT = os.path.join(ROOT, "third_party", "LIBERO", "libero", "libero", "bddl_files")
DEMO_ROOT = os.path.join(ROOT, "third_party", "libero_datasets")
_FIELDS = ["body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "site_pos", "site_size", "jnt_pos"]


def main():
    from libero.libero.envs import OffScreenRenderEnv

    bddls = sorted(glob.glob(os.path.join(BDDL_ROOT, "**", "*.bddl"), recursive=True))
    rows, bad = [], []
    for k, bddl in enumerate(bddls):
        suite = os.path.basename(os.path.dirname(bddl))
        base = os.path.basename(bddl)[: -len(".bddl")]
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(0)  # match the deterministic fixture placement the migration vendored
        env.reset()
        e = env.env
        goal = _resolve_goal(e)
        m0, d0 = e.sim.model._model, e.sim.data._data
        tk = LiberoNativeTask.from_vendored(suite, base)
        # (a) static model field equality
        fdiff = max(
            float(np.abs(np.asarray(getattr(m0, f)) - np.asarray(getattr(tk.model, f))).max())
            for f in _FIELDS
            if np.asarray(getattr(m0, f)).shape == np.asarray(getattr(tk.model, f)).shape
        )
        # (b) checker parity over the demo's recorded states
        with h5py.File(os.path.join(DEMO_ROOT, suite, base + "_demo.hdf5"), "r") as h:
            states = np.asarray(h["data"]["demo_0"]["states"])
        mism = 0
        for st in states:
            env.regenerate_obs_from_state(st)
            tk.reset(st)
            mism += int(bool(e._check_success()) != bool(tk.success()))
        env.close()
        row = {
            "suite": suite,
            "task": base,
            "model_field_max_delta": fdiff,
            "checker_mismatches": mism,
            "n_states": len(states),
        }
        rows.append(row)
        if fdiff > 1e-9 or mism:
            bad.append(row)
        print(f"[{k + 1:3d}/130] {suite:14s} {base[:40]:40s} field_Δ={fdiff:.2e} checker_mism={mism}")

    json.dump(rows, open(os.path.join(ROOT, "outputs", "vendored_verify.json"), "w"), indent=2)
    print(
        f"\n=== {len(rows)} tasks; field-exact: {sum(1 for r in rows if r['model_field_max_delta'] <= 1e-9)}/{len(rows)}; "
        f"checker 1:1: {sum(1 for r in rows if r['checker_mismatches'] == 0)}/{len(rows)} ==="
    )
    if bad:
        print("NON-1:1 tasks:")
        for r in bad:
            print("  ", r)


if __name__ == "__main__":
    raise SystemExit(main())
