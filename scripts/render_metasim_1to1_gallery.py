"""1:1 side-by-side gallery for the report: MetaSim-native vs reference, all 25 tasks.

For every task: run the MetaSim-API task and the proven _native env (== upstream) from the same seed
+ the same scripted reach-close-lift motion, in separate subprocesses (SAPIEN is process-global),
and compose a side-by-side mp4:

    [ MetaSim-native (ScenarioCfg+handler) | reference _native (=upstream) | abs-diff x30 ]

The diff panel is (near-)black -> the two are 1:1. Reuses the builders from spike_metasim_full_parity.

Run:  JAX_PLATFORMS=cpu python scripts/render_metasim_1to1_gallery.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spike_metasim_full_parity import ALL, SEED, _metasim, _native

OUT = "/tmp/metasim_1to1_gallery"
DIFF_GAIN = 30.0


def _scripted(n_down=5, n_close=3, n_lift=6):
    a = []
    for _ in range(n_down):
        a.append(np.array([0, 0, -0.06, 0, 0, 0, 1.0], dtype=np.float32))
    for _ in range(n_close):
        a.append(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    for _ in range(n_lift):
        a.append(np.array([0, 0, 0.06, 0, 0, 0, -1.0], dtype=np.float32))
    return a


def run(which, task):
    env = _native(task) if which == "native" else _metasim(task)
    env.reset(seed=SEED)
    rc = env.render_color if hasattr(env, "render_color") else env.scene_obj.render_color

    def raw():
        return np.clip(rc() * 255, 0, 255).astype(np.uint8)

    frames = [raw()]
    for a in _scripted():
        env.step(a)
        frames.append(raw())
    np.save(f"/tmp/r1_{which}_{task}.npy", np.stack(frames))


def compose():
    import cv2
    import imageio

    os.makedirs(OUT, exist_ok=True)
    for task in ALL:
        m = np.load(f"/tmp/r1_metasim_{task}.npy").astype(np.float64)
        n = np.load(f"/tmp/r1_native_{task}.npy").astype(np.float64)
        k = min(len(m), len(n))
        m, n = m[:k], n[:k]
        h = m.shape[1]
        bar = np.full((h, 4, 3), 255, np.uint8)
        out = []
        for i in range(k):
            diff = np.clip(np.abs(m[i] - n[i]) * DIFF_GAIN, 0, 255).astype(np.uint8)
            row = np.concatenate([m[i].astype(np.uint8), bar, n[i].astype(np.uint8), bar, diff], axis=1)
            cv2.putText(row, "MetaSim-native", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(
                row, "reference (=upstream)", (m.shape[2] + 14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            cv2.putText(row, "diff x30", (2 * m.shape[2] + 20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.append(row)
        out += [out[-1]] * 3
        imageio.mimsave(f"{OUT}/{task}.mp4", out, fps=7)
        md = float(np.abs(m - n).mean())
        print(f"OK {task} -> {OUT}/{task}.mp4 mean-abs {md:.4f}/255")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "metasim", "compose", "all"], default="all")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.mode in ("native", "metasim"):
        return run(a.mode, a.task)
    if a.mode == "compose":
        return compose()
    env = dict(os.environ, JAX_PLATFORMS="cpu", LOGURU_LEVEL="WARNING")
    for task in ALL:
        for w in ("native", "metasim"):
            if (
                subprocess.run([sys.executable, __file__, "--mode", w, "--task", task], env=env, check=False).returncode
                != 0
            ):
                print(f"FAIL {task} {w}")
                sys.exit(1)
    subprocess.run([sys.executable, __file__, "--mode", "compose"], env=env, check=False)


if __name__ == "__main__":
    main()
