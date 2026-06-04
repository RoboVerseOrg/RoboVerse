"""Build a native-vs-RoboVerse side-by-side 1:1 verification video for a RoboTwin task.

One command produces the proof artifact: it renders the RoboTwin ground truth and
the RoboVerse replay from the *identical* camera and the *identical* trajectory,
then composites them frame-for-frame with labels.

- The native side uses ``native_render.py --replay-bridge`` (robotwin env): it
  drives the RoboTwin env from the bridge trajectory (teleport robot qpos + object
  poses/joints per frame) rather than re-planning, so it is the SAME episode.
- The RoboVerse side uses ``mesh_replay_robotwin.py --observer-cam --rt`` (roboverse
  env) with the same camera pose.

Because the two envs have conflicting sapien/torch builds, each render runs in its
own conda env via that env's python; this orchestrator only needs imageio + PIL
(run it in the ``roboverse`` env).

Run::

    conda run -n roboverse python tools/robotwin_integration/sidebyside.py --task move_can_pot
"""

from __future__ import annotations

import argparse
import os
import subprocess

import imageio.v2 as imageio_w
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

_ROBOTWIN_PY = os.path.expanduser("~/miniconda3/envs/robotwin/bin/python")
_ROBOVERSE_PY = os.path.expanduser("~/miniconda3/envs/roboverse/bin/python")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_NATIVE = os.path.join(_HERE, "native_render.py")
_REPLAY = os.path.join(_HERE, "mesh_replay_robotwin.py")

# Per-task camera presets (pos, look_at, fovy) framing each task's objects; the
# default frames the table centre and suits most tabletop tasks.
CAM_PRESETS = {
    "default": ([0.0, 0.6, 1.35], [0.0, -0.3, 0.78], 55.0),
    "open_microwave": ([0.9, 0.45, 1.15], [-0.05, 0.05, 0.88], 50.0),
    "open_laptop": ([0.8, 0.5, 1.1], [-0.05, 0.05, 0.85], 50.0),
    "put_object_cabinet": ([0.9, 0.5, 1.2], [0.0, 0.1, 0.9], 52.0),
}


def _render(py: str, script: str, extra: list[str]) -> None:
    # Pin CWD to the repo root so the scripts' relative output paths resolve there.
    env = dict(os.environ, MUJOCO_GL="egl", SAPIEN_HEADLESS="1")
    subprocess.run([py, script, *extra], check=True, env=env, cwd=_REPO_ROOT)


def _label(width: int, text: str) -> np.ndarray:
    im = Image.new("RGB", (width, 26), (20, 23, 30))
    ImageDraw.Draw(im).text((8, 6), text, fill=(232, 128, 106))
    return np.array(im)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--bridge", default=None, help="bridge pkl (default ~/projects/robotwin/data/_rv_bridge/<task>.pkl)"
    )
    ap.add_argument("--cam", default=None, help="camera preset name (default: per-task or 'default')")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    bridge = args.bridge or os.path.expanduser(f"~/projects/robotwin/data/_rv_bridge/{args.task}.pkl")
    if not os.path.exists(bridge):
        raise FileNotFoundError(f"bridge pkl not found: {bridge} (collect it with collect_bridge.py first)")
    import pickle

    # native setup_demo must use the bridge's seed (object placement + stability
    # checks are seed-dependent; the bridge recorded the seed that succeeded).
    seed = int(pickle.load(open(bridge, "rb")).get("seed", 0))
    pos, look, fovy = CAM_PRESETS.get(args.cam or args.task, CAM_PRESETS["default"])
    cam = ["--cam-pos", *map(str, pos), "--cam-lookat", *map(str, look), "--fovy", str(fovy)]
    work = os.path.join(_REPO_ROOT, "outputs", "robotwin_coverage")
    os.makedirs(work, exist_ok=True)
    nat = os.path.join(work, f"_sbs_nat_{args.task}.mp4")
    rv = os.path.join(work, f"_sbs_rv_{args.task}.mp4")
    out = args.out or os.path.join(work, f"sidebyside_{args.task}.mp4")

    print(f"[{args.task}] rendering native (bridge-replay, robotwin env) ...")
    _render(
        _ROBOTWIN_PY, _NATIVE, ["--task", args.task, "--seed", str(seed), "--replay-bridge", bridge, *cam, "--out", nat]
    )
    print(f"[{args.task}] rendering RoboVerse replay (roboverse env) ...")
    # mesh_replay names its own video; remove any stale one, render, then move it.
    rv_tmp = os.path.join(work, f"mesh_replay_{args.task}_kinematic.mp4")
    if os.path.exists(rv_tmp):
        os.remove(rv_tmp)
    _render(
        _ROBOVERSE_PY, _REPLAY, ["--bridge", bridge, "--mode", "kinematic", "--video", "--observer-cam", "--rt", *cam]
    )
    if not os.path.exists(rv_tmp):
        raise RuntimeError(f"RoboVerse replay produced no video at {rv_tmp}")
    os.replace(rv_tmp, rv)

    natv, rvv = iio.imread(nat), iio.imread(rv)
    n = min(len(natv), len(rvv))
    print(f"[{args.task}] native {len(natv)}f, replay {len(rvv)}f -> compositing {n} frames")
    S = args.size

    def rs(im):
        return np.array(Image.fromarray(im[..., :3]).resize((S, S)))

    bar = np.concatenate(
        [_label(S, "RoboTwin native"), np.full((26, 6, 3), 20, np.uint8), _label(S, "RoboVerse replay")], axis=1
    )
    frames = [
        np.concatenate(
            [bar, np.concatenate([rs(natv[f]), np.full((S, 6, 3), 255, np.uint8), rs(rvv[f])], axis=1)], axis=0
        )
        for f in range(n)
    ]
    imageio_w.mimsave(out, frames, fps=args.fps)
    print(f"SAVED side-by-side -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
