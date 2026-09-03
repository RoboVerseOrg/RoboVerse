"""Generate a fresh native-LIBERO vs MetaSim side-by-side video for the docs.

Builds a LIBERO(-plus) task, replays a demo's actions for real motion, captures
the native LIBERO `agentview` render at each step, then re-renders each state
through MetaSim's own MuJoCo handler (the texture-fixed native_repro loader).
Composes a side-by-side (native | MetaSim) mp4 + a poster frame.

Run (liberoplus env):
    LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \\
    python -m tools.parity.gen_libero_sidebyside --suite libero_object \\
        --base pick_up_the_alphabet_soup_and_place_it_in_the_basket \\
        --variant _table_5 --frames 60 --out docs/source/_static/integrations/libero/sb_object_texture
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from roboverse_pack.tasks.libero.native_repro import make_native_handler, render_cam, set_flat_state
from roboverse_pack.tasks.libero_plus import _passthrough as pt

DEMO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "libero_datasets")


def _u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def run(suite, base, variant, frames, sz, out):
    names = pt.list_liberoplus_tasks(suite)
    if variant:
        tid = next(i for i, n in enumerate(names) if n.startswith(base) and variant in n)
    else:
        tid = next(i for i, n in enumerate(names) if n.startswith(base))
    task_name = names[tid]
    print(f"task: {task_name}  (tid={tid})")

    with h5py.File(os.path.join(DEMO_ROOT, suite, f"{base}_demo.hdf5"), "r") as h:
        actions = np.asarray(h["data"]["demo_0"]["actions"])[:frames]
        init_state = np.asarray(h["data"]["demo_0"]["states"])[0]

    # native LIBERO rollout: replay demo actions, capture agentview + flat states.
    # The raw robosuite agentview_image is OpenGL bottom-up; openvla's get_libero_image
    # rotates it 180 deg ([::-1, ::-1]) for the display/training-correct view.
    env = pt.make_liberoplus_env(suite, tid, camera_heights=sz, camera_widths=sz, seed=0)
    model_xml = env.env.sim.model.get_xml()
    obs = env.set_init_state(init_state)
    native, states = [], []
    for a in actions:
        native.append(_u8(np.asarray(obs["agentview_image"])[::-1, ::-1]))  # right-side-up
        states.append(np.asarray(env.env.sim.get_state().flatten()))
        obs, *_ = env.step(a.tolist())
    env.close()
    states = np.array(states)

    # MetaSim render of each state through its own MuJoCo handler
    handler, _xmlp, _info, cam = make_native_handler(model_xml, image_size=sz)
    meta = []
    for st in states:
        set_flat_state(handler, st)
        meta.append(_u8(render_cam(handler, cam, width=sz, height=sz, visual_only=True)))

    # orient the MetaSim panel to match the (correctly-oriented) native panel:
    # search the 4 axis flips and pick the one with lowest MAE on frame 0.
    transforms = {"id": lambda x: x, "v": lambda x: x[::-1], "h": lambda x: x[:, ::-1], "vh": lambda x: x[::-1, ::-1]}
    best = min(transforms, key=lambda k: np.abs(native[0].astype(int) - transforms[k](meta[0]).astype(int)).mean())
    meta = [transforms[best](m) for m in meta]
    mae = float(np.mean([np.abs(n.astype(int) - m.astype(int)).mean() for n, m in zip(native, meta, strict=False)]))
    print(f"MetaSim orientation match = '{best}'; per-frame native-vs-MetaSim pixel MAE = {mae:.2f}/255")

    sbs = [np.concatenate([n, m], axis=1) for n, m in zip(native, meta, strict=False)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio

    imageio.mimsave(
        f"{out}.mp4",
        sbs,
        fps=20,
        codec="libx264",
        quality=6,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    imageio.imwrite(f"{out}_poster.png", sbs[int(len(sbs) * 0.6)])
    # optimized GIF — plays inline everywhere incl. GitHub; an always-visible
    # fallback (the mp4 is the quality version). Shrink frames+palette until <480KB.
    for stride, res, psize in [(3, 3, 96), (4, 3, 64), (4, 4, 64), (5, 4, 48), (6, 4, 32)]:
        gif = [s[::res, ::res] for s in sbs[::stride]]
        imageio.mimsave(f"{out}.gif", gif, fps=8, loop=0, palettesize=psize, subrectangles=True)
        if os.path.getsize(f"{out}.gif") < 480 * 1024:
            break
    print(
        f"wrote {out}.mp4 ({os.path.getsize(out + '.mp4') // 1024} KB) + "
        f"{out}.gif ({os.path.getsize(out + '.gif') // 1024} KB) + poster"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--base", default="pick_up_the_alphabet_soup_and_place_it_in_the_basket")
    ap.add_argument("--variant", default="", help="substring of the perturbation variant, e.g. _table_5 / _light_3")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--sz", type=int, default=384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args.suite, args.base, args.variant, args.frames, args.sz, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
