"""Render the RoboTwin ALOHA-AgileX standing rollout in MetaSim/Sapien3.

Output: an mp4 at `reports/robotwin_integration/runs/aloha_demo/render.mp4`
showing the dual-arm robot under MetaSim physics. Quick visual proof
that the embodiment loaded + the new passive-joints handler patch is
behaving sensibly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/ghr/projects/RoboVerse/reports/robotwin_integration")
    p.add_argument("--n-steps", type=int, default=60)
    args = p.parse_args(argv)
    out_dir = Path(args.out)

    import sapien

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.sapien.sapien3 import Sapien3Handler
    from roboverse_pack.robots.aloha_agilex_cfg import AlohaAgilexCfg

    cfg = AlohaAgilexCfg()
    if not cfg.urdf_path:
        print("[aloha_render] no URDF — see aloha_demo.py for setup steps")
        return 1

    sc = ScenarioCfg(
        robots=[cfg],
        sim_params=SimParamCfg(dt=0.01),
        decimation=4,
        simulator="sapien3",
        num_envs=1,
        headless=True,
        add_default_ground=True,  # show the robot on a floor
    )

    handler = Sapien3Handler(sc)
    handler.launch()
    scene = handler.scene

    cam = scene.add_camera("rgb_demo", width=480, height=320, fovy=1.0, near=0.05, far=10)
    # Stand off to the side and look at the robot's torso.
    cam.set_local_pose(sapien.Pose([1.4, -0.6, 1.4], [0.40, 0.20, 0.50, 0.74]))

    frames: list[np.ndarray] = []
    for _ in range(args.n_steps):
        handler.simulate()
        scene.update_render()
        cam.take_picture()
        rgb = np.asarray(cam.get_picture("Color"))
        if rgb.dtype != np.uint8 and rgb.max() <= 1.5:
            rgb = (rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb[..., :3].astype(np.uint8)
        frames.append(rgb)

    handler.close()

    out_mp4 = out_dir / "runs" / "aloha_demo" / "render.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        imageio.mimwrite(out_mp4, frames, fps=25, codec="libx264", quality=7)
        print(f"[aloha_render] wrote {out_mp4}")
    except Exception as e:
        print(f"[aloha_render] imageio failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
