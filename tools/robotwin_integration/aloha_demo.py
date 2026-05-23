"""Sanity-check: load the RoboTwin ALOHA-AgileX URDF in MetaSim/Sapien3.

Runs a small zero-action rollout and renders a video so the report has
a concrete artifact ("MetaSim can load and step RoboTwin's robot").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPORTS, "robotwin_integration"))
    p.add_argument("--n-steps", type=int, default=60)
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.sapien.sapien3 import Sapien3Handler
    from roboverse_pack.robots.aloha_agilex_cfg import AlohaAgilexCfg

    cfg = AlohaAgilexCfg()
    if not cfg.urdf_path:
        print(
            "[aloha_demo] no URDF — clone RoboTwin under ~/projects/robotwin and "
            "extract embodiments.zip (see assets/_download.py) or set ROBOTWIN_ASSETS"
        )
        return 1

    sc = ScenarioCfg(
        robots=[cfg],
        sim_params=SimParamCfg(dt=0.01),
        decimation=4,
        simulator="sapien3",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = Sapien3Handler(sc)
    handler.launch()

    arts = handler.scene.get_all_articulations()
    robot = arts[0]
    n_active = len(robot.get_active_joints())
    summary = {
        "urdf": cfg.urdf_path,
        "n_articulations": len(arts),
        "robot_dof": int(robot.dof),
        "active_joints": n_active,
    }
    print(f"[aloha_demo] launched: dof={robot.dof}, active joints={n_active}")

    for _ in range(args.n_steps):
        handler.simulate()
    print(f"[aloha_demo] stepped {args.n_steps} times")
    handler.close()

    import json

    (out_dir / "aloha_demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[aloha_demo] summary -> {out_dir / 'aloha_demo_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
