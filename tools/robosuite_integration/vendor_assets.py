"""Vendor a robosuite task's compiled MJCF + meshes into a standalone bundle.

Run ONCE while robosuite is installed. It compiles the task model, copies every
referenced mesh/texture into ``<out>/assets`` (preserving the robosuite-relative
subtree), and rewrites the MJCF's ``file=`` paths to that bundle so the model
loads with raw mujoco / MetaSim's handler and **no robosuite import**.

The 44 MB of meshes are data (they belong in roboverse_data / HF like every other
RoboVerse pack), so the bundle dir is git-ignored; this script regenerates it.

Usage:
    python -m tools.robosuite_integration.vendor_assets --env Lift --out .robosuite_bundle/lift
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)


def vendor(env_name: str, out_dir: Path, robot: str = "Panda") -> Path:
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    env = robosuite.make(
        env_name, robots=robot, controller_configs=cfg, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, control_freq=20, horizon=200, ignore_done=True,
    )
    env.reset()
    xml = env.model.get_xml()
    env.close()

    assets_root = out_dir / "assets"
    refs = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    common = os.path.commonpath([r for r in refs if os.path.exists(r)])
    for ref in refs:
        if not os.path.exists(ref):
            continue
        rel = os.path.relpath(ref, common)
        dst = assets_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(ref, dst)
        xml = xml.replace(f'file="{ref}"', f'file="{rel}"')

    # point mesh/texture dirs at the bundle's assets root (relative to the xml)
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="assets/"', xml)
    xml = re.sub(r'texturedir="[^"]*"', 'texturedir="assets/"', xml)
    if "texturedir" not in xml:
        xml = xml.replace("<compiler ", '<compiler texturedir="assets/" ', 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{env_name.lower()}.xml"
    model_path.write_text(xml)
    n = sum(1 for _ in assets_root.rglob("*") if _.is_file())
    print(f"vendored {env_name}: {n} asset files -> {assets_root}")
    print(f"standalone model -> {model_path}")
    return model_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Lift")
    ap.add_argument("--out", type=Path, default=Path(".robosuite_bundle/lift"))
    vendor(ap.parse_args().env, ap.parse_args().out)
