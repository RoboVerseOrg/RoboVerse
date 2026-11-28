"""Regenerate robots.md from the robot configs in ``roboverse_pack.robots`` (no hand-written rows).

python docs/source/dataset_benchmark/dataset/generate_robot_docs.py
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "metasim"))

import roboverse_pack.robots as robots_pkg
from metasim.scenario.robot import RobotCfg

FIELDS = (("mjcf_path", "MuJoCo"), ("usd_path", "Isaac Sim"), ("urdf_path", "URDF"), ("mjx_mjcf_path", "MJX"))


def main() -> None:
    rows = []
    for mod in pkgutil.iter_modules(robots_pkg.__path__):
        if mod.name.startswith("_") or mod.name in ("robot_template",):
            continue
        try:
            module = importlib.import_module(f"roboverse_pack.robots.{mod.name}")
        except Exception as err:  # a robot that needs an optional library is listed with the error, not hidden
            rows.append((mod.name, "", "", f"import failed: {type(err).__name__}", ""))
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, RobotCfg) or cls is RobotCfg or cls.__module__ != module.__name__:
                continue
            try:
                cfg = cls()
            except Exception as err:
                rows.append((mod.name, cls.__name__, "", f"construct failed: {type(err).__name__}", ""))
                continue
            assets = [
                label for field, label in FIELDS if isinstance(getattr(cfg, field, None), str) and getattr(cfg, field)
            ]
            rows.append((
                mod.name,
                cls.__name__,
                str(getattr(cfg, "num_joints", "") or ""),
                ", ".join(assets),
                getattr(cfg, "name", ""),
            ))
    rows.sort(key=lambda r: r[0])
    lines = [
        "# Robots",
        "",
        "Generated from `roboverse_pack/robots` by `generate_robot_docs.py`; do not edit by hand. A robot is listed",
        "with the asset formats its config declares; `tools/assets/check_manifest.py` verifies on every pull request",
        "that each declared file exists on the Hub. Robots whose assets did not exist were retired on 2026-11-28",
        "(`roboverse_pack/tasks/RETIRED_TASKS.md`).",
        "",
        "| config module | class | `name` | joints | asset formats |",
        "|---|---|---|---|---|",
    ]
    for mod_name, cls_name, joints, assets, name in rows:
        lines.append(f"| `{mod_name}` | `{cls_name}` | `{name}` | {joints} | {assets} |")
    lines += ["", f"{sum(1 for r in rows if r[1])} robot configs."]
    out = Path(__file__).with_name("robots.md")
    out.write_text("\n".join(lines) + "\n")
    sys.stderr.write(f"{out}: {len(rows)} rows\n")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
