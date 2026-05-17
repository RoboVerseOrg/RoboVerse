"""Capture MetaSim's exported scenario XML by intercepting dm_control's
`mjcf.export_with_assets` call inside `MujocoHandler.launch()`.

We monkey-patch the function to also write a copy to a stable directory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import mujoco

from . import inventory as inv
from .full_runner import SCENARIO_BUILDERS


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    out_dir = Path("/tmp/mjlab_diag_cartpole")
    out_dir.mkdir(exist_ok=True)
    capture_dir = out_dir / "metasim_export"
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir()

    from dm_control import mjcf

    real_export = mjcf.export_with_assets

    def _patched(model, out_dir=None, *args, **kwargs):
        if out_dir is None and args:
            out_dir = args[0]
            args = args[1:]
        try:
            result = real_export(model, out_dir=out_dir, **kwargs)
        except TypeError:
            result = real_export(model, out_dir, *args, **kwargs)
        # Mirror to our capture dir
        try:
            for f in Path(out_dir).iterdir():
                shutil.copy(f, capture_dir / f.name)
        except Exception as e:
            print(f"(mirror failed: {e})")
        return result

    mjcf.export_with_assets = _patched

    task = inv.task_by_metasim_name("mjlab.cartpole_balance")
    builder = SCENARIO_BUILDERS["mjlab.cartpole_balance"]
    scenario = builder(task)
    raw = mujoco.MjModel.from_xml_path(str(task.asset.xml_path))
    scenario.sim_params.dt = float(raw.opt.timestep)

    from metasim.sim.mujoco import MujocoHandler

    handler = MujocoHandler(scenario)
    handler.launch()
    handler.close()

    files = sorted(capture_dir.iterdir())
    print(f"Captured {len(files)} files into {capture_dir}:")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

    raw_copy = out_dir / "raw_mjlab_cartpole.xml"
    shutil.copy(task.asset.xml_path, raw_copy)
    print(f"raw mjlab xml copied to {raw_copy}")


if __name__ == "__main__":
    main()
