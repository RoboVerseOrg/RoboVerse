# ENVIRONMENTS.md

This file defines the local environment correspondence for simulator-backed development and testing.

Update it when the actual conda env names on the machine differ from the defaults below.

## Simulator Environment Correspondence

- `isaacgym` -> conda env `isaacgym`
- `mujoco` -> conda env `isaacgym`
- `newton` -> conda env `newton`
- `isaacsim` -> conda env `isaacsim`
- `blender` -> conda env `isaacsim`
- `superdex` -> a Python 3.12 venv with `.[dev,superdex]` (the superdex wheels are 3.12-only)

## Notes

- Keep this file simple and human-editable.
- If a simulator env is unknown or not configured, ask the user before running simulator-backed tests.

## Rendering validation environments

The portable visual-recipe tests (`metasim/test/randomization/test_visual_render.py`) and
`examples/6_advanced_rendering.py` need one environment per renderer. The combination that
was validated for this code:

- Blender: `bpy==4.2.0` with `usd-core==25.5` (Blender's USD import of authored asset
  materials; `usd-core==24.5` crashed on import next to `bpy` 4.2), MuJoCo 3.13, NumPy 2.
- Isaac Sim 5.0 with Isaac Lab 2.2.1 (NumPy < 2). Do not install USD Python bindings into
  the Isaac environment; it uses the bindings bundled with Kit.

Both can be plain venvs created with `--system-site-packages` on top of an existing
simulator environment, with both packages of this monorepo installed editable.

Vulkan is checked separately from CUDA. If the default NVIDIA GLX Vulkan ICD cannot expose
`vkCreateInstance` (headless nodes), point Kit at NVIDIA's EGL entry point with a
process-local ICD copy instead of touching system driver files:

```bash
python - <<'PY'
import json
from pathlib import Path
icd = json.loads(Path("/etc/vulkan/icd.d/nvidia_icd.json").read_text())
icd["ICD"]["library_path"] = "libEGL_nvidia.so.0"
Path("nvidia_egl_icd.json").write_text(json.dumps(icd))
PY
export VK_ICD_FILENAMES=$PWD/nvidia_egl_icd.json
export OMNI_KIT_ACCEPT_EULA=YES
```

Reference: [NVIDIA installed components](https://download.nvidia.com/XFree86/Linux-x86_64/580.65.06/README/installedcomponents.html).

Optional assets for the example: PBR texture sets (`<name>_BaseColor/_Normal/_ORM.png`,
e.g. `materials/arnold/...` from `RoboVerseOrg/roboverse_data`) for `--material-library`,
HDRIs for `--hdri-library`, and interior scenes for `--background` (see
`roboverse_pack/asset/setup_interior_agent_assets.py` and the InteriorAgent terms of use).
Recipes store absolute asset paths, so keep libraries at a stable location.

## Rendering smoke commands

Run from `packages/metasim` in the corresponding renderer environment. They exercise rendered
image changes, recipe replay, actual camera calibration, depth of field, box UV projection and
per-environment isolation; the Isaac suite runs with one, two and four environments.

```bash
MUJOCO_GL=egl python -m pytest metasim/test/randomization/test_visual_render.py -k blender -q

VK_ICD_FILENAMES=... OMNI_KIT_ACCEPT_EULA=YES MUJOCO_GL=egl \
  python -m pytest metasim/test/randomization/test_visual_render.py -k isaacsim -q
```
