# box_task

OpenArm Wuji bimanual replay task. The robot stands in front of a fixed
table that carries three rigid objects (`cardboard_box`,
`feast_soda_can`, `feast_scented_candle`). The shipped trajectory drives
the arms through a packing motion across 849 frames.

Registered names:

- `box_task.replay` — canonical
- `box_task` — alias

## One-shot asset preparation

The recording artifact lives outside the repo at
`box_task_replay_render_bundle_clean/`. Stage its assets into the
canonical `roboverse_data/` layout with:

```bash
python scripts/prepare_box_task_assets.py \
    --bundle ~/projects/RoboVerse/box_task_replay_render_bundle_clean
```

This copies the robot MJCF + meshes, patches a few naming quirks so
the file matches `OpenarmBimanualWujiCfg`, copies the per-object MJCFs
alongside their USDs, and converts the legacy v2 trajectory pkl into
the canonical form.

Staged paths:

| Path | Contents |
| --- | --- |
| `roboverse_data/robots/openarm_wuji/` | Robot MJCF + meshes |
| `roboverse_data/assets/box_task/local_pack_box/` | Per-object USD + MJCF |
| `roboverse_data/trajs/box_task/task3_openarm_bimanual_wuji_v2.pkl` | 849-frame v2 trajectory |

The trajectory was originally recorded under the older robot naming
(`openarm_wuji` + `{side}_hand_finger{i}_joint{j}`). The current robot
uses `openarm_bimanual_wuji` and `{side}_finger{i}_joint{j}`; the
prepare script converts pkl keys and joint names so no runtime remap
hook is needed.

## Replay rendering

`examples/replay_multi_scene_render.py` is the task-agnostic
replay-render CLI.

### Mujoco (no extra setup)

Mujoco needs neither USD scenes nor a GPU rendering pipeline; pass
`--simulator mujoco --scenes none` to render the bare scenario:

```bash
MUJOCO_GL=egl python examples/replay_multi_scene_render.py \
    --task box_task.replay \
    --simulator mujoco --scenes none \
    --duration-sec 10 --fps 30 --width 512 --height 512 \
    --out-video out/box_task_mujoco_10s.mp4
```

### IsaacSim (photoreal, switchable Kujiale backgrounds)

```bash
python examples/replay_multi_scene_render.py \
    --task box_task.replay \
    --scenes 0021,0022,0024,0025,0031 \
    --render-mode raytracing \
    --out-video out/box_task_switch5_ray.mp4
```

Higher-quality 22.5 s pathtraced version:

```bash
python examples/replay_multi_scene_render.py \
    --task box_task.replay \
    --scenes 0021,0022,0024,0025,0031 \
    --render-mode pathtracing \
    --duration-sec 22.5 --fps 30 --width 800 --height 800 \
    --out-video out/box_task_path_22p5s.mp4
```

Kujiale scene cfgs live in `roboverse_pack.scenes`. If a scene's USD
is missing locally, prepare it with the existing helper:

```python
from roboverse_pack.asset.setup_interior_agent_assets import (
    download_roboverse_usd, copy_usd_assets, ensure_kujiale_scene,
)
download_roboverse_usd()
copy_usd_assets(scene_ids=["0021", "0022", "0024", "0025", "0031"])
for sid in ["0021", "0022", "0024", "0025", "0031"]:
    ensure_kujiale_scene(sid)
```
