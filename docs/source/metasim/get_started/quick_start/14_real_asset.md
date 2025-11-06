#  14. Real Asset

[![GitHub](https://img.shields.io/badge/GitHub-EmbodiedGen-black?logo=github)](https://github.com/HorizonRobotics/EmbodiedGen)
[![📖 Documentation](https://img.shields.io/badge/📖-Documentation-blue)](https://horizonrobotics.github.io/EmbodiedGen/)

In this tutorial, we will show you how to use real asset generated from [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) in MetaSim.

You can visited more all-simulator-ready assets in [here](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).

You can generate any objects or scene assets you need, [reference](https://github.com/RoboVerseOrg/RoboVerse/tree/main/generation).

## Common Usage
```bash
python get_started/14_real_assets.py  --sim <simulator>
```

In headless mode:
```bash
python get_started/14_real_assets.py  --sim isaacsim --headless
python get_started/14_real_assets.py  --sim isaacgym --headless
MUJOCO_GL=egl python get_started/14_real_assets.py  --sim mujoco --headless
python get_started/14_real_assets.py  --sim genesis --headless
python get_started/14_real_assets.py  --sim sapien3 --headless
python get_started/14_real_assets.py  --sim pybullet --headless
```

You will get the following image:
---
| Isaac Lab | Isaac Gym | Mujoco |
|:---:|:---:|:---:|
| ![Isaac Lab](../../../_static/standard_output/14_real_assets_isaacsim.png) | ![Isaac Gym](../../../_static/standard_output/14_real_assets_isaacgym.png) | ![Mujoco](../../../_static/standard_output/14_real_assets_mujoco.png) |

| Genesis | Sapien | PyBullet |
|:---:|:---:|:---:|
| ![Genesis](../../../_static/standard_output/14_real_assets_genesis.png) | ![Sapien](../../../_static/standard_output/14_real_assets_sapien3.png) | ![Pybullet](../../../_static/standard_output/14_real_assets_pybullet.png) |

## Asset Converter

Use [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) generated assets with correct physical collisions and consistent visual effects in MetaSim.
([isaacsim](https://github.com/isaac-sim/IsaacSim), [mujoco](https://github.com/google-deepmind/mujoco), [genesis](https://github.com/Genesis-Embodied-AI/Genesis), [pybullet](https://github.com/bulletphysics/bullet3), [isaacgym](https://github.com/isaac-sim/IsaacGymEnvs), [sapien](https://github.com/haosulab/SAPIEN)).
Example in `generation/tests/test_asset_converter.py`.

| Simulator | Conversion Class |
|-----------|------------------|
| [isaacsim](https://github.com/isaac-sim/IsaacSim) | MeshtoUSDConverter |
| [mujoco](https://github.com/google-deepmind/mujoco) | MeshtoMJCFConverter |
| [genesis](https://github.com/Genesis-Embodied-AI/Genesis) / [sapien](https://github.com/haosulab/SAPIEN) / [isaacgym](https://github.com/isaac-sim/IsaacGymEnvs) / [pybullet](https://github.com/bulletphysics/bullet3) | [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) generated .urdf can be used directly |

<img src="../../../_static/standard_output/14_real_assets_collision.jpg" alt="simulators_collision" width="600">


```py
from huggingface_hub import snapshot_download
from generation.asset_converter import AssetConverterFactory, AssetType

data_dir = "roboverse_data/assets/EmbodiedGenData"
snapshot_download(
    repo_id="HorizonRobotics/EmbodiedGenData",
    repo_type="dataset",
    local_dir=data_dir,
    allow_patterns="demo_assets/*",
)

target_asset_type = AssetType.MJCF # or AssetType.USD

urdf_paths = [
    f"{data_dir}/demo_assets/remote_control/result/remote_control.urdf",
]

if target_asset_type == AssetType.MJCF:
    output_files = [
        f"{data_dir}/demo_assets/remote_control/mjcf/remote_control.mjcf",
    ]
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.MJCF,
        source_type=AssetType.URDF,
    )
elif target_asset_type == AssetType.USD:
    output_files = [
        f"{data_dir}/demo_assets/remote_control/usd/remote_control.usd",
    ]
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.USD,
        source_type=AssetType.MESH,
    )

with asset_converter:
    for urdf_path, output_file in zip(urdf_paths, output_files):
        asset_converter.convert(urdf_path, output_file)
```