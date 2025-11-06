# EmbodiedGen for 3D Asset and Interactive Scene Generation

[![📖 Documentation](https://img.shields.io/badge/📖-Documentation-blue)](https://horizonrobotics.github.io/EmbodiedGen/)
[![GitHub](https://img.shields.io/badge/GitHub-EmbodiedGen-black?logo=github)](https://github.com/HorizonRobotics/EmbodiedGen)
[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2506.10600)
[![🎥 Video](https://img.shields.io/badge/🎥-Video-red)](https://www.youtube.com/watch?v=rG4odybuJRk)
[![中文介绍](https://img.shields.io/badge/中文介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/HH1cPBhK2xcDbyCK4BBTbw)

[![🤗 Asset Gallery](https://img.shields.io/badge/🤗-EmbodiedGen_Asset_Gallery-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer)
[![🤗 Image-to-3D Demo](https://img.shields.io/badge/🤗-Image_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Image-to-3D)
[![🤗 Text-to-3D Demo](https://img.shields.io/badge/🤗-Text_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D)
[![🤗 Texture Generation Demo](https://img.shields.io/badge/🤗-Texture_Gen_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen)

---

> 🚀 **EmbodiedGen** provides a unified framework for generating **sim-ready 3D assets** and **interactive 3D scenes**, fully compatible with RoboVerse and multiple popular simulators.

We use [EmbodiedGen](https://horizonrobotics.github.io/EmbodiedGen) as the foundation platform for generating realistic, physically consistent, and simulation-ready 3D contents.  
You can seamlessly import generated assets into any RoboVerse simulator following these tutorials:
- [Import Assets](https://roboverse.wiki/metasim/get_started/quick_start/14_real_asset)
- [Import 3D Scene](https://roboverse.wiki/metasim/get_started/quick_start/16_embodiedgen_layout)
- [Import 3DGS Background](https://roboverse.wiki/metasim/get_started/quick_start/15_gs_background)

Explore the [EmbodiedGen Asset Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer) to browse and use generated sim-ready assets and follow [Any Simulators Tutorial](#any-simulators). To generate diverse sim-ready 3D assets and interactive 3D scenes by yourself, please [install EmbodiedGen](https://horizonrobotics.github.io/EmbodiedGen/install).


## 🧭 Overview

<img src="assets/overall.jpg" alt="Overall Framework" width="700"/>


<h2 id="image-to-3d">🖼️ Image-to-3D</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Image_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Image-to-3D)

Generate physically plausible 3D asset URDF from single input image, offering high-quality support for digital twin systems.
(HF space is a simplified demonstration. For the full functionality, please refer to [img3d-cli](https://horizonrobotics.github.io/EmbodiedGen/tutorials/image_to_3d)). Use generated 3D assets in RoboVerse, see [Import Assets Tutorial](https://roboverse.wiki/metasim/get_started/quick_start/14_real_asset).

```sh
img3d-cli --image_path apps/assets/example_image/sample_00.jpg apps/assets/example_image/sample_01.jpg apps/assets/example_image/sample_19.jpg \
--n_retry 1 --output_root outputs/imageto3d

# See result(.urdf/mesh.obj/mesh.glb/gs.ply) in ${output_root}/sample_xx/result
```

<h2 id="text-to-3d">📝 Text-to-3D</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Text_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D) 

Create 3D assets from text descriptions for a wide range of geometry and styles. (HF space is a simplified demonstration. For the full functionality, please refer to [text3d-cli](https://horizonrobotics.github.io/EmbodiedGen/tutorials/text_to_3d)). Use generated 3D assets in RoboVerse, see [Import Assets Tutorial](https://roboverse.wiki/metasim/get_started/quick_start/14_real_asset).

```sh
text3d-cli --prompts "small bronze figurine of a lion" "A globe with wooden base" "wooden table with embroidery" \
    --n_image_retry 1 --n_asset_retry 1 --n_pipe_retry 1 --seed_img 0 \
    --output_root outputs/textto3d
```


<h2 id="texture-generation">🎨 Texture Generation</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Texture_Gen_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen)

Generate visually rich textures for 3D mesh, detailed guide: [Texture Editing Tutorial](https://horizonrobotics.github.io/EmbodiedGen/tutorials/texture_edit/).

```sh
texture-cli --mesh_path "apps/assets/example_texture/meshes/robot_text.obj" \
"apps/assets/example_texture/meshes/horse.obj" \
--prompt "举着牌子的写实风格机器人，大眼睛，牌子上写着“Hello”的文字" \
"A gray horse head with flying mane and brown eyes" \
--output_root "outputs/texture_gen" \
--seed 0
```

<h2 id="3d-scene-generation">🌍 3D Scene Generation</h2>

Automatically generate background scenes (color mesh + 3D Gaussian Splatting) from text prompts. Typical runtime: ~30 minutes per scene. Details: [Scene Generation Generation Tutorial](https://horizonrobotics.github.io/EmbodiedGen/tutorials/scene_gen), [Import 3DGS Background](https://roboverse.wiki/metasim/get_started/quick_start/15_gs_background).

```sh
CUDA_VISIBLE_DEVICES=0 scene3d-cli \
--prompts "Art studio with easel and canvas" \
--output_dir outputs/bg_scenes/ \
--seed 0 \
--gs3d.max_steps 4000 \
--disable_pano_check
```

<h2 id="layout-generation">🏞️ Layout(Interactive 3D Worlds) Generation</h2>

Generating one interactive 3D scene from task description with [Layout Generation Tutorial](https://horizonrobotics.github.io/EmbodiedGen/tutorials/layout_gen/) takes approximately 30 minutes. Use generated layout in RoboVerse, see [Import 3D Scene](https://roboverse.wiki/metasim/get_started/quick_start/16_embodiedgen_layout).

```sh
layout-cli --task_descs "Place the pen in the mug on the desk" "Put the fruit on the table on the plate" \
--bg_list "outputs/bg_scenes/scene_list.txt" --output_root "outputs/layouts_gen" --insert_robot
```

<h2 id="any-simulators">🎮 Any Simulators</h2>

Use EmbodiedGen-generated assets with correct physical collisions and consistent visual effects in any simulator, see [Any Simulator Tutorial](https://horizonrobotics.github.io/EmbodiedGen/tutorials/any_simulators).

| Simulator | Conversion Class |
|-----------|------------------|
| [isaacsim](https://github.com/isaac-sim/IsaacSim) | MeshtoUSDConverter |
| [mujoco](https://github.com/google-deepmind/mujoco) / [genesis](https://github.com/Genesis-Embodied-AI/Genesis) | MeshtoMJCFConverter |
| [sapien](https://github.com/haosulab/SAPIEN) / [isaacgym](https://github.com/isaac-sim/IsaacGymEnvs) / [pybullet](https://github.com/bulletphysics/bullet3) | EmbodiedGen generated .urdf can be used directly |

## 📚 Citation

If you use EmbodiedGen in your research or projects, please cite:

```bibtex
@misc{wang2025embodiedgengenerative3dworld,
      title={EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence},
      author={Xinjie Wang and Liu Liu and Yu Cao and Ruiqi Wu and Wenkang Qin and Dehui Wang and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2506.10600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.10600},
}
```
