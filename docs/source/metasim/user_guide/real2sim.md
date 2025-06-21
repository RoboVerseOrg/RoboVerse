# Real2Sim

## Prerequisite
1. [supersplat](https://github.com/playcanvas/supersplat)
2. [colmap](https://colmap.github.io/) / [glomap](https://github.com/colmap/glomap)
3. [robogs](https://github.com/louhz/robogs)
4. [StableNormal](https://github.com/Stable-X/StableNormal)


### note for installation
for stablenormal, please follow this issue if you have installation issue: https://github.com/Stable-X/StableNormal/issues/34

for detail of the process 4-13 please follow the docoment from robogs

After you have the URDF(mjcf), You can utilize the infra of Roboverse for training and sim2real deployment.

## Process
1. Take a video
2. Run colmap
3. Extract normal map (StableNormal)
4. Run Reconstruction
5. Extract mesh 
6. Recenter and reorientation
7. Segment 3DGS and mesh
8. Assign ID for 3DGS
9. Fix kinemics & dynamics parameters
10. Align coordinate and scale (between mesh, 3DGS, and physics engines)
11. Construct URDF
12. Physics-awared 3DGS rendering (with FK, IK, and collision detection)
13. Load URDF for simulation

```{mermaid}
flowchart-elk LR
    start{Start} --1--> Video --2--> Cameras
    Video & Cameras --3--> 3DGS
    Video --4--> Normal
    Video & Normal --5--> Mesh
    Mesh -->|6,7,9,10| Mesh
    3DGS -->|6,7,8,9,10| 3DGS
    Mesh --11--> URDF
    3DGS --12--> 3DGS_render{3DGS Rendering}
    URDF --13--> Simulation{Physics Simulation}
```

### 1. Take a video
360 degree video around the object, table, arm, etc.

### 2. Run colmap
Extract camera poses and sparse point cloud from video.


    
### 3. Extract normal map (StableNormal)
Predict normal map from video.

inference every images we use for reconstruction and save it to folder name:normals


# citation

if you find this is helpful, please cite robogs and stablenormal

```bibtex
@misc{lou2024robogsphysicsconsistentspatialtemporal,
  title={Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm with Hybrid Representation}, 
  author={Haozhe Lou and Yurong Liu and Yike Pan and Yiran Geng and Jianteng Chen and Wenlong Ma and Chenglong Li and Lin Wang and Hengzhen Feng and Lu Shi and Liyi Luo and Yongliang Shi},
  year={2024},
  eprint={2408.14873},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2408.14873}, 
}

@misc{ye2024stablenormalreducingdiffusionvariance,
      title={StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal}, 
      author={Chongjie Ye and Lingteng Qiu and Xiaodong Gu and Qi Zuo and Yushuang Wu and Zilong Dong and Liefeng Bo and Yuliang Xiu and Xiaoguang Han},
      year={2024},
      eprint={2406.16864},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.16864}, 
}
```
