# Real2Sim

## Prerequisite
1. [supersplat](https://github.com/playcanvas/supersplat)
2. [colmap](https://colmap.github.io/) / [glomap](https://github.com/colmap/glomap)
3. [gsplat](https://github.com/nerfstudio-project/gsplat)
4. [StableNormal](https://github.com/Stable-X/StableNormal)
5. [2DGS](https://github.com/hugoycj/2d-gaussian-splatting-great-again)

## Process
1. Take a video
2. Run colmap
3. Run gsplat
4. Extract normal map (StableNormal)
5. Extract mesh (2DGS)
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

### 3. Run gsplat
Generate dense 3DGS from video.

### 4. Extract normal map (StableNormal)
Predict normal map from video.

### 5. Extract mesh (2DGS)
Extract mesh from video.

### 6. Recenter and reorientation
Recenter and reorientation the 3DGS and mesh.

### 7. Segment 3DGS and mesh
Segment the 3DGS and mesh.

### 8. Assign ID for 3DGS
Assign ID for each segment of 3DGS.

### 9. Fix kinemics & dynamics parameters
Fix kinemics and dynamics parameters for objects and robotic body links.

### 10. Align coordinate and scale (between mesh, 3DGS, and physics engines)
Align the coordinate and scale (between mesh, 3DGS, and physics engines).

### 11. Construct URDF
Construct URDF from the 3DGS.

### 12. Physics-awared 3DGS rendering (with FK, IK, and collision detection)
Physics-awared 3DGS rendering (with FK, IK, and collision detection).

### 13. Load URDF for simulation
Load URDF for simulation.
