# Real2Sim

## Prerequisite
1. [supersplat](https://github.com/playcanvas/supersplat)
2. [colmap](https://colmap.github.io/) / [glomap](https://github.com/colmap/glomap)
3. [gsplat](https://github.com/nerfstudio-project/gsplat)
4. [StableNormal](https://github.com/Stable-X/StableNormal)
5. [2DGS](https://github.com/hugoycj/2d-gaussian-splatting-great-again)

### note for installation
for stablenormal, please follow this issue if you have installation issue: https://github.com/Stable-X/StableNormal/issues/34



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
Generate dense 3DGS from video:
python vis/gsplat_trainer.py default \
    --data_dir $data_path$ \
    --data_factor 1 \
    --result_dir $gs_data_path$

export gs ply:
python vis/extract_ply.py default \
    --ckpt  $gs_data_path$/ckpts/ckpt_29999_rank0.pt \
    --data_factor 1 \
    --export_ply_path  $gs_data_path$/scan30000.ply \
    --data_dir $data_path$
    
### 4. Extract normal map (StableNormal)
Predict normal map from video.

inference every images we use for reconstruction and save it to folder name:normals

### 5. Extract mesh (2DGS)
Extract mesh from video.

for 2dgs, you just need to install the submodules after you install the gsplat: pip install submodules/diff-surfel-rasterization
here is the running command: 
-s means the data source $data_path$

-m is the result path $mesh_result_path$

python train.py -s  $data_path$  -r 2  --contribution_prune_ratio 0.5 --lambda_normal_prior 1 --lambda_dist 10 --densify_until_iter 3000 --iteration 7000  -m $mesh_result_path$ --w_normal_prior normals


python render.py -s  $data_path$ -m $mesh_result_path$

### 6. Recenter and reorientation(fix coordinate)
Recenter and reorientation the 3DGS and mesh.

demo video of how to recenter and reorientation 

keep the recenter and reorientation vector 

### 7. Segment 3DGS and mesh
Segment the 3DGS and mesh.


a demo video here for manual:


A sam based method will be released later: 


### 8. Assign ID for 3DGS
Assign ID for each segment of 3DGS.

run assign.py with your custom id setting

### 9. Fix kinemics & dynamics parameters
Fix kinemics and dynamics parameters for objects and robotic body links.

fix the mdh hyperparameter


### 10. Align coordinate and scale (between mesh, 3DGS, and physics engines)
Align the coordinate and scale (between mesh, 3DGS, and physics engines).

fix the coordinate and scale(normally it is automatic)

run icp for the scale registration

### 11. Construct URDF(MJCF)
Construct URDF(MJCF) from the 3DGS.

run the ik from the mdh we cumstomizer, the fk to the new urdf that is aligned with the real world robot setting.

Setup the table, object mesh and the relative pose in regard of the robotic arm base(by compute the bbox and center of mass)

generate the urdf and mjcf with the physics parameter infered from VLM.


### 12. Load URDF(MJCF) for simulation
Load URDF(MJCF) for simulation.


### 13. Physics-awared 3DGS rendering (with FK, IK, and collision detection)

fix the binding and perform 4d rendering 

Physics-awared 3DGS rendering (with FK, IK, and collision detection).


