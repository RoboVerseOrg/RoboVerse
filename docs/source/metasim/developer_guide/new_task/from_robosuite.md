# From RoboSuite

## Asset convert from mjcf
### 1. Find anf pre-process mjcf for Arena, object
- For Arena: No changes
- For Single Object: Delete the **<\body>** label between **<\world_body>** and **<\object>**
- For Articulated Object:
### 2. Convert mjcf to usd
```
python scripts/convert_mjcf.py \
data_isaaclab/robosuite/objects/square-nut-delete-body.xml \
tmp/nut_assembly/square_nut/square_nut.usd --import-sites --make-instanceable
```

### 3. Post-process USD
**Note that every RigidBody in sim need**
1. CoACD mesh Processing
1. Set default prim for each usd and add rigid body with collision reset
```
pip install coacd
```
```
python scripts/coacd_preprocess.py \
-i tmp/nut_assembly/square_nut/square_nut_object.obj \
-o tmp/nut_assembly/square_nut/square_nut_coacd.obj
```
- For Arena except table: Decouple USD into serveral USD
- For Table: Find the size from task definition in the robosuite
- For Object: export mesh (maybe using blender) and then coacd

### 4.Troubleshooting
- The transformation from Blender maybe different with original USD
