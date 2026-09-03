# Asset Conversion

General pipeline: first convert MJCF(xmls) to URDF, then URDF to USD.

## MJCF to URDF
The script is located in `scripts/mjcf2urdf.py`. The general logic is to first load the MJCF to mujoco, parse the links and joints information with mujoco APIs and convert the asset and texture meshes (obj, mtl and png files). Users are expected to run the following scripts in mujoco environment. Below are some currently known problems that will be fixed in future iterations of the asset converter.

1: texture misalignment for .msh mesh files

2: joint parent_link and child_link mismatch for robot models

Converting a single MJCF file:
```bash
python scripts/mjcf2urdf.py path/to/mjcf path/to/urdf
```
This will create a URDF file to the path specified. Note that there may be an asset folder in the same directory as the URDF file when needed.

Converting a directory of MJCF files:
```bash
python scripts/batch_mjcf2urdf.py path/to/folder
```
This will generate all the URDF files recursively in the folder and its subfolders.

## URDF to USD
The script is located in `scripts/urdf2usd.py`. Everything is copied from isaacsim GitHub repo. Note that users are expected to install IsaacLab environment to use this script.

Converting a single USD file:
```bash
python scripts/urdf2usd.py path/to/urdf path/to/usd
```
This will create a USD file to the path specified.

Converting a directory of URDF files:
```bash
python scripts/batch_urdf2usd.py path/to/folder
```
This will generate all the USD files recursively in the folder and its subfolders. The script spawns isaacsim environments for each URDF conversion. USD files are saved in the same directory once the user closes the isaacsim environment.
