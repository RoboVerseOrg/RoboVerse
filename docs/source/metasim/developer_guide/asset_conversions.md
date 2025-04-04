# Asset Conversion

General pipeline: first convert MJCF(xmls) to URDF, then URDF to USD.

## MJCF to URDF
The script is located in `metasim/scripts/convert_mjcf_to_urdf.py`. The general logic is to first load the MJCF to mujoco, parse the links and joints information with mujoco APIs and convert the asset and texture meshes (obj, mtl and png files)

### Prerequisites and file structure ettiquettes

Current asset converter, unfortunately and of course, is not robust enough to handle all variation of MJCF files and the corresponding textures. Current limitations include accurate conversion of .msh files to .obj files with correctly aligned .mtl textures.

Developers are expected to strictly follow the following guidelines to correctly position/name the MJCF, mesh and texture files.

#### Case 0: Converting a simple MJCF file with no mesh and no texture:
~~Ur MJCF asset is boring and doesn't deserve a conversion.~~ The current asset converter actually does skip these MJCF file since boxes and cylinders are natively supported by isaaclab and isaacgym. Developers are expected to see an empty URDF file generated, but this might be changed in future iterations of the asset converter.

#### Case 1: MJCF asset does have a mesh file, but the mesh file does not import any texture:

The overall folder should include a mujoco xml file, with a mesh file (.msh, .obj, .stl) in the same directory with the xml file or in a subfolder. As long as the mesh file is present, the converter will automatically select the correct mesh file and do the conversion.

#### Case 2: MJCF asset does have a .obj or .stl mesh file, and the mesh file imports a texture:

Developers have to make sure the name of the .obj file shares the same name as the .mjcf file. The converter will skip the .msh to .obj conversion and use the .obj file directly as current msh2obj converter does not support texture conversion. The .obj file should clearly list the path to the .mtl texture file and the .mtl texture file should clearly list the path to the .png texture map.

#### Case 3: MJCF asset only has a .msh file, and the .msh file imports a texture file:

This is an extreme fringe case which is technically not really supported by the current asset converter. We do observe that for some of the time, the generated texture is suprisingly aligned. For most of the time, developers should avoid having a .msh file with texture map without a .obj file.

### Conversion

Converting a single MJCF file:
```bash
python scripts/mjcf2urdf.py path/to/mjcf path/to/urdf
```
This will create a URDF file to the path specified. Note that there may be an asset folder in the same directory as the URDF file when needed.

Converting a directory of MJCF files:
```bash
python scripts/batch_mjcf2urdf.py path/to/folder
```
This will generate all the URDF files recursively in the folder and its subfolders. Note that additional folders are generated for saving the texture and mesh files.

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
