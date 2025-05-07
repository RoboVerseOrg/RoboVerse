# Scenes in RoboVerse

RoboVerse supports a wide variety of **USD**-formatted assets that contribute to building immersive worlds. These assets are categorized for easy identification and usage. While not all assets are utilized in our demonstration tasks, they are available for you to incorporate into your own custom tasks. Below is an overview of the different types of assets available:

- [Scenes in RoboVerse](#scenes-in-roboverse)
  - [Ground Materials](#ground-materials)

## Ground Materials
Ground Materials are mainly adapted from [ARNOLD](https://arnold-benchmark.github.io/). These materials are ideal for texturing environments and can be applied easily in your tasks. Here’s an example of how you can use them:

```python
omni_python src/scripts/demo.py task=PickCube controller=ReplayController robot=Franka num_envs=4 demo_collection=true ground.material_mdl_path=data/source_data/arnold/materials/Wood/Bamboo.mdl

# other options
# data/source_data/arnold/materials/Wood/Walnut_Planks.mdl
# data/source_data/arnold/materials/Wood/Walnut.mdl
# data/source_data/arnold/materials/Wood/Birch_Planks.mdl
# data/source_data/arnold/materials/Wood/Cherry.mdl
# data/source_data/arnold/materials/Wood/Parquet_Floor.md
# data/source_data/arnold/materials/Wood/Mahogany.mdl
# data/source_data/arnold/materials/Wood/Cork.mdl
# data/source_data/arnold/materials/Wood/Plywood.mdl
# data/source_data/arnold/materials/Wood/Cherry_Planks.mdl
# data/source_data/arnold/materials/Wood/Timber_Cladding.mdl
# data/source_data/arnold/materials/Wood/Bamboo_Planks.mdl
# data/source_data/arnold/materials/Wood/Ash_Planks.mdl
# data/source_data/arnold/materials/Wood/Oak.mdl
# data/source_data/arnold/materials/Wood/Ash.mdl
# data/source_data/arnold/materials/Wood/Timber.mdl
# data/source_data/arnold/materials/Wood/Mahogany_Planks.mdl
# data/source_data/arnold/materials/Wood/Oak_Planks.mdl
# data/source_data/arnold/materials/Wood/Birch.mdl
# data/source_data/arnold/materials/Wood/Bamboo.mdl
# data/source_data/arnold/materials/Wood/Beadboard.mdl
```
