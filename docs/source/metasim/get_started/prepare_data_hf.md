# Store Data Locally
If you are not a developer, you don't have to follow this instruction. The required data will be downloaded automatically from HuggingFace when you run the code.

For the current stage, the HuggingFace dataset is private. So you need a token to access it. See tips for using HuggingFace [here](../developer_guide/tips/huggingface.md).

## Data on HuggingFace

See tips for using HuggingFace [here](../developer_guide/tips/huggingface.md).

Download the release-ready data by running:
```bash
git clone git@hf.co:datasets/RoboVerseOrg/roboverse_data
```

```{note}
If you haven't used Git LFS on your machine, run the following command to install it:
```bash
sudo apt install git-lfs
git lfs install
```

### File structure

```text
roboverse_data
|-- robots
    |-- {robot}
        |-- usd
            |-- *.usd
        |-- urdf
            |-- *.urdf
        |-- mjcf
            |-- *.xml

|-- trajs
    |-- {benchmark}
        |-- {task}
            |-- v2
                |-- {robot}_v2.pkl

|-- assets
    |-- {benchmark}
        |-- {task}
            |-- COMMON
            |-- {object}
                |-- textures
                    |-- *.png
                |-- usd
                    |-- *.usd
                |-- urdf
                    |-- *.urdf
                |-- mjcf
                    |-- *.xml
|-- materials
    |-- arnold
        |-- {Category}
            |-- *.mdl
    |-- vMaterial_2
        |-- {Category}
            |-- *.mdl
    |-- ...

|-- scenes
    |-- {source} (e.g. arnold, physcene)
        |-- {scene}
            |-- textures
                |-- *.png
            |-- usd
                |-- *.usd
            |-- urdf
                |-- *.urdf
            |-- mjcf
                |-- *.xml

metasim
|-- data
    |-- quickstarts
        |-- ...
    |-- robots (deprecated)
        |-- {robot}
            |-- usd
                |-- *.usd
            |-- urdf
                |-- *.urdf
            |-- mjcf
                |-- *.xml
```
Explanation:
- When assets are reused in multiple tasks, they can be stored in a `COMMON` folder to avoid redundancy.

Naming convention:
- The {benchmark} is in lowercase
- The {task} is in snake_case
- The {robot} is in snake_case
- The {object} is in snake_case
