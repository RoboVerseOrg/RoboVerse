# Prepare Data (Deprecated)
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

## Data on Google Drive (Deprecated)

1. Please download the data from [here](https://drive.google.com/drive/folders/1ORMP3__KIlXettN8eUCF3YQNybZQxzkw) and put it under `./data`.

2. Download the converted isaaclab data from [here](https://drive.google.com/drive/folders/1nF-5SU4nC6S_vgC_NL7E7Rt63Zl9sYqW) and put it under `./data_isaaclab`.

3. Then link `./data/source_data` to `./data_isaaclab/source_data`.
    ```bash
    cd data_isaaclab
    ln -s ../data/source_data source_data
    ```

### Sync data with Google Drive
We recommend using rclone to sync the data with google drive.

#### Setup rclone
Create a rclone config under the [official guide](https://rclone.org/drive/), during which you need to create your own client id and token by following [this](https://rclone.org/drive/#making-your-own-client-id).

The final rclone config should look like this:
```
[roboverse]
type = drive
client_id = <your_client_id>
client_secret = <your_client_secret>
scope = drive
token = <your_token>
team_drive =
root_folder_id = 1ORMP3__KIlXettN8eUCF3YQNybZQxzkw

[roboverse_isaaclab]
type = drive
client_id = <your_client_id>
client_secret = <your_client_secret>
scope = drive
token = <your_token>
team_drive =
root_folder_id = 1nF-5SU4nC6S_vgC_NL7E7Rt63Zl9sYqW
```

#### Download data
```bash
rclone copy -P roboverse: data --exclude='demo/**'
rclone copy -P roboverse_isaaclab: data_isaaclab
```
You may want to speed up by setting `--multi-thread-streams=16` (default is 4).

#### Upload data
For example, if you are a developer who is migrating the rlbench task, you can upload the assets and source demos to google drive by running:
```bash
rclone copy -P data/assets/rlbench roboverse_isaaclab:assets/rlbench --exclude='.thumbs/**'
rclone copy -P data/source_data/rlbench roboverse:data/source_data --include='trajectory-unified.pkl'
```
You ***should*** replace `-P` with `-n` (means dry run) to check before uploading.
