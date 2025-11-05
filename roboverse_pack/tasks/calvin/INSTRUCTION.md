# Calvin Task Migration Guide

## What's Finished Now

[x] Loading data from the Calvin dataset
[x] Static environment setup for calvin_scene_A
[x] Joint-state based control of the robot

## Missing Blocks

[ ] Implement the functions of light and buttom in the Calvin Env.
[ ] Split the large dataset into small clips and save as trajectories.
[ ] Write the cfg file for all scenes (A, B, C, D)

## TODO

1. Implement the functions of light and buttom in the Calvin Env. (Optional)

The detailed functions of light and buttom can be checked from the original repo of calvin env: calvin_env/scene/objects

2. Split the large dataset into small clips and save as trajectories.

Currently, the dataset consists of long sequences of robot motion. As error accumulates over time, the later part of the sequence may not be very accurate. The result is that the robot may miss the object when it tries to pick it up. To solve this problem, we can split the long sequence into small clips, each of which contains one or two robot-object interactions. This way, we can reduce the error accumulation and improve the quality of the trajectories.

3. Write the cfg file for all scenes (A, B, C, D)

Currently, we only wrote the cfg file for calvin_scene_A. We need to write the cfg file for all scenes (A, B, C, D) so that we can use them in the training and evaluation.

## Deliverables

[ ] Cfg files for calvin_scene_A/B/C/D, which are implemented in `roboverse_pack/tasks/calvin/scene_A/B/C/D.py`, can inherit from `base_table.py`.

[ ] A dataset of small clips saved as trajectories. Can be loaded by `scripts/advanced/replay_demo.py`.

[ ] (Optional) Correct behaviors of lights in the environment, implemented in `roboverse_pack/tasks/calvin/base_table.py`

## Instructions

### Original Calvin Dataset

https://github.com/mees/calvin

This repo contains the raw dataset of calvin.

In dataset/README.md, you can see how the dataset is structured, and how to download the dataset.

### Original Calvin Environment

https://github.com/mees/calvin_env/tree/main

This repo contains the environment setup for calvin.

### RoboVerse Infra

You can refer to the [RoboVerse Wiki](https://roboverse.wiki/) for how to write a new task.

The sections you need to look at are:

- User Guide/Concept/RoboVerse Project Architecture
- User Guide/Concept/Configuration System
- User Guide/Get Started/0. Static Scene
- User Guide/Get Started/1. Control Robot
