# Contributing Trajectories

With a task defined, we can also define the related trajectories for the task, to be the demonstration for the environment. 

## 1. Record the Trajectories

Recording can be done in any fasion, such as teleoperation, scripted policy, or RL policy.

The recorded trajectories should be saved in the v2 trajectory format, which is a dumped dict defined in [State](https://roboverse.wiki/metasim/concept/state#trajectory).

## 2. Replay the Trajectories

By running the following command, you can replay trajectory:

```bash
python scripts/advanced/replay_demo.py --sim=${simulator} --robot=${robot_name} --task=${task_name} --traj_path=${path_to_traj}
```

`traj_path` parameter can also be ignored to use the default trajectory path defined by the task class.

## 3. Uploading the Trajectories

Add the trajectories to the [RoboVerse data repository](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data) by creating a PR. Please follow the [data structure](https://roboverse.wiki/metasim/developer_guide/data_structure). We thank you for your contribution!