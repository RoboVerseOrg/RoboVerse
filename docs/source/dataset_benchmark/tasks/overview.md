# Tasks Overview

RoboVerse provides a diverse set of tasks for robots to learn and evaluate their capabilities. We mainly migrate the tasks from existing benchmarks and simulation environments. We further migrate trajectories, collect demonstrations, roll out existing policies or heuristics, and train policies on them. Please follow the license of the original tasks and datasets when using them.

## Task Organization

We organize the tasks with our standard metasim configuration, as shown in `roboverse_pack/tasks`. You can easily use them to instantiate simulations, or inherit them to create modified tasks.

When instantiating tasks, you can specify the task name at config level. The task is a preset to be loaded into your `ScenarioCfg`.

```python
ScenarioCfg(
    task=task_name,
    robot=robot_name,
    scene=scene_type,
    cameras=[camera_config],
    sim=sim_name,
    num_envs=num_envs,
    ...
)
```

<!-- You can find how to create/modify task configurations at [here](https://roboverse.wiki/metasim/concept/config.html). -->
