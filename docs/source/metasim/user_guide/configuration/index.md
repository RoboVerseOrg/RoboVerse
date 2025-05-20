# Configuration System

## Introduction

The configuration system is the core of the metasim infrastructure.

We use python dataclasses as the datastructure to store configuration information. Dataclasses are nested to form hierarchies and abstractions. The root level of the configuration system is ScenarioCfg, it necessarily contains everything related to instantiating an environment.

```python
@configclass
class ScenarioCfg:
    """Scenario configuration."""

    task: BaseTaskCfg | None = None  # This item should be removed?
    """None means no task specified"""
    robot: BaseRobotCfg = MISSING
    scene: SceneCfg | None = None
    """None means no scene"""
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = [PinholeCameraCfg()]
    checker: BaseChecker = EmptyChecker()
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "pybullet", "mujoco"] = "isaaclab"
    renderer: Literal["isaaclab", "isaacgym", "pybullet", "mujoco"] | None = None

    ## Others
    num_envs: int = 1
    decimation: int = 1
    episode_length: int = 10000000  # never timeout
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False
```

By setting correct values to the dataclass, you can further instantiate the configuration into a running simulation environment.

At first glance, it may seem too much : ) . There are too many things to know and write! How is it easier than simply learning a new simulator! Luckily we offer a way to fill in every term with presets: the `task` configuration.

When setting the `task` configuration term to a specific value, you are simply setting the default values of the `ScenarioCfg` with the values inside of the `task` configuration. We provided hundreds of tasks for your choice.

If you are not satisfied with the preset tasks, you can always modify anything according to your need, which brings us to the detailed explaination of each and every term in the `ScenarioCfg`.

## Configuration 101

Here we provide case studies to help you getting familiar with the configuration system:

```{toctree}
:titlesonly:

objects
robots
tasks
```
