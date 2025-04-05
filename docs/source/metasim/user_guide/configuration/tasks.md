# Tasks Config

Tasks are predefined configurations for different tasks. They are stored in `metasim/cfg/tasks`. You can use them to instantiate simulations, or inherit them to create modified tasks.

The basic structure of a `TaskCfg` looks like this:

```python
@configclass
class BaseTaskCfg:
    """Base task configuration.

    Attributes:
        decimation: The decimation factor for the task.
        episode_length: The length of the episode.
        objects: The list of object configurations.
        traj_filepath: The file path to the trajectory.
        source_benchmark: The source benchmark.
        task_type: The type of the task.
        checker: The checker for the task.
        can_tabletop: Whether the task can be tabletop.
        reward_functions: The list of reward functions.
        reward_weights: The list of reward weights.
    """

    decimation: int = 3
    episode_length: int = MISSING
    objects: list[BaseObjCfg] = MISSING
    traj_filepath: str = MISSING
    source_benchmark: BenchmarkType = MISSING
    task_type: TaskType = MISSING
    checker: BaseChecker = BaseChecker()
    can_tabletop: bool = False
    reward_functions: list[callable[[list[EnvState], str | None], torch.FloatTensor]] = MISSING
    reward_weights: list[float] = MISSING
```

The `TaskCfg` configuration is a preset to be loaded into your `ScenarioCfg`. Here is an example `TaskCfg` for box-closing task:

```python
@configclass
class CloseBoxCfg(RLBenchTaskCfg):
    episode_length = 250
    objects = [
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]
    traj_filepath = "metasim/data/quick_start/trajs/rlbench/close_box/v2"
    checker = JointPosChecker(
        obj_name="box_base",
        joint_name="box_joint",
        mode="le",
        radian_threshold=-14 / 180 * math.pi,
    )

    def reward_fn(self, states):
        # HACK: metasim_body_panda_hand may not be universal across all robots
        try:
            ee_poses = torch.stack([state["metasim_body_panda_hand"]["pos"] for state in states])
        except KeyError as e:
            log.error(f"KeyError: {e}")
            ee_poses = torch.zeros(len(states), 3)
        distance = torch.norm(ee_poses, dim=-1)
        return -distance
```
