# 7. Collect Demonstrations
```bash
./roboverse_learn/il/collect_demo.sh
```

**collect_demo.sh** collects demos for imitation learning, using `~/RoboVerse/scripts/advanced/collect_demo.py` and converts the metadata into Zarr format for efficient dataloading.

**Outputs**: Collects demos are stored in `~/RoboVerse/roboverse_demo/demo_{sim_set}/{task_name_set}-{cust_name}/robot-franka`. Converted dataset is stored in `~/RoboVerse/data_policy`

When collecting demo, the RAM will grow up, during which the collected rendered data are gathered before writing to the disk. After a while, the RAM occupation should become steady.

On RTX 4090, the ideal num_envs is 64.

## Parameters in .sh
| Argument            | Description                                                  | Example                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `task_name_set`     | Name of the task                                             | `close_box`                                                  |
| `sim_set`           | Name of the selected simulator                               | `isaacsim`                                                   |
| `max_demo_idx`      | Maximum index of demos to collect                            | `100`                                                        |
| `expert_data_num`   | Number of expert demonstrations to process                   | `100`                                                        |
| `run_all`           | Rollout all trajectories, overwrite existing demos           | `run_all`                                                    |
| `run_unfinished`    | Rollout unfinished trajectories                              | `bool`                                                       |
| `run_failed`        | Rollout unfinished and failed trajectories                   | `bool`                                                       |
| `metadata_dir`      | Path to the directory containing demonstration metadata saved by collect_demo |`~/RoboVerse/roboverse_demo/demo_isaacsim/close_box-/robot-franka` |
| `action_space`      | Type of action space to use (options: 'joint_pos' or 'ee')   | `joint_pos`                                                  |
| `observation_space` | Type of observation space to use (options: 'joint_pos' or 'ee') | `joint_pos`                                                  |
| `delta_ee`          | (optional) Delta control (0: absolute, 1: delta; default 0)  | `0`                                                          |
| `cust_name`         | User defined name                                            | `noDR`                                                       |

## Domain Randomization
                                                 
### Domain randomization options:
- `enable_randomization`（bool）: Enable domain randomization during demo collection.
- `randomize_materials`（bool）: Enable material randomization (when randomization is enabled).
- `randomize_lights`（bool）: Enable light randomization (when randomization is enabled).
- `randomize_cameras`（bool）: Enable camera randomization (when randomization is enabled).
- `randomize_physics`（bool）: Enable physics (mass/friction/pose) randomization using ObjectRandomizer.
- `randomization_frequency`（Literal）: When to apply randomization: per_demo (once at start) or per_episode (every episode).
- `randomization_seed` (int): Seed for reproducible randomization. If None, uses random seed.

### Steps to add DR in custom .py (for developer):

**Step 1:** Adding arguments of domain randomization and related log information. Refer to Lines 55-69 and Lines 81-101 in collect_demo.py.

**Step 2:** Import randomization components. Refer to Lines 129-145 in collect_demo.py.

**Step 3:** Adding class `DomainRandomizationManager` and other auxiliary functions. Refer to Lines 148-402 in collect_demo.py.

**Step 4:** Adding key DR callbacks in mian(). 
- Initialize `DomainRandomizationManager`： line 677. 
- Randomization for initial parallel demos： lines 737-739. 
- Randomization for next parallel demos： lines 789, 837. 
- Randomization for failed parallel demos： line 821. 





<!-- First, prepare the material dataset [vMaterial 2.4](https://developer.nvidia.com/vmaterials).
```bash
wget https://developer.nvidia.com/downloads/designworks/vmaterials/secure/2.4/nvidia-vmaterials_2-linux-x86-64-2.4.0-373000.4039.run
sudo sh ./nvidia-vmaterials_2-linux-x86-64-2.4.0-373000.4039.run
# The installed path should be /opt/nvidia/mdl/vMaterials_2
# If not, change the path to the actual intallation path on your machine
ln -s /opt/nvidia/mdl/vMaterials_2 third_party
``` -->
