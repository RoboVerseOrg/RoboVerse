# 0. Quick start

Imitation Learning is a powerful tool for training agents to perform tasks by leveraging expert demonstrations, especially when reward signals are sparse or hard to specify.

Here we will show the ready-to-run commands to quickly install the dependencies of imitation learning and get it up.


## 1. Installation

```bash
bash roboverse_learn/il/il_setup.sh
```
This command can install all relevant libraries for diffusion policy and action chunking.

## 2. Running 

```bash
 bash roboverse_learn/il/il_run.sh --task_name_set close_box --algo_choose act --demo_num 100 --sim_set mujoco
```
This command performs demo collection, model training, and evaluation.


| Argument            | Description                                                  | Example                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `task_name_set`     | Name of the task                                             | `close_box`，`stack_cube`, `pick_cube`, `pick_butter`        |
| `sim_set`           | Name of the selected simulator                               | `isaacsim`, `mujoco`                                         |
| `demo_num`          | Numbers of demos to collect, train, and eval                 | `100`                                                        |
| `algo_choose`       | IL algorithm                                                 | `act`, `dp_DDPM`, `dp_DDIM`, `dp_FM_UNet`, `dp_FM_DiT`, `dp_Score`, `dp_VITA`        |
