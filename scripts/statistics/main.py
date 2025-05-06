import os
import pickle as pkl

import numpy as np
from loguru import logger as log
from matplotlib import pyplot as plt

from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import TaskType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_task

save_dir = "statistics/plots"
os.makedirs(save_dir, exist_ok=True)

franka_joint_names = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
franka_joint_limits = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
    (0.0, 0.04),
    (0.0, 0.04),
]


def get_available_tasks():
    from metasim.cfg import tasks

    tasks = [task for task_name, task in vars(tasks).items() if task_name.endswith("Cfg")]

    tasks = [task for task in tasks if issubclass(task, BaseTaskCfg) and task != BaseTaskCfg]
    tasks = [task for task in tasks if str(task).find("open6d") == -1]

    available_tasks = []

    for task in tasks:
        task_inst = task()
        task_name = task_inst.__class__.__name__.removesuffix("Cfg")
        if task_inst.task_type != TaskType.TABLETOP_MANIPULATION:
            log.info(f"Skipping {task_name} because it is not tabletop manipulation")
            continue
        if not task_inst.traj_filepath.endswith("_v2.pkl"):
            log.info(f"Skipping {task_name} because it is not v2")
            continue
        if not os.path.exists(task_inst.traj_filepath):
            log.info(f"Skipping {task_name} because the trajectory file does not exist")
            continue

        available_tasks.append(task_name)

    return available_tasks


def plot(actions: np.ndarray, save_path: str):
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i in range(7):
        joint_actions = actions[:, i]
        ax = axes[i]
        ax.hist(joint_actions, bins=300, range=franka_joint_limits[i])
        ax.set_title(f"Joint {i + 1} Action Distribution")
        ax.set_xlabel("Joint Position Target")
        ax.set_ylabel("Density")
        ax.grid(True)

    axes[-1].remove()  # Remove the empty subplot
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    available_tasks = get_available_tasks()
    print("available_tasks:", available_tasks)

    all_task_actions = []
    all_task_actions_100 = []
    for task_name in available_tasks:
        ## Load actions
        task_inst = get_task(task_name)
        data = pkl.load(open(task_inst.traj_filepath, "rb"))
        robots = list(data.keys())
        _, all_actions, _ = get_traj(task_inst, get_robot(robots[0]), None)

        ## Parse actions
        actions = []
        for demo_idx in range(len(all_actions)):
            for step in range(len(all_actions[demo_idx])):
                action = [all_actions[demo_idx][step]["dof_pos_target"][k] for k in franka_joint_names]
                actions.append(action)
        actions = np.array(actions)

        ## Plot actions
        plot(actions, f"{save_dir}/{task_name}.png")

        all_task_actions.append(actions)
        all_task_actions_100.append(actions[:100])

    all_task_actions = np.concatenate(all_task_actions, axis=0)
    plot(all_task_actions, f"{save_dir}/all_tasks.png")

    all_task_actions_100 = np.concatenate(all_task_actions_100, axis=0)
    plot(all_task_actions_100, f"{save_dir}/all_tasks_100.png")


if __name__ == "__main__":
    main()
