import copy
import os

import numpy as np
import torch

from metasim.utils.math import quat_from_euler_np


def get_simulation_state_template(
    # --- Inputs ---
    obs_sene,
    obs_robot,
):
    state_dict = {
        "table": {
            "pos": [0.0, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": {
                "base__button": float(obs_sene[2]),
                "base__drawer": float(obs_sene[1]),
                "base__slide": float(obs_sene[0]),
                "base__switch": float(obs_sene[3]),
            },
        },
        "pink_cube": {
            "pos": obs_sene[18:21].tolist(),
            "rot": quat_from_euler_np(obs_sene[21], obs_sene[22], obs_sene[23]).tolist(),
            "dof_pos": {},
        },
        "blue_cube": {
            "pos": obs_sene[12:15].tolist(),
            "rot": quat_from_euler_np(obs_sene[15], obs_sene[16], obs_sene[17]).tolist(),
            "dof_pos": {},
        },
        "red_cube": {
            "pos": obs_sene[6:9].tolist(),
            "rot": quat_from_euler_np(obs_sene[9], obs_sene[10], obs_sene[11]).tolist(),
            "dof_pos": {},
        },
        "franka": {
            "pos": [-0.34, -0.46, 0.24],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": {
                "panda_finger_joint1": float(obs_robot[6] / 2),
                "panda_finger_joint2": float(obs_robot[6] / 2),
                "panda_joint1": float(obs_robot[7]),
                "panda_joint2": float(obs_robot[8]),
                "panda_joint3": float(obs_robot[9]),
                "panda_joint4": float(obs_robot[10]),
                "panda_joint5": float(obs_robot[11]),
                "panda_joint6": float(obs_robot[12]),
                "panda_joint7": float(obs_robot[13]),
            },
        },
    }

    return state_dict


def update_env_state_from_dataset_obs(current_env_state_list, scene_obs: np.ndarray, robot_obs: np.ndarray):
    new_state_list = copy.deepcopy(current_env_state_list)
    state_to_update = new_state_list[0]

    TABLE_JOINT_MAP = {
        0: "base__slide",  # 'sliding door'
        1: "base__drawer",  # 'drawer'
        2: "base__button",  # 'button'
        3: "base__switch",  # 'switch'
    }

    BLOCK_MAP = {"red_block": "red_cube", "blue_block": "blue_cube", "pink_block": "pink_cube"}

    if "table" in state_to_update["objects"]:
        table_dof_pos = state_to_update["objects"]["table"]["dof_pos"]
        if table_dof_pos is not None:
            for obs_idx, joint_name in TABLE_JOINT_MAP.items():
                if joint_name in table_dof_pos:
                    table_dof_pos[joint_name] = scene_obs[obs_idx]

    obs_block_data = {
        "red_block": {"pos": scene_obs[6:9], "rot_euler": scene_obs[9:12]},
        "blue_block": {"pos": scene_obs[12:15], "rot_euler": scene_obs[15:18]},
        "pink_block": {"pos": scene_obs[18:21], "rot_euler": scene_obs[21:24]},
    }

    for obs_name, state_name in BLOCK_MAP.items():
        if state_name in state_to_update["objects"]:
            block_data = obs_block_data[obs_name]

            state_to_update["objects"][state_name]["pos"] = torch.tensor(block_data["pos"], dtype=torch.float32)

            quat_wxyz = quat_from_euler_np(
                block_data["rot_euler"][0], block_data["rot_euler"][1], block_data["rot_euler"][2]
            )
            state_to_update["objects"][state_name]["rot"] = torch.tensor(quat_wxyz, dtype=torch.float32)

    ROBOT_ARM_JOINTS = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    if "franka" in state_to_update["robots"]:
        franka_dof_pos = state_to_update["robots"]["franka"]["dof_pos"]
        if franka_dof_pos is not None:
            arm_states = robot_obs[7:14]
            for i, joint_name in enumerate(ROBOT_ARM_JOINTS):
                if joint_name in franka_dof_pos:
                    franka_dof_pos[joint_name] = arm_states[i]

            gripper_action = robot_obs[14]
            gripper_val = 0.04 if gripper_action == 1 else 0.0  # 1=open, -1=close

            if "panda_finger_joint1" in franka_dof_pos:
                franka_dof_pos["panda_finger_joint1"] = gripper_val
            if "panda_finger_joint2" in franka_dof_pos:
                franka_dof_pos["panda_finger_joint2"] = gripper_val

    return new_state_list


def _prepare_actions(actions_arr: np.ndarray, robot_obs: np.ndarray) -> np.ndarray:
    g = actions_arr[:, 6].copy()
    g[g == -1] = 0.0
    g *= 0.04

    return list(np.concatenate([g[:, None], g[:, None], robot_obs[:, 7:14]], axis=1))
    # return list(np.concatenate([robot_obs[:, 6:7]/2, robot_obs[:, 6:7]/2, robot_obs[:, 7:14]], axis=1))


def _find_segments(scene_obs: np.ndarray, min_len: int = 300, diff_eps: float = 1e-4):
    T = scene_obs.shape[0]
    if T < 2:
        return [(0, T)]
    diffs = np.abs(scene_obs[1:][6:] - scene_obs[:-1][6:]).sum(axis=1)

    candidates = np.flatnonzero(diffs < diff_eps) + 1
    segments = []
    start = 0
    for cut in candidates:
        if cut - start >= min_len:
            segments.append((start, cut))
            start = cut
    if start < T:
        segments.append((start, T))
    return segments


import pickle

import numpy as np

from metasim import task  # noqa
from metasim.task.registry import get_task_class
from roboverse_pack.tasks.calvin.base_table import BaseCalvinTableTask  # noqa: F401

if __name__ == "__main__":
    file_list = ["env_D_val"]

    for env_files in file_list:
        if env_files == "env_A":
            task_cls = get_task_class("calvin.base_table_A")  # e.g., "example.my_task"

            scenario = task_cls.scenario.update(
                simulator="pybullet",
            )
        elif env_files == "env_B":
            task_cls = get_task_class("calvin.base_table_B")  # e.g., "example.my_task"

            scenario = task_cls.scenario.update(
                simulator="pybullet",
            )
        elif env_files == "env_C":
            task_cls = get_task_class("calvin.base_table_C")  # e.g., "example.my_task"

            scenario = task_cls.scenario.update(
                simulator="pybullet",
            )
        elif env_files == "env_D" or env_files == "env_D_val":
            task_cls = get_task_class("calvin.base_table")  # e.g., "example.my_task"

            scenario = task_cls.scenario.update(
                simulator="pybullet",
            )
        env = task_cls(scenario=scenario)
        obs0 = env.handler.get_states(mode="dict")

        base_dir = f"/home/dyz/RoboVerse/roboverse_pack/tasks/calvin/data_preparation/{env_files}/"
        base_dir_o = f"/home/dyz/RoboVerse/roboverse_pack/tasks/calvin/data_preparation/{env_files}_out/"
        if not os.path.isdir(base_dir):
            continue
        files = os.listdir(base_dir)

        for file in files:
            npz_dir = f"{base_dir}{file}"
            out_dir = f"{base_dir_o}{file}"
            npz_path = f"{npz_dir}/consolidated_data.npz"
            if not os.path.isfile(npz_path):
                continue

            t = np.load(npz_path, allow_pickle=True)
            scene_obs = t["scene_obs"]  # (T, 24)
            robot_obs = t["robot_obs"]  # (T, 15)
            actions = t["actions"]  # (T, ?)

            T = scene_obs.shape[0]
            if T < 2:
                continue

            act_all = _prepare_actions(actions[:-1], robot_obs[1:])  # (T, 9)

            get_state = get_simulation_state_template
            states_all = [get_state(scene_obs[i], robot_obs[i]) for i in range(T)]

            segments = _find_segments(scene_obs, min_len=300, diff_eps=1e-4)

            for start, end in segments:
                if end - start < 2:
                    continue

                init_state = update_env_state_from_dataset_obs(obs0, scene_obs[start], robot_obs[start])

                env.reset(init_state)
                reset_state = env.handler.get_states(mode="dict")
                # import ipdb; ipdb.set_trace()

                init_state = reset_state[0]["objects"]

                # import ipdb; ipdb.set_trace()

                actions_seg = act_all[start : end - 1]

                states_seg = states_all[start + 1 : end]

                dict_traj = {
                    "init_state": init_state,
                    "reset_state": reset_state,
                    "actions": [a for a in actions_seg],
                    "states": states_seg,
                    "env_meta": {
                        "env_name": env_files,
                        "source_dir": f"{env_files}/{file}",
                    },
                }
                trajs = {"franka": [dict_traj]}
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/trajectory_{env_files}_{end}_v2.pkl"
                with open(out_path, "wb", buffering=1024 * 1024) as f:
                    pickle.dump(trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
