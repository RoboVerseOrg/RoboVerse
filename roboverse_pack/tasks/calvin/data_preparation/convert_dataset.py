import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from gymnasium import make_vec

import metasim

metasim.register_gym_envs()

if __name__ == "__main__":
    # No personal default: point at a CALVIN split via --path or $CALVIN_DATASET_DIR.
    default_path = os.environ.get("CALVIN_DATASET_DIR")
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument(
        "--path",
        type=str,
        default=default_path,
        required=default_path is None,
        help="Path to dir containing scene_info.npy",
    )
    parser.add_argument("-d", "--data", nargs="*", default=["rgb_static", "rgb_gripper"], help="Data to visualize")
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        print(f"Path {args.path} is either not a directory, or does not exist.")  # noqa: T201
        exit()

    indices = next(iter(np.load(f"{args.path}/scene_info.npy", allow_pickle=True).item().values()))
    indices = list(range(indices[0], indices[1] + 1))

    annotations = np.load(f"{args.path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], strict=False))

    idx = 0
    ann_idx = -1

    trace_list = []

    env_id = "RoboVerse/calvin.base_table"

    env = make_vec(env_id, simulator="pybullet", headless=False, device="cuda", action_mode="joint", num_envs=1)
    for idx in range(len(indices)):
        t = np.load(f"{args.path}/episode_{indices[idx]:07d}.npz", allow_pickle=True)

        # print(t["robot_obs"])
        action = np.concatenate([t["actions"][6:7] * 0.04, t["actions"][6:7] * 0.04, t["robot_obs"][7:14]])

        # print(t["scene_obs"])

        # pos = t["actions"][:3]
        # quat = quat_from_euler_np(
        #     roll=t["actions"][3],
        #     pitch=t["actions"][4],
        #     yaw=t["actions"][5],
        # )
        # action = np.concatenate([pos, quat, t["actions"][6:7]*0.04])
        action = action[None, :]
        # print(action)
        env.step(action)
