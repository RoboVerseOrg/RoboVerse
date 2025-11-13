import argparse
import json
import logging
import os
import shutil

import imageio.v2 as iio
import numpy as np
import torch
import zarr
from tqdm import tqdm

try:
    from pytorch3d import transforms
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Process Meta Data To ZARR For Diffusion Policy.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="StackCube_franka",
        help="The name of the task (e.g., StackCube_franka)",
    )
    parser.add_argument(
        "--expert_data_num",
        type=int,
        default=200,
        help="Number of episodes to process (e.g., 200)",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="~/RoboVerse/data_isaaclab/demo/StackCube/robot-franka",
        help="The path of metadata",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=1,
        help="The downsample ratio of metadata",
    )

    parser.add_argument(
        "--observation_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The observation space to use (e.g., joint_pos, ee)",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The action space to use (e.g., joint_pos, ee)",
    )

    parser.add_argument("--delta_ee", type=int, choices=[0, 1], default=0)

    parser.add_argument(
        "--joint_pos_padding",
        type=int,
        default=0,
        help="If > 0, pad joint positions to this length when using joint_pos observation/action space",
    )

    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    load_dir = args.metadata_dir
    downsample_ratio = args.downsample_ratio

    print("Metadata load dir:", load_dir)
    save_dir = f"data_policy/{task_name}_{num}.zarr"
    print("ZARR save dir:", save_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # Batch processing settings
    batch_size = 100
    head_camera_arrays = []
    action_arrays = []
    state_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
    # current_demo_index = 0

    if args.joint_pos_padding > 0 and args.observation_space == "ee" and args.action_space == "ee":
        logging.warning("Padding is not supported for ee observation and action spaces.")

    for current_ep in tqdm(range(num), desc=f"Processing {num} MetaData"):
        demo_id = str(current_ep).zfill(4)
        demo_dir = os.path.join(load_dir, f"demo_{demo_id}")
        # current_ep += 1

        if not os.path.isdir(demo_dir):
            print(f"Skipping episode {current_ep} as directory {demo_dir} does not exist.")
            continue
        else:
            demo_id = str(current_ep).zfill(4)
            demo_dir = os.path.join(load_dir, f"demo_{demo_id}")
            # current_demo_index += 1

         # Check metadata.json
        metadata_path = os.path.join(demo_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
          print(f"Skipping episode {current_ep} as metadata.json does not exist.")
          continue
        else:
            with open(os.path.join(demo_dir, "metadata.json"), encoding="utf-8") as f:
                # print("metadata load dir:", demo_dir)
                metadata = json.load(f)

        data_length = len(metadata["joint_qpos"])
        rgbs = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))
        for i, rgb in enumerate(rgbs):
            if i % downsample_ratio != 0:
                continue

            # you can change state and action here
            if args.observation_space == "joint_pos":
                state = metadata["joint_qpos"][i]
                # Apply padding if specified and using joint_pos
                if args.joint_pos_padding > 0 and len(state) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(state))
                    state = np.concatenate([state, padding])
            elif args.observation_space == "ee":
                robot_pos, robot_quat = (
                    torch.tensor(metadata["robot_root_state"][i][0:3]),
                    torch.tensor(metadata["robot_root_state"][i][3:7]),
                )

                # Convert both current and next EE state into local coordinates
                local_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                )
                local_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state"][i][3:7])
                )

                gripper_state = metadata["joint_qpos"][i][-2:]
                state = np.concatenate([local_ee_pos, local_ee_quat, gripper_state])
                assert state.shape == (9,)
            else:
                raise ValueError(f"Unknown observation space: {args.observation_space}")

            if args.action_space == "joint_pos":
                action = metadata["joint_qpos_target"][i]
                # Apply padding if specified and using joint_pos
                if args.joint_pos_padding > 0 and len(action) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(action))
                    action = np.concatenate([action, padding])
            elif args.action_space == "ee":
                robot_pos, robot_quat = (
                    torch.tensor(metadata["robot_root_state"][i][0:3]),
                    torch.tensor(metadata["robot_root_state"][i][3:7]),
                )

                # Convert both current and next EE state into local coordinates
                local_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                )
                local_next_ee_pos = transforms.quaternion_apply(
                    transforms.quaternion_invert(robot_quat),
                    torch.tensor(metadata["robot_ee_state_target"][i][0:3]) - robot_pos,
                )

                local_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state"][i][3:7])
                )
                local_next_ee_quat = transforms.quaternion_multiply(
                    transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state_target"][i][3:7])
                )
                gripper_action = metadata["joint_qpos_target"][i][-2:]

                if not args.delta_ee:
                    action = np.concatenate([local_next_ee_pos, local_next_ee_quat, gripper_action])
                else:
                    # Compute the delta in local coordinates
                    local_ee_delta_pos = local_next_ee_pos - local_ee_pos
                    local_ee_delta_quat = transforms.quaternion_multiply(
                        transforms.quaternion_invert(local_ee_quat), local_next_ee_quat
                    )
                    action = np.concatenate([local_ee_delta_pos, local_ee_delta_quat, gripper_action])

                assert action.shape == (9,), f"Action shape is {action.shape}, expected (9,)"
            else:
                raise ValueError(f"Unknown action space: {args.action_space}")

            action = list(action)
            # Append data to batch arrays
            head_camera_arrays.append(rgb)
            state_arrays.append(state)
            action_arrays.append(action)
            total_count += 1

        episode_ends_arrays.append(total_count)

        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == num:
            # Convert arrays to NumPy and format head_camera
            head_camera_arrays = np.array(head_camera_arrays)
            head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
            # print(head_camera_arrays)
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)

            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "head_camera",
                    shape=(0, *head_camera_arrays.shape[1:]),
                    chunks=(batch_size, *head_camera_arrays.shape[1:]),
                    dtype=head_camera_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )

            # Append data to ZARR datasets
            zarr_data["head_camera"].append(head_camera_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)

            print(f"Batch {current_batch + 1} written with {len(head_camera_arrays)} samples.")

            # Clear arrays for next batch
            head_camera_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1

    # Save metadata to a JSON file
    metadata = {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "delta_ee": args.delta_ee,
        "joint_pos_padding": args.joint_pos_padding,
        "task_name": args.task_name,
        "num_episodes": args.expert_data_num,
        "downsample_ratio": args.downsample_ratio,
    }

    # Save metadata to zarr group
    for key, value in metadata.items():
        zarr_meta.attrs[key] = value

    # Also save as a separate JSON file for easier access
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
