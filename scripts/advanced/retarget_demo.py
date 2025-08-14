from __future__ import annotations

#########################################
### Add command line arguments
#########################################
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    src_robot: str
    tgt_robots: list[str]
    tasks: list[str] | None = None
    src_path: str | None = None
    tgt_path: str | None = None
    device: str = "cuda:0"
    ignore_ground_collision: bool = False
    binary_action: bool = False
    """Use binary action for the target robot. If `True`, the resulting action for each step will be either `gripper_open_q` or `gripper_close_q`, otherwise the resulting action will be scaled to the new gripper joint limits"""
    viz: bool = False

    def __post_init__(self):
        assert self.tasks is not None or (self.src_path is not None and self.tgt_path is not None), (
            "Either tasks, or both source_path and target_path must be provided"
        )


args = tyro.cli(Args)


#########################################
### Normal code
#########################################
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from glob import glob

import numpy as np
import torch
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler
from tqdm.rich import tqdm, trange

from metasim.utils.demo_util.loader import load_traj_file, save_traj_file
from metasim.utils.kinematics_utils import ee_pose_from_tcp_pose, get_curobo_models, tcp_pose_from_ee_pose
from metasim.utils.setup_util import get_robot, get_task

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
NUM_SEED = 20


def single(src_path, tgt_path, src_robot, tgt_robot):
    if args.viz:
        from plotly import graph_objects as go

        from metasim.utils.viz_utils import plot_point_cloud

    # Robot configurations
    src_robot_cfg = get_robot(src_robot)
    tgt_robot_cfg = get_robot(tgt_robot)

    _, src_fk, src_ik = get_curobo_models(src_robot_cfg, args.ignore_ground_collision)
    _, _, tgt_ik = get_curobo_models(tgt_robot_cfg, args.ignore_ground_collision)

    # Prepare file structure
    demo_data = load_traj_file(src_path)

    if tgt_robot == "ur5e_2f85":
        log.warning("Retargetting to robot ur5e_2f85, removing first 30 actions")
        for k, v in demo_data.items():
            for i in range(len(v)):
                demo_data[k][i]["actions"] = demo_data[k][i]["actions"][30:]
                if demo_data[k][i].get("states", None) is not None:
                    demo_data[k][i]["states"] = demo_data[k][i]["states"][30:]

    src_demos = demo_data[src_robot_cfg.name]
    n_demo = len(src_demos)
    episode_lengths = torch.tensor([len(demo["actions"]) for demo in src_demos], dtype=torch.int, device=args.device)
    max_episode_len = episode_lengths.max().item()

    # Compute source trajectory: (JS pose ->) EE pose -> TCP pose
    arm_joint_names = [jn for jn, jn_cfg in src_robot_cfg.actuators.items() if not jn_cfg.is_ee]
    gripper_joint_names = [jn for jn, jn_cfg in src_robot_cfg.actuators.items() if jn_cfg.is_ee]
    curobo_joints = src_ik.robot_config.cspace.joint_names

    robot_pos = np.stack([np.array(demo["init_state"][src_robot_cfg.name]["pos"]) for demo in src_demos], axis=0)
    robot_quat = np.stack([np.array(demo["init_state"][src_robot_cfg.name]["rot"]) for demo in src_demos], axis=0)
    robot_pos = torch.from_numpy(robot_pos).to(args.device).float()
    robot_quat = torch.from_numpy(robot_quat).to(args.device).float()

    robot_q_traj = np.stack(
        [
            np.pad(
                np.asarray([
                    [a["dof_pos_target"][j] for j in (arm_joint_names + gripper_joint_names)] for a in demo["actions"]
                ]),
                [(0, max_episode_len - len(demo["actions"])), (0, 0)],
                mode="edge",
            )
            for demo in src_demos
        ],
        axis=0,
    )

    # Align with cuRobo: Some robot does not have gripper joints
    robot_q_traj = torch.from_numpy(robot_q_traj).to(args.device).float()[..., : len(curobo_joints)]

    tgt_ee_act_rel = np.stack(
        [
            np.pad(
                np.asarray([[a["dof_pos_target"][j] for j in gripper_joint_names] for a in demo["actions"]]),
                [(0, max_episode_len - len(demo["actions"])), (0, 0)],
                mode="edge",
            )
            for demo in src_demos
        ],
        axis=0,
    )
    tgt_ee_act_rel = torch.from_numpy(tgt_ee_act_rel).to(args.device).float()

    if "ee_pose_target" in src_demos[0]["actions"][0]:
        src_ee_pos_traj = np.stack(
            [
                np.pad(
                    np.asarray([a["ee_pose_target"]["pos"] for a in demo["actions"]]),
                    [(0, max_episode_len - len(demo["actions"])), (0, 0)],
                    mode="edge",
                )
                for demo in src_demos
            ],
            axis=0,
        )
        src_ee_quat_traj = np.stack(
            [
                np.pad(
                    np.asarray([a["ee_pose_target"]["rot"] for a in demo["actions"]]),
                    [(0, max_episode_len - len(demo["actions"])), (0, 0)],
                    mode="edge",
                )
                for demo in src_demos
            ],
            axis=0,
        )
        tgt_ee_act_rel = np.stack(
            [
                np.pad(
                    np.array([a["ee_pose_target"]["gripper_joint_pos"] for a in demo["actions"]]),
                    [(0, max_episode_len - len(demo["actions"])), (0, 0)],
                    mode="edge",
                )
                for demo in src_demos
            ],
            axis=0,
        )
        src_ee_pos_traj = torch.from_numpy(src_ee_pos_traj).to(args.device).float()
        src_ee_quat_traj = torch.from_numpy(src_ee_quat_traj).to(args.device).float()
    else:
        # The EE information are missing, compute from JS pose
        log.warning("EE pose information is missing, computing from joint space pose")
        src_ee_pos_traj, src_ee_quat_traj = [], []
        for i_traj in trange(len(src_demos), desc=f"Computing EE pose for source robot: {src_robot_cfg.name}"):
            # JS pose -> EE pose (local) -> EE pose (world)
            ee_traj, quat_traj = src_fk(robot_q_traj[i_traj].contiguous())
            src_ee_pos_traj.append(ee_traj.clone())
            src_ee_quat_traj.append(quat_traj.clone())

        # These are still in the robot's local frame (for solving IK)
        src_ee_pos_traj = torch.stack(src_ee_pos_traj, dim=0)
        src_ee_quat_traj = torch.stack(src_ee_quat_traj, dim=0)

    tcp_pos_traj, tcp_quat_traj = tcp_pose_from_ee_pose(src_robot_cfg, src_ee_pos_traj, src_ee_quat_traj)

    tcp_pos_traj = tcp_pos_traj.reshape(n_demo, max_episode_len, 3)
    tcp_quat_traj = tcp_quat_traj.reshape(n_demo, max_episode_len, 4)

    # Visualziation
    to_plot = []
    if args.viz:
        to_plot.append(
            plot_point_cloud(
                src_ee_pos_traj[0].cpu().numpy(),
                name=f"Source: {src_robot_cfg.name}",
                marker=dict(color=tgt_ee_act_rel[0].cpu().numpy()),
            )
        )

    # For target embodiment, compute target trajecotry via IK: TCP pose -> EE pose -> JS pose
    succ = torch.ones([n_demo], dtype=torch.bool, device=args.device)

    target_dof = len(tgt_ik.robot_config.cspace.joint_names)
    tgt_arm_qpos = torch.zeros([n_demo, max_episode_len, target_dof], device=args.device)

    tgt_ee_pos_traj = torch.zeros([n_demo, max_episode_len, 3], device=args.device)
    tgt_ee_quat_traj = torch.zeros([n_demo, max_episode_len, 4], device=args.device)

    log.info(f"Retargetting: {src_robot_cfg.name} -> {tgt_robot_cfg.name}")
    tt = trange(max_episode_len, desc=f"{src_robot_cfg.name} -> {tgt_robot_cfg.name}")
    for t in tt:
        # Only solve IK for unfinished and unfailed trajectories
        valid_idx = torch.where((t <= episode_lengths) | succ)[0]
        if len(valid_idx) == 0:
            log.error("gg")
            break

        n_succ, n_fail = ((t >= episode_lengths - 1) | succ).sum(), (~succ).sum()

        tgt_ee_pos_traj[valid_idx, t], tgt_ee_quat_traj[valid_idx, t] = ee_pose_from_tcp_pose(
            tgt_robot_cfg, tcp_pos_traj[valid_idx, t], tcp_quat_traj[valid_idx, t]
        )

        if t > 0:
            seed_config = tgt_arm_qpos[:, t - 1 : t].tile([1, NUM_SEED, 1])
        else:
            seed_config = torch.zeros_like(tgt_arm_qpos[:, t - 1 : t]).tile([1, NUM_SEED, 1])
            seed_config[:, :, 1] = -20
        result = tgt_ik.solve_batch(Pose(tgt_ee_pos_traj[:, t], tgt_ee_quat_traj[:, t]), seed_config=seed_config)

        tgt_arm_qpos[valid_idx, t] = result.solution[valid_idx, 0]
        succ[valid_idx] = succ[valid_idx] * result.success.squeeze()[valid_idx]

        tt.set_description(
            f"{src_robot_cfg.name} -> {tgt_robot_cfg.name} | {n_succ.item()!s} succ | {n_fail.item()!s} fail | {n_demo!s} total"
        )
        tt.refresh()

    if args.viz:
        to_plot.append(plot_point_cloud(tgt_ee_pos_traj[0].cpu().numpy(), name=tgt_robot_cfg.name))

    src_release_q = torch.tensor(src_robot_cfg.gripper_open_q, device=args.device)[None, None, :]
    src_actuate_q = torch.tensor(src_robot_cfg.gripper_close_q, device=args.device)[None, None, :]
    tgt_release_q = torch.tensor(tgt_robot_cfg.gripper_open_q, device=args.device)[None, None, :]
    tgt_actuate_q = torch.tensor(tgt_robot_cfg.gripper_close_q, device=args.device)[None, None, :]
    src_ee_act_rel = torch.abs(tgt_ee_act_rel - src_release_q) / torch.abs(src_actuate_q - src_release_q)
    src_ee_act_rel = src_ee_act_rel.mean(axis=-1)  # Avg over two fingers

    if args.binary_action:
        src_ee_act_rel = (src_ee_act_rel > 0.5).float()
    tgt_ee_act_rel = src_ee_act_rel[:, :, None] * tgt_actuate_q + (1 - src_ee_act_rel[:, :, None]) * tgt_release_q

    tgt_robot_qpos = torch.cat([tgt_arm_qpos, tgt_ee_act_rel], dim=-1)
    tgt_robot_joint_names = list(tgt_robot_cfg.actuators.keys())

    log.info(f"Success rate: {(succ.sum().item() / n_demo * 100):.2f}%, saving trajectories...")
    tgt_demos = []
    for i_succ in tqdm(torch.where(succ)[0], desc=f"Saving trajectories for {tgt_robot_cfg.name}"):
        demo = {"actions": [], "states": [], "init_state": {}}
        for i_step in range(episode_lengths[i_succ]):
            demo["actions"].append({
                "dof_pos_target": dict(zip(tgt_robot_joint_names, tgt_robot_qpos[i_succ, i_step].tolist()))
            })
        demo["init_state"] = src_demos[i_succ]["init_state"]
        del demo["init_state"][src_robot_cfg.name]
        demo["init_state"][tgt_robot_cfg.name] = {
            "dof_pos": dict(zip(tgt_robot_joint_names, tgt_robot_qpos[i_succ, 0].tolist())),
            "pos": robot_pos[i_succ].tolist(),
            "rot": robot_quat[i_succ].tolist(),
        }
        tgt_demos.append(demo)
    log.info(f"Saving trajectories for {tgt_robot_cfg.name}")

    if args.viz:
        go.Figure(to_plot).show()

    if not os.path.exists(os.path.dirname(tgt_path)):
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    save_traj_file({tgt_robot_cfg.name: tgt_demos}, tgt_path)
    log.info(f"Retarget finished -> {tgt_path}")


def main():
    if args.tasks is not None:
        for task_name in args.tasks:
            task = get_task(task_name)
            src_path = task.traj_filepath
            if os.path.isdir(src_path):
                paths = glob(os.path.join(src_path, f"{args.src_robot}_v2.*"))
                assert len(paths) >= 1, f"No trajectories found for {args.src_robot}"
                src_path = paths[0]
            for tgt_robot in args.tgt_robots:
                tgt_path = os.path.join(os.path.dirname(src_path), f"{tgt_robot}_v2.pkl.gz")
                single(src_path, tgt_path, args.src_robot, tgt_robot)
    else:
        single(args.src_path, args.tgt_path)


if __name__ == "__main__":
    main()
