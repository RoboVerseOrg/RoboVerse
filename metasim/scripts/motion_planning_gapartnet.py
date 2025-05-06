from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import json
import os
import time

import numpy as np
import torch
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import quat_from_euler_xyz
from metasim.utils.setup_util import get_robot, get_sim_env_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim", type=str, default="isaaclab", choices=["isaaclab", "isaacgym", "mujoco"])
    parser.add_argument("--task", type=str, default="PickCube")
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        choices=[
            "franka",
            "ur5e_2f85",
            "sawyer",
            "franka_with_gripper_extension",
            "h1_2_without_hand",
            "h1",
            "h1_simple_hand",
            "sawyer_mujoco",
            "fetch",
        ],
    )
    parser.add_argument("--add_table", action="store_true")
    parser.add_argument(
        "--joints", default=None, nargs="+", type=str, help="Joints to randomize, if None, randomize all joints"
    )
    args = parser.parse_args()
    return args


# def plan_to_pose_curobo(position, quaternion, max_attempts=100, start_state=None):
#     """
#     start_state: JointState
#         if None, use current state as start state
#         else, use given start_state

#     position: list or np.array
#         target position
#     quaternion: list or np.array
#         target orientation
#     """
#     if start_state == None:
#         start_state = JointState.from_position(robot_dof_qpos_qvel[:, :7, 0])
#     goal_state = Pose(
#         torch.tensor(
#             torch.tensor(position) - torch.tensor(self.cfgs["asset"]["franka_pose_p"]),
#             device=self.device,
#             dtype=torch.float64,
#         ),
#         quaternion=torch.tensor(quaternion, device=self.device, dtype=torch.float64),
#     )
#     result = self.motion_gen.plan_single(start_state, goal_state, MotionGenPlanConfig(max_attempts=max_attempts))

#     traj = result.get_interpolated_plan()
#     # if result.optimized_dt == None or result.success[0] == False:
#     #     return None
#     try:
#         print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
#     except:
#         print("Trajectory Generated: ", result.success)
#     return traj


def get_gapartnet_anno(anno_path):
    """Get gapartnet annotation"""
    info = {}
    # load object annotation
    anno = json.loads(open(anno_path).read())
    num_link_anno = len(anno)
    gapart_raw_valid_anno = []
    for link_i in range(num_link_anno):
        anno_i = anno[link_i]
        if anno_i["is_gapart"]:
            gapart_raw_valid_anno.append(anno_i)
    info["gapart_cates"] = [anno_i["category"] for anno_i in gapart_raw_valid_anno]
    info["gapart_init_bboxes"] = np.array([np.asarray(anno_i["bbox"]) for anno_i in gapart_raw_valid_anno])
    info["gapart_link_names"] = [anno_i["link_name"] for anno_i in gapart_raw_valid_anno]
    return info


def control_to_pose(
    env,
    num_envs,
    robot,
    ee_pos_target,
    rotation,
    gripper_widths,
    robot_ik,
    curobo_n_dof,
    ee_n_dof,
    seed_config,
    steps=3,
):
    pass
    # Solve IK
    result = robot_ik.solve_batch(Pose(ee_pos_target, rotation=rotation), seed_config=seed_config)

    q = torch.zeros((num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = gripper_widths

    actions = [{"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(num_envs)]

    for i_step in range(steps):
        obs, _, _, _, _ = env.step(actions)
        env.render()
    return actions, obs


def main():
    args = parse_args()
    num_envs: int = args.num_envs
    camera = PinholeCameraCfg(pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        try_add_table=args.add_table,
        sim=args.sim,
        cameras=[camera],
        num_envs=num_envs,
    )

    ##############################################
    # change urdf path
    import glob

    import tqdm

    # paths = glob.glob("metasim/cfg/tasks/gapartnet/GAPartNet/assets/*/mobility_annotation_gapartnet.urdf")
    paths = glob.glob("roboverse_data/assets/gapartnet/*/mobility_annotation_gapartnet.urdf")
    available_gapart_ids = []
    for path in tqdm.tqdm(paths, total=len(paths)):
        # get gapart id and anno
        gapart_id = path.split("/")[-2]
        gapart_anno_path = "/".join(path.split("/")[:-1]) + "/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path))
        for link_i, link_anno in enumerate(gapart_anno):
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                available_gapart_ids.append((gapart_id, link_anno["link_name"], link_i, gapart_anno))

    import random

    # random select one gapart id
    gapart_id, link_name, link_i, gapart_anno = random.choice(available_gapart_ids)
    scenario.task.objects[
        0
    ].urdf_path = f"roboverse_data/assets/gapartnet/{gapart_id}/mobility_annotation_gapartnet.urdf"

    tic = time.time()
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robot)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")
    ##############################################

    robot = get_robot(args.robot)
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_release_q)

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    ##############################################
    init_states[0]["cabinet"]["dof_pos"] = {joint_name: 0 for joint_name in env.handler._joint_info["cabinet"]["names"]}
    scenario.task.checker.joint_name = env.handler._joint_info["cabinet"]["names"][-1]
    ##############################################

    ## Main
    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # Generate random actions
    urdf_path = scenario.task.objects[0].urdf_path
    anno_path = urdf_path.replace("mobility_annotation_gapartnet.urdf", "link_annotation_gapartnet.json")
    gapartnet_anno = get_gapartnet_anno(anno_path)

    ee_rot_target = torch.rand((num_envs, 3), device="cuda:0") * torch.pi
    ee_quat_target = quat_from_euler_xyz(ee_rot_target[..., 0], ee_rot_target[..., 1], ee_rot_target[..., 2])

    # Compute targets
    bbox_id = -1
    all_bbox_now = gapartnet_anno["gapart_init_bboxes"] * scenario.task.objects[0].scale + np.array(
        init_states[0]["cabinet"]["pos"]
    )
    # get the part bbox and calculate the handle direction
    all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32).to("cuda:0").reshape(-1, 8, 3)
    all_bbox_center_front_face = torch.mean(all_bbox_now[:, 0:4, :], dim=1)
    handle_out = all_bbox_now[:, 0, :] - all_bbox_now[:, 4, :]
    handle_out /= torch.norm(handle_out, dim=1, keepdim=True)
    handle_long = all_bbox_now[:, 0, :] - all_bbox_now[:, 1, :]
    handle_long /= torch.norm(handle_long, dim=1, keepdim=True)
    handle_short = all_bbox_now[:, 0, :] - all_bbox_now[:, 3, :]
    handle_short /= torch.norm(handle_short, dim=1, keepdim=True)

    init_position = all_bbox_center_front_face[bbox_id]
    handle_out_ = handle_out[bbox_id]

    gripper_widths = torch.tensor(robot.gripper_release_q).to("cuda:0")

    rotation_transform_for_franka = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        device="cuda:0",
    )
    rotation_target = torch.stack(
        [
            -handle_out[bbox_id] + 1e-2,
            -handle_short[bbox_id] + 1e-2,
            handle_long[bbox_id] + 1e-2,
        ],
        dim=0,
    )
    rotation = rotation_target @ rotation_transform_for_franka

    actions_list = []
    log.info("move to pre-grasp position")
    for i in range(1):
        curr_robot_q = obs["joint_qpos"].cuda()
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

        actions, obs = control_to_pose(
            env,
            num_envs,
            robot,
            init_position + 0.1 * handle_out_,
            rotation,
            gripper_widths,
            robot_ik,
            curobo_n_dof,
            ee_n_dof,
            seed_config,
            steps=3,
        )
        actions_list.append(actions)
    # move the object to the grasp position
    log.info("move to grasp position")
    for i in range(20):
        curr_robot_q = obs["joint_qpos"].cuda()
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

        actions, obs = control_to_pose(
            env,
            num_envs,
            robot,
            init_position + (0.1 - 0.015 * i) * handle_out_,
            rotation,
            gripper_widths,
            robot_ik,
            curobo_n_dof,
            ee_n_dof,
            seed_config,
            steps=4,
        )

        actions_list.append(actions)
        actions_list.append(actions)
    # close the gripper
    log.info("close the gripper")

    gripper_widths = torch.tensor(robot.gripper_release_q)
    gripper_widths[:] = 0.0
    actions[0]["dof_pos_target"]["panda_finger_joint1"] = 0.0
    actions[0]["dof_pos_target"]["panda_finger_joint2"] = 0.0
    for i in range(10):
        obs, _, _, _, _ = env.step(actions)
        actions_list.append(actions)
        env.render()

    # move the object to the lift position]
    log.info("move the object to the lift position")

    for i in range(15):
        curr_robot_q = obs["joint_qpos"].cuda()
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

        actions, obs = control_to_pose(
            env,
            num_envs,
            robot,
            init_position + (-0.2 + i * 0.01) * handle_out_,
            rotation,
            gripper_widths,
            robot_ik,
            curobo_n_dof,
            ee_n_dof,
            seed_config,
            steps=4,
        )
        actions_list.append(actions)

    log.info("done")

    ori_json = json.load(open(scenario.task.traj_filepath))
    ori_json["franka"][0]["actions"] = [act[0] for act in actions_list]
    json.dump(ori_json, open(scenario.task.traj_filepath, "w"), indent=4)


if __name__ == "__main__":
    main()
