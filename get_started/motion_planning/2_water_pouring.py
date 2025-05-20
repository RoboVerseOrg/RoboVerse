"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import rootutils
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from get_started.utils import convert_to_ply

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from scipy.spatial.transform import Rotation as R

from get_started.utils import ObsSaver, get_pcd_from_rgbd
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass

# from get_started.motion_planning.util_gsnet import GSNet
from metasim.utils.camera_util import get_cam_params
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)


# initialize scenario
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.3, 1.5), look_at=(1.0, 0.3, 0.0))]
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    RigidObjCfg(
        name="cup1",
        scale=(0.005, 0.005, 0.008),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/motion_planning/Cup/1/Collected_cup_z/cup.usd",
    ),
    RigidObjCfg(
        name="cup2",
        scale=(0.01, 0.01, 0.01),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/motion_planning/Cup/1/Collected_cup_z/cup.usd",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cup1": {
                "pos": torch.tensor([0.5, -0.2, 0.01]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "cup2": {
                "pos": torch.tensor([0.5, 0.1, 0.01]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
    for _ in range(args.num_envs)
]


robot = scenario.robot
*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_open_q)

obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/5_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)


def get_point_cloud_from_obs(obs, save_pcd=False):
    """Get the point cloud from the observation."""
    img = obs.cameras["camera0"].rgb
    depth = obs.cameras["camera0"].depth
    extr, intr = get_cam_params(
        cam_pos=torch.tensor([scenario.cameras[i].pos for i in range(len(scenario.cameras))]),
        cam_look_at=torch.tensor([scenario.cameras[i].look_at for i in range(len(scenario.cameras))]),
        width=scenario.cameras[0].width,
        height=scenario.cameras[0].height,
        focal_length=scenario.cameras[0].focal_length,
        horizontal_aperture=scenario.cameras[0].horizontal_aperture,
    )

    pcd = get_pcd_from_rgbd(-depth.cpu()[0], img.cpu()[0], intr[0], extr[0])
    if save_pcd:
        convert_to_ply(np.array(pcd.points), "get_started/output/motion_planning.ply")
    return pcd


def move_to_pose(
    obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=10, open_gripper=False
):
    """Move the robot to a given pose."""
    curr_robot_q = obs.robots[robot.name].joint_pos

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    ]
    for i in range(steps):
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)
    return obs


step = 0
robot_joint_limits = scenario.robot.joint_limits
for step in range(1):
    log.debug(f"Step {step}")
    states = env.handler.get_states()
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    gripper_out = torch.tensor([1, 0, 0])
    gripper_long = torch.tensor([0, 1, 0])
    gripper_short = torch.tensor([0, 0, 1])
    # rotation = np.dot(rotation, delta_m)

    # breakpoint()
    rotation_transform_for_franka = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )
    rotation_target = torch.stack(
        [
            gripper_out + 1e-2,
            gripper_long + 1e-2,
            gripper_short + 1e-2,
        ],
        dim=0,
    ).float()
    rotation = rotation_target @ rotation_transform_for_franka

    quat = R.from_matrix(rotation).as_quat()

    ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
    ee_quat_target = torch.zeros((args.num_envs, 4), device="cuda:0")

    position = torch.tensor([0.48, -0.2, 0.1], device="cuda:0")

    ee_pos_target[0] = torch.tensor(position, device="cuda:0")
    ee_quat_target[0] = torch.tensor(quat, device="cuda:0")

    rotation_unit_vect = torch.tensor(rotation[:, 2], device="cuda:0")
    pre_grasp_pos = ee_pos_target.clone()
    grasp_pos = ee_pos_target.clone()
    lift_pos = ee_pos_target.clone()
    # breakpoint()
    pre_grasp_pos[:] -= rotation_unit_vect * 0.2
    grasp_pos[:] -= rotation_unit_vect * 0.08
    lift_pos[:] -= rotation_unit_vect * 0.08
    lift_pos[:, 2] += 0.3
    lift_pos[:, 1] += 0.2
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, pre_grasp_pos, ee_quat_target, steps=50, open_gripper=True
    )
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, grasp_pos, ee_quat_target, steps=50, open_gripper=True
    )
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, grasp_pos, ee_quat_target, steps=50, open_gripper=False
    )
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, lift_pos, ee_quat_target, steps=50, open_gripper=False
    )

    gripper_out2 = torch.tensor([1, 0, 0])
    gripper_long2 = torch.tensor([0, 0, -1])
    gripper_short2 = torch.tensor([0, 1, 0])
    rotation_target2 = torch.stack(
        [
            gripper_out2 + 1e-2,
            gripper_long2 + 1e-2,
            gripper_short2 + 1e-2,
        ],
        dim=0,
    ).float()
    rotation2 = rotation_target2 @ rotation_transform_for_franka

    quat2 = R.from_matrix(rotation2).as_quat()
    ee_quat_target[0] = torch.tensor(quat2, device="cuda:0")
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, lift_pos, ee_quat_target, steps=50, open_gripper=False
    )

    # result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    # q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    # ik_succ = result.success.squeeze(1)
    # q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    # q[:, -ee_n_dof:] = 0.04
    # actions = [
    #     {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    # ]

    # obs, reward, success, time_out, extras = env.step(actions)

    # pcd = get_point_cloud_from_obs(obs)

    # points = np.array(pcd.points)
    # points[:, 2] = -points[:, 2]
    # points[:, 1] = -points[:, 1]
    # pcd.points = o3d.utility.Vector3dVector(points)
    # gsnet = GSNet()
    # gg = gsnet.inference(np.array(pcd.points))
    # gsnet.visualize(pcd, gg[:1])
    # breakpoint()

    # obs_saver.add(obs)
    step += 1

obs_saver.save()


# if __name__ == "__main__":
#     import open3d as o3d

#     cloud = o3d.io.read_point_cloud("third_party/gsnet/assets/test.ply")

#     gsnet = GSNet()
#     gg = gsnet.inference(np.array(cloud.points))
#     gsnet.visualize(cloud, gg)


# grasp_position = filtered_gg[0].translation
# grasp_position[2] = -grasp_position[2]

# print(grasp_position, place_position)
# delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
# # import pdb; pdb.set_trace()
# rotation_output = filtered_gg[0].rotation_matrix.copy()
# print(rotation_output)
# rotation_output[2, :] = -rotation_output[2, :]
# rotation_output[:, 2] = -rotation_output[:, 2]
# print(rotation_output)


# # grippers = filtered_gg[:1].to_open3d_geometry_list()
# # cloud = o3d.geometry.PointCloud()
# # cloud.points = o3d.utility.Vector3dVector(points_envs[0])
# # o3d.visualization.draw_geometries([cloud, *grippers])

# rotation_output = np.dot(rotation_output, delta_m)
# print(rotation_output)
# grasp_quat_R = R.from_matrix(rotation_output).as_quat()
# print(grasp_quat_R)
# rotation_input = grasp_quat_R
# rotation_input = np.array([rotation_input[3],rotation_input[0],rotation_input[1],rotation_input[2]])
# grasp_pose = np.concatenate([grasp_position, rotation_input])

# place_pose = np.concatenate([place_position, rotation_input])

# # R: xyzw
# # IsaacGym: xyzw

# # traj = self.plan_to_pose_curobo(torch.tensor(grasp_pose[:3], dtype = torch.float32), torch.tensor(rotation_input, dtype = torch.float32))
# # self.move_to_traj(traj, close_gripper=False, save_video=save_video, save_root = save_root, start_step = step_num)

# # self.refresh_observation(get_visual_obs=False)

# # R.from_quat(grasp_pose[3:]).as_matrix()
# # R.from_quat(rotation_input).as_matrix()
# # R.from_quat(self.hand_rot.cpu().numpy()).as_matrix()


# rotation_unit_vect = rotation_output[:,2]

# grasp_pre_grasp = grasp_pose.copy()
# grasp_pre_grasp[:3] -= rotation_unit_vect*0.2

# grasp_grasp = grasp_pose.copy()
# grasp_grasp[:3] -= rotation_unit_vect*0.05

# grasp_lift = grasp_pose.copy()
# grasp_lift[2] += 0.3
# # grasp_lift[:3] -= rotation_unit_vect*0.2

# place_pose[:3] -= rotation_unit_vect*0.05
# place_position_lift = place_pose.copy()
# place_position_lift[2] += 0.3
# place_position_place = place_pose.copy()
# place_position_place[2] += 0.05
# place_position_up = place_pose.copy()
# place_position_up[2] += 0.3

# finger_front = np.array([0, 0, -1])
# finger_side = np.array([0, 1, 0])
# finger_front_norm = finger_front / np.linalg.norm(finger_front)
# finger_side_norm = finger_side / np.linalg.norm(finger_side)
# finger_face_norm = np.cross(finger_side_norm, finger_front_norm)

# quaternion = R.from_matrix(np.concatenate([finger_face_norm.reshape(-1,1), finger_side_norm.reshape(-1,1), finger_front_norm.reshape(-1,1)], axis = 1)).as_quat()

# # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
# # prompt = grasp_obj_name
# # masks, bbox_axis_aligned_envs, grasp_envs = self.inference_gsam(rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt=prompt, save_dir=self.cfgs["SAVE_ROOT"])

# # grasp_envs[0] += 0.00

# step_num = 0
# #import pdb; pdb.set_trace()
# # move to pre-grasp
# print("grasp_pre_grasp: ", grasp_pre_grasp)


# self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])
# step_num, traj = self.control_to_pose(grasp_pre_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
# import pdb; pdb.set_trace()
# points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)

# trajs = []
# fig_data = []
# for _ in range(5):
#     # add noise
#     _=0
#     target = filtered_gg[_].translation
#     target[2] = -target[2]
#     grasp_grasp[:3] = target
#     step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
#     trajs.append(traj)
#     config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
#     urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"
#     base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
#     ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
#     tensor_args = TensorDeviceType()
#     robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
#     kin_model = CudaRobotModel(robot_cfg.kinematics)
#     qpos = torch.tensor(traj.position, **vars(tensor_args))
#     out = kin_model.get_state(qpos)
#     traj_p = out.ee_position.cpu().numpy()
#     fig_data.append(go.Scatter3d(x=traj_p[:,0], y=traj_p[:,1], z=traj_p[:,2], mode='markers', name='waypoints', marker=dict(size=10, color='red')))
#     for i in range(0, traj_p[:,0].shape[0] - 1): fig_data.append(go.Scatter3d(x=traj_p[:,0][i:i+2], y=traj_p[:,1][i:i+2], z=traj_p[:,2][i:i+2], mode='lines', name='path', line=dict(width=10, color='yellow')))

# fig_data.append(go.Scatter3d(x=points_envs[0][:,0], y=points_envs[0][:,1], z=points_envs[0][:,2], mode='markers', name='waypoints', marker=dict(size=4, color=colors_envs[0])))
# # add lines between waypoints
# fig = go.Figure(data = fig_data)
# fig.show()
# fig.write_html("test.html")


# # move to grasp
# print("grasp_grasp: ", grasp_grasp)
# step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
# step_num = self.move_gripper(close_gripper = True, save_video=save_video, save_root = save_root, start_step = step_num)

# # move to lift
# print("grasp_lift: ", grasp_lift)
# step_num, traj = self.control_to_pose(grasp_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)

# # move to pre-place
# print("place_position_lift: ", place_position_lift)
# step_num, traj = self.control_to_pose(place_position_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)

# # move to place
# print("place_position_place: ", place_position_place)
# step_num, traj = self.control_to_pose(place_position_place, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)
# step_num = self.move_gripper(close_gripper = False, save_video=save_video, save_root = save_root, start_step = step_num)

# # move to pre-place

# print("place_position_up: ", place_position_up)
# step_num, traj = self.control_to_pose(place_position_up, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
