import os
import pathlib
import time

import hydra
import torch
import tqdm
from omegaconf import OmegaConf

from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.utils.real_world.attr_dict import AttrDict

os.environ["WANDB_SILENT"] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import numpy as np
from termcolor import cprint

from roboverse_learn.il.utils.real_world.multi_realsense import MultiRealsenseWrapper
from roboverse_learn.il.utils.real_world.franka_ros_client import FrankaRobotClient
from roboverse_learn.il.utils.real_world.franka_ros import FrankaRobot

class RealWorldEnv:
    """
    The deployment is running on the local computer of the robot.
    """

    def __init__(
        self,
        device="gpu",
        use_server_robot=True,
        use_rs_pntcloud=False,
        task_name="RealworldLiberoPickButter",
        use_tianji=True
    ):
        # camera
        # self.camera = MultiRealsenseWrapper(set_auto_exposure=False, exposure_time=1500, gain=20, use_rs_pcd=use_rs_pntcloud,
        #                                     task_name=task_name, use_post_process=True, restore_depth=True, num_points=4096)
        # NOTE: `use_rs_pcd` controls whether camera outputs `xyzrgb` point cloud.
        self.camera = MultiRealsenseWrapper(
            set_auto_exposure=True,
            use_rs_pcd=use_rs_pntcloud,
            task_name=task_name,
            num_points=4096,
        )
        if use_server_robot:
            self.robot = FrankaRobotClient()
        else:
            self.robot = FrankaRobot()
        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def step(self, single_step_action, use_ee_control=False):
        if not use_ee_control:
        # Stepping the robot
            action = self._action_to_ros_state(single_step_action)
            # import pdb; pdb.set_trace()
            self.robot.goto(action)
        else:
            self.robot.goto_ee_state(single_step_action["franka"]["dof_pos_target"]["ee_state"])
        # After execution
        cam_dict = self.camera()
        robot_state, robot_ee_state = map(torch.tensor, self.robot.get_state(return_ee=True))
        obs_dict = {
            "agent_pos": robot_state.unsqueeze(0).to(self.device),
            "cameras": cam_dict,
            "robots":{
                "franka":{
                    "joint_pos": robot_state.to(self.device),
                    "robot_ee_state": robot_ee_state.to(self.device)
                }
            }
        }
        if  len(obs_dict["cameras"]["camera0"]["depth"].shape) == 2:
            obs_dict["cameras"]["camera0"]["depth"] = obs_dict["cameras"]["camera0"]["depth"].unsqueeze(-1)
        if (
            not obs_dict["cameras"]["camera0"]["rgb"].shape[-1] == 3
            or not obs_dict["cameras"]["camera0"]["depth"].shape[-1] == 1
            or not len(obs_dict["cameras"]["camera0"]["rgb"].shape) == 3
            or not len(obs_dict["cameras"]["camera0"]["depth"].shape) == 3
        ):
            raise ValueError(
                f"Please check the camera output shape. Expected RGB shape: (H, W, C) and Depth shape: (H, W, C), but got {obs_dict['cameras']['camera0']['rgb'].shape} and {obs_dict['cameras']['camera0']['depth'].shape}"
            )
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert the entire obs_dict to AttrDict for consistency
        return obs_dict

    def step_ee(self, single_step_action):

        self.robot.goto_ee_state(single_step_action[0]["franka"]["ee_state_target"])
        # After execution
        cam_dict = self.camera()
        robot_state, robot_ee_state = map(torch.tensor, self.robot.get_state(return_ee=True))
        obs_dict = {
            "agent_pos": robot_state.unsqueeze(0).to(self.device),
            "cameras": cam_dict,
            "robots":{
                "franka":{
                    "joint_pos": robot_state.to(self.device),
                    "robot_ee_state": robot_ee_state.to(self.device)
                }
            }
        }
        if  len(obs_dict["cameras"]["camera0"]["depth"].shape) == 2:
            obs_dict["cameras"]["camera0"]["depth"] = obs_dict["cameras"]["camera0"]["depth"].unsqueeze(-1)
        if (
            not obs_dict["cameras"]["camera0"]["rgb"].shape[-1] == 3
            or not obs_dict["cameras"]["camera0"]["depth"].shape[-1] == 1
            or not len(obs_dict["cameras"]["camera0"]["rgb"].shape) == 3
            or not len(obs_dict["cameras"]["camera0"]["depth"].shape) == 3
        ):
            raise ValueError(
                f"Please check the camera output shape. Expected RGB shape: (H, W, C) and Depth shape: (H, W, C), but got {obs_dict['cameras']['camera0']['rgb'].shape} and {obs_dict['cameras']['camera0']['depth'].shape}"
            )
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert the entire obs_dict to AttrDict for consistency
        return obs_dict


    def reset(self):
        # reset robot
        if self.tianji_arm:
            self.tianji_arm.home()
        self.robot.do_homing()
        time.sleep(3)  # wait for the robot to finish homing
        print("Robot ready!")

        # ======== INIT ==========
        cam_dict = self.camera()
        robot_state, robot_ee_state = self.robot.get_state(return_ee=True)
        robot_state = torch.tensor(robot_state)
        robot_ee_state = torch.tensor(robot_ee_state)
        obs_dict = {
            "agent_pos": robot_state.unsqueeze(0).to(self.device),
            "cameras": cam_dict,
            "robots":{
                "franka":{
                    "joint_pos": robot_state.to(self.device),
                    "robot_ee_state": robot_ee_state.to(self.device)
                }
            }
        }
        if len(obs_dict["cameras"]["camera0"]["depth"].shape) == 2:
            obs_dict["cameras"]["camera0"]["depth"] = obs_dict["cameras"]["camera0"]["depth"].unsqueeze(-1)
        if (
            not obs_dict["cameras"]["camera0"]["rgb"].shape[-1] == 3
            or not obs_dict["cameras"]["camera0"]["depth"].shape[-1] == 1
            or not len(obs_dict["cameras"]["camera0"]["rgb"].shape) == 3
            or not len(obs_dict["cameras"]["camera0"]["depth"].shape) == 3
        ):
            raise ValueError(
                f"Please check the camera output shape. Expected RGB shape: (H, W, C) and Depth shape: (H, W, C), but got {obs_dict['cameras']['camera0']['rgb'].shape} and {obs_dict['cameras']['camera0']['depth'].shape}"
            )
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert the entire obs_dict to AttrDict for consistency
        return obs_dict

    def rotate(self):
        assert self.tianji_arm is not None
        self.tianji_arm.rotate()

    def close(self):
        return

    def _robot_polymetis_state_to_tensor_state(self, robot_state):
        """
        Args:
            robot_state: polymetis_pb2.RobotState{
                timestamp: Dict
                joint_positions: tuple()
                joint_velocities: tuple()
                joint_torques_computed: tuple()
                prev_joint_torques_computed_safened: tuple()
                motor_torques_measured: tuple()
                motor_torques_external: tuple()
                motor_torques_desired: tuple()
                prev_controller_latency_ms: float
                prev_command_successful: bool
            }
        Returns:
            tensor_state: torch.Tensor([7,]): state_dim = 7
        """
        tensor_state = torch.tensor(robot_state.joint_positions[0:7]).to("cuda")  # (7,)
        return tensor_state

    def _gripper_polymetis_state_to_tensor_state(self, gripper_state):
        width = gripper_state.width
        return torch.tensor([width/2]*2).to("cuda")  # (2,)

    def _action_to_ros_state(self, action):
        """
        Args:
            action: Dict{robot_name: {'dof_pos_target':{"joint_name": tensor([1,])}}}
        Returns:
            robot_state: torch.Tensor([7,])
            gripper_width: float
        """
        if isinstance(action, list):
            if not len(action) == 1:
                raise ValueError(f"Expected action to be a list of length 1, but got {len(action)}")
            action = action[0]
        elif not isinstance(action, dict):
            raise ValueError(f"Expected action to be a dict or list, but got {type(action)}")

        robot_joint_name_sequence = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        gripper_joint_name_sequence = ["panda_finger_joint1", "panda_finger_joint2"]
        robot_name = "franka"
        robot_state = [
            action[robot_name]["dof_pos_target"][joint_name] for joint_name in robot_joint_name_sequence
        ]
        gripper_state = [
            action[robot_name]["dof_pos_target"][joint_name] for joint_name in gripper_joint_name_sequence
        ]
        action = gripper_state + robot_state
        return action
