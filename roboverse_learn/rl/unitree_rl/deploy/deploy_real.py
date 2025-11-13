from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)
import time

import numpy as np
import rootutils
import torch
from collections import deque
from copy import deepcopy
rootutils.setup_root(__file__, pythonpath=True)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from roboverse_learn.rl.unitree_rl.deploy.common.command_helper import (
    MotorMode,
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from roboverse_learn.rl.unitree_rl.deploy.common.remote_controller import KeyMap, RemoteController
from roboverse_learn.rl.unitree_rl.deploy.common.rotation_helper import get_gravity_orientation, transform_imu_data
from roboverse_learn.rl.unitree_rl.deploy.config import G1Config
from roboverse_learn.rl.unitree_rl.deploy.utils import get_euler_xyz


class Controller:
    def __init__(self, config: G1Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network

        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_body_dof_pos = config.default_body_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.obs_queue = deque(
            [deepcopy(self.obs) for _ in range(self.config.obs_len_history)],
            maxlen=self.config.obs_len_history,
        )
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.last_action = np.zeros((config.num_actions,), dtype=np.float32)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo | LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        default_body_pos = self.config.default_body_angles.copy()
        body_kps = self.config.default_body_kps.copy()
        body_kds = self.config.default_body_kds.copy()
        body_size = len(default_body_pos)
        # record the current pos
        init_body_dof_pos = np.zeros(body_size, dtype=np.float32)
        for i in range(body_size):
            init_body_dof_pos[i] = self.low_state.motor_state[i].q

        # move to default pos
        for i in range(num_step + 1):
            alpha = i / num_step
            for j in range(body_size):
                self.low_cmd.motor_cmd[j].q = init_body_dof_pos[j] * (1 - alpha) + default_body_pos[j] * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = body_kps[j]
                self.low_cmd.motor_cmd[j].kd = body_kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        default_body_pos = self.config.default_body_angles.copy()
        body_kps = self.config.default_body_kps.copy()
        body_kds = self.config.default_body_kds.copy()
        body_size = len(default_body_pos)
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(body_size):
                self.low_cmd.motor_cmd[i].q = default_body_pos[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = body_kps[i]
                self.low_cmd.motor_cmd[i].kd = body_kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for default_idx, sorted_idx in self.config.body_default_sorted_idx_tuples:
            self.qj[sorted_idx] = self.low_state.motor_state[default_idx].q
            self.dqj[sorted_idx] = self.low_state.motor_state[default_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        # angular velocity handling
        # for torso IMU, transform requires shape (1, 3); otherwise keep (3,)
        if self.config.imu_type == "torso":
            ang_vel_in = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel_in
            )
        else:
            ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        base_euler_xyz = get_euler_xyz(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.obs[0:3] = self.cmd * self.config.cmd_scale
        self.obs[3:6] = ang_vel
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs_queue.append(deepcopy(self.obs))

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(np.concatenate(list(self.obs_queue), axis=0)).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        delay = 0.2
        self.action = (1 - delay) * self.action + delay * self.last_action
        self.last_action = self.action.copy()

        # transform action to target_body_dof_pos
        target_body_dof_pos = self.config.default_body_angles.copy()
        target_body_dof_pos[self.config.body_default_sorted_idx_tuples[:, 0]] += (
            self.action[self.config.body_default_sorted_idx_tuples[:, 1]] * self.config.action_scale
        )

        # Build low cmd

        for i in range(target_body_dof_pos.shape[0]):
            self.low_cmd.motor_cmd[i].q = target_body_dof_pos[i]
            self.low_cmd.motor_cmd[i].qd = 0
            self.low_cmd.motor_cmd[i].kp = self.config.default_body_kps[i]
            self.low_cmd.motor_cmd[i].kd = self.config.default_body_kds[i]
            self.low_cmd.motor_cmd[i].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # # Load config
    config_path = f"roboverse_learn/rl/unitree_rl/deploy/configs/{args.config}"
    config = G1Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
