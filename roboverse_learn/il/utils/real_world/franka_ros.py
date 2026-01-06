import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.action import Homing, Move, Grasp
from franka_msgs.msg import GraspEpsilon
from sensor_msgs.msg import JointState
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from roboverse_learn.collect_real_world.state_reader import FrankaStateReader
import threading
import time

from pytorch_kinematics import Transform3d, matrix_to_quaternion, quaternion_to_matrix
import pytorch_kinematics as pk
import os
import torch
import zmq
import xml.etree.ElementTree as ET
import numpy as np
def state8_to_matrix(pos, xyzw):
    pos = np.array(pos, dtype=float)
    x, y, z, w = xyzw

    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = pos
    return T

def roll_pitch_yaw_to_rot_matrix(roll, pitch, yaw):
    alpha, beta, gamma = yaw, pitch, roll
    mat = np.eye(3)
    mat[0,0] = np.cos(alpha) * np.cos(beta)
    mat[0,1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    mat[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    mat[1,0] = np.sin(alpha) * np.cos(beta)
    mat[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    mat[1,2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)


class FrankaArm(Node):
    def __init__(self):
        super().__init__('joint_trajectory_action_client')
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )
        # PickButter
        self.home = [-0.06616484708418602, -0.06023259713774687, -0.02127205782529562, -1.6294949431447576, -0.04775707762955839, 1.5328541535079623, 0.7959323557075442]
        self.init_ee_solver()

    def init_ee_solver(self):
        urdf_path = "/ros2_ws/fr3.urdf"
        urdf_xml = open(urdf_path, 'r').read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_xml, "fr3_link7")
        lim = torch.tensor(self.chain.get_joint_limits(), device=torch.device('cpu'))  # (DOF,2)
        self.ik = pk.PseudoInverseIK(
            self.chain, max_iterations=90, num_retries=1, joint_limits=lim.T,
            early_stopping_any_converged=True, early_stopping_no_improvement="all",
            debug=False, lr=0.05
        )

    def _select_solution(self, sol):
        solutions = sol.solutions[0]     # (num_retries, DOF)
        converged = sol.converged[0]     # (num_retries,)
        for idx, ok in enumerate(converged.tolist()):
            if ok:
                return solutions[idx]
        errs = sol.err_pos[0] + sol.err_rot[0]
        best = torch.argmin(errs)
        return solutions[best]

    # def _pose_err_deg_mm(self, q, goal_tf):
    #     fk_tf = self.chain.forward_kinematics(q[None, :])  # (1,4,4) 或等价
    #     # 取位置
    #     p_fk = fk_tf[0,:3,3]
    #     p_goal = goal_tf.get_matrix()[0,:3,3]  # 或 goal_tf.pos
    #     pos_err_mm = torch.norm(p_fk - p_goal).item() * 1000.0

    #     # 取姿态，统一到同一四元数顺序（比如都用 wxyz），并“同号半球”
    #     q_fk   = rotmat_to_wxyz_quat(fk_tf[0,:3,:3])
    #     q_goal = goal_tf_quat_wxyz(goal_tf)
    #     dot = abs((q_fk * q_goal).sum().item())
    #     dot = min(max(dot, -1.0), 1.0)
    #     ang_err_deg = 2.0 * math.degrees(math.acos(dot))

    #     return pos_err_mm, ang_err_deg

    # def _select_solution(self, sol, goal_tf):
    #     solutions = sol.solutions[0]   # (num_retries, DOF)
    #     converged = sol.converged[0]   # (num_retries,)
    #     best_idx, best_score = None, float('inf')
    #     for idx in range(solutions.shape[0]):
    #         q = solutions[idx]
    #         pos_mm, ang_deg = self._pose_err_deg_mm(q, goal_tf)
    #         score = pos_mm + 2.0 * ang_deg  # 自定义加权
    #         if score < best_score:
    #             best_score, best_idx = score, idx
    #     return solutions[best_idx]

    def goto_joint_degree(self, joint_degrees):
        if len(joint_degrees) != 7:
            raise ValueError(f"Expected 7 joint angles in degrees, got {len(joint_degrees)}")
        joint_radians = [angle * (3.141592653589793 / 180.0) for angle in joint_degrees]
        self.get_logger().info(f"Converting degrees to radians: {joint_radians}")
        self.goto(joint_radians)

    def goto(self, joint_positions, time_to_reach=0.5):
        self.action_client.wait_for_server()
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'fr3_joint1','fr3_joint2','fr3_joint3','fr3_joint4',
            'fr3_joint5','fr3_joint6','fr3_joint7'
        ]
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(seconds=float(time_to_reach)).to_msg()
        goal_msg.trajectory.points.append(point)
        self.get_logger().info('Sending goal request...')
        self.action_client.send_goal_async(goal_msg)

    def do_homing(self):
        if hasattr(self, 'home_degrees'):
            self.goto_joint_degree(self.home_degrees)
        elif hasattr(self, 'home'):
            self.goto(self.home, time_to_reach=3.0)

    def _rotmat_to_quat(self, R: np.ndarray) -> np.ndarray:
        m = R
        trace = m[0,0] + m[1,1] + m[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (m[2,1] - m[1,2]) * s
            qy = (m[0,2] - m[2,0]) * s
            qz = (m[1,0] - m[0,1]) * s
        else:
            if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
                qw = (m[2,1] - m[1,2]) / s
                qx = 0.25 * s
                qy = (m[0,1] + m[1,0]) / s
                qz = (m[0,2] + m[2,0]) / s
            elif m[1,1] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
                qw = (m[0,2] - m[2,0]) / s
                qx = (m[0,1] + m[1,0]) / s
                qy = 0.25 * s
                qz = (m[1,2] + m[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
                qw = (m[1,0] - m[0,1]) / s
                qx = (m[0,2] + m[2,0]) / s
                qy = (m[1,2] + m[2,1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=np.float32)
        return quat

    def goto_ee_pos(self, pos16, duration=5.0, rate_hz=10, curr_qpos = None):
        arr = torch.tensor(pos16, dtype=torch.float32).reshape(4, 4)
        position = arr[:3, 3]
        R = arr[:3, :3]
        goal_tf = Transform3d(pos=position, rot=R, device=position.device)
        print('Solving IK')

        if curr_qpos is None:
            sol = self.ik.solve(goal_tf)
        else:
            self.ik.initial_config = torch.tensor(curr_qpos, dtype = self.ik.initial_config.dtype, device = self.ik.initial_config.device)
            sol = self.ik.solve(goal_tf)
        joint_pos = self._select_solution(sol)
        print(f"IK Result: {joint_pos}")
        self.goto(joint_pos.tolist(), time_to_reach=10)


class FrankaGripper(Node):
    def __init__(self):
        super().__init__('gripper_action_client')
        self.homing_client = ActionClient(self, Homing, '/franka_gripper_wrapper/homing')
        self.move_client = ActionClient(self, Move, '/franka_gripper_wrapper/move')
        self.grasp_client = ActionClient(self, Grasp, '/franka_gripper_wrapper/grasp')
        self.threshold = 0.001
        self.prev_width = None
        self.min = 100

    # def do_homing(self) -> bool:
    #     time.sleep(1.0)
    #     self.do_move(self.home)
    #     time.sleep(2.0)
    #     return True
    def do_homing(self) -> bool:
        self.get_logger().info(">>> 等待 Homing Server...")
        if not self.homing_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Homing Server 不可用！")
            return False

        goal = Homing.Goal()
        self.get_logger().info(">>> 发送 Homing 请求...")

        send_future = self.homing_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Homing 请求被拒绝！")
            return False
        self.get_logger().info(">>> Homing 已接受，等待结果...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        #result = result_future.result().result
        self.get_logger().info(">>> Homing 完成！")

        return True

    def do_move(self, width: float, speed: float = 0.10) -> bool:
        self.get_logger().info(f'>>> 等待 Move Server... 目标宽度={width:.3f} m')
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Move Server 不可用！')
            return False
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        self.get_logger().info('>>> 发送 Move 请求...')
        self.move_client.send_goal_async(goal)
        return True

    def do_grasp(self, width: float, speed: float = 5.0, force: float = 70.0) -> bool:
        self.get_logger().info(f'>>> 等待 Grasp Server... 目标宽度={width:.3f} m')
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Grasp Server 不可用！')
            return False
        goal=Grasp.Goal()
        goal.width = width
        goal.epsilon = GraspEpsilon(inner=0.04, outer=0.04)
        goal.speed = speed
        goal.force = force

        def feedback_cb(msg):
            cur = msg.feedback.current_width
            self.get_logger().info(f' [反馈] 当前宽度 = {cur:.4f} m')

        self.get_logger().info('>>> 发送 Grasp 请求...')
        self.grasp_client.send_goal_async(goal, feedback_callback=feedback_cb)
        return True

    def goto(self, width: float, current_width: float) -> bool:
        # if width > 0.03:
        #     #width = 0.079
        #     self.do_move(width)
        # elif width <= 0.03:
        #     #width = 0.0
        #     self.do_grasp(width)
        # return True
        if self.prev_width is None:
            self.prev_width = width
            return True
        if width < 0.04:
            width = 0.0
        else:
            width = 0.079
        if abs(width-self.prev_width) <0.01:
            self.prev_width = width
            return True

        self.do_grasp(width)
        self.prev_width = width

        # if self.prev_width is None:
        #     self.prev_width = width
        #     self.do_move(width)
        #     return True

        # if abs(width - self.prev_width) < 0.01:
        #     return True

        # self.prev_width = width
        # self.do_grasp(width)


class FrankaRobot():
    def __init__(self):
        try:
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            rclpy.init()

        self.arm_client = FrankaArm()
        self.gripper_client = FrankaGripper()
        self.state_reader = FrankaStateReader()
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.state_reader)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        print(f"ik joint seq:{self.arm_client.chain.get_joint_parameter_names()}")

        # self._lock = threading.Lock()
        # self._latest_joint_state = None       # list[7]
        # self._latest_gripper_state = None     # list[2]
        # self._got_joint_evt = threading.Event()
        # self._got_grip_evt = threading.Event()

        # self.arm_client.create_subscription(
        #     JointState, '/franka/joint_states', self._callback_joint_states, 10
        # )
        # self.gripper_client.create_subscription(
        #     JointState, '/franka_gripper/joint_states', self._callback_gripper_states, 10
        # )

        # self.executor = SingleThreadedExecutor()
        # self.executor.add_node(self.arm_client)
        # self.executor.add_node(self.gripper_client)

        # self._spin_thread = threading.Thread(target=self._spin_forever, daemon=True)
        # self._spin_thread.start()

        self.desired_seq_joints = [
            "fr3_joint1","fr3_joint2","fr3_joint3","fr3_joint4",
            "fr3_joint5","fr3_joint6","fr3_joint7"
        ]
        self.desired_seq_gripper = [
            "fr3_finger_joint1","fr3_finger_joint2",
        ]
        # self._joint_names = [
        #     'fr3_joint1','fr3_joint2','fr3_joint3',
        #     'fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7'
        # ]

    # def _spin_forever(self):
    #     try:
    #         self.executor.spin()
    #     except Exception as e:
    #         print(f"[FrankaRobot] executor stopped: {e}")

    # def _callback_joint_states(self, msg: JointState):
    #     # name2idx = {n: i for i, n in enumerate(msg.name)}
    #     # try:
    #     #     state = [msg.position[name2idx[j]] for j in self.desired_seq_joints]
    #     # except KeyError:
    #     #     return
    #     # with self._lock:
    #     #     self._latest_joint_state = state
    #     # self._got_joint_evt.set()
    #     name2idx = {n:i for i,n in enumerate(msg.name)}
    #     arm_joints = [(msg.position[name2idx[jn]] if jn in name2idx else None)
    #                   for jn in self._joint_names]
    #     self._latest_joint_state = arm_joints

    # def _callback_gripper_states(self, msg: JointState):
    #     # name2idx = {n: i for i, n in enumerate(msg.name)}
    #     # try:
    #     #     state = [msg.position[name2idx[j]] for j in self.desired_seq_gripper]
    #     # except KeyError:
    #     #     return
    #     # with self._lock:
    #     #     self._latest_gripper_state = state
    #     # self._got_grip_evt.set()
    #     joint_name = "fr3_finger_joint1"
    #     if joint_name not in msg.name:
    #         raise ValueError(f"Expected joint name '{joint_name}' not found in message names {msg.name}")
    #     state = [msg.position[msg.name.index(joint_name)]] * 2
    #     self._latest_gripper_state = state


    def get_state(self, return_ee = False):
        # if wait_first_frame:
        #     self._got_joint_evt.wait(timeout=timeout)
        #     self._got_grip_evt.wait(timeout=timeout)

        # with self._lock:
        #     if self._latest_gripper_state is None or self._latest_joint_state is None:
        #         return None
        # return self._latest_gripper_state + self._latest_joint_state
        joint_pos, gripper_width, ee_pos, ee_rot, joint_stamp, gripper_stamp, ee_stamp = self.state_reader.get_state()
        gripper_pos = [gripper_width/2] * 2
        print(f"Sending state: {gripper_pos + joint_pos}")
        if not return_ee:
            return gripper_pos + joint_pos
        robot_ee_state = torch.cat([torch.tensor(ee_pos), torch.tensor(ee_rot), torch.tensor([gripper_width])]).tolist()
        return (gripper_pos + joint_pos, robot_ee_state)

    def goto(self, goal):
        if not len(goal) == len(self.desired_seq_gripper) + len(self.desired_seq_joints):
            raise ValueError(
                f"目标关节数 {len(goal)} 与期望 {len(self.desired_seq_gripper) + len(self.desired_seq_joints)} 不匹配"
            )
        cur = self.get_state()
        if cur is None:
            raise RuntimeError("尚未收到任何状态，无法执行 goto")

        current_gripper_width = sum(cur[0:2])
        joint_pos = goal[-7:]
        gripper_width = sum(goal[0:2])

        self.arm_client.goto(joint_pos, time_to_reach=1)
        self.gripper_client.goto(gripper_width, current_gripper_width)

    def goto_joint_degree(self, joint_degrees):
        if not len(joint_degrees) == len(self.desired_seq_joints):
            raise ValueError(
                f"目标关节数 {len(joint_degrees)} 与期望 {len(self.desired_seq_joints)} 不匹配"
            )
        cur = self.get_state()
        if cur is None:
            raise RuntimeError("尚未收到任何状态，无法执行 goto_joint_degree")

        current_gripper_width = sum(cur[0:2])
        joint_pos_deg = joint_degrees[-7:]
        gripper_width = sum(joint_degrees[0:2])

        self.arm_client.goto_joint_degree(joint_pos_deg)
        self.gripper_client.goto(gripper_width, current_gripper_width)

    def do_homing(self):
        self.arm_client.do_homing()
        self.gripper_client.do_homing()
        time.sleep(3.0)
        state = self.get_state()
        print("已收到第一条关节状态：", state)

    def goto_ee_pos(self, pos16, curr_qpos=None):
        print("goto_ee_pos called with pos16:", pos16)
        self.arm_client.goto_ee_pos(pos16, curr_qpos)

    def goto_ee_state(self, state8):
        half_gripper_width = state8[-1]
        pos = state8[0:3]
        xyzw = state8[3:7]
        pos16 = state8_to_matrix(pos, xyzw)
        cur = self.get_state()
        curr_qpos = cur[2:]
        print(f"---------------------------------------")
        print(f"pos:{pos}")
        print(f"xyzw:{xyzw}")
        print(f"curr_qpos:{curr_qpos}")
        self.goto_ee_pos(pos16, curr_qpos)
        current_gripper_width = sum(cur[0:2])
        self.gripper_client.goto(half_gripper_width * 2, current_gripper_width)
        time.sleep(1)
        new_state = self.get_state(return_ee = True)
        new_qpos = new_state[0][2:]
        new_ee = new_state[1]
        print(f"new qpos:{new_qpos}")
        print(f"new_ee:{new_ee}")
        print(f"---------------------------------------\n\n\n")


    def get_ee_pos_and_quat(self):
        """
        Returns:
            pos: (3,) np.ndarray
            rot: (3,) np.ndarray, XYZ euler angles in radians
        """
        joint_pos = self.get_state()[2:]
        ret = self.chain.forward_kinematics(joint_pos, end_only=True)
        # get transform matrix (1,4,4), then convert to separate position and unit quaternion
        m = ret['lbr_iiwa_link_7'].get_matrix()
        pos = m[:, :3, 3]
        rot = pk.matrix_to_euler_angles(m[:, :3, :3], order='XYZ')
        return pos, rot

    def goto_delta_osc_pose(self, pos7):
        """
        Args:
            pos7: (7,) np.ndarray, [dx, dy, dz, droll, dpitch, dyaw, gripper_action], where gripper_action == -1 -> close, == 1 -> open
        """
        dx, dy, dz, droll, dpitch, dyaw, gripper_action = pos7
        target_gripper_width = 0.079 if gripper_action > 0 else 0.0
        pos, rot = self.get_ee_pos_and_quat()
        new_pos = pos + np.array([dx, dy, dz], dtype=np.float32)
        new_rot = rot + np.array([droll, dpitch, dyaw], dtype=np.float32)
        new_R = quaternion_to_matrix(quaternions=pk.euler_angles_to_matrix(new_rot, order='XYZ'))
        new_mat16 = np.eye(4, dtype=np.float32)
        new_mat16[:3, :3] = new_R
        new_mat16[:3, 3] = new_pos
        self.goto_ee_pos(new_mat16)

    def shutdown(self):
        try:
            self.executor.shutdown()
            self.spin_thread.join()
        except Exception:
            pass
        try:
            self.arm_client.destroy_node()
            self.gripper_client.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

class FrankaRobotServer():
    def __init__(self, socket_number = 5555):
        self.robot = FrankaRobot()
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{socket_number}")
        print(f"Franka Robot Server Listening on tcp://127.0.0.1:{socket_number}")
        self.socket = socket

    def run(self):
        while True:
            message = self.socket.recv_json()
            print(f"Received request: {message}")
            if message['command'] == 'goto':
                self.robot.goto(message['goal'])
                response = {'status': 'success', 'message': 'Goal reached'}
            elif message['command'] == 'homing':
                self.robot.do_homing()
                response = {'status': 'success', 'message': 'Homing completed'}
            elif message['command'] == 'get_state':
                state = self.robot.get_state(return_ee = message["return_ee"])
                response = {'status': 'success', 'state': state}
            elif message['command'] == 'goto_ee_pose':
                self.robot.goto_ee_pos(message['goal'])
                response = {'status': 'success', 'message': 'EE pose reached'}
            elif message['command'] == 'goto_ee_state':
                self.robot.goto_ee_state(message['goal'])
            else:
                response = {'status': 'error', 'message': 'Unknown action'}
            self.socket.send_json(response)
