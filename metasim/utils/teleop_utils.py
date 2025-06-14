from __future__ import annotations

import numpy as np
import pygame
from loguru import logger as log

# todo: add pygame, websockets to requirements.txt


class PygameKeyboardClient:
    def __init__(self, width=640, height=480, title="Keyboard Control"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        self.pressed_keys = set()

        self.instructions = [
            "===== Keyboard Controls Instructions =====",
            "support combination of multiple keys",
            "example: press UP+RIGHT for diagonal move",
            "",
            "   Movement (world EE coords):",
            "     UP    - Move +X",
            "     DOWN  - Move -X",
            "     LEFT  - Move +Y",
            "     RIGHT - Move -Y",
            "     e     - Move +Z",
            "     d     - Move -Z",
            "",
            "   Rotation (local EE coords):",
            "     q     - Roll + ",
            "     w     - Roll - ",
            "     a     - Pitch + ",
            "     s     - Pitch - ",
            "     z     - Yaw + ",
            "     x     - Yaw - ",
            "",
            "   Gripper:",
            "     Space - Close(hold) / Open(release)",
            "",
            "     ESC   - Quit Simulation",
            "==========================================",
        ]

    def update(self) -> bool:
        """
        fresh pygame event, update pressed_keys.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
        return True

    def is_pressed(self, key: int) -> bool:
        return (key in self.pressed_keys) or pygame.key.get_pressed()[key]

    def close(self):
        pygame.quit()

    def draw_instructions(self):
        font = pygame.font.Font(
            pygame.font.match_font("DejaVu Sans Mono"), 25
        )  # monospaced font Courier
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        self.screen.fill(bg_color)

        y_offset = 20
        for line in self.instructions:
            text_surface = font.render(line, True, text_color)
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 30

        pygame.display.flip()


def process_kb_input(
    keyboard_client: PygameKeyboardClient, dpos: float, drot: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    delta_pos = np.zeros(3, dtype=np.float32)
    delta_rot = np.zeros(3, dtype=np.float32)

    key_to_action = {
        pygame.K_UP: (delta_pos, np.array([dpos, 0.0, 0.0], dtype=np.float32)),
        pygame.K_DOWN: (delta_pos, np.array([-dpos, 0.0, 0.0], dtype=np.float32)),
        pygame.K_LEFT: (delta_pos, np.array([0.0, dpos, 0.0], dtype=np.float32)),
        pygame.K_RIGHT: (delta_pos, np.array([0.0, -dpos, 0.0], dtype=np.float32)),
        pygame.K_e: (delta_pos, np.array([0.0, 0.0, dpos], dtype=np.float32)),
        pygame.K_d: (delta_pos, np.array([0.0, 0.0, -dpos], dtype=np.float32)),
        pygame.K_q: (delta_rot, np.array([drot, 0.0, 0.0], dtype=np.float32)),
        pygame.K_w: (delta_rot, np.array([-drot, 0.0, 0.0], dtype=np.float32)),
        pygame.K_a: (delta_rot, np.array([0.0, drot, 0.0], dtype=np.float32)),
        pygame.K_s: (delta_rot, np.array([0.0, -drot, 0.0], dtype=np.float32)),
        pygame.K_z: (delta_rot, np.array([0.0, 0.0, drot], dtype=np.float32)),
        pygame.K_x: (delta_rot, np.array([0.0, 0.0, -drot], dtype=np.float32)),
    }

    # cache pressed keys first rather query keyboard_client.is_pressed(key) every time to save computation
    pressed_keys = {key: keyboard_client.is_pressed(key) for key in key_to_action.keys()}
    for key, (target, value) in key_to_action.items():
        if pressed_keys.get(key):
            target += value

    close_gripper = keyboard_client.is_pressed(pygame.K_SPACE)

    return delta_pos, delta_rot, close_gripper


#############################################################
################### android phone
#############################################################

import asyncio
import json
import threading
import time

import websockets


class PhoneServer:
    def __init__(self, translation_step, host="0.0.0.0", port=8765, update_dt=1 / 50):
        self.host = host
        self.port = port
        self.update_dt = update_dt
        self.translation_step = translation_step

        self._lock = threading.Lock()

        self._latest_data = {
            "buttonStates": [False] * 6,
            "isSwitch1On": False,
            "isSwitch2On": False,
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }

        self.q_world = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.delta_position = np.zeros(3)
        self.rotation_flag = False
        self.gripper_flag = False

    def key_map(self, data):
        delta_position = np.zeros(3)

        button_states = data["buttonStates"]
        isSwitch1On = data["isSwitch1On"]
        isSwitch2On = data["isSwitch2On"]

        rotation_flag = True if isSwitch1On else False
        gripper_flag = True if isSwitch2On else False

        # kep map
        if button_states[0]:
            delta_position[2] = -self.translation_step
        if button_states[1]:
            delta_position[2] = self.translation_step
        if button_states[2]:  # todo: check why the y axis is inversed
            delta_position[1] = -self.translation_step
        if button_states[5]:
            delta_position[1] = self.translation_step
        if button_states[4]:
            delta_position[0] = -self.translation_step
        if button_states[3]:
            delta_position[0] = self.translation_step

        q_world = data.get("rotation", [0.0, 0.0, 0.0, 1.0])[:4]
        q_world = np.array(q_world, dtype=np.float64)

        return q_world, delta_position, gripper_flag

    async def handler(self, websocket, path):
        """
        whenever a client connects, this handler will be called.
        upon receiving data, update self._latest_data.
        """

        log.info("A client connected.")
        time.sleep(3)
        self.receive_flag = True

        while True:
            try:
                message = await websocket.recv()
                # print("Received from client:", message)

                data = json.loads(message)

                q_world, delta_position, gripper_flag = self.key_map(data)

                # update latest data (thread-safe)
                with self._lock:
                    self.q_world = q_world
                    self.delta_position = delta_position
                    self.gripper_flag = gripper_flag

            except websockets.ConnectionClosed:
                self.receive_flag = False
                log.info("Connection closed.")
                break
            except Exception as e:
                self.receive_flag = False
                log.error(f"Error in handler: {e}")
                break
            except KeyboardInterrupt:
                self.receive_flag = False
                log.info("KeyboardInterrupt.")
                break

    async def main_server(self):
        """
        start the websockets server (async)
        """
        async with websockets.serve(self.handler, self.host, self.port):
            log.info(f"WebSocket Server started, listening on {self.host}:{self.port}")
            await asyncio.Future()  # run forever

    def start_server(self):
        """
        start the server in a new thread
        """

        def run_asyncio():
            asyncio.run(self.main_server())

        server_thread = threading.Thread(target=run_asyncio, daemon=True)
        server_thread.start()

    def get_latest_data(self):
        """
        get the latest position, orientation (thread-safe).
        return: (position: np.ndarray, orientation: np.ndarray)
        """
        with self._lock:
            # pos = self._latest_data["position"].copy()
            q_world = self.q_world.copy()
            delta_position = self.delta_position.copy()
            gripper_flag = self.gripper_flag
        return q_world, delta_position, gripper_flag

    @property
    def _latest_position(self):
        """
        internal attribute: get the latest position (no lock, for internal use only)
        """
        return self._latest_data["position"]


def quaternion_inverse(q):
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-12:
        log.warning("Warning: Quaternion norm is too small, potential divide by zero.")
        return np.array([0, 0, 0, 1])  # Return a default unit quaternion
    q_inv = q * np.array([1, -1, -1, -1])
    return q_inv / norm_q


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def get_delta_quaternion(quaternion1, quaternion2):
    """delta q = q2 * q1^-1"""
    q1_inv = quaternion_inverse(quaternion1)
    delta_q = quaternion_multiply(quaternion2, q1_inv)
    return delta_q


def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def rotate_vector_with_quaternion(q, v):
    v_quaternion = np.array([0, v[0], v[1], v[2]])  # 将向量 v 转化为四元数
    q_conjugate = quaternion_conjugate(q)
    # v' = q * v * q^-1
    rotated_quaternion = quaternion_multiply(quaternion_multiply(q, v_quaternion), q_conjugate)
    return rotated_quaternion[1:]


TRANSFORMATION_MATRIX = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def transform_position(joystick_coords):
    transformed_position = np.dot(TRANSFORMATION_MATRIX, joystick_coords)
    return transformed_position


def transform_orientation(quaternion_orientation):
    rotation_quaternion = rotation_matrix_to_quaternion(TRANSFORMATION_MATRIX)
    transformed_quaternion = quaternion_multiply(
        quaternion_multiply(rotation_quaternion, quaternion_orientation),
        quaternion_conjugate(rotation_quaternion),
    )
    return transformed_quaternion


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    norm_q = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm_q < 1e-12:
        log.warning("Warning: Quaternion norm is too small, potential divide by zero.")
        return np.eye(3)  # Return the identity matrix if the quaternion is invalid
    qx, qy, qz, qw = qx / norm_q, qy / norm_q, qz / norm_q, qw / norm_q
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )
    return R


#############################################################
################### Pico VR
#############################################################
import xrobotoolkit_sdk as xrt


class XrClient:
    """Client for the XrClient SDK to interact with XR devices."""

    def __init__(self):
        """Initializes the XrClient and the SDK."""
        xrt.init()

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Returns the pose of the specified device by name.
        Valid names: "left_controller", "right_controller", "headset".
        Pose is [x, y, z, qx, qy, qz, qw]."""
        if name == "left_controller":
            return xrt.get_left_controller_pose()
        elif name == "right_controller":
            return xrt.get_right_controller_pose()
        elif name == "headset":
            return xrt.get_headset_pose()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_controller', 'right_controller', 'headset'."
            )

    def get_key_value_by_name(self, name: str) -> float:
        """Returns the trigger/grip value by name (float).
        Valid names: "left_trigger", "right_trigger", "left_grip", "right_grip".
        """
        if name == "left_trigger":
            return xrt.get_left_trigger()
        elif name == "right_trigger":
            return xrt.get_right_trigger()
        elif name == "left_grip":
            return xrt.get_left_grip()
        elif name == "right_grip":
            return xrt.get_right_grip()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
            )

    def get_button_state_by_name(self, name: str) -> bool:
        """Returns the button state by name (bool).
        Valid names: "A", "B", "X", "Y",
                      "left_menu_button", "right_menu_button",
                      "left_axis_click", "right_axis_click"
        """
        if name == "A":
            return xrt.get_A_button()
        elif name == "B":
            return xrt.get_B_button()
        elif name == "X":
            return xrt.get_X_button()
        elif name == "Y":
            return xrt.get_Y_button()
        elif name == "left_menu_button":
            return xrt.get_left_menu_button()
        elif name == "right_menu_button":
            return xrt.get_right_menu_button()
        elif name == "left_axis_click":
            return xrt.get_left_axis_click()
        elif name == "right_axis_click":
            return xrt.get_right_axis_click()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'A', 'B', 'X', 'Y', "
                "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
            )

    def get_timestamp_ns(self) -> int:
        """Returns the current timestamp in nanoseconds (int)."""
        return xrt.get_time_stamp_ns()

    def close(self):
        xrt.close()


R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
)  # rotation matrix from headset to world frame
