import copy

import numpy as np
from loguru import logger as log

from metasim.constants import BenchmarkType
from scenario_cfg.cameras import PinholeCameraCfg
from scenario_cfg.tasks import BaseTaskCfg

_phi_theta_candidates = np.loadtxt("metasim/sim/isaaclab/cfg/randomization/camera_pos_candidates.txt")


def randomize_camera_pose(
    original_camera: PinholeCameraCfg,
    obj_pos: tuple[float, float, float],
    robot_quat: tuple[float, float, float, float],
    mode: str,
    task: BaseTaskCfg,
) -> PinholeCameraCfg:
    ## XXX: We assume that when robot front side faces X+ in its local coordinate,

    randomized_camera = copy.deepcopy(original_camera)
    if mode == "semisphere":
        distance = np.random.uniform(2, 4)
        theta_to_robot = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 70 / 180 * np.pi)
        look_at_offset = np.random.uniform([-0.2, -0.2, 0], [0.2, 0.2, 0])
    elif mode == "front":
        distance = 2.0
        theta_to_robot = 0.0
        phi = 60.0 / 180.0 * np.pi
        look_at_offset = np.array([0.0, 0.0, 0.0])
    elif mode == "front_uniform_random":
        distance = np.random.uniform(1.5, 2.0)
        theta_to_robot = np.random.normal(0, np.pi * (2 / 3))
        phi = np.random.uniform(70, 50) / 180.0 * np.pi
        look_at_offset = np.array([0.0, 0.0, 0.0])
    elif mode == "front_triangular_random":
        distance = np.random.uniform(1.5, 2.0)
        theta_to_robot = np.random.triangular(-np.pi * (2 / 3), 0, np.pi * (2 / 3))
        phi = np.random.uniform(70, 50) / 180.0 * np.pi
        look_at_offset = np.array([0.0, 0.0, 0.0])
    elif mode == "front_select":
        distance = 1.5
        phi, theta_to_robot = _phi_theta_candidates[np.random.randint(0, _phi_theta_candidates.shape[0])]
        look_at_offset = np.array([0.0, 0.0, 0.0])
        # XXX: Avoid table blocking the view
        if task.source_benchmark == BenchmarkType.CALVIN:
            distance = 2.0
            theta_to_robot = (theta_to_robot - np.pi / 2) / 240 * 180

    else:
        raise ValueError(f"Unknown mode: {mode}")

    robot_theta = np.arctan2(robot_quat[1], robot_quat[0])
    log.debug(f"robot_theta: {robot_theta / np.pi * 180:.2f}, theta_to_robot: {theta_to_robot / np.pi * 180:.2f}")
    theta = robot_theta + theta_to_robot

    pos = (np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]) * distance).tolist()
    look_at = (np.array(obj_pos) + look_at_offset).tolist()
    randomized_camera.pos = pos
    randomized_camera.look_at = look_at
    randomized_camera.focus_distance = distance
    return randomized_camera
