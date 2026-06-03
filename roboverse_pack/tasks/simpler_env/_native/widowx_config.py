"""Vendored WidowX (wx250s) config + grasp checker — values verbatim, no upstream import.

Reproduces WidowXDefaultConfig / WidowXSinkCameraSetupConfig (agents/configs/widowx/defaults.py)
and WidowX.check_grasp (agents/robots/widowx.py) for the deployed Bridge control mode
``arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`` (control_freq 5, sim_freq 500).
"""

from __future__ import annotations

import numpy as np
from transforms3d.quaternions import mat2quat

from .control import PDEEPoseControllerConfig, PDJointPosMimicControllerConfig
from .control._vendor_utils import compute_angle_between, normalize_vector
from .grasp import compute_total_impulse, get_pairwise_contacts

WIDOWX_SIM_FREQ = 500
WIDOWX_CONTROL_FREQ = 5
WIDOWX_SUBSTEPS = WIDOWX_SIM_FREQ // WIDOWX_CONTROL_FREQ  # 100

ARM_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
GRIPPER_JOINT_NAMES = ["left_finger", "right_finger"]
EE_LINK_NAME = "ee_gripper_link"

ARM_STIFFNESS = [
    1169.7891719504198,
    730.0,
    808.4601346394447,
    1229.1299089624076,
    1272.2760456418862,
    1056.3326605132252,
]
ARM_DAMPING = [330.0, 180.0, 152.12036565582588, 309.6215302722146, 201.04998711007383, 269.51458932695414]
ARM_FORCE_LIMIT = [200, 200, 100, 100, 100, 100]
ARM_FRICTION = 0.0
GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_FORCE_LIMIT = 1000, 200, 60

# wx250s init qpos (base_env _initialize_agent widowx branch) + init pose
INIT_QPOS = np.array([-0.01840777, 0.0398835, 0.22242722, -0.00460194, 1.36524296, 0.00153398, 0.037, 0.037])
# widowx_sink_camera_setup uses a different arm qpos (eggplant task)
SINK_QPOS = np.array([-0.2600599, -0.12875618, 0.04461369, -0.00652761, 1.7033415, -0.26983038, 0.037, 0.037])
ROBOT_INIT_HEIGHT = 0.870
SCENE_OFFSET = np.array([-2.0634, -2.8313, 0.0])  # bridge_table_1_v1
SCENE_POSE_Q = [0.707, 0.707, 0, 0]

CAMERA_INTRINSIC = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])
CAMERA_W, CAMERA_H = 640, 480

# urdf gripper finger friction material
_URDF_GRIPPER_MAT = dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
_GRIPPER_LINKS = ["left_finger_link", "right_finger_link"]


def look_at_q(eye, target, up=(0, 0, 1)):
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(np.array(up, dtype=float))
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return mat2quat(rotation)


# default 3rd_view_camera (robot="widowx") and sink-setup camera (eggplant)
DEFAULT_CAMERA = dict(p=[0.0, -0.16, 0.36], q=look_at_q([0, 0, 0], [1, 0.553, -1.085]), actor_uid="base_link")
SINK_CAMERA = dict(p=[-0.00300001, -0.21, 0.39], q=[-0.907313, 0.0782, -0.36434, -0.194741], actor_uid="base_link")


def widowx_deployed_controller_configs():
    """arm_pd_ee_target_delta_pose_align2 + gripper_pd_joint_pos (the deployed Bridge mode)."""
    arm = PDEEPoseControllerConfig(
        ARM_JOINT_NAMES,
        -1.0,
        1.0,
        np.pi / 2,
        ARM_STIFFNESS,
        ARM_DAMPING,
        ARM_FORCE_LIMIT,
        friction=ARM_FRICTION,
        ee_link=EE_LINK_NAME,
        frame="ee_align2",
        use_target=True,
        normalize_action=False,
    )
    extra = 0.001  # extra_gripper_clearance
    gripper = PDJointPosMimicControllerConfig(
        GRIPPER_JOINT_NAMES,
        0.015 - extra,
        0.037 + extra,
        GRIPPER_STIFFNESS,
        GRIPPER_DAMPING,
        GRIPPER_FORCE_LIMIT,
        normalize_action=True,
        drive_mode="force",
    )
    return {"arm": arm, "gripper": gripper}


def apply_widowx_urdf_materials(scene, robot):
    """Set the gripper finger friction material (WidowXDefaultConfig.urdf_config)."""
    mat = scene.create_physical_material(**_URDF_GRIPPER_MAT)
    links = {lk.get_name(): lk for lk in robot.get_links()}
    for name in _GRIPPER_LINKS:
        for cs in links[name].get_collision_shapes():
            cs.set_physical_material(mat)


def apply_robot_render_material(robot, specular=0.9, roughness=0.3):
    """base_env._load_agent: set_articulation_render_material(robot, specular=0.9, roughness=0.3)."""
    for link in robot.get_links():
        for b in link.get_visual_bodies():
            for s in b.get_render_shapes():
                m = s.material
                m.specular = specular
                m.roughness = roughness


class WidowXGraspChecker:
    """WidowX.check_grasp (max_angle=60; left/right finger links)."""

    def __init__(self, robot, scene):
        self.scene = scene
        links = {lk.get_name(): lk for lk in robot.get_links()}
        self.finger_left_link = links["left_finger_link"]
        self.finger_right_link = links["right_finger_link"]

    def check_grasp(self, actor, min_impulse=1e-6, max_angle=60):
        contacts = self.scene.get_contacts()
        limp = compute_total_impulse(get_pairwise_contacts(contacts, self.finger_left_link, actor))
        rimp = compute_total_impulse(get_pairwise_contacts(contacts, self.finger_right_link, actor))
        ldir = self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        rdir = self.finger_right_link.pose.to_transformation_matrix()[:3, 1]
        langle = compute_angle_between(ldir, limp)
        rangle = compute_angle_between(-rdir, rimp)
        lflag = (np.linalg.norm(limp) >= min_impulse) and np.rad2deg(langle) <= max_angle
        rflag = (np.linalg.norm(rimp) >= min_impulse) and np.rad2deg(rangle) <= max_angle
        return bool(lflag and rflag)
