import os
import sys

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import torch
from yourdfpy import URDF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import third_party.pyroki.examples.pyroki_snippets as pks

from metasim.utils.hf_util import check_and_download_single


class get_pyroki_model:
    """Class to get the Pyroki robot model.

    This class loads a robot model from a URDF file and provides an
    inverse kinematics (IK) solver interface using Pyroki.

    Attributes:
        urdf_path (str): Path to the URDF file describing the robot.
        ee_link_name (str): The end-effector link name used for IK.
        urdf (URDF): Loaded URDF model.
        pk_robot (pk.Robot): Pyroki robot model instance.
    """

    def __init__(self, robot_cfg):
        """Initialize the model with robot configuration.

        Args:
            robot_cfg: An instance of BaseRobotCfg or similar, which must contain:
                - urdf_path (str): Path to the robot's URDF file.
                - ee_body_name (str): Name of the end-effector link.
        """
        self.urdf_path = robot_cfg.urdf_path
        check_and_download_single(self.urdf_path)
        self.ee_link_name = getattr(robot_cfg, "ee_body_name", None)
        if self.ee_link_name is None:
            raise ValueError("robot_cfg must have 'ee_body_name' defined")

        # Load URDF model from file
        self.urdf = URDF.load(self.urdf_path, load_meshes=False)

        # Initialize Pyroki robot model from URDF
        self.pk_robot = pk.Robot.from_urdf(self.urdf)

    def solve_ik(self, pos_target: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
        """Solve inverse kinematics for a single target pose.

        Converts the input PyTorch tensors to JAX numpy arrays,
        calls Pyroki IK solver, and converts the solution back to a
        PyTorch tensor.

        Args:
            pos_target (torch.Tensor): Target position tensor (3D).
            quat_target (torch.Tensor): Target orientation quaternion tensor (wxyz).

        Returns:
            torch.Tensor: Joint angle solution tensor.
        """
        # Convert PyTorch tensors to JAX numpy arrays for Pyroki compatibility
        pos = jnp.array(pos_target.detach().cpu().numpy())
        quat = jnp.array(quat_target.detach().cpu().numpy())

        # Solve IK using Pyroki
        solution = pks.solve_ik(
            self.pk_robot,
            self.ee_link_name,
            target_wxyz=quat,
            target_position=pos,
        )

        # Append fixed joint values [0.04, 0.04] as required by your robot setup
        q_list = np.concatenate([solution, [0.04, 0.04]])

        # Convert numpy array back to PyTorch tensor, move to GPU if available
        q_tensor = torch.tensor(q_list, dtype=torch.float32)
        return q_tensor.cuda() if torch.cuda.is_available() else q_tensor
