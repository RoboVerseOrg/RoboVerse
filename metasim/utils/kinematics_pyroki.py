import pyroki as pk
import third_party.pyroki.examples.pyroki_snippets as pks
import torch
from yourdfpy import URDF

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
        """Solve IK for target pose.

        Args:
            pos_target: Target position (3D)
            quat_target: Target quaternion (wxyz)

        Returns:
            Joint angle solution
        """
        import jax.numpy as jnp

        # Try CUDA first, fallback to CPU if JAX doesn't support CUDA
        try:
            # Convert PyTorch to JAX via DLPack (try CUDA)
            pos = jnp.from_dlpack(pos_target)
            quat = jnp.from_dlpack(quat_target)
        except RuntimeError as e:
            # JAX doesn't support CUDA, use CPU
            pos = jnp.from_dlpack(pos_target.cpu())
            quat = jnp.from_dlpack(quat_target.cpu())
        # Solve IK
        solution = pks.solve_ik(
            self.pk_robot,
            self.ee_link_name,
            target_wxyz=quat,
            target_position=pos,
        )

        # Convert JAX to PyTorch via DLPack
        q_tensor = torch.from_dlpack(solution)

        # Move to same device as input if needed
        if pos_target.is_cuda and q_tensor.device.type == "cpu":
            q_tensor = q_tensor.cuda()

        return q_tensor
