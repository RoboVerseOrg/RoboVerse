import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from deploy.arm_hand_deployment.consts import CONFIG_PATH
from deploy.arm_hand_deployment.franka.communication.client import FrankaClient
from deploy.arm_hand_deployment.utils.client_context import robot_client_context


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    port: int = int(cfg.rpc_port)
    server_ip: str = cfg.server_ip

    logger.info(f"Connecting to server {server_ip}:{port}")

    with robot_client_context(server_ip, port, FrankaClient) as client:
        client: FrankaClient

        # 7 joint positions for the arm + 1 for the gripper (width of the gripper)
        positions = client.GetJointPositions()

        # xyz+rpy
        ee_pose = client.GetEndEffectorPose()
        print(f"Joint positions: {positions}")
        print(f"End effector pose: {ee_pose}")

        target_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        # 7 joint positions for the arm
        assert client.MoveToJointPositions(target_joint_positions)

        positions = client.GetJointPositions()
        ee_pose = client.GetEndEffectorPose()
        print(f"Joint positions: {positions}")
        print(f"End effector pose: {ee_pose}")

        # target_end_effector_pose = [0.5, 0.1, 0.3,  np.pi, 0, 0]

        # # xyz+rpy
        # assert client.MoveToEndEffectorPose(target_end_effector_pose)

        # positions = client.GetJointPositions()
        # ee_pose = client.GetEndEffectorPose()
        # print(f"Joint positions: {positions}")
        # print(f"End effector pose: {ee_pose}")

        # # See /home/abrar/hsc/deoxys_control/deoxys/deoxys/franka_interface/franka_interface.py
        # # action<0 open the gripper target width of the gripper is -action*0.08
        # # action>0 close the gripper

        # # We have to open the gripper first so that deoxys can remember?
        # # Honestly I am not very familiar with deoxys's gripper control.
        # client.SetGripperAction(-1)

        # sleep(1)

        # client.SetGripperAction(1)

        # sleep(3)

        # # open
        # client.SetGripperAction(-1)

        # sleep(3)


if __name__ == "__main__":
    main()
