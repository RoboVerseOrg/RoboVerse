from time import time

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.deploy.arm_hand_deployment.consts import CONFIG_PATH
from scripts.deploy.arm_hand_deployment.franka.communication.client import FrankaClient
from scripts.deploy.arm_hand_deployment.utils.client_context import robot_client_context


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()  # Record start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result

    return wrapper


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    port: int = int(cfg.rpc_port)
    server_ip: str = cfg.server_ip

    logger.info(f"Connecting to server {server_ip}:{port}")

    initial_arm_joint_positions = [
        -0.14377304911613464,
        -0.41615936160087585,
        0.1558820903301239,
        -2.799217700958252,
        0.09156578779220581,
        2.3855371475219727,
        1.5030053853988647,
    ]

    actions = np.load("scripts/action/actions.npy")

    with robot_client_context(server_ip, port, FrankaClient) as client:
        client: FrankaClient

        assert client.MoveToJointPositions(initial_arm_joint_positions)

        qpos = client.GetJointPositions()[:7]

        # gripper command, greater than 0 is close , 0  to -1 is open
        # gripper_action=-0.5
        # client.SetGripperAction(gripper_action)
        # sleep_time = 1
        # sleep(sleep_time)

        print(f"Initial qpos:\n{qpos}")

        @timing_decorator
        def execute(action):
            assert client.ControlJointPositions(action=action)

        # warmup
        execute(qpos)

        print("-" * 80)

        for action in actions:
            action = action[:7].tolist()
            print(action)
            execute(action)

        # end_joint_positions = actions[-1]
        # end_joint_positions = end_joint_positions[:7].tolist()
        # assert client.MoveToJointPositions(end_joint_positions)


if __name__ == "__main__":
    main()
