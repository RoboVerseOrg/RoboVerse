from __future__ import annotations

import numpy as np
import torch
from loguru import logger as log
from pyrep import PyRep
from pyrep.objects.object import Object
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from rlbench.backend.robot import Robot

from metasim.sim import BaseSimHandler
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, TimeOut
from metasim.utils import to_snake_case

# TODO: try best to be independent from RLBench

## Joint names correspondence dict
jd = {
    "franka": {
        "Panda_joint1": "panda_joint1",
        "Panda_joint2": "panda_joint2",
        "Panda_joint3": "panda_joint3",
        "Panda_joint4": "panda_joint4",
        "Panda_joint5": "panda_joint5",
        "Panda_joint6": "panda_joint6",
        "Panda_joint7": "panda_joint7",
        "Panda_gripper_joint1": "panda_finger_joint1",
        "Panda_gripper_joint2": "panda_finger_joint2",
    }
}


class PyrepHandler(BaseSimHandler):
    def launch(self) -> None:
        self.sim = PyRep()
        self.sim.launch("third_party/rlbench/rlbench/task_design.ttt", headless=False)  # TODO: set headless from cfg
        # self.sim.set_simulation_timestep(1 / 60)  # Control frequency 60Hz, TODO: set control frequency from cfg
        task_name = to_snake_case(self.task.__class__.__name__.replace("Cfg", ""))
        base_object = self.sim.import_model(f"third_party/rlbench/rlbench/task_ttms/{task_name}.ttm")

        self.arm, self.gripper = Panda(), PandaGripper()
        self.robot_inst = Robot(self.arm, self.gripper)
        self.sim.start()
        self.sim.step()
        # TODO: initialize rlbench scene and task

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        action = action[0]["dof_pos_target"]
        arm_action = [action[jd[self.robot.name][joint.get_name()]] for joint in self.arm.joints]
        gripper_action = [action[jd[self.robot.name][joint.get_name()]] for joint in self.gripper.joints]
        self.arm.set_joint_target_positions(arm_action)
        self.gripper.set_joint_target_positions(gripper_action)
        self.sim.step()
        self.arm.set_joint_target_positions(self.arm.get_joint_positions())
        self.gripper.set_joint_target_positions(self.gripper.get_joint_positions())
        return None, None, torch.tensor([False]), torch.tensor([False]), None

    def reset(self) -> tuple[Obs, Extra]:
        return None, None

    def close(self) -> None:
        self.sim.stop()
        self.sim.shutdown()

    ############################################################
    ## Set states
    ############################################################
    def set_states(self, states: list[EnvState]) -> None:
        assert len(states) == 1  # PyRep only supports one env
        state = states[0]

        ## Set robot
        joint_pos = np.zeros(len(self.arm.joints))
        for i, joint_inst in enumerate(self.arm.joints):
            joint_name = joint_inst.get_name()
            joint_pos[i] = state[self.robot.name]["dof_pos"][joint_name]
        self.arm.set_joint_positions(joint_pos[:7])
        self.gripper.set_joint_positions(joint_pos[7:])
        # TODO: set robot pose #!

        ## Set objects
        for obj in self.objects:
            obj_inst = Object.get_object(obj.name)
            obj_inst.set_position(state[obj.name]["pos"].numpy())
            obj_inst.set_orientation(state[obj.name]["rot"].numpy())

            log.debug("Set object", obj.name, "to", state[obj.name]["pos"], state[obj.name]["rot"])

    ############################################################
    ## Get states
    ############################################################
    def _get_states(self) -> list[EnvState]:
        pass
