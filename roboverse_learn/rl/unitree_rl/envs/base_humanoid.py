from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.humanoid_robot_util import contact_forces_tensor, gait_phase_tensor
from metasim.utils.state import TensorState
from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg
from roboverse_learn.rl.unitree_rl.envs.base_legged import LeggedRobot
from roboverse_learn.rl.unitree_rl.helper.utils import (
    get_body_reindexed_indices_from_substring,
    get_joint_reindexed_indices_from_substring,
)


class Humanoid(LeggedRobot):
    """
    Inherit from LeggedRobot to implement a humanoid robot environment.
    The main difference is the additional joints and rigid bodies specific to humanoid robots, e.g., knees, elbows, wrists, and torso.
    """

    cfg: BaseLeggedTaskCfg

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._parse_joint_indices(scenario.robots[0])  # new funcs for utilies

    # region: Parse configs & Get the necessary parametres
    def _parse_rigid_body_indices(self, robot):
        """
        Parse rigid body indices from robot cfg.
        """
        # parse for foot. termination_contact, penalised_contact
        super()._parse_rigid_body_indices(robot)
        knee_names = robot.knee_links
        elbow_names = robot.elbow_links
        wrist_names = robot.wrist_links
        torso_names = robot.torso_links

        # get sorted indices for specific body links
        self.knee_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, knee_names, device=self.device
        )
        self.elbow_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, elbow_names, device=self.device
        )
        self.wrist_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, wrist_names, device=self.device
        )
        self.torso_indices = get_body_reindexed_indices_from_substring(
            self.handler, robot.name, torso_names, device=self.device
        )

        # attach to cfg for reward computation.
        self.cfg.knee_indices = self.knee_indices
        self.cfg.elbow_indices = self.elbow_indices
        self.cfg.wrist_indices = self.wrist_indices
        self.cfg.torso_indices = self.torso_indices

    def _parse_joint_indices(self, robot):
        """
        Parse joint indices.
        """
        left_yaw_roll_names = robot.left_yaw_roll_joints
        right_yaw_roll_names = robot.right_yaw_roll_joints
        upper_body_names = robot.upper_body_joints
        self.cfg.left_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.handler, robot.name, left_yaw_roll_names, device=self.device
        )
        self.cfg.right_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.handler, robot.name, right_yaw_roll_names, device=self.device
        )
        self.cfg.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.handler, robot.name, upper_body_names, device=self.device
        )
        # keep the waist stable
        if hasattr(robot, "waist_joints"):
            self.cfg.waist_joint_indices = get_joint_reindexed_indices_from_substring(
                self.handler, robot.name, robot.waist_joints, device=self.device
            )

    # endregion

    # region: Parse states for reward computation
    def _parse_state_for_reward(self, envstate: TensorState):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """
        # self._parse_gait_phase(envstate)
        envstate.robots[self.robot.name].extra["gait_phase"] = self._get_gait_phase()
        self._parse_feet_clearance(envstate)
        super()._parse_state_for_reward(envstate)

    # def _parse_gait_phase(self, envstate: TensorState):
    #     # period = self.cfg.reward_cfg.feet_cycle_time
    #     # offset = 0.5
    #     # phase = (self.episode_length_buf * self.dt) % period / period
    #     # phase_left = phase
    #     # phase_right = (phase + offset) % 1
    #     # envstate.robots[self.robot.name].extra["leg_phase"] = torch.cat(
    #     #     [phase_left.unsqueeze(1), phase_right.unsqueeze(1)], dim=-1
    #     # )
    #     envstate.robots[self.robot.name].extra["gait_phase"] = self._get_gait_phase()

    # NOTE: A Rewritten Function
    def _parse_feet_air_time(self, envstate: TensorState):
        contact = contact_forces_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 1.0
        stance_mask = gait_phase_tensor(envstate, self.robot.name)
        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt * self.decimation
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        envstate.robots[self.robot.name].extra["feet_air_time"] = air_time
        # envstate.robots[self.robot.name].extra["req_airTime"] = rew_airTime

    def _parse_feet_clearance(self, envstate: TensorState):
        """Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.


        Directly calculates reward since no intermediate variables are reused for other reward.
        """
        contact = contact_forces_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 5.0
        feet_z = envstate.robots[self.robot.name].body_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = torch.logical_not(self._get_gait_phase())

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.reward_cfg.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        envstate.robots[self.robot.name].extra["feet_clearance"] = rew_pos

    # endregion

    # region: Utility functions
    def _get_gait_phase(self):
        """Add phase into states"""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, len(self.feet_indices)), dtype=torch.bool, device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < self.cfg.reward_cfg.feet_full_contact_time / 2.0] = True
        return stance_mask.to(torch.bool)

    def _get_phase(self):
        feet_cycle_time = self.cfg.reward_cfg.feet_cycle_time
        # ep
        phase = self._episode_steps * self.dt % feet_cycle_time / feet_cycle_time
        return phase

    # endregion
