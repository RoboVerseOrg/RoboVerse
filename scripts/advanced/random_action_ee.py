from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class Args:
    robot: str = "franka"
    num_envs: int = 1
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] = "isaaclab"
    decimation: int = 40


args = tyro.cli(Args)

#########################################
### Normal code
#########################################

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import quat_apply, quat_from_euler_xyz, quat_inv
from metasim.utils.setup_util import get_robot, get_sim_env_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def main():
    num_envs: int = args.num_envs
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(robots=[robot], cameras=[camera], sim=args.sim, decimation=args.decimation)

    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    try:
        from omni.isaac.core.objects import FixedSphere
        from omni.isaac.core.prims import XFormPrim
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedSphere
        from isaacsim.core.prims import SingleXFormPrim as XFormPrim

    ## Reset
    states, extras = env.reset()

    ## cuRobo controller
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)

    if args.sim == "isaaclab":
        FixedSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            scale=torch.tensor([0.05, 0.05, 0.05]),
            position=torch.tensor([0.0, 0.0, 0.0]),
            color=torch.tensor([1.0, 0.0, 0.0]),
        )

    step = 0
    q_min = torch.ones(len(robot.joint_limits.values()), device="cuda") * 100
    q_max = torch.ones(len(robot.joint_limits.values()), device="cuda") * -100
    while True:
        log.debug(f"Step {step}")

        # Generate random actions
        random_gripper_widths = torch.rand((num_envs, len(robot.gripper_open_q)))
        random_gripper_widths = torch.tensor(robot.gripper_open_q) + random_gripper_widths * (
            torch.tensor(robot.gripper_close_q) - torch.tensor(robot.gripper_open_q)
        )

        ee_rot_target = torch.rand((num_envs, 3), device="cuda") * torch.pi
        ee_quat_target = quat_from_euler_xyz(ee_rot_target[..., 0], ee_rot_target[..., 1], ee_rot_target[..., 2])

        # Compute targets
        reorder_idx = env.handler.get_joint_reindex(robot.name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        curr_robot_q = states.robots[robot.name].joint_pos[:, inverse_reorder_idx]
        robot_pos, robot_quat = states.robots[robot.name].root_state[0, :7].split([3, 4])

        if robot.name == "iiwa":
            ee_pos_target = torch.distributions.Uniform(-0.5, 0.5).sample((num_envs, 3)).cuda()
            ee_pos_target[:, 2] += 0.5
        elif robot.name == "franka" or robot.name == "kinova_gen3_robotiq_2f85":
            ee_pos_target = torch.distributions.Uniform(-0.5, 0.5).sample((num_envs, 3)).cuda()
            ee_pos_target[:, 2] += 0.5
        elif robot.name == "sawyer":
            ee_pos_target = torch.stack(
                [
                    torch.distributions.Uniform(-0.8, 0.8).sample((num_envs, 1)),
                    torch.distributions.Uniform(-0.8, 0.8).sample((num_envs, 1)),
                    torch.distributions.Uniform(0.2, 0.8).sample((num_envs, 1)),
                ],
                dim=-1,
            ).cuda()
        else:
            raise ValueError(f"Unsupported robot: {robot.name}")

        target_prim = XFormPrim("/World/envs/env_0/target", name="target")
        target_prim.set_world_pose(position=quat_apply(quat_inv(robot_quat), ee_pos_target) + robot_pos)

        # Solve IK
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
        result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

        # Compose robot command
        q = curr_robot_q.clone()
        ik_succ = result.success.squeeze(1)
        q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
        q[:, -ee_n_dof:] = random_gripper_widths

        actions = [
            {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
            for i_env in range(num_envs)
        ]
        q_min = torch.min(torch.stack([q_min, q[0]], -1), -1)[0]
        q_max = torch.max(torch.stack([q_max, q[0]], -1), -1)[0]

        log.info(f"q: {' '.join([f'{x:.2f}' for x in q[0].tolist()])}")
        log.info(f"q_min: {' '.join([f'{x:.2f}' for x in q_min.tolist()])}")
        log.info(f"q_max: {' '.join([f'{x:.2f}' for x in q_max.tolist()])}")

        env.step(actions)
        env.handler.refresh_render()
        step += 1

        states = env.handler.get_states()

    env.handler.close()


if __name__ == "__main__":
    main()
