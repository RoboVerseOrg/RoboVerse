"""Stage 3: Track task for trajectory tracking with bowl object (ground layout).

This task:
- Targets the bowl as name="object" (required by PickPlaceBase)
- Uses a hardcoded initial state
- Uses hardcoded 5 trajectory markers (required by PickPlaceBase)
- Forces object_grasped=True so tracking rewards are active without requiring a grasp state file
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from roboverse_pack.tasks.pick_place.base import PickPlaceBase


def _interpolate_waypoints(waypoints: torch.Tensor, steps_per_segment: list[int]) -> torch.Tensor:
    """Linear interpolation: include the first waypoint, then for each segment add N points (excluding segment start).

    If waypoints has shape (K, 3) and steps_per_segment has length K-1,
    output shape will be (1 + sum(steps_per_segment), 3).
    """
    assert waypoints.ndim == 2 and waypoints.shape[1] == 3
    assert len(steps_per_segment) == waypoints.shape[0] - 1

    out = [waypoints[0]]
    for i, n in enumerate(steps_per_segment):
        start = waypoints[i]
        end = waypoints[i + 1]
        n = int(n)
        if n <= 0:
            continue
        # t = 1..n (exclude start, include end when t==n)
        for t in range(1, n + 1):
            alpha = float(t) / float(n)
            out.append(start + (end - start) * alpha)
    return torch.stack(out, dim=0)


@register_task("pick_place.track_bowl", "pick_place_track_bowl")
class PickPlaceTrackBowl(PickPlaceBase):
    """Trajectory tracking task for bowl (ground layout)."""

    # Trajectory interpolation (object_init -> marker_0 -> ... -> marker_4)
    INTERP_SEGMENT_STEPS = [40, 20, 20, 20, 20]  # total 120 steps, 121 waypoints with start
    REACH_THRESHOLD = 0.10
    TERMINATION_DISTANCE = 0.3
    FORESIGHT_STEPS = 0

    # Tracking reward (match HandTraj-style: only position + rotation tracking)
    K_POS = 10.0
    K_ROT = 5.0
    W_POS = 0.5
    W_ROT = 0.3

    scenario = ScenarioCfg(
        objects=[
            # Target: bowl
            RigidObjCfg(
                name="object",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/usd/0f296af3df66565c9e1a7c2bc7b35d72.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/0f296af3df66565c9e1a7c2bc7b35d72.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/mjcf/0f296af3df66565c9e1a7c2bc7b35d72.xml",
            ),
            # Context objects
            RigidObjCfg(
                name="basket",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/usd/663158968e3f5900af1f6e7cecef24c7.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/663158968e3f5900af1f6e7cecef24c7.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/basket/663158968e3f5900af1f6e7cecef24c7/mjcf/663158968e3f5900af1f6e7cecef24c7.xml",
            ),
            RigidObjCfg(
                name="cutting_tools",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/usd/c5810e7c2c785fe3940372b205090bad.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/c5810e7c2c785fe3940372b205090bad.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/cutting_tools/c5810e7c2c785fe3940372b205090bad/mjcf/c5810e7c2c785fe3940372b205090bad.xml",
            ),
            RigidObjCfg(
                name="lighting_fixtures",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/usd/03f09dca16db5598a67f0715cf3fb157.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/03f09dca16db5598a67f0715cf3fb157.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/lighting_fixtures/03f09dca16db5598a67f0715cf3fb157/mjcf/03f09dca16db5598a67f0715cf3fb157.xml",
            ),
            RigidObjCfg(
                name="screwdriver",
                scale=(1.5, 1.5, 1.5),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/usd/ae51f060e3455e9f84a4fec81cc9284b.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/ae51f060e3455e9f84a4fec81cc9284b.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/screwdriver/ae51f060e3455e9f84a4fec81cc9284b/mjcf/ae51f060e3455e9f84a4fec81cc9284b.xml",
            ),
            RigidObjCfg(
                name="spoon",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/usd/2f1c3077a8d954e58fc0bf75cf35e849.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/2f1c3077a8d954e58fc0bf75cf35e849.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/assets/spoon/2f1c3077a8d954e58fc0bf75cf35e849/mjcf/2f1c3077a8d954e58fc0bf75cf35e849.xml",
            ),
            # Markers
            *[
                RigidObjCfg(
                    name=f"traj_marker_{i}",
                    urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                    mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                    usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                    scale=0.2,
                    physics=PhysicStateType.XFORM,
                    enabled_gravity=False,
                    collision_enabled=False,
                    fix_base_link=True,
                )
                for i in range(5)
            ],
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    max_episode_steps = 200

    def __init__(self, scenario, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.object_grasped = None
        super().__init__(scenario, device)

        # Build dense interpolated trajectory from initial object pose and marker poses.
        init0 = self._get_initial_states()[0]
        obj0_pos = init0["objects"]["object"]["pos"].to(self.device).float()
        obj0_rot = init0["objects"]["object"]["rot"].to(self.device).float()
        base_markers = [init0["objects"][f"traj_marker_{i}"]["pos"].to(self.device).float() for i in range(5)]
        all_waypoints = torch.stack([obj0_pos] + base_markers, dim=0)  # (6, 3)
        dense = _interpolate_waypoints(all_waypoints, self.INTERP_SEGMENT_STEPS).to(self.device).float()

        self.waypoint_positions = dense
        self.num_waypoints = int(dense.shape[0])
        self.target_orientation = obj0_rot  # fixed target orientation (HandTraj-style)

        # Track-only reward: position + rotation
        self.reward_functions = [self._reward_pos_rot_tracking]
        self.reward_weights = [1.0]

    def _prepare_states(self, states, env_ids):
        """Disable randomization for track task."""
        return states

    def _get_initial_states(self) -> list[dict] | None:
        # Hardcoded initial state (objects + robot + traj_marker_0..4)
        initial_states = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([1.060000, -0.380000, 0.130000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, -0.000000]),
                    },
                    "basket": {
                        "pos": torch.tensor([0.550000, -0.470000, 0.200000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "cutting_tools": {
                        "pos": torch.tensor([1.140000, -0.180000, 0.040000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "lighting_fixtures": {
                        "pos": torch.tensor([0.970000, 0.070000, 0.210000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "screwdriver": {
                        "pos": torch.tensor([1.220000, -0.500000, 0.100000]),
                        "rot": torch.tensor([0.947354, 0.023689, 0.319209, 0.007982]),
                    },
                    "spoon": {
                        "pos": torch.tensor([1.390000, -0.280000, 0.020000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_0": {
                        "pos": torch.tensor([0.990000, -0.380000, 0.230000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.930000, -0.380000, 0.322500]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.790000, -0.380000, 0.385000]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.690000, -0.380000, 0.377500]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.580000, -0.380000, 0.362500]),
                        "rot": torch.tensor([0.999687, -0.024997, 0.000000, 0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.910000, -0.790000, 0.030000]),
                        "rot": torch.tensor([-0.666275, -0.000000, 0.000000, -0.745703]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.040000,
                            "panda_finger_joint2": 0.040000,
                            "panda_joint1": 0.000000,
                            "panda_joint2": -0.785398,
                            "panda_joint3": 0.000000,
                            "panda_joint4": -2.356194,
                            "panda_joint5": 0.000000,
                            "panda_joint6": 1.570796,
                            "panda_joint7": 0.785398,
                        },
                    }
                },
            }
            for _ in range(self.num_envs)
        ]

        return initial_states

    def _get_waypoint_indices(self) -> torch.Tensor:
        """Waypoint index = current episode step (optionally with foresight)."""
        # RLTaskEnv.step increments _episode_steps BEFORE computing reward.
        step0 = torch.clamp(self._episode_steps - 1, min=0)
        idx = step0 + int(self.FORESIGHT_STEPS)
        return idx.clamp(0, self.num_waypoints - 1).long()

    def _reward_pos_rot_tracking(self, env_states) -> torch.Tensor:
        """Only position + rotation tracking reward (HandTraj-style)."""
        obj_pos = env_states.objects["object"].root_state[:, 0:3]
        obj_quat = env_states.objects["object"].root_state[:, 3:7]

        idx = self._get_waypoint_indices()
        target_pos = self.waypoint_positions[idx]

        pos_error = torch.norm(obj_pos - target_pos, dim=-1)
        pos_reward = torch.exp(-float(self.K_POS) * pos_error)

        # Fixed target orientation (from initial object pose)
        target_quat = self.target_orientation.unsqueeze(0).expand_as(obj_quat)
        # Angle between quaternions: angle = 2*acos(|dot(q1,q2)|)
        dot = (target_quat * obj_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        rot_reward = torch.exp(-float(self.K_ROT) * angle)

        return float(self.W_POS) * pos_reward + float(self.W_ROT) * rot_reward

    def _terminated(self, env_states) -> torch.Tensor:
        """Terminate if object deviates too far from current target waypoint."""
        obj_pos = env_states.objects["object"].root_state[:, 0:3]
        idx = self._get_waypoint_indices()
        target_pos = self.waypoint_positions[idx]
        dist = torch.norm(obj_pos - target_pos, dim=-1)
        return dist > float(self.TERMINATION_DISTANCE)
