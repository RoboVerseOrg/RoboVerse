"""Box-task replay env for the OpenArm Wuji bimanual robot."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

ROBOT_NAME = "openarm_wuji"
ASSET_ROOT = "roboverse_data/assets/box_task/local_pack_box"
TRAJ_PATH = "roboverse_data/trajs/box_task/task3_openarm_wuji_v2.pkl"


@register_task("box_task.replay", "box_task")
class BoxTaskReplayEnv(BaseTaskEnv):
    """Replay env for the box-packing demo trajectory.

    The scenario carries a fixed front table plus three rigid objects;
    the robot is openarm_wuji (bimanual). The trajectory records joint
    targets for every frame, so the replay loop just calls
    ``set_states`` per frame — no physics simulation needed.
    """

    traj_filepath = TRAJ_PATH

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="front_table",
                size=(0.60, 0.70, 0.04),
                mass=80.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.85, 0.78, 0.62),
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="cardboard_box",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=f"{ASSET_ROOT}/cardboard_box/cardboard_box.usd",
                mjcf_path=f"{ASSET_ROOT}/cardboard_box/cardboard_box.xml",
            ),
            RigidObjCfg(
                name="feast_soda_can",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=f"{ASSET_ROOT}/feast_soda_can/feast_soda_can.usd",
                mjcf_path=f"{ASSET_ROOT}/feast_soda_can/feast_soda_can.xml",
            ),
            RigidObjCfg(
                name="feast_scented_candle",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=f"{ASSET_ROOT}/feast_scented_candle/feast_scented_candle.usd",
                mjcf_path=f"{ASSET_ROOT}/feast_scented_candle/feast_scented_candle.xml",
            ),
        ],
        robots=[ROBOT_NAME],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    def _get_initial_states(self) -> list[dict]:
        return [
            {
                "objects": {
                    "front_table": {
                        "pos": torch.tensor([0.55, 0.0, 0.33], dtype=torch.float32),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    }
                },
                "robots": {
                    ROBOT_NAME: {
                        "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    }
                },
                "cameras": {},
                "extras": {},
            }
        ]
