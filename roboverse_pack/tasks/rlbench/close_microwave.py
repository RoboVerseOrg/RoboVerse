from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_microwave", "close_microwave", "franka.close_microwave")
class CloseMicrowaveTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="microwave_frame_resp",
                usd_path="roboverse_data/assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_microwave/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.open_microwave", "open_microwave", "franka.open_microwave")
class OpenMicrowaveTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="microwave_frame_resp",
                usd_path="roboverse_data/assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_microwave/v2/franka_v2.pkl.gz"
    # TODO: add checker
