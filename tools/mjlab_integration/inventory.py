"""Static inventory of mjlab tasks and the raw MJCF assets that back them.

We hold the catalog as plain data so it can be consumed by both the diff
harness and the MetaSim task wrappers in `roboverse_pack.tasks.mjlab`.

The `mjlab_repo_path()` helper locates the mjlab source checkout. We support
either a sibling clone at `~/projects/mjlab` (default for this machine) or an
override via the `MJLAB_REPO` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def mjlab_repo_path() -> Path:
    env = os.environ.get("MJLAB_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MJLAB_REPO points at missing path: {p}")
        return p
    candidates = [
        Path.home() / "projects" / "mjlab",
        Path("/home/ghr/projects/mjlab"),
    ]
    for c in candidates:
        if (c / "src" / "mjlab").exists():
            return c
    raise FileNotFoundError("Could not locate mjlab. Clone https://github.com/mujocolab/mjlab or set MJLAB_REPO.")


def mjlab_src() -> Path:
    return mjlab_repo_path() / "src" / "mjlab"


@dataclass
class MjlabAsset:
    """A standalone MJCF model shipped by mjlab."""

    name: str  # short id, used in report tables
    relpath: str  # path under mjlab/src/mjlab
    family: str  # "task" | "robot" | "scene"
    description: str = ""

    @property
    def xml_path(self) -> Path:
        return mjlab_src() / self.relpath


@dataclass
class MjlabTask:
    """A registered mjlab task (subset of `Mjlab-*` ids)."""

    task_id: str  # mjlab task id, e.g. "Mjlab-Cartpole-Balance"
    metasim_name: str  # mirrored MetaSim task id, e.g. "mjlab.cartpole_balance"
    asset: MjlabAsset
    decimation: int
    episode_length_s: float
    control_mode: str  # "effort" | "joint_pos"
    # action_scale is informational; the diff harness writes ctrl directly.
    action_scale: float = 1.0
    description: str = ""
    # init_joint_pos lets the harness reproduce mjlab reset state for diffing.
    init_joint_pos: dict[str, float] = field(default_factory=dict)
    init_joint_vel: dict[str, float] = field(default_factory=dict)


CARTPOLE_ASSET = MjlabAsset(
    name="cartpole",
    relpath="tasks/cartpole/cartpole.xml",
    family="task",
    description="dm_control-style cartpole with a slide+hinge, 1 motor.",
)

G1_ASSET = MjlabAsset(
    name="unitree_g1",
    relpath="asset_zoo/robots/unitree_g1/xmls/g1.xml",
    family="robot",
    description="Unitree G1 humanoid (29 DoF) with collision capsules.",
)

GO1_ASSET = MjlabAsset(
    name="unitree_go1",
    relpath="asset_zoo/robots/unitree_go1/xmls/go1.xml",
    family="robot",
    description="Unitree Go1 quadruped (12 DoF).",
)

YAM_ASSET = MjlabAsset(
    name="i2rt_yam",
    relpath="asset_zoo/robots/i2rt_yam/xmls/yam.xml",
    family="robot",
    description="i2rt YAM 6-DoF arm with parallel-jaw gripper.",
)

ALL_ASSETS = [CARTPOLE_ASSET, G1_ASSET, GO1_ASSET, YAM_ASSET]


ALL_TASKS = [
    MjlabTask(
        task_id="Mjlab-Cartpole-Balance",
        metasim_name="mjlab.cartpole_balance",
        asset=CARTPOLE_ASSET,
        decimation=5,
        episode_length_s=50.0,
        control_mode="effort",
        action_scale=1.0,
        description="Balance an upright cartpole. Hinge starts near 0.",
        init_joint_pos={"slider": 0.0, "hinge_1": 0.0},
    ),
    MjlabTask(
        task_id="Mjlab-Cartpole-Swingup",
        metasim_name="mjlab.cartpole_swingup",
        asset=CARTPOLE_ASSET,
        decimation=5,
        episode_length_s=50.0,
        control_mode="effort",
        action_scale=1.0,
        description="Swing the pole up from hanging-down (hinge_1=pi).",
        init_joint_pos={"slider": 0.0, "hinge_1": 3.141592653589793},
    ),
    # Locomotion / manipulation tasks below share assets; ctrl semantics depend
    # on mjlab actuator configuration that the comparison harness reapplies.
    MjlabTask(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        metasim_name="mjlab.velocity_flat_go1",
        asset=GO1_ASSET,
        decimation=4,
        episode_length_s=20.0,
        control_mode="joint_pos",
        action_scale=0.25,
        description="Quadruped velocity tracking on flat terrain (Go1).",
    ),
    MjlabTask(
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        metasim_name="mjlab.velocity_flat_g1",
        asset=G1_ASSET,
        decimation=4,
        episode_length_s=20.0,
        control_mode="joint_pos",
        action_scale=0.5,
        description="Humanoid velocity tracking on flat terrain (G1).",
    ),
    MjlabTask(
        task_id="Mjlab-Lift-Cube-Yam",
        metasim_name="mjlab.lift_cube_yam",
        asset=YAM_ASSET,
        decimation=4,
        episode_length_s=5.0,
        control_mode="joint_pos",
        action_scale=0.5,
        description="YAM arm lifts a cube to a target height.",
    ),
]


def task_by_metasim_name(name: str) -> MjlabTask:
    for t in ALL_TASKS:
        if t.metasim_name == name:
            return t
    raise KeyError(name)
