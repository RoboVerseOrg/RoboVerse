from __future__ import annotations

import sys
from importlib.machinery import PathFinder
from pathlib import Path


METASIM_GITHUB_REF = "git+https://github.com/RoboVerseOrg/MetaSim.git@main"
METASIM_DEPENDENCY = f"metasim @ {METASIM_GITHUB_REF}"
METASIM_SIMULATOR_EXTRAS = [
    "genesis",
    "isaacgym",
    "isaacsim",
    "isaacsim211",
    "mjx",
    "mujoco",
    "newton",
    "pybullet",
    "robosplatter",
    "sapien2",
    "sapien3",
]


def _load_pyproject() -> dict:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib

    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_metasim_import_resolves_outside_roboverse_checkout():
    repo_root = Path(__file__).resolve().parents[1]
    local_spec = PathFinder.find_spec("metasim", [str(repo_root)])

    assert local_spec is None

    for module_name in list(sys.modules):
        if module_name == "metasim" or module_name.startswith("metasim."):
            del sys.modules[module_name]

    sys.path.insert(0, str(repo_root))
    try:
        import metasim
    finally:
        sys.path.remove(str(repo_root))

    metasim_path = Path(metasim.__file__).resolve()

    assert repo_root not in metasim_path.parents


def test_roboverse_pack_discovered_by_metasim():
    from metasim.utils.package_discovery import get_package_candidates

    assert "roboverse_pack.tasks" in get_package_candidates("tasks")
    assert "roboverse_pack.robots" in get_package_candidates("robots")


class _RecordingStateHandler:
    def __init__(self):
        self.calls = []

    def get_states(self, env_ids=None, mode="dict"):
        self.calls.append({"env_ids": env_ids, "mode": mode})
        return {"env_ids": env_ids, "mode": mode}


def test_humanoid_agent_task_get_states_matches_handler_contract():
    from roboverse_pack.tasks.humanoid.base.base_agent import AgentTask

    _assert_task_get_states_matches_handler_contract(AgentTask)


def test_beyondmimic_legged_task_get_states_matches_handler_contract():
    from roboverse_pack.tasks.beyondmimic.metasim.envs.base_legged_robot import LeggedRobotTask

    _assert_task_get_states_matches_handler_contract(LeggedRobotTask)


def _assert_task_get_states_matches_handler_contract(task_cls):
    task = object.__new__(task_cls)
    task.handler = _RecordingStateHandler()

    assert task.get_states() == {"env_ids": None, "mode": "dict"}
    assert task.get_states(env_ids=[1], mode="tensor") == {"env_ids": [1], "mode": "tensor"}

    assert task.handler.calls == [
        {"env_ids": None, "mode": "dict"},
        {"env_ids": [1], "mode": "tensor"},
    ]


def test_roboverse_runtime_dependencies_are_downstream_owned():
    pyproject = _load_pyproject()

    assert pyproject["project"]["dependencies"] == [
        METASIM_DEPENDENCY,
        "gymnasium",
        "loguru",
        "numpy",
        "pyyaml",
        "scipy",
        "torch",
    ]


def test_roboverse_optional_dependencies_forward_metasim_extras():
    optional = _load_pyproject()["project"]["optional-dependencies"]

    assert optional["dev"] == ["pytest", "pytest-cov", "ruff", f"metasim[dev] @ {METASIM_GITHUB_REF}"]
    assert optional["assets"] == ["huggingface-hub", "rootutils", "tyro"]
    assert optional["examples"] == [
        "imageio[ffmpeg]",
        "opencv-python",
        "pygame",
        "rich",
        "rootutils",
        "stable-baselines3",
        "torchvision",
        "tqdm",
        "tyro",
    ]
    assert optional["learn"] == [
        "diffusers",
        "einops",
        "h5py",
        "hydra-core",
        "imageio[ffmpeg]",
        "matplotlib",
        "numcodecs",
        "numpy-quaternion",
        "omegaconf",
        "opencv-python",
        "pandas",
        "pillow",
        "pymunk",
        "robomimic",
        "rsl-rl-lib",
        "seaborn",
        "stable-baselines3",
        "tensordict",
        "torchcfm",
        "torchvision",
        "tqdm",
        "wandb",
        "zarr",
    ]
    assert optional["sensors"] == ["trimesh"]
    assert optional["vla"] == [
        "lerobot",
        "tensorflow",
        "tensorflow-datasets",
        "tensorflow-hub",
        "transformers",
    ]

    for extra in METASIM_SIMULATOR_EXTRAS:
        assert optional[extra] == [f"metasim[{extra}] @ {METASIM_GITHUB_REF}"]
