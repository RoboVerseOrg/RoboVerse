from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _parse_module(path: str) -> tuple[ast.Module, str]:
    source_path = REPO_ROOT / path
    source = source_path.read_text(encoding="utf-8")
    return ast.parse(source), source


def _get_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found")


def _get_function(class_node: ast.ClassDef, func_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"Function {func_name!r} not found in class {class_node.name!r}")


def _get_annotated_field(class_node: ast.ClassDef, field_name: str) -> ast.AnnAssign:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field_name:
            return node
    raise AssertionError(f"Field {field_name!r} not found in class {class_node.name!r}")


def _annotation_text(source: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    text = ast.get_source_segment(source, node)
    if text is None:
        raise AssertionError("Unable to recover annotation source text")
    return text


@pytest.mark.general
def test_metasim_production_code_does_not_import_monorepo_siblings():
    forbidden_roots = {
        "dashboard",
        "get_started",
        "roboverse_learn",
        "roboverse_pack",
        "scripts",
    }
    offenders: list[str] = []

    for source_path in (REPO_ROOT / "metasim").rglob("*.py"):
        relative_path = source_path.relative_to(REPO_ROOT)
        if "__pycache__" in relative_path.parts or relative_path.parts[:2] == ("metasim", "test"):
            continue
        module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in forbidden_roots:
                        offenders.append(f"{relative_path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".", 1)[0]
                if root in forbidden_roots:
                    offenders.append(f"{relative_path}:{node.lineno}: from {node.module} import ...")

    assert offenders == []


@pytest.mark.general
def test_pyproject_builds_metasim_distribution_only():
    pyproject = _load_pyproject()

    assert pyproject["build-system"]["requires"] == ["setuptools>=61.0", "wheel"]
    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"

    project = pyproject["project"]
    assert project["name"] == "metasim"
    assert project["description"] == "MetaSim: A unified simulation framework for robotics"
    assert project["license"] == {"file": "LICENSE"}
    assert project["requires-python"] == ">=3.8"

    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_find["include"] == ["metasim", "metasim.*"]
    assert "where" not in package_find

    assert pyproject["tool"]["setuptools"]["package-data"]["metasim"] == ["data/**/*"]


@pytest.mark.general
def test_pyproject_runtime_dependencies_are_not_test_or_roboverse_content_dependencies():
    pyproject = _load_pyproject()
    runtime_deps = set(pyproject["project"]["dependencies"])

    assert "pytest" not in runtime_deps
    assert not any(dep.startswith("roboverse") for dep in runtime_deps)

    dev_deps = set(pyproject["project"]["optional-dependencies"]["dev"])
    assert {"pytest", "pytest-cov", "ruff"}.issubset(dev_deps)


@pytest.mark.general
def test_pyproject_runtime_dependencies_are_metasim_owned():
    pyproject = _load_pyproject()

    assert pyproject["project"]["dependencies"] == [
        "gymnasium",
        "huggingface-hub",
        "imageio[ffmpeg]",
        "loguru",
        "numpy",
        "numpy-quaternion",
        "opencv-python>=4.11,<4.12",
        "pillow",
        "portalocker",
        "pyyaml",
        "scipy",
        "tomli; python_version < '3.11'",
        "torch",
        "torchvision",
    ]


@pytest.mark.general
def test_pyproject_optional_dependencies_separate_examples_and_visualization():
    optional = _load_pyproject()["project"]["optional-dependencies"]

    assert optional["dev"] == ["pytest", "pytest-cov", "ruff"]
    assert optional["examples"] == ["rich", "rootutils", "tyro"]
    assert optional["teleop"] == ["pygame"]
    assert optional["visualization"] == ["rerun-sdk", "trimesh", "viser", "yourdfpy"]


@pytest.mark.general
def test_metasim_task_package_does_not_ship_ad_hoc_test_module():
    assert not (REPO_ROOT / "metasim/task/test.py").exists()


@pytest.mark.general
def test_primitive_frame_is_a_builtin_asset_not_an_external_download():
    from metasim.scenario.objects import PrimitiveFrameCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.utils.hf_util import FileDownloader

    scenario = ScenarioCfg(
        simulator="isaacsim",
        objects=[PrimitiveFrameCfg(name="frame")],
    )

    downloader = FileDownloader(scenario)

    assert downloader.files_to_download == []


@pytest.mark.general
def test_primitive_frame_builtin_usd_is_packaged_with_metasim():
    import pkgutil

    frame_usd = pkgutil.get_data("metasim", "data/quick_start/assets/COMMON/frame/usd/frame.usd")

    assert frame_usd is not None


@pytest.mark.general
def test_state_dataclasses_use_explicit_shape_aliases():
    module, source = _parse_module("metasim/types.py")

    expected_annotations = {
        "ObjectState": {
            "root_state": "RootStateTensor",
            "body_state": "BodyStateTensor | None",
            "joint_pos": "JointStateTensor | None",
            "joint_vel": "JointStateTensor | None",
        },
        "RobotState": {
            "root_state": "RootStateTensor",
            "body_state": "BodyStateTensor",
            "joint_pos": "JointStateTensor",
            "joint_vel": "JointStateTensor",
            "joint_pos_target": "JointStateTensor | None",
            "joint_vel_target": "JointStateTensor | None",
            "joint_effort_target": "JointStateTensor | None",
        },
        "CameraState": {
            "rgb": "CameraRgbTensor | None",
            "depth": "CameraDepthTensor | None",
            "instance_id_seg": "CameraSegmentationTensor | None",
            "instance_seg": "CameraSegmentationTensor | None",
            "pos": "CameraPosTensor | None",
            "quat_world": "CameraQuatTensor | None",
            "intrinsics": "CameraIntrinsicsTensor | None",
        },
    }

    for class_name, fields in expected_annotations.items():
        class_node = _get_class(module, class_name)
        for field_name, expected_annotation in fields.items():
            field_node = _get_annotated_field(class_node, field_name)
            actual_annotation = _annotation_text(source, field_node.annotation)
            assert actual_annotation == expected_annotation, (
                f"{class_name}.{field_name} should use {expected_annotation!r}, got {actual_annotation!r}"
            )


@pytest.mark.general
def test_task_info_alias_is_used_in_public_task_apis():
    expectations = {
        "metasim/task/base.py": {
            "BaseTaskEnv": {
                "step": "tuple[Obs, Reward, Success, TimeOut, Info | None]",
                "reset": "tuple[Obs, Info | None]",
            }
        },
        "metasim/task/rl_task.py": {
            "RLTaskEnv": {
                "reset": "tuple[torch.Tensor, Info]",
                "step": "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]",
            }
        },
    }

    for path, class_expectations in expectations.items():
        module, source = _parse_module(path)
        for class_name, method_expectations in class_expectations.items():
            class_node = _get_class(module, class_name)
            for method_name, expected_annotation in method_expectations.items():
                func_node = _get_function(class_node, method_name)
                actual_annotation = _annotation_text(source, func_node.returns)
                assert actual_annotation == expected_annotation, (
                    f"{path}:{class_name}.{method_name} should return {expected_annotation!r}, got {actual_annotation!r}"
                )


@pytest.mark.general
def test_isaacsim_world_state_to_env_local_does_not_mutate_source():
    from metasim.sim.isaacsim.isaacsim import _world_state_to_env_local

    world_state = torch.tensor(
        [
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [2.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ]
    )
    original_world_state = world_state.clone()
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    local_state = _world_state_to_env_local(world_state, env_origins)

    assert torch.equal(world_state, original_world_state)
    expected_local_pos = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    assert torch.allclose(local_state[:, :3], expected_local_pos)
    assert local_state.data_ptr() != world_state.data_ptr()

    world_body_state = world_state[:, None, :].repeat(1, 2, 1)
    original_world_body_state = world_body_state.clone()

    local_body_state = _world_state_to_env_local(world_body_state, env_origins)

    assert torch.equal(world_body_state, original_world_body_state)
    assert torch.allclose(
        local_body_state[:, :, :3],
        expected_local_pos[:, None, :].repeat(1, 2, 1),
    )
    assert local_body_state.data_ptr() != world_body_state.data_ptr()


@pytest.mark.general
def test_assert_close_accepts_tensor_expected_without_copy_warning():
    from metasim.test.test_utils import assert_close

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_close(torch.tensor([1.0]), torch.tensor([1.0]))

    warning_messages = [str(warning.message) for warning in caught]
    assert not any("To copy construct from a tensor" in message for message in warning_messages)
