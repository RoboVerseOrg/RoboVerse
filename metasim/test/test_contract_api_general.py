from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        "roboverse_pack/tasks/humanoid/base/base_legged_robot.py": {
            "LeggedRobotTask": {
                "step": "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]",
            }
        },
        "roboverse_pack/tasks/beyondmimic/metasim/envs/base_legged_robot.py": {
            "LeggedRobotTask": {
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
