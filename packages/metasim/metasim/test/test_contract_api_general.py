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
    assert project["name"] == "roboverse-metasim"  # PyPI name; `metasim` is an unrelated project there
    assert project["description"] == "MetaSim: A unified simulation framework for robotics"
    assert project["license"] == {"file": "LICENSE"}
    assert project["requires-python"] == ">=3.10,<3.13"

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
        "defusedxml",  # M4: hardened XML parser for untrusted MJCF/URDF
        "gymnasium",
        "huggingface-hub",
        "imageio[ffmpeg]",
        "loguru",
        "numpy",
        "numpy-quaternion",
        "opencv-python>=4.11,<4.12",
        "packaging>=23",
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
            # body_state mirrors ObjectState: ``None`` allowed for backends
            # that don't expose per-body data (currently pybullet).
            "body_state": "BodyStateTensor | None",
            "joint_pos": "JointStateTensor | None",
            "joint_vel": "JointStateTensor | None",
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

    world_state = torch.tensor([
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [2.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ])
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
def test_render_cfg_mode_includes_realtime_pathtracing():
    """RenderCfg.mode exposes the RTX Real-Time 2.0 value on its Literal surface.

    AST/source assertion because ``configclass`` does not enforce ``Literal`` at runtime, so
    constructing ``RenderCfg(mode=...)`` would not actually validate the accepted values.
    """
    module, source = _parse_module("metasim/scenario/render.py")
    mode_field = _get_annotated_field(_get_class(module, "RenderCfg"), "mode")
    annotation = _annotation_text(source, mode_field.annotation)
    for value in ("rasterization", "raytracing", "pathtracing", "realtime_pathtracing"):
        assert f'"{value}"' in annotation, annotation


@pytest.mark.general
def test_isaacsim_render_mode_to_rtx_rendermode_mapping():
    """_load_render_settings maps the public modes to correctly-cased /rtx/rendermode tokens."""
    _, source = _parse_module("metasim/sim/isaacsim/isaacsim.py")
    # RTX - Real-Time 2.0
    assert 'set_string("/rtx/rendermode", "RealTimePathTracing")' in source
    # RTX - Real-Time (legacy), correctly cased: regression guard against "RayTracedLighting"
    assert 'set_string("/rtx/rendermode", "RaytracedLighting")' in source
    assert "RayTracedLighting" not in source
    # offline path tracing left untouched
    assert 'set_string("/rtx/rendermode", "PathTracing")' in source


@pytest.mark.general
def test_isaacsim_realtime_pathtracing_fails_fast_when_unavailable():
    """The new mode reads back /rtx/rendermode and raises if the renderer did not accept it."""
    _, source = _parse_module("metasim/sim/isaacsim/isaacsim.py")
    assert 'applied = settings.get_as_string("/rtx/rendermode")' in source
    assert 'if applied != "RealTimePathTracing":' in source


@pytest.mark.general
def test_isaacsim_bvh_refresh_covers_realtime_pathtracing():
    """The stale-BVH-after-material-edit workaround also runs for RTX Real-Time 2.0."""
    _, source = _parse_module("metasim/sim/isaacsim/isaacsim.py")
    assert '("raytracing", "realtime_pathtracing")' in source


@pytest.mark.general
def test_isaacsim_boot_kit_args_helper_behavior():
    """``_kit_args_with_boot_render_mode`` registers RT-PT at boot and is idempotent.

    The helper is dependency-free by design, so it is extracted from the module AST and
    executed directly — the module itself cannot be imported without isaacsim installed.
    """
    module, _ = _parse_module("metasim/sim/isaacsim/isaacsim.py")
    wanted = {"_BOOT_KIT_ARGS_BY_RENDER_MODE", "_kit_args_with_boot_render_mode"}
    nodes = [
        n
        for n in module.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))
    ]
    assert len(nodes) == 2, "boot-flag mapping + helper must both exist at module level"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    extracted = ast.Module(body=[future, *nodes], type_ignores=[])
    ast.fix_missing_locations(extracted)
    ns: dict = {}
    exec(compile(extracted, "<isaacsim-extract>", "exec"), ns)
    fn = ns["_kit_args_with_boot_render_mode"]

    persistent = "--/persistent/rtx/modes/rt2/enabled=true"
    transient = "--/rtx-transient/rt2Enabled=true"
    assert fn("realtime_pathtracing", None) == f"{persistent} {transient}"
    assert fn("realtime_pathtracing", "") == f"{persistent} {transient}"
    assert fn("realtime_pathtracing", "--ext-folder=/x") == f"--ext-folder=/x {persistent} {transient}"
    # idempotent: present flags are not duplicated, missing ones are appended
    assert fn("realtime_pathtracing", f"{persistent} {transient}") == f"{persistent} {transient}"
    assert fn("realtime_pathtracing", persistent) == f"{persistent} {transient}"
    # runtime-switchable modes need no boot flags; existing kit_args pass through
    assert fn("raytracing", None) is None
    assert fn("pathtracing", "--ext-folder=/x") == "--ext-folder=/x"
    assert fn(None, None) is None


@pytest.mark.general
def test_isaacsim_realtime_pathtracing_registered_at_kit_boot():
    """``_init_scene`` injects the rt2 registration flags into AppLauncher kit_args.

    The RealTimePathTracing renderer only joins the render-mode list when
    ``/rtx-transient/rt2Enabled`` is on at Kit boot (derived from the persistent
    ``/persistent/rtx/modes/rt2/enabled`` preference); without that registration every
    ``/rtx/rendermode`` write — runtime, boot argv, or SimulationApp ``renderer``
    config — is silently refused (probed on Isaac Sim 5.0). The runtime readback in
    ``_load_render_settings`` remains as engagement verification.
    """
    module, source = _parse_module("metasim/sim/isaacsim/isaacsim.py")
    init_scene = _get_function(_get_class(module, "IsaacsimHandler"), "_init_scene")
    init_src = ast.get_source_segment(source, init_scene)
    assert init_src is not None
    assert "_kit_args_with_boot_render_mode(" in init_src
    assert "args.kit_args" in init_src
    # the injection must land in the namespace BEFORE AppLauncher(args) consumes it
    assert init_src.index("args.kit_args") < init_src.index("AppLauncher(args)")


@pytest.mark.general
def test_dr_manager_groups_realtime_pathtracing_with_pathtracing():
    """DR light-intensity ranges treat RTX Real-Time 2.0 like path tracing (brighter ranges)."""
    _, source = _parse_module("metasim/randomization/dr_manager.py")
    assert '("pathtracing", "realtime_pathtracing")' in source


@pytest.mark.general
def test_assert_close_accepts_tensor_expected_without_copy_warning():
    from metasim.test.test_utils import assert_close

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_close(torch.tensor([1.0]), torch.tensor([1.0]))

    warning_messages = [str(warning.message) for warning in caught]
    assert not any("To copy construct from a tensor" in message for message in warning_messages)
