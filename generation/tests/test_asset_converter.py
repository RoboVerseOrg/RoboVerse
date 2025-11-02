import os
import sys

import pytest
from huggingface_hub import snapshot_download

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, "../.."))
from generation.asset_converter import AssetConverterFactory, AssetType


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Download the EmbodiedGenData dataset from Hugging Face."""
    data_dir = tmp_path_factory.mktemp("EmbodiedGenData")
    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns="demo_assets/remote_control/*",
        local_dir_use_symlinks=False,
    )
    return data_dir


def test_MeshtoMJCFConverter(data_dir):
    """Test MeshtoMJCFConverter with a sample URDF file."""
    urdf_path = data_dir / "demo_assets/remote_control/result/remote_control.urdf"
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"

    output_file = data_dir / "demo_assets/remote_control/mjcf/remote_control.mjcf"
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.MJCF,
        source_type=AssetType.MESH,
    )

    with asset_converter:
        asset_converter.convert(str(urdf_path), str(output_file))

    assert output_file.exists(), f"Output not generated: {output_file}"
    assert output_file.stat().st_size > 0


def test_MeshtoUSDConverter(data_dir):
    """Test MeshtoUSDConverter with a sample URDF file."""
    urdf_path = data_dir / "demo_assets/remote_control/result/remote_control.urdf"
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"

    output_file = data_dir / "demo_assets/remote_control/usd/remote_control.usd"
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.USD,
        source_type=AssetType.MESH,
    )

    with asset_converter:
        asset_converter.convert(str(urdf_path), str(output_file))

    assert output_file.exists(), f"Output not generated: {output_file}"
    assert output_file.stat().st_size > 0
