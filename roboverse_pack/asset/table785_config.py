"""Table configurations for tables with height 0.750 from embodiedgen_modified/table750.

This file contains pre-configured RigidObjCfg for all tables in the table750 directory.
All tables are configured with fix_base_link=True and physics=PhysicStateType.RIGIDBODY.
"""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg

# Base directory for table750 assets
TABLE750_BASE_DIR = "roboverse_data/assets/EmbodiedGenData/dataset/basic_furniture/table"

# Table configurations
TABLE_1 = RigidObjCfg(
    name="table_1",
    scale=(1.2, 1.5, 1.0),
    physics=PhysicStateType.RIGIDBODY,
    usd_path=f"{TABLE750_BASE_DIR}/126f60baf12759ea957fb6c38ba7458d/usd/126f60baf12759ea957fb6c38ba7458d.usd",
    urdf_path=f"{TABLE750_BASE_DIR}/126f60baf12759ea957fb6c38ba7458d/126f60baf12759ea957fb6c38ba7458d.urdf",
    mjcf_path=f"{TABLE750_BASE_DIR}/126f60baf12759ea957fb6c38ba7458d/mjcf/126f60baf12759ea957fb6c38ba7458d.xml",
    default_position=(0.3, 0.0, 0.37),
    fix_base_link=True,
)

TABLE_2 = RigidObjCfg(
    name="table_2",
    scale=(1.2, 1.4, 1.0),
    physics=PhysicStateType.RIGIDBODY,
    usd_path=f"{TABLE750_BASE_DIR}/1522dad65f0859758dad5636ba348bf8/usd/1522dad65f0859758dad5636ba348bf8.usd",
    urdf_path=f"{TABLE750_BASE_DIR}/1522dad65f0859758dad5636ba348bf8/1522dad65f0859758dad5636ba348bf8.urdf",
    mjcf_path=f"{TABLE750_BASE_DIR}/1522dad65f0859758dad5636ba348bf8/mjcf/1522dad65f0859758dad5636ba348bf8.xml",
    default_position=(0.3, 0.0, 0.37),
    fix_base_link=True,
)

TABLE_3 = RigidObjCfg(
    name="table_3",
    scale=(1.2, 1.6, 1.0),
    physics=PhysicStateType.RIGIDBODY,
    usd_path=f"{TABLE750_BASE_DIR}/18848428c54456aa82070f2fd33f7bb4/usd/18848428c54456aa82070f2fd33f7bb4.usd",
    urdf_path=f"{TABLE750_BASE_DIR}/18848428c54456aa82070f2fd33f7bb4/18848428c54456aa82070f2fd33f7bb4.urdf",
    mjcf_path=f"{TABLE750_BASE_DIR}/18848428c54456aa82070f2fd33f7bb4/mjcf/18848428c54456aa82070f2fd33f7bb4.xml",
    default_position=(0.3, 0.0, 0.37),
    fix_base_link=True,
)


TABLE_4 = RigidObjCfg(
    name="table_4",
    scale=(2.0, 1.6, 1.0),
    physics=PhysicStateType.RIGIDBODY,
    usd_path=f"{TABLE750_BASE_DIR}/848396479c0b5da3bc05d0ef74d4dcfb/usd/848396479c0b5da3bc05d0ef74d4dcfb.usd",
    urdf_path=f"{TABLE750_BASE_DIR}/848396479c0b5da3bc05d0ef74d4dcfb/848396479c0b5da3bc05d0ef74d4dcfb.urdf",
    mjcf_path=f"{TABLE750_BASE_DIR}/848396479c0b5da3bc05d0ef74d4dcfb/mjcf/848396479c0b5da3bc05d0ef74d4dcfb.xml",
    default_position=(0.3, 0.0, 0.37),
    fix_base_link=True,
)

TABLE_5 = RigidObjCfg(
    name="table_5",
    scale=(1.3, 1.3, 1.0),
    physics=PhysicStateType.RIGIDBODY,
    usd_path=f"{TABLE750_BASE_DIR}/b4b40966ebda5393bd4d7fc634062519/usd/b4b40966ebda5393bd4d7fc634062519.usd",
    urdf_path=f"{TABLE750_BASE_DIR}/b4b40966ebda5393bd4d7fc634062519/b4b40966ebda5393bd4d7fc634062519.urdf",
    mjcf_path=f"{TABLE750_BASE_DIR}/b4b40966ebda5393bd4d7fc634062519/mjcf/b4b40966ebda5393bd4d7fc634062519.xml",
    default_position=(0.3, 0.0, 0.37),
    fix_base_link=True,
)

# List of all table configurations
ALL_TABLE750_CONFIGS = [
    TABLE_1,
    TABLE_2,
    TABLE_3,
    TABLE_4,
    TABLE_5,
]

# Dictionary mapping table names to configurations
TABLE750_CONFIG_DICT = {
    "table_1": TABLE_1,
    "table_2": TABLE_2,
    "table_3": TABLE_3,
    "table_4": TABLE_4,
    "table_5": TABLE_5,
}
