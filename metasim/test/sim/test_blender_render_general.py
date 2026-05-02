from __future__ import annotations

import pytest
import torch

pytest.importorskip("bpy")


@pytest.mark.general
def test_blender_name_normalization_removes_numeric_suffix() -> None:
    from metasim.sim.blender.blender import _normalized_blender_name

    assert _normalized_blender_name("left_palm_link") == "left_palm_link"
    assert _normalized_blender_name("left_palm_link.001") == "left_palm_link"
    assert _normalized_blender_name("mesh.123") == "mesh"
    assert _normalized_blender_name("joint.abc") == "joint.abc"


@pytest.mark.general
def test_blender_body_object_selection_prefers_empty_with_children() -> None:
    import bpy
    from metasim.sim.blender.blender import _choose_body_object

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    empty = bpy.data.objects.new("left_palm_link", None)
    mesh_data = bpy.data.meshes.new("left_palm_link_mesh_data")
    mesh = bpy.data.objects.new("left_palm_link.001", mesh_data)
    child_mesh = bpy.data.objects.new("left_palm_link_visual", bpy.data.meshes.new("visual_mesh_data"))

    bpy.context.collection.objects.link(empty)
    bpy.context.collection.objects.link(mesh)
    bpy.context.collection.objects.link(child_mesh)
    child_mesh.parent = empty

    assert _choose_body_object([mesh, empty]) is empty


@pytest.mark.general
def test_blender_body_object_selection_prefers_visible_import_instance() -> None:
    import bpy
    from metasim.sim.blender.blender import _choose_body_object

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    prototype = bpy.data.objects.new("left_palm_link", None)
    prototype_child = bpy.data.objects.new("mesh", bpy.data.meshes.new("prototype_mesh"))
    instance = bpy.data.objects.new("left_palm_link.001", None)
    instance_child = bpy.data.objects.new("visuals", None)
    proto_collection = bpy.data.collections.new("proto")
    instance_child.instance_type = "COLLECTION"
    instance_child.instance_collection = proto_collection

    bpy.context.collection.objects.link(instance)
    bpy.context.collection.objects.link(instance_child)
    prototype_child.parent = prototype
    instance_child.parent = instance

    # USD instance prototypes are not linked into the active scene collection,
    # but they can share the body name with the visible instance transform.
    assert prototype.visible_get() is False
    assert instance.visible_get() is True

    assert _choose_body_object([prototype, instance]) is instance


@pytest.mark.general
def test_blender_matrix_from_root_state_uses_wxyz_quaternion() -> None:
    from metasim.sim.blender.blender import _matrix_from_root_state

    root_state = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0])
    matrix = _matrix_from_root_state(root_state)

    assert tuple(round(value, 5) for value in matrix.translation) == (1.0, 2.0, 3.0)
    assert matrix.to_quaternion().w == pytest.approx(1.0)
    assert matrix.to_quaternion().x == pytest.approx(0.0)
    assert matrix.to_quaternion().y == pytest.approx(0.0)
    assert matrix.to_quaternion().z == pytest.approx(0.0)


@pytest.mark.general
def test_blender_apply_root_state_sets_world_pose_for_parented_object() -> None:
    import bpy
    from metasim.sim.blender.blender import _apply_root_state_to_object

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    parent = bpy.data.objects.new("parent", None)
    child = bpy.data.objects.new("child", None)
    bpy.context.collection.objects.link(parent)
    bpy.context.collection.objects.link(child)
    parent.location = (10.0, 0.0, 0.0)
    child.parent = parent
    child.scale = (2.0, 3.0, 4.0)
    bpy.context.view_layer.update()
    initial_world_scale = tuple(round(value, 5) for value in child.matrix_world.to_scale())

    root_state = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    _apply_root_state_to_object(child, root_state)

    assert tuple(round(value, 5) for value in child.matrix_world.translation) == (1.0, 2.0, 3.0)
    assert tuple(round(value, 5) for value in child.matrix_world.to_scale()) == initial_world_scale


@pytest.mark.general
def test_blender_handler_applies_robot_body_state_to_body_objects() -> None:
    import bpy
    from metasim.sim.blender import BlenderHandler
    from metasim.types import RobotState

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    body_obj = bpy.data.objects.new("left_palm_link", None)
    bpy.context.collection.objects.link(body_obj)

    handler = object.__new__(BlenderHandler)
    handler._body_objs = {"robot": {"left_palm_link": body_obj}}

    robot_state = RobotState(
        root_state=torch.zeros((1, 13), dtype=torch.float32),
        body_names=["left_palm_link"],
        body_state=torch.tensor(
            [[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32
        ),
        joint_pos=torch.zeros((1, 0), dtype=torch.float32),
        joint_vel=torch.zeros((1, 0), dtype=torch.float32),
        joint_pos_target=None,
        joint_vel_target=None,
        joint_effort_target=None,
    )

    handler._apply_robot_body_state("robot", robot_state)

    assert tuple(round(value, 5) for value in body_obj.location) == (1.0, 2.0, 3.0)


@pytest.mark.general
def test_blender_handler_strips_usd_unit_scale_for_robot_body_pose() -> None:
    import bpy
    from metasim.sim.blender import BlenderHandler
    from metasim.types import RobotState

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    body_obj = bpy.data.objects.new("body", None)
    visual = bpy.data.objects.new("visuals", None)
    proto = bpy.data.collections.new("proto")
    visual.instance_type = "COLLECTION"
    visual.instance_collection = proto
    bpy.context.collection.objects.link(body_obj)
    bpy.context.collection.objects.link(visual)
    visual.parent = body_obj
    body_obj.scale = (0.01, 0.01, 0.01)

    handler = object.__new__(BlenderHandler)
    handler._body_objs = {"robot": {"body": body_obj}}

    robot_state = RobotState(
        root_state=torch.zeros((1, 13), dtype=torch.float32),
        body_names=["body"],
        body_state=torch.tensor(
            [[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32
        ),
        joint_pos=torch.zeros((1, 0), dtype=torch.float32),
        joint_vel=torch.zeros((1, 0), dtype=torch.float32),
        joint_pos_target=None,
        joint_vel_target=None,
        joint_effort_target=None,
    )

    handler._apply_robot_body_state("robot", robot_state)

    assert tuple(round(value, 5) for value in body_obj.matrix_world.translation) == (1.0, 2.0, 3.0)
    assert tuple(round(value, 5) for value in body_obj.matrix_world.to_scale()) == (1.0, 1.0, 1.0)


@pytest.mark.general
def test_blender_handler_uses_renderable_visual_object_for_body_pose() -> None:
    import bpy
    from metasim.sim.blender import BlenderHandler
    from metasim.types import RobotState

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    body_obj = bpy.data.objects.new("openarm_left_link1", None)
    visual_obj = bpy.data.objects.new("openarm_left_link1_visual", None)
    visual_mesh = bpy.data.objects.new("mesh", bpy.data.meshes.new("visual_mesh"))
    bpy.context.collection.objects.link(body_obj)
    bpy.context.collection.objects.link(visual_obj)
    bpy.context.collection.objects.link(visual_mesh)
    visual_mesh.parent = visual_obj

    handler = object.__new__(BlenderHandler)
    handler._body_objs = {"robot": {"openarm_left_link1": body_obj, "openarm_left_link1_visual": visual_obj}}

    robot_state = RobotState(
        root_state=torch.zeros((1, 13), dtype=torch.float32),
        body_names=["openarm_left_link1"],
        body_state=torch.tensor(
            [[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32
        ),
        joint_pos=torch.zeros((1, 0), dtype=torch.float32),
        joint_vel=torch.zeros((1, 0), dtype=torch.float32),
        joint_pos_target=None,
        joint_vel_target=None,
        joint_effort_target=None,
    )

    handler._apply_robot_body_state("robot", robot_state)

    assert tuple(round(value, 5) for value in body_obj.matrix_world.translation) == (0.0, 0.0, 0.0)
    assert tuple(round(value, 5) for value in visual_obj.matrix_world.translation) == (1.0, 2.0, 3.0)


@pytest.mark.general
def test_blender_repairs_empty_imported_material_node_tree() -> None:
    import bpy
    from metasim.sim.blender.blender import _repair_imported_materials, _rgba_srgb_to_linear

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    mesh_obj = bpy.data.objects.new("mesh", bpy.data.meshes.new("mesh_data"))
    material = bpy.data.materials.new("material_A0A0A0")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    mesh_obj.data.materials.append(material)
    bpy.context.collection.objects.link(mesh_obj)

    _repair_imported_materials([mesh_obj])

    output = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    surface = output.inputs["Surface"]
    expected_color = _rgba_srgb_to_linear((0.6274509803921569, 0.6274509803921569, 0.6274509803921569, 1.0))
    assert len(surface.links) == 1
    bsdf = surface.links[0].from_node
    assert bsdf.bl_idname == "ShaderNodeBsdfPrincipled"
    assert tuple(round(value, 5) for value in bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_color
    )


@pytest.mark.general
def test_blender_repairs_openarm_materials_without_recoloring_hands() -> None:
    import bpy
    from metasim.sim.blender.blender import _repair_imported_materials, _rgba_srgb_to_linear

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    shared_material = bpy.data.materials.new("material_E5EAED")
    shared_material.use_nodes = True
    shared_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    arm = bpy.data.objects.new("openarm_left_link1", None)
    arm_mesh = bpy.data.objects.new("arm_mesh", bpy.data.meshes.new("arm_mesh_data"))
    hand = bpy.data.objects.new("left_palm_link", None)
    hand_mesh = bpy.data.objects.new("hand_mesh", bpy.data.meshes.new("hand_mesh_data"))
    arm_mesh.data.materials.append(shared_material)
    hand_mesh.data.materials.append(shared_material)
    for obj in (arm, arm_mesh, hand, hand_mesh):
        bpy.context.collection.objects.link(obj)
    arm_mesh.parent = arm
    hand_mesh.parent = hand

    _repair_imported_materials([arm, hand])

    arm_bsdf = next(
        node
        for node in arm_mesh.material_slots[0].material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    hand_bsdf = next(
        node
        for node in hand_mesh.material_slots[0].material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )

    expected_arm_color = _rgba_srgb_to_linear((0.24705882, 0.24705882, 0.24705882, 1.0))
    expected_hand_color = _rgba_srgb_to_linear((0.89804, 0.91765, 0.92941, 1.0))

    assert arm_mesh.material_slots[0].material is not hand_mesh.material_slots[0].material
    assert tuple(round(value, 5) for value in arm_bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_arm_color
    )
    assert tuple(round(value, 5) for value in hand_bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_hand_color
    )


@pytest.mark.general
def test_blender_uses_openarm_silver_material_slots() -> None:
    import bpy
    from metasim.sim.blender.blender import _repair_imported_materials, _rgba_srgb_to_linear

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    black_slot_material = bpy.data.materials.new("mat_0_004")
    silver_slot_material = bpy.data.materials.new("mat_1_004")
    for material in (black_slot_material, silver_slot_material):
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    link = bpy.data.objects.new("openarm_left_link4", None)
    mesh = bpy.data.objects.new("mesh", bpy.data.meshes.new("mesh_data"))
    mesh.data.materials.append(black_slot_material)
    mesh.data.materials.append(silver_slot_material)
    for obj in (link, mesh):
        bpy.context.collection.objects.link(obj)
    mesh.parent = link

    _repair_imported_materials([link])

    black_bsdf = next(
        node
        for node in mesh.material_slots[0].material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    silver_bsdf = next(
        node
        for node in mesh.material_slots[1].material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    expected_black = _rgba_srgb_to_linear((0.24705882, 0.24705882, 0.24705882, 1.0))
    expected_silver = _rgba_srgb_to_linear((0.79607843, 0.79607843, 0.79607843, 1.0))

    assert tuple(round(value, 5) for value in black_bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_black
    )
    assert tuple(round(value, 5) for value in silver_bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_silver
    )


@pytest.mark.general
def test_blender_reuses_context_material_for_revisited_usd_prototype_mesh() -> None:
    import bpy
    from metasim.sim.blender.blender import _repair_imported_materials, _rgba_srgb_to_linear

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    source_material = bpy.data.materials.new("matte_source")
    source_material.use_nodes = True
    source_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    link = bpy.data.objects.new("openarm_left_link1", None)
    instance = bpy.data.objects.new("visuals", None)
    visual_root = bpy.data.objects.new("openarm_left_link1_visual", None)
    mesh = bpy.data.objects.new("mesh", bpy.data.meshes.new("mesh_data"))
    prototype_collection = bpy.data.collections.new("proto")
    for obj in (link, instance):
        bpy.context.collection.objects.link(obj)
    for obj in (mesh, visual_root):
        prototype_collection.objects.link(obj)
    mesh.data.materials.append(source_material)
    instance.parent = link
    instance.instance_type = "COLLECTION"
    instance.instance_collection = prototype_collection
    mesh.parent = visual_root

    _repair_imported_materials([link])

    context_materials = [
        material for material in bpy.data.materials if material.get("_metasim_source_material_name") == "matte_source"
    ]
    expected_color = _rgba_srgb_to_linear((0.24705882, 0.24705882, 0.24705882, 1.0))
    bsdf = next(
        node
        for node in mesh.material_slots[0].material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )

    assert len(context_materials) == 1
    assert tuple(round(value, 5) for value in bsdf.inputs["Base Color"].default_value) == tuple(
        round(value, 5) for value in expected_color
    )


@pytest.mark.general
def test_blender_handler_maps_wuji_hand_body_names_to_usd_names() -> None:
    import bpy
    from metasim.sim.blender import BlenderHandler
    from metasim.types import RobotState

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    finger_obj = bpy.data.objects.new("left_finger1_link1", None)
    palm_obj = bpy.data.objects.new("left_palm_link", None)
    bpy.context.collection.objects.link(finger_obj)
    bpy.context.collection.objects.link(palm_obj)

    handler = object.__new__(BlenderHandler)
    handler._body_objs = {"robot": {"left_finger1_link1": finger_obj, "left_palm_link": palm_obj}}

    robot_state = RobotState(
        root_state=torch.zeros((1, 13), dtype=torch.float32),
        body_names=["left_hand_finger1_link1", "left_hand_palm_link"],
        body_state=torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
                    [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=torch.float32,
        ),
        joint_pos=torch.zeros((1, 0), dtype=torch.float32),
        joint_vel=torch.zeros((1, 0), dtype=torch.float32),
        joint_pos_target=None,
        joint_vel_target=None,
        joint_effort_target=None,
    )

    handler._apply_robot_body_state("robot", robot_state)

    assert tuple(round(value, 5) for value in finger_obj.location) == (1.0, 2.0, 3.0)
    assert tuple(round(value, 5) for value in palm_obj.location) == (4.0, 5.0, 6.0)


@pytest.mark.general
def test_blender_handler_renders_primitive_cube_rgb(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.objects import PrimitiveCubeCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.blender import BlenderHandler
    from metasim.types import ObjectState, TensorState

    monkeypatch.chdir(tmp_path)

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.2, 0.2, 0.2],
                mass=1.0,
                physics=PhysicStateType.XFORM,
                color=[0.8, 0.1, 0.1],
                default_position=(0.0, 0.0, 0.0),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        cameras=[
            PinholeCameraCfg(
                name="overview",
                data_types=["rgb"],
                width=32,
                height=24,
                pos=(0.6, -1.0, 0.5),
                look_at=(0.0, 0.0, 0.0),
                focal_length=24.0,
                horizontal_aperture=24.0,
            )
        ],
        simulator="blender",
        headless=True,
    )
    state = TensorState(
        objects={
            "cube": ObjectState(
                root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            )
        },
        robots={},
        cameras={},
    )

    handler = BlenderHandler(scenario)
    try:
        handler.launch()
        handler.set_states(state)
        rendered = handler.get_states(mode="tensor")
    finally:
        handler.close()

    assert set(rendered.cameras) == {"overview"}
    rgb = rendered.cameras["overview"].rgb
    assert rgb is not None
    assert rgb.shape == (1, 24, 32, 3)
    assert rgb.dtype == torch.uint8
    assert int(rgb.max()) > int(rgb.min())


@pytest.mark.general
def test_blender_handler_get_states_retains_tensor_state_objects_and_robots(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bpy
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.objects import PrimitiveCubeCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.blender import BlenderHandler
    from metasim.types import ObjectState, RobotState, TensorState

    monkeypatch.chdir(tmp_path)

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.2, 0.2, 0.2],
                mass=1.0,
                physics=PhysicStateType.XFORM,
                color=[0.8, 0.1, 0.1],
                default_position=(0.0, 0.0, 0.0),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        cameras=[
            PinholeCameraCfg(
                name="overview",
                data_types=["rgb"],
                width=32,
                height=24,
                pos=(0.6, -1.0, 0.5),
                look_at=(0.0, 0.0, 0.0),
                focal_length=24.0,
                horizontal_aperture=24.0,
            )
        ],
        simulator="blender",
        headless=True,
    )
    state = TensorState(
        objects={
            "cube": ObjectState(
                root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            )
        },
        robots={
            "robot": RobotState(
                root_state=torch.zeros((1, 13), dtype=torch.float32),
                body_names=["body"],
                body_state=torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32),
                joint_pos=torch.zeros((1, 0), dtype=torch.float32),
                joint_vel=torch.zeros((1, 0), dtype=torch.float32),
                joint_pos_target=None,
                joint_vel_target=None,
                joint_effort_target=None,
            )
        },
        cameras={},
    )

    handler = BlenderHandler(scenario)
    try:
        handler.launch()
        body_obj = bpy.data.objects.new("body", None)
        bpy.context.collection.objects.link(body_obj)
        handler._body_objs = {"robot": {"body": body_obj}}
        handler.set_states(state)
        rendered = handler.get_states(mode="tensor")
    finally:
        handler.close()

    assert set(rendered.objects) == {"cube"}
    assert set(rendered.robots) == {"robot"}
    assert set(rendered.cameras) == {"overview"}


@pytest.mark.general
def test_blender_handler_default_get_states_returns_camera_tensor_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.objects import PrimitiveCubeCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.blender import BlenderHandler
    from metasim.types import ObjectState, TensorState

    monkeypatch.chdir(tmp_path)

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.2, 0.2, 0.2],
                mass=1.0,
                physics=PhysicStateType.XFORM,
                color=[0.8, 0.1, 0.1],
                default_position=(0.0, 0.0, 0.0),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        cameras=[
            PinholeCameraCfg(
                name="overview",
                data_types=["rgb"],
                width=32,
                height=24,
                pos=(0.6, -1.0, 0.5),
                look_at=(0.0, 0.0, 0.0),
                focal_length=24.0,
                horizontal_aperture=24.0,
            )
        ],
        simulator="blender",
        headless=True,
    )
    state = TensorState(
        objects={
            "cube": ObjectState(
                root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            )
        },
        robots={},
        cameras={},
    )

    handler = BlenderHandler(scenario)
    try:
        handler.launch()
        handler.set_states(state)
        rendered = handler.get_states()
    finally:
        handler.close()

    assert isinstance(rendered, TensorState)
    assert rendered.cameras["overview"].rgb is not None


@pytest.mark.general
@pytest.mark.parametrize("refresh_method", ["refresh_render", "render"])
def test_blender_handler_refresh_invalidates_cached_camera_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch, refresh_method: str
) -> None:
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.objects import PrimitiveCubeCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.blender import BlenderHandler
    from metasim.types import ObjectState, TensorState

    monkeypatch.chdir(tmp_path)

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.2, 0.2, 0.2],
                mass=1.0,
                physics=PhysicStateType.XFORM,
                color=[0.8, 0.1, 0.1],
                default_position=(0.0, 0.0, 0.0),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        cameras=[
            PinholeCameraCfg(
                name="overview",
                data_types=["rgb"],
                width=32,
                height=24,
                pos=(0.6, -1.0, 0.5),
                look_at=(0.0, 0.0, 0.0),
                focal_length=24.0,
                horizontal_aperture=24.0,
            )
        ],
        simulator="blender",
        headless=True,
    )
    state = TensorState(
        objects={
            "cube": ObjectState(
                root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            )
        },
        robots={},
        cameras={},
    )

    handler = BlenderHandler(scenario)
    try:
        handler.launch()
        handler.set_states(state)
        first_camera_state = handler.get_states(mode="tensor").cameras["overview"]
        handler._objs["cube"].location.x = 0.2
        getattr(handler, refresh_method)()
        second_camera_state = handler.get_states(mode="tensor").cameras["overview"]
    finally:
        handler.close()

    assert second_camera_state is not first_camera_state


@pytest.mark.general
def test_blender_handler_set_dof_targets_fails_fast() -> None:
    from metasim.sim.blender import BlenderHandler

    handler = object.__new__(BlenderHandler)

    with pytest.raises(NotImplementedError, match="render-only"):
        handler.set_dof_targets([])


@pytest.mark.general
def test_blender_offline_renderer_writes_camera_frames(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.objects import PrimitiveCubeCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.blender import BlenderOfflineRenderCfg, render_state_sequence
    from metasim.types import ObjectState, TensorState

    monkeypatch.chdir(tmp_path)

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.2, 0.2, 0.2],
                mass=1.0,
                physics=PhysicStateType.XFORM,
                color=[0.8, 0.1, 0.1],
                default_position=(0.0, 0.0, 0.0),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        cameras=[
            PinholeCameraCfg(
                name="overview",
                data_types=["rgb"],
                width=32,
                height=24,
                pos=(0.6, -1.0, 0.5),
                look_at=(0.0, 0.0, 0.0),
                focal_length=24.0,
                horizontal_aperture=24.0,
            )
        ],
        simulator="mujoco",
        headless=True,
    )
    states = [
        TensorState(
            objects={
                "cube": ObjectState(
                    root_state=torch.tensor(
                        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
                    )
                )
            },
            robots={},
            cameras={},
        )
    ]

    outputs = render_state_sequence(
        scenario,
        states,
        BlenderOfflineRenderCfg(output_dir=tmp_path / "renders", samples=8, device="CPU"),
    )

    assert outputs == [tmp_path / "renders" / "overview" / "frame_000000.png"]
    assert outputs[0].is_file()
    assert outputs[0].stat().st_size > 0
