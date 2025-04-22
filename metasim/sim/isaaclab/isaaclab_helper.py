import os

import torch
from loguru import logger as log

from metasim.cfg.lights import BaseLightCfg, CylinderLightCfg, DistantLightCfg
from metasim.cfg.objects import (
    ArticulationObjCfg,
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveFrameCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.cfg.robots import BaseRobotCfg

try:
    from .empty_env import EmptyEnv
except:
    pass


def add_object(env: "EmptyEnv", obj: BaseObjCfg) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

    assert isinstance(obj, BaseObjCfg)
    prim_path = f"/World/envs/env_.*/{obj.name}"
    ## Rigid object
    if isinstance(obj, RigidObjCfg):
        if obj.fix_base_link:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True)
        else:
            rigid_props = sim_utils.RigidBodyPropertiesCfg()
        if obj.collision_enabled:
            collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        else:
            collision_props = None

        ## Primitive object
        if isinstance(obj, PrimitiveCubeCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCuboidCfg(
                        size=tuple([x * s for x, s in zip(obj.size, obj.scale)]),
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveSphereCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshSphereCfg(
                        radius=obj.radius,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveCylinderCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCylinderCfg(
                        radius=obj.radius,
                        height=obj.height,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveFrameCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/fs/Downloads/frame_prim.usd",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True, kinematic_enabled=True
                        ),  # fixed
                        collision_props=None,  # no collision
                        scale=obj.scale,
                    ),
                )
            )
            return

        ## File-based object
        usd_file_cfg = sim_utils.UsdFileCfg(
            usd_path=obj.usd_path,
            rigid_props=rigid_props,
            collision_props=collision_props,
            scale=obj.scale,
        )
        if isinstance(obj, RigidObjCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(RigidObjectCfg(prim_path=prim_path, spawn=usd_file_cfg))
            return

    ## Articulation object
    if isinstance(obj, ArticulationObjCfg):
        env.scene.articulations[obj.name] = Articulation(
            ArticulationCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(usd_path=obj.usd_path, scale=obj.scale),
                actuators={},
            )
        )
        return
    raise ValueError(f"Unsupported object type: {type(obj)}")


def add_objects(env: "EmptyEnv", objects: list[BaseObjCfg]) -> None:
    for obj in objects:
        add_object(env, obj)


def add_robot(env: "EmptyEnv", robot: BaseRobotCfg) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import Articulation, ArticulationCfg
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(),
        ),
        actuators={
            jn: ImplicitActuatorCfg(joint_names_expr=[jn], stiffness=None, damping=None) for jn in robot.actuators
        },
    )
    cfg.prim_path = f"/World/envs/env_.*/{robot.name}"
    cfg.spawn.usd_path = os.path.abspath(robot.usd_path)
    cfg.spawn.rigid_props.disable_gravity = not robot.enabled_gravity
    cfg.spawn.articulation_props.enabled_self_collisions = robot.enabled_self_collisions
    for joint_name, actuator in robot.actuators.items():
        cfg.actuators[joint_name].velocity_limit = actuator.velocity_limit

    robot_inst = Articulation(cfg)
    env.scene.articulations[robot.name] = robot_inst
    env.robot = robot_inst


def _add_light(env: "EmptyEnv", light: BaseLightCfg, prim_path: str) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim.spawners import spawn_light
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

    if isinstance(light, DistantLightCfg):
        spawn_light(
            prim_path,
            sim_utils.DistantLightCfg(
                intensity=light.intensity,
            ),
            orientation=light.quat,
        )
    elif isinstance(light, CylinderLightCfg):
        spawn_light(
            prim_path,
            sim_utils.CylinderLightCfg(
                intensity=light.intensity,
                length=light.length,
                radius=light.radius,
            ),
            translation=light.pos,
            orientation=light.rot,
        )
    else:
        raise ValueError(f"Unsupported light type: {type(light)}")


def add_lights(env: "EmptyEnv", lights: list[BaseLightCfg]) -> None:
    for i, light in enumerate(lights):
        if light.is_global:
            _add_light(env, light, f"/World/lights/light_{i}")
        else:
            _add_light(env, light, f"/World/envs/env_0/lights/light_{i}")


def get_pose(
    env: "EmptyEnv", obj_name: str, obj_subpath: str | None = None, env_ids: list[int] | None = None
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    try:
        from omni.isaac.core.prims import RigidPrimView

        ISAACLAB_VERSION = 1
    except ModuleNotFoundError:
        from isaacsim.core.prims import RigidPrim as RigidPrimView

        ISAACLAB_VERSION = 2

    if env_ids is None:
        env_ids = list(range(env.num_envs))

    if obj_name in env.scene.rigid_objects:
        obj_inst = env.scene.rigid_objects[obj_name]
    elif obj_name in env.scene.articulations:
        obj_inst = env.scene.articulations[obj_name]
    else:
        raise ValueError(f"Object {obj_name} not found")

    if obj_subpath is None:
        pos = obj_inst.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        rot = obj_inst.data.root_quat_w[env_ids]
    else:
        ## TODO: Following code has bug with IsaacLab2.0 (IsaacSim 4.5)
        if ISAACLAB_VERSION == 1:
            view = RigidPrimView(
                obj_inst.cfg.prim_path + "/" + obj_subpath,
                name=f"{obj_name}_{obj_subpath}_view",
                reset_xform_properties=False,
            )
            pos, rot = view.get_world_poses(indices=env_ids)
            pos = pos - env.scene.env_origins[env_ids]
        else:
            log.warning("IsaacLab2.0 (IsaacSim 4.5) does not support creating RigidPrimView, so we return zeros.")
            pos = torch.zeros((len(env_ids), 3), device=env.device)
            rot = torch.zeros((len(env_ids), 4), device=env.device)

    assert pos.shape == (len(env_ids), 3)
    assert rot.shape == (len(env_ids), 4)
    return pos, rot


def joint_is_implicit_actuator(joint_name: str, obj_inst) -> bool:
    try:
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import Articulation
    except ModuleNotFoundError:
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation

    assert isinstance(obj_inst, Articulation)
    actuators = [actuator for actuator in obj_inst.actuators.values() if joint_name in actuator.joint_names]
    if len(actuators) == 0:
        log.error(f"Joint {joint_name} could not be found in actuators of {obj_inst.cfg.prim_path}")
        return False
    if len(actuators) > 1:
        log.warning(f"Joint {joint_name} is found in multiple actuators of {obj_inst.cfg.prim_path}")
    actuator = actuators[0]
    return isinstance(actuator, ImplicitActuatorCfg)
