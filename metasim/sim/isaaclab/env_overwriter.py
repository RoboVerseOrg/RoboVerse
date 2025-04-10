import torch
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.camera_util import get_cam_params

from .isaaclab_helper import add_lights, add_objects, add_robot, get_pose

try:
    from .empty_env import EmptyEnv
except:
    pass

## Constants

table_height = 0.75
table_thickness = 0.05
table_size = 1.5

wall_dist = 3.0
wall_height = 4.0
wall_thickness = 0.1

SCENE_PRIM_PATH = "/World/envs/env_0/house"


################################################################################
## Since IsaacLab is a OOP style framework, we need to overwrite some methods
################################################################################
class IsaaclabEnvOverwriter:
    def __init__(self, scenario: ScenarioCfg):
        self.scenario = scenario
        self.task = scenario.task
        self.robot = scenario.robot
        self.cameras = scenario.cameras
        self.objects = scenario.objects
        self.scene = scenario.scene
        self.checker = scenario.checker
        self.lights = scenario.lights
        self.checker_debug_viewers = self.checker.get_debug_viewers()

        ## XXX for initialization
        self.first_reset = True

    def _reset_idx(self, env: "EmptyEnv", env_ids: list[int] | None = None) -> None:
        if self.first_reset:
            log.info("Resetting for the first time, initializing randomizer")
            from .utils.usd_util import MdlRandomizer

            self.table_randomizer = MdlRandomizer(
                "/World/envs/env_0/metasim_table", case="table", split=self.scenario.split
            )
            self.ground_randomizer = MdlRandomizer("/World/ground", case="ground", split=self.scenario.split)
            self.wall_randomizer = MdlRandomizer(
                "/World/envs/env_0/metasim_wall", case="wall", split=self.scenario.split
            )

        for camera in self.cameras:
            ## Randomize camera pose
            if self.scenario.random.camera:
                from .utils.camera_random_util import randomize_camera_pose

                ## Get object in interest
                ## XXX: temporarily choose the first object as object in interest
                obj_pos, obj_quat = get_pose(env, self.objects[0].name)
                obj_pos = obj_pos[0]  # FIXME: only support one environment

                ## Get robot
                robot_pos, robot_quat = get_pose(env, self.robot.name)
                robot_quat = robot_quat[0]  # FIXME: only support one environment

                camera = randomize_camera_pose(camera, obj_pos.tolist(), robot_quat.tolist(), "front_select", self.task)

            if self.first_reset or self.scenario.random.camera:
                eyes = torch.tensor(camera.pos, dtype=torch.float32, device=env.device)[None, :]
                targets = torch.tensor(camera.look_at, dtype=torch.float32, device=env.device)[None, :]
                eyes = eyes + env.scene.env_origins
                targets = targets + env.scene.env_origins
                camera_inst = env.scene.sensors[camera.name]
                camera_inst.set_world_poses_from_view(eyes=eyes, targets=targets)

                ## XXX: update config to make sure alignment
                camera.pos = eyes[0].tolist()
                camera.look_at = targets[0].tolist()

        ## Randomize ground
        if self.scenario.random.ground:
            self.ground_randomizer.do_it()

        ## Randomize wall
        if self.scenario.random.wall:
            self.wall_randomizer.do_it()

        ## Randomize table
        if self.scenario.random.table and self.task.can_tabletop and self.scenario.try_add_table:
            try:
                from omni.isaac.core.prims import GeometryPrim
            except ModuleNotFoundError:
                from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim

            from metasim.utils.math import quat_from_matrix

            self.table_randomizer.do_it()

            for env_id in range(env.num_envs):
                table_prim = GeometryPrim(f"/World/envs/env_{env_id}/metasim_table", name=f"table_prim_{env_id}")
                pos, _ = table_prim.get_world_pose()
                theta = torch.rand(1) * 2 * torch.pi
                rot_mat = torch.tensor([
                    [torch.cos(theta), -torch.sin(theta), 0.0],
                    [torch.sin(theta), torch.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ])
                quat = quat_from_matrix(rot_mat)
                table_prim.set_world_pose(position=pos, orientation=quat)

        ## Randomize objects reflection
        if self.scenario.random.reflection:
            from .utils.usd_util import ReflectionRandomizer

            for env_id in range(env.num_envs):
                for obj in self.objects:
                    if isinstance(obj, PrimitiveSphereCfg) or isinstance(obj, PrimitiveCubeCfg):
                        continue
                    ReflectionRandomizer(f"/World/envs/env_{env_id}/{obj.name}").do_it()

            if self.task.can_tabletop and self.scenario.try_add_table:
                ReflectionRandomizer("/World/envs/env_0/metasim_table").do_it()
            ReflectionRandomizer("/World/ground").do_it()

        ## Randomize lights
        if self.scenario.random.light:
            import numpy as np

            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils
            from pxr import Gf

            nrow = 2
            ncol = 3

            light_radius = np.random.uniform(0.05, 0.1)
            light_length = np.random.uniform(0.5, 2.0)
            light_intensity = np.random.uniform(4e4, 1.2e5)
            light_color_temperature = np.random.uniform(4000, 8000)

            for row in range(nrow):
                for col in range(ncol):
                    light_prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/lights/light_{row}_{col}")

                    light_spacing = (1.0, 2.5 * light_length)
                    light_prim.GetAttribute("xformOp:translate").Set((
                        light_spacing[1] * (col - (ncol - 1) / 2),
                        light_spacing[0] * (row - (nrow - 1) / 2),
                        wall_height - (table_height + table_thickness) - wall_thickness,
                    ))
                    light_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                    light_prim.GetAttribute("inputs:radius").Set(light_radius)
                    light_prim.GetAttribute("inputs:length").Set(light_length)
                    light_prim.GetAttribute("inputs:intensity").Set(light_intensity)
                    light_prim.GetAttribute("inputs:enableColorTemperature").Set(True)
                    light_prim.GetAttribute("inputs:colorTemperature").Set(light_color_temperature)

        if self.first_reset:
            self.first_reset = False

    def _setup_scene(self, env: "EmptyEnv") -> None:
        try:
            import omni.isaac.lab.sim as sim_utils
            from omni.isaac.core.prims import GeometryPrim
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
        except ModuleNotFoundError:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import TiledCamera, TiledCameraCfg
            from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
            from isaacsim.core.utils.stage import add_reference_to_stage

        from .utils.ground_util import create_ground, set_ground_material, set_ground_material_scale
        from .utils.usd_util import ReflectionRandomizer, ShaderFixer

        use_scene = False
        if self.scene is not None or self.scenario.random.scene:
            if self.scenario.random.scene:
                from .utils.usd_util import SceneRandomizer

                scene_randomizer = SceneRandomizer()
                scene_cfg_dict = scene_randomizer.do_it()
            else:
                scene_cfg_dict = {
                    "filepath": self.scene.usd_path,
                    "position": self.scene.default_position,
                    "quat": self.scene.quat,
                    "scale": self.scene.scale,
                }

            add_reference_to_stage(
                scene_cfg_dict["filepath"],
                SCENE_PRIM_PATH,
            )
            house = GeometryPrim(SCENE_PRIM_PATH)
            quat = scene_cfg_dict["quat"]
            house.set_local_scale(scene_cfg_dict["scale"])
            house.set_world_pose(position=scene_cfg_dict["position"], orientation=quat)
            ShaderFixer(scene_cfg_dict["filepath"], SCENE_PRIM_PATH).fix_all()
            use_scene = True

        add_robot(env, self.robot)
        add_objects(env, self.objects + self.checker_debug_viewers[:1])  # TODO: now only support one checker viewer

        ## Fix shader texture map path
        for obj in self.objects:
            if isinstance(obj, RigidObjCfg) or isinstance(obj, ArticulationObjCfg):
                fixer = ShaderFixer(obj.usd_path, f"/World/envs/env_0/{obj.name}")
                fixer.fix_all()

        ## Add ground
        ## TODO: test if this will conflict with the scene
        if not use_scene:
            create_ground()

        ## Add table
        if self.task is not None and self.task.can_tabletop and self.scenario.try_add_table:
            try:
                import omni.isaac.core.utils.prims as prim_utils
                from omni.isaac.core.prims import GeometryPrim
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils
                from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim

            from .utils.custom_cuboid import FixedCuboid
            from .utils.ground_util import GROUND_PRIM_PATH

            ## Move ground down
            ground_prim = GeometryPrim(GROUND_PRIM_PATH, name="ground_prim")
            ground_prim.set_world_pose(position=(0.0, 0.0, -table_height), orientation=(1.0, 0.0, 0.0, 0.0))

            ## Add table
            prim_utils.create_prim("/World/envs/env_0/metasim_table")
            FixedCuboid(
                prim_path="/World/envs/env_0/metasim_table/surface",
                name="fixed_table",
                scale=torch.tensor([table_size, table_size, table_thickness]),
                position=torch.tensor([0.0, 0.0, -table_thickness / 2]),
            )
            for i, (x, y) in enumerate([
                (-table_size * 3 / 8, -table_size * 3 / 8),
                (table_size * 3 / 8, -table_size * 3 / 8),
                (-table_size * 3 / 8, table_size * 3 / 8),
                (table_size * 3 / 8, table_size * 3 / 8),
            ]):
                FixedCuboid(
                    prim_path=f"/World/envs/env_0/metasim_table/leg_{i}",
                    name=f"fixed_table_leg_{i}",
                    scale=torch.tensor([0.05, 0.05, table_height - table_thickness]),
                    position=torch.tensor([x, y, -(table_height + table_thickness) / 2]),
                )
        ## Add wall
        if self.scenario.random.wall:
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            from .utils.custom_cuboid import FixedCuboid

            prim_utils.create_prim("/World/envs/env_0/metasim_wall")

            for i, (x, y, sx, sy) in enumerate([
                (0, wall_dist, wall_dist * 2, wall_thickness),
                (0, -wall_dist, wall_dist * 2, wall_thickness),
                (wall_dist, 0, wall_thickness, wall_dist * 2),
                (-wall_dist, 0, wall_thickness, wall_dist * 2),
            ]):
                FixedCuboid(
                    f"/World/envs/env_0/metasim_wall/wall_{i}",
                    name=f"wall_{i}",
                    scale=torch.tensor([sx, sy, wall_height]),
                    position=torch.tensor([x, y, wall_height / 2 - (table_height + table_thickness)]),
                )

            ## Add roof
            if self.scenario.random.light:
                FixedCuboid(
                    prim_path="/World/envs/env_0/metasim_wall/roof",
                    name="roof",
                    scale=torch.tensor([wall_dist * 2, wall_dist * 2, wall_thickness]),
                    position=torch.tensor([
                        0.0,
                        0.0,
                        wall_height + wall_thickness / 2 - (table_height + table_thickness),
                    ]),
                )

        ## Set default ground material
        if not use_scene:
            set_ground_material("metasim/data/quick_start/materials/Ash.mdl")
            set_ground_material_scale(10)

        ## Clone, filter, and replicate
        env.scene.clone_environments(copy_from_source=False)
        env.scene.filter_collisions(global_prim_paths=[])

        ## Randomize material
        ## XXX: First call must be in _setup_scene, otherwise it will raise strange error
        if self.scenario.random.reflection:
            for env_id in range(env.num_envs):
                for obj in self.objects:
                    if isinstance(obj, PrimitiveSphereCfg) or isinstance(obj, PrimitiveCubeCfg):
                        continue
                    ReflectionRandomizer(f"/World/envs/env_{env_id}/{obj.name}").do_it()

        ## Add lights
        if self.scenario.random.light:
            ## Randomization mode: use cylinder light
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            nrow = 2
            ncol = 3

            for row in range(nrow):
                for col in range(ncol):
                    prim_utils.create_prim(
                        f"/World/envs/env_0/lights/light_{row}_{col}",
                        "CylinderLight",
                    )
        else:
            add_lights(env, self.lights)

        ## Add camera
        for camera in self.cameras:
            env.scene.sensors[camera.name] = TiledCamera(
                TiledCameraCfg(
                    prim_path=f"/World/envs/env_.*/{camera.name}",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                    data_types=camera.data_types,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=camera.focal_length,
                        focus_distance=camera.focus_distance,
                        horizontal_aperture=camera.horizontal_aperture,
                        clipping_range=camera.clipping_range,
                    ),
                    width=camera.width,
                    height=camera.height,
                )
            )

    def _pre_physics_step(self, env: "EmptyEnv", actions: torch.Tensor) -> None:
        ## TODO: Clip action or not?
        env.actions = actions

    def _apply_action(self, env: "EmptyEnv") -> None:
        env.robot.set_joint_position_target(env.actions)

    def _get_observations(self, env: "EmptyEnv") -> None:
        ## TODO: get proprioception observations
        for camera in self.cameras:
            camera_inst = env.scene.sensors[camera.name]

            ## Vision
            rgb_data = camera_inst.data.output.get("rgb", None)
            depth_data = camera_inst.data.output.get("depth", None)

            ## Camera
            cam_pos = torch.tensor(camera.pos, device=env.device)[None, :].repeat(env.num_envs, 1)
            cam_look_at = torch.tensor(camera.look_at, device=env.device)[None, :].repeat(env.num_envs, 1)
            cam_intr0 = camera_inst.data.intrinsic_matrices
            cam_extr, cam_intr = get_cam_params(
                cam_pos,
                cam_look_at,
                width=camera.width,
                height=camera.height,
                focal_length=camera.focal_length,
                horizontal_aperture=camera.horizontal_aperture,
            )
            assert torch.allclose(cam_intr0, cam_intr)

            ## Robot State
            joint_qpos_target = env.actions.cpu()
            joint_qpos = env.robot.data.joint_pos.cpu()
            robot_root_state = env.robot.data.root_state_w.cpu()
            robot_body_state = env.robot.data.body_link_state_w.cpu()
            robot_root_state[..., 0:3] -= env.scene.env_origins.cpu()
            robot_body_state[..., 0:3] -= env.scene.env_origins.cpu().unsqueeze(1)
            if self.robot.ee_prim_path is not None:
                ee_pos, ee_quat = get_pose(env, self.robot.name, self.robot.ee_prim_path)
                robot_ee_state = torch.cat([ee_pos, ee_quat, joint_qpos[:, -1:]], dim=-1)
            else:
                log.warning(f"No end-effector prim path for {self.robot.name}")
                robot_ee_state = None

            data_dict = {
                ## Vision
                "rgb": rgb_data,
                "depth": depth_data,
                ## Camera
                "cam_pos": cam_pos,
                "cam_look_at": cam_look_at,
                "cam_intr": cam_intr,
                "cam_extr": cam_extr,
                ## State
                "joint_qpos_target": joint_qpos_target,
                "joint_qpos": joint_qpos,  # align with old version
                "robot_ee_state": robot_ee_state,  # align with old version
                "robot_root_state": robot_root_state,
                "robot_body_state": robot_body_state,
            }

            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.cpu()
            torch.cuda.empty_cache()

            return data_dict

    def _get_rewards(self, env: "EmptyEnv") -> None:
        pass

    def _get_dones(self, env: "EmptyEnv") -> tuple[torch.Tensor, torch.Tensor]:
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        is_success = torch.zeros_like(time_out)
        return is_success, time_out
