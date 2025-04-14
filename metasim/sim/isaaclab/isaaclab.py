import argparse
import time
from typing import Type

import gymnasium as gym
import torch
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, BaseObjCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, EnvWrapper, IdentityEnvWrapper
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, TimeOut
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState

from .env_overwriter import IsaaclabEnvOverwriter
from .isaaclab_helper import get_pose

try:
    from omni.isaac.lab.app import AppLauncher
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher

try:
    from .empty_env import EmptyEnv
except:
    pass


class IsaaclabHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []

    ############################################################
    ## Launch
    ############################################################

    def launch(self) -> None:
        env_overwriter = IsaaclabEnvOverwriter(self.scenario)
        gym.register(
            id="MetaSimEmptyTaskEnv",
            entry_point="metasim.sim.isaaclab.empty_env:EmptyEnv",
            disable_env_checker=True,
            order_enforce=False,
            kwargs={
                "env_cfg_entry_point": "metasim.sim.isaaclab.empty_env:EmptyEnvCfg",
                "_setup_scene": env_overwriter._setup_scene,
                "_reset_idx": env_overwriter._reset_idx,
                "_pre_physics_step": env_overwriter._pre_physics_step,
                "_apply_action": env_overwriter._apply_action,
                "_get_observations": env_overwriter._get_observations,
                "_get_rewards": env_overwriter._get_rewards,
                "_get_dones": env_overwriter._get_dones,
            },
        )
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.enable_cameras = True
        args.headless = self.headless

        ## Only set args.renderer seems not enough
        if self.scenario.render.mode == "raytracing":
            args.renderer = "RayTracedLighting"
        elif self.scenario.render.mode == "pathtracing":
            args.renderer = "PathTracing"
        elif self.scenario.render.mode == "rasterization":
            raise ValueError("Isaaclab does not support rasterization")
        else:
            raise ValueError(f"Unknown render mode: {self.scenario.render.mode}")

        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        try:
            from omni.isaac.lab_tasks.utils import parse_env_cfg
        except ModuleNotFoundError:
            from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg("MetaSimEmptyTaskEnv")
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.decimation = self.scenario.decimation
        env_cfg.episode_length_s = self.scenario.episode_length * (1 / 60) * self.scenario.decimation

        self.env: EmptyEnv = gym.make("MetaSimEmptyTaskEnv", cfg=env_cfg)

        ## Render mode setting, must be done after isaaclab is launched
        ## For more info, see the import below
        import carb
        import omni.replicator.core as rep

        # from omni.rtx.settings.core.widgets.pt_widgets import PathTracingSettingsFrame

        rep.settings.set_render_rtx_realtime()  # fix noising rendered images

        settings = carb.settings.get_settings()
        if self.scenario.render.mode == "pathtracing":
            settings.set_string("/rtx/rendermode", "PathTracing")
        elif self.scenario.render.mode == "raytracing":
            settings.set_string("/rtx/rendermode", "RayTracedLighting")
        elif self.scenario.render.mode == "rasterization":
            raise ValueError("Isaaclab does not support rasterization")
        else:
            raise ValueError(f"Unknown render mode: {self.scenario.render.mode}")

        log.info(f"Render mode: {settings.get_as_string('/rtx/rendermode')}")
        log.info(f"Render totalSpp: {settings.get('/rtx/pathtracing/totalSpp')}")
        log.info(f"Render spp: {settings.get('/rtx/pathtracing/spp')}")
        log.info(f"Render adaptiveSampling/enabled: {settings.get('/rtx/pathtracing/adaptiveSampling/enabled')}")
        log.info(f"Render maxBounces: {settings.get('/rtx/pathtracing/maxBounces')}")

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        self._actions_cache = action
        joint_names = self.get_object_joint_names(self.robot)
        action_env_tensors = torch.zeros((self.num_envs, len(joint_names)), device=self.env.device)
        for env_id in range(self.num_envs):
            action_env = action[env_id]
            for i, joint_name in enumerate(joint_names):
                # Convert numpy float32 to torch tensor and move to correct device
                action_env_tensors[env_id, i] = torch.tensor(
                    action_env["dof_pos_target"][joint_name], device=self.env.device
                )

        _, _, _, time_out, extras = self.env.step(action_env_tensors)
        time_out = time_out.cpu()
        success = self.checker.check(self)
        states = self.get_states()
        return states, None, success, time_out, extras

    def reset(self, env_ids: list[int] | None = None) -> tuple[list[EnvState], Extra]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        log.info(f"Resetting envs {env_ids}")

        tic = time.time()
        _, extras = self.env.reset(env_ids=env_ids)
        toc = time.time()
        log.trace(f"Reset isaaclab env time: {toc - tic:.2f}s")

        tic = time.time()
        self.scenario.checker.reset(self, env_ids=env_ids)
        toc = time.time()
        log.trace(f"Reset checker time: {toc - tic:.2f}s")

        ## Force rerender, see https://isaac-sim.github.io/IsaacLab/main/source/refs/issues.html#blank-initial-frames-from-the-camera
        tic = time.time()
        # XXX: previously 12 is not enough for pick_cube, if this is the case again, try 18
        for _ in range(12):
            # XXX: previously sim.render() is not enough for pick_cube, if this is the case again, try calling sim.step()
            # self.env.sim.step()
            self.env.sim.render()
        toc = time.time()
        log.trace(f"Reset render time: {toc - tic:.2f}s")

        ## Update camera buffer
        tic = time.time()
        for sensor in self.env.scene.sensors.values():
            sensor.update(dt=0)
        toc = time.time()
        log.trace(f"Reset sensor buffer time: {toc - tic:.2f}s")

        ## Update obs
        tic = time.time()
        states = self.get_states()
        toc = time.time()
        log.trace(f"Reset getting obs time: {toc - tic:.2f}s")

        return states, extras

    def close(self) -> None:
        self.env.close()
        self.simulation_app.close()

    ############################################################
    ## Utils
    ############################################################
    def refresh_render(self) -> None:
        for sensor in self.env.scene.sensors.values():
            sensor.update(dt=0)
        self.env.sim.render()

    ############################################################
    ## Set states
    ############################################################
    def _set_object_pose(
        self,
        object: BaseObjCfg,
        position: torch.Tensor,  # (num_envs, 3)
        rotation: torch.Tensor,  # (num_envs, 4)
        env_ids: list[int] | None = None,
    ) -> None:
        """
        Set the pose of an object, set the velocity to zero
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert position.shape == (len(env_ids), 3)
        assert rotation.shape == (len(env_ids), 4)

        if isinstance(object, ArticulationObjCfg):
            obj_inst = self.env.scene.articulations[object.name]
        elif isinstance(object, RigidObjCfg):
            obj_inst = self.env.scene.rigid_objects[object.name]
        else:
            raise ValueError(f"Invalid object type: {type(object)}")

        pose = torch.concat(
            [
                position.to(self.env.device, dtype=torch.float32) + self.env.scene.env_origins[env_ids],
                rotation.to(self.env.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        obj_inst.write_root_pose_to_sim(pose, env_ids=torch.tensor(env_ids, device=self.env.device))
        obj_inst.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.env.device, dtype=torch.float32),
            env_ids=torch.tensor(env_ids, device=self.env.device),
        )  # ! critical
        obj_inst.write_data_to_sim()

    def _set_object_joint_pos(
        self,
        object: BaseObjCfg,
        joint_pos: torch.Tensor,  # (num_envs, num_joints)
        env_ids: list[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert joint_pos.shape[0] == len(env_ids)
        pos = joint_pos.to(self.env.device)
        vel = torch.zeros_like(pos)
        obj_inst = self.env.scene.articulations[object.name]
        obj_inst.write_joint_state_to_sim(pos, vel, env_ids=torch.tensor(env_ids, device=self.env.device))
        obj_inst.write_data_to_sim()

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states_flat = [states[i]["objects"] | states[i]["robots"] for i in range(self.num_envs)]
        for obj in self.objects + [self.robot] + self.checker.get_debug_viewers():
            if obj.name not in states_flat[0]:
                log.warning(f"Missing {obj.name} in states, setting its velocity to zero")
                pos, rot = get_pose(self.env, obj.name, env_ids=env_ids)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)
                continue

            if states_flat[0][obj.name].get("pos", None) is None or states_flat[0][obj.name].get("rot", None) is None:
                log.warning(f"No pose found for {obj.name}, setting its velocity to zero")
                pos, rot = get_pose(self.env, obj.name, env_ids=env_ids)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)
            else:
                pos = torch.stack([states_flat[env_id][obj.name]["pos"] for env_id in env_ids]).to(self.env.device)
                rot = torch.stack([states_flat[env_id][obj.name]["rot"] for env_id in env_ids]).to(self.env.device)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)

            if isinstance(obj, ArticulationObjCfg):
                if states_flat[0][obj.name].get("dof_pos", None) is None:
                    log.warning(f"No dof_pos found for {obj.name}")
                else:
                    dof_dict = [states_flat[env_id][obj.name]["dof_pos"] for env_id in env_ids]
                    joint_names = self.get_object_joint_names(obj)
                    joint_pos = torch.zeros((len(env_ids), len(joint_names)), device=self.env.device)
                    for i, joint_name in enumerate(joint_names):
                        if joint_name in dof_dict[0]:
                            joint_pos[:, i] = torch.tensor([x[joint_name] for x in dof_dict], device=self.env.device)
                        else:
                            log.warning(f"Missing {joint_name} in {obj.name}, setting its position to zero")

                    self._set_object_joint_pos(obj, joint_pos, env_ids=env_ids)
                    if obj == self.robot:
                        robot_inst = self.env.scene.articulations[obj.name]
                        robot_inst.set_joint_position_target(
                            joint_pos, env_ids=torch.tensor(env_ids, device=self.env.device)
                        )
                        robot_inst.write_data_to_sim()

    ############################################################
    ## Get states
    ############################################################
    def get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj in self.objects:
            if isinstance(obj, ArticulationObjCfg):
                obj_inst = self.env.scene.articulations[obj.name]
                joint_reindex = self.get_object_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=obj_inst.data.root_state_w,
                    joint_pos=obj_inst.data.joint_pos[:, joint_reindex],
                    joint_vel=obj_inst.data.joint_vel[:, joint_reindex],
                )
            else:
                obj_inst = self.env.scene.rigid_objects[obj.name]
                state = ObjectState(
                    root_state=obj_inst.data.root_state_w,
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in [self.robot]:
            ## TODO: dof_pos_target, dof_vel_target, dof_torque
            obj_inst = self.env.scene.articulations[obj.name]
            joint_reindex = self.get_object_joint_reindex(obj.name)
            state = RobotState(
                root_state=obj_inst.data.root_state_w,
                joint_pos=obj_inst.data.joint_pos[:, joint_reindex],
                joint_vel=obj_inst.data.joint_vel[:, joint_reindex],
                joint_pos_target=obj_inst.data.joint_pos_target[:, joint_reindex],
                joint_vel_target=obj_inst.data.joint_vel_target[:, joint_reindex],
                joint_effort_target=obj_inst.data.joint_effort_target[:, joint_reindex],
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.env.scene.sensors[camera.name]
            rgb_data = camera_inst.data.output.get("rgb", None)
            depth_data = camera_inst.data.output.get("depth", None)
            camera_states[camera.name] = CameraState(rgb=rgb_data, depth=depth_data)

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states)

    def get_pos(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        pos, _ = get_pose(self.env, obj_name, env_ids=env_ids)
        assert pos.shape == (len(env_ids), 3)
        return pos

    def get_rot(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        _, rot = get_pose(self.env, obj_name, env_ids=env_ids)
        assert rot.shape == (len(env_ids), 4)
        return rot

    def get_dof_pos(self, obj_name: str, joint_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        dof_pos = torch.zeros(len(env_ids))
        for i, env_id in enumerate(env_ids):
            dof_pos[i] = self.env.scene.articulations[obj_name].data.joint_pos[env_id][
                self.env.scene.articulations[obj_name].joint_names.index(joint_name)
            ]
        return dof_pos

    ############################################################
    ## Misc
    ############################################################
    def simulate(self):
        pass

    def get_object_joint_names(self, object: BaseObjCfg) -> list[str]:
        if isinstance(object, ArticulationObjCfg):
            return self.env.scene.articulations[object.name].joint_names
        else:
            return []

    @property
    def episode_length_buf(self) -> list[int]:
        return self.env.episode_length_buf.tolist()

    def get_joint_limits(self, obj_name: str, joint_name: str) -> torch.FloatTensor:
        obj_inst = self.env.scene.articulations[obj_name]
        # Get joint limits with shape (num_envs, num_joints, 2)
        joint_limits = obj_inst.root_physx_view.get_dof_limits()
        joint_names = obj_inst.joint_names
        joint_index = joint_names.index(joint_name)
        # Process limits for each environment in the batch, converting limits to tuples
        # Initialize dictionary to store joint limits for each environment
        batch_joint_limits = joint_limits[:, joint_index, :]
        return batch_joint_limits

    def set_camera_pose(self, position: tuple[float, float, float], look_at: tuple[float, float, float]) -> None:
        camera_inst = self.env.scene.sensors[self.cameras[0].name]
        eyes = torch.tensor(position, dtype=torch.float32, device=self.env.device)[None, :]
        targets = torch.tensor(look_at, dtype=torch.float32, device=self.env.device)[None, :]
        eyes = eyes + self.env.scene.env_origins
        targets = targets + self.env.scene.env_origins
        camera_inst.set_world_poses_from_view(eyes=eyes, targets=targets)

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return self.env.device


IsaaclabEnv: Type[EnvWrapper[IsaaclabHandler]] = IdentityEnvWrapper(IsaaclabHandler)
