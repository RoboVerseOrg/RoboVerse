"""Tensorized state of the simulation."""

from __future__ import annotations

from itertools import chain

import numpy as np
import torch

from metasim.types import Action, CameraState, DictEnvState, ObjectState, RobotState, TensorState

try:
    from metasim.sim.base import BaseSimHandler
except:
    pass


def join_tensor_states(tensor_states: list[TensorState]) -> TensorState:
    """Join a list of tensor states with num_envs = 1 into a single tensor state."""
    rst = TensorState(objects={}, robots={}, cameras={})

    if not tensor_states:
        return rst

    # Get all unique keys from each category
    all_object_keys = set()
    all_robot_keys = set()
    all_camera_keys = set()
    # all_sensor_keys = set()

    for state in tensor_states:
        all_object_keys.update(state.objects.keys())
        all_robot_keys.update(state.robots.keys())
        all_camera_keys.update(state.cameras.keys())
        # all_sensor_keys.update(state.sensors.keys())

    # Join objects
    for key in all_object_keys:
        object_states = [state.objects[key] for state in tensor_states if key in state.objects]
        if object_states:
            rst.objects[key] = ObjectState(
                root_state=torch.cat([obj.root_state for obj in object_states], dim=0),
                body_names=object_states[0].body_names,
                body_state=torch.cat([obj.body_state for obj in object_states], dim=0)
                if object_states[0].body_state is not None
                else None,
                joint_pos=torch.cat([obj.joint_pos for obj in object_states], dim=0)
                if object_states[0].joint_pos is not None
                else None,
                joint_vel=torch.cat([obj.joint_vel for obj in object_states], dim=0)
                if object_states[0].joint_vel is not None
                else None,
            )

    # Join robots
    for key in all_robot_keys:
        robot_states = [state.robots[key] for state in tensor_states if key in state.robots]
        if robot_states:
            rst.robots[key] = RobotState(
                root_state=torch.cat([robot.root_state for robot in robot_states], dim=0),
                body_names=robot_states[0].body_names,
                body_state=torch.cat([robot.body_state for robot in robot_states], dim=0)
                if robot_states[0].body_state is not None
                else None,
                joint_pos=torch.cat([robot.joint_pos for robot in robot_states], dim=0)
                if robot_states[0].joint_pos is not None
                else None,
                joint_vel=torch.cat([robot.joint_vel for robot in robot_states], dim=0)
                if robot_states[0].joint_vel is not None
                else None,
                joint_pos_target=torch.cat([robot.joint_pos_target for robot in robot_states], dim=0)
                if robot_states[0].joint_pos_target is not None
                else None,
                joint_vel_target=torch.cat([robot.joint_vel_target for robot in robot_states], dim=0)
                if robot_states[0].joint_vel_target is not None
                else None,
                joint_effort_target=torch.cat([robot.joint_effort_target for robot in robot_states], dim=0)
                if robot_states[0].joint_effort_target is not None
                else None,
            )

    # Join cameras
    for key in all_camera_keys:
        camera_states = [state.cameras[key] for state in tensor_states if key in state.cameras]
        if camera_states:
            rst.cameras[key] = CameraState(
                rgb=torch.cat([cam.rgb for cam in camera_states], dim=0) if camera_states[0].rgb is not None else None,
                depth=torch.cat([cam.depth for cam in camera_states], dim=0)
                if camera_states[0].depth is not None
                else None,
                pos=torch.cat([cam.pos for cam in camera_states], dim=0) if camera_states[0].pos is not None else None,
                quat_world=torch.cat([cam.quat_world for cam in camera_states], dim=0)
                if camera_states[0].quat_world is not None
                else None,
                intrinsics=torch.cat([cam.intrinsics for cam in camera_states], dim=0)
                if camera_states[0].intrinsics is not None
                else None,
            )

    # Join sensors (assuming similar structure to objects)
    # for key in all_sensor_keys:
    #     sensor_states = [state.sensors[key] for state in tensor_states if key in state.sensors]
    #     if sensor_states:
    #         # Note: SensorState structure is not defined, so this is a placeholder
    #         rst.sensors[key] = sensor_states[0]  # This would need to be implemented based on SensorState structure

    return rst


def _dof_tensor_to_dict(dof_tensor: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    """Convert a DOF tensor to a dictionary of joint positions."""
    assert isinstance(dof_tensor, torch.Tensor)
    joint_names = sorted(joint_names)
    return {jn: dof_tensor[i].item() for i, jn in enumerate(joint_names)}


def _dof_array_to_dict(dof_array, joint_names: list[str]) -> dict[str, float]:
    """Convert a DOF array to a dictionary of joint positions."""
    assert isinstance(dof_array, (list, np.ndarray))
    joint_names = sorted(joint_names)
    return {jn: dof_array[i] for i, jn in enumerate(joint_names)}


def _body_tensor_to_dict(body_tensor: torch.Tensor, body_names: list[str]) -> dict[str, float]:
    """Convert a body tensor to a dictionary of body positions."""
    body_names = sorted(body_names)
    return {
        bn: {
            "pos": body_tensor[i][:3].cpu(),
            "rot": body_tensor[i][3:7].cpu(),
            "vel": body_tensor[i][7:10].cpu(),
            "ang_vel": body_tensor[i][10:13].cpu(),
        }
        for i, bn in enumerate(body_names)
    }


def state_tensor_to_nested(handler: BaseSimHandler, tensor_state: TensorState) -> list[DictEnvState]:
    """Convert a tensor state to a list of env states. All the tensors will be converted to cpu for compatibility."""
    num_envs = next(iter(chain(tensor_state.objects.values(), tensor_state.robots.values()))).root_state.shape[0]
    env_states = []
    for env_id in range(num_envs):
        object_states = {}
        for obj_name, obj_state in tensor_state.objects.items():
            object_states[obj_name] = {
                "pos": obj_state.root_state[env_id, :3].cpu(),
                "rot": obj_state.root_state[env_id, 3:7].cpu(),
                "vel": obj_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": obj_state.root_state[env_id, 10:13].cpu(),
            }
            if obj_state.body_state is not None:
                bns = handler.get_body_names(obj_name)
                object_states[obj_name]["body"] = _body_tensor_to_dict(obj_state.body_state[env_id], bns)
            if obj_state.joint_pos is not None:
                jns = handler.get_joint_names(obj_name)
                object_states[obj_name]["dof_pos"] = _dof_tensor_to_dict(obj_state.joint_pos[env_id], jns)
            if obj_state.joint_vel is not None:
                jns = handler.get_joint_names(obj_name)
                object_states[obj_name]["dof_vel"] = _dof_tensor_to_dict(obj_state.joint_vel[env_id], jns)

        robot_states = {}
        for robot_name, robot_state in tensor_state.robots.items():
            jns = handler.get_joint_names(robot_name)
            robot_states[robot_name] = {
                "pos": robot_state.root_state[env_id, :3].cpu(),
                "rot": robot_state.root_state[env_id, 3:7].cpu(),
                "vel": robot_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": robot_state.root_state[env_id, 10:13].cpu(),
            }
            robot_states[robot_name]["dof_pos"] = _dof_tensor_to_dict(robot_state.joint_pos[env_id], jns)
            robot_states[robot_name]["dof_vel"] = _dof_tensor_to_dict(robot_state.joint_vel[env_id], jns)
            robot_states[robot_name]["dof_pos_target"] = (
                _dof_tensor_to_dict(robot_state.joint_pos_target[env_id], jns)
                if robot_state.joint_pos_target is not None
                else None
            )
            robot_states[robot_name]["dof_vel_target"] = (
                _dof_tensor_to_dict(robot_state.joint_vel_target[env_id], jns)
                if robot_state.joint_vel_target is not None
                else None
            )
            robot_states[robot_name]["dof_torque"] = (
                _dof_tensor_to_dict(robot_state.joint_effort_target[env_id], jns)
                if robot_state.joint_effort_target is not None
                else None
            )
            if robot_state.body_state is not None:
                bns = handler.get_body_names(robot_name)
                robot_states[robot_name]["body"] = _body_tensor_to_dict(robot_state.body_state[env_id], bns)

        camera_states = {}
        for camera_name, camera_state in tensor_state.cameras.items():
            cam_dict = {}
            if camera_state.rgb is not None:
                cam_dict["rgb"] = camera_state.rgb[env_id].cpu()
            if camera_state.depth is not None:
                cam_dict["depth"] = camera_state.depth[env_id].cpu()
            camera_states[camera_name] = cam_dict

        extra_states = {}
        if isinstance(tensor_state.extras, dict):
            for extra_key, extra_val in tensor_state.extras.items():
                if isinstance(extra_val, torch.Tensor):
                    extra_states[extra_key] = extra_val[env_id].cpu()

        env_state = {
            "objects": object_states,
            "robots": robot_states,
            "cameras": camera_states,
            "extras": extra_states,
        }

        env_states.append(env_state)
    return env_states


def _alloc_state_tensors(n_env: int, n_body: int | None = None, n_jnt: int | None = None, device="gpu"):
    root = torch.zeros((n_env, 13), device=device)

    n_body = n_body or 0
    body = torch.zeros((n_env, n_body, 13), device=device) if n_body else None

    n_jnt = n_jnt or 0
    jpos = torch.zeros((n_env, n_jnt), device=device) if n_jnt else None
    jvel = torch.zeros_like(jpos) if jpos is not None else None
    return root, body, jpos, jvel


def list_state_to_tensor(
    handler: BaseSimHandler,
    env_states: list[DictEnvState],
    device: torch.device | str = "cpu",
) -> TensorState:
    """Convert nested python list-states to a batched TensorState."""
    obj_names = sorted({n for es in env_states for n in es["objects"].keys()})
    robot_names = sorted({n for es in env_states for n in es["robots"].keys()})
    cam_names = sorted({n for es in env_states if "cameras" in es for n in es["cameras"].keys()})
    extra_names = sorted({n for es in env_states if "extras" in es for n in es["extras"].keys()})

    n_env = len(env_states)
    dev = device

    objects: dict[str, ObjectState] = {}
    robots: dict[str, RobotState] = {}
    cameras: dict[str, CameraState] = {}
    extras: dict[str, torch.Tensor] = {}

    # -------- objects --------------------------------------------------
    for name in obj_names:
        bnames = handler.get_body_names(name)
        jnames = handler.get_joint_names(name)

        root, body, jpos, jvel = _alloc_state_tensors(n_env, len(bnames) or None, len(jnames) or None, dev)

        for e, es in enumerate(env_states):
            if name not in es["objects"]:
                continue
            s = es["objects"][name]

            vel = s.get("vel", torch.zeros(3, device=dev))
            ang_vel = s.get("ang_vel", torch.zeros(3, device=dev))

            root[e, :3] = s["pos"]
            root[e, 3:7] = s["rot"]
            root[e, 7:10] = vel
            root[e, 10:13] = ang_vel

            if body is not None and "body" in s:
                for i, bn in enumerate(sorted(bnames)):
                    if bn not in s["body"]:
                        continue
                    bi = s["body"][bn]
                    body[e, i, :3], body[e, i, 3:7] = bi["pos"], bi["rot"]
                    body[e, i, 7:10], body[e, i, 10:13] = bi["vel"], bi["ang_vel"]

            if jpos is not None and "dof_pos" in s:
                for i, jn in enumerate(sorted(jnames)):
                    if jn in s["dof_pos"]:
                        jpos[e, i] = s["dof_pos"][jn]
            if jvel is not None and "dof_vel" in s:
                for i, jn in enumerate(sorted(jnames)):
                    if jn in s["dof_vel"]:
                        jvel[e, i] = s["dof_vel"][jn]

        objects[name] = ObjectState(root_state=root, body_state=body, joint_pos=jpos, joint_vel=jvel)

    # -------- robots ---------------------------------------------------
    for name in robot_names:
        jnames = handler.get_joint_names(name)
        bnames = handler.get_body_names(name)

        root, body, jpos, jvel = _alloc_state_tensors(n_env, len(bnames) or None, len(jnames) or None, dev)
        jpos_t, jvel_t, jeff_t = (
            torch.zeros_like(jpos) if jpos is not None else None,
            torch.zeros_like(jvel) if jvel is not None else None,
            torch.zeros_like(jvel) if jvel is not None else None,
        )

        for e, es in enumerate(env_states):
            if name not in es["robots"]:
                continue
            s = es["robots"][name]

            pos = s["pos"]
            rot = s["rot"]
            vel = s.get("vel", torch.zeros(3, device=dev))
            ang_vel = s.get("ang_vel", torch.zeros(3, device=dev))

            root[e, :3] = pos
            root[e, 3:7] = rot
            root[e, 7:10] = vel
            root[e, 10:13] = ang_vel
            for i, jn in enumerate(sorted(jnames)):
                if "dof_pos" in s and s["dof_pos"] is not None and jn in s["dof_pos"]:
                    jpos[e, i] = s["dof_pos"][jn]
                if "dof_vel" in s and s["dof_vel"] is not None and jn in s["dof_vel"]:
                    jvel[e, i] = s["dof_vel"][jn]
                if "dof_pos_target" in s and s["dof_pos_target"] is not None and jn in s["dof_pos_target"]:
                    jpos_t[e, i] = s["dof_pos_target"][jn]
                if "dof_vel_target" in s and s["dof_vel_target"] is not None and jn in s["dof_vel_target"]:
                    jvel_t[e, i] = s["dof_vel_target"][jn]
                if "dof_torque" in s and s["dof_torque"] is not None and jn in s["dof_torque"]:
                    jeff_t[e, i] = s["dof_torque"][jn]

            if body is not None and "body" in s:
                for i, bn in enumerate(sorted(bnames)):
                    if bn not in s["body"]:
                        continue
                    bi = s["body"][bn]
                    body[e, i, :3], body[e, i, 3:7], body[e, i, 7:10], body[e, i, 10:13] = (
                        bi["pos"],
                        bi["rot"],
                        bi["vel"],
                        bi["ang_vel"],
                    )

        robots[name] = RobotState(
            root_state=root,
            body_names=bnames,
            body_state=body,
            joint_pos=jpos,
            joint_vel=jvel,
            joint_pos_target=jpos_t,
            joint_vel_target=jvel_t,
            joint_effort_target=jeff_t,
        )

    # -------- cameras ---------------------------------------------
    for cam in cam_names:
        rgb = torch.stack(
            [es["cameras"][cam]["rgb"] for es in env_states if "cameras" in es and cam in es["cameras"]], dim=0
        ).to(dev)
        depth = torch.stack(
            [es["cameras"][cam]["depth"] for es in env_states if "cameras" in es and cam in es["cameras"]], dim=0
        ).to(dev)
        cameras[cam] = CameraState(rgb=rgb, depth=depth)

    # -------- extras ----------------------------------------------
    for extra_key in extra_names:
        extra_vec = torch.stack(
            [es["extras"][extra_key] for es in env_states if "extras" in es and extra_key in es["extras"]], dim=0
        ).to(dev)
        extras[extra_key] = extra_vec

    return TensorState(objects=objects, robots=robots, cameras=cameras, extras=extras)


def adapt_actions_to_dict(
    handler: BaseSimHandler, actions: list[Action] | TensorState
) -> dict[str, dict[str, dict[str, float]]]:
    """Adapt actions to the format of single env handlers.

    Args:
        handler: The handler of the simulation.
        actions: The actions to adapt.
    """
    if isinstance(actions, torch.Tensor):
        if len(actions.shape) == 2:
            actions = actions[0]
        actions = {
            handler.robot.name: {
                "dof_pos_target": _dof_tensor_to_dict(actions, handler.get_joint_names(handler.robot.name))
            }
        }
    elif isinstance(actions, np.ndarray):
        if len(actions.shape) == 2:
            actions = actions[0]
        actions = {
            handler.robot.name: {
                "dof_pos_target": _dof_array_to_dict(actions, handler.get_joint_names(handler.robot.name))
            }
        }
    elif isinstance(actions, list):
        actions = actions[0]
    return actions
