"""Tensorized state of the simulation."""

from __future__ import annotations

from itertools import chain

import numpy as np
import torch

from metasim.types import (
    Action,
    ActionBatch,
    CameraState,
    CompatActionInput,
    DictEnvState,
    ObjectState,
    RobotState,
    TensorState,
)

try:
    from metasim.sim.base import BaseSimHandler
except ImportError:
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
                # Segmentation was silently dropped here, so a backend that
                # populates it (e.g. isaacsim) lost it the moment the state
                # passed through a parallel join. The id2label maps are per-env
                # identical (same label space across envs), so take the first.
                instance_id_seg=torch.cat([cam.instance_id_seg for cam in camera_states], dim=0)
                if camera_states[0].instance_id_seg is not None
                else None,
                instance_id_seg_id2label=camera_states[0].instance_id_seg_id2label,
                instance_seg=torch.cat([cam.instance_seg for cam in camera_states], dim=0)
                if camera_states[0].instance_seg is not None
                else None,
                instance_seg_id2label=camera_states[0].instance_seg_id2label,
            )

    # Join sensors (assuming similar structure to objects)
    # for key in all_sensor_keys:
    #     sensor_states = [state.sensors[key] for state in tensor_states if key in state.sensors]
    #     if sensor_states:
    #         # Note: SensorState structure is not defined, so this is a placeholder
    #         rst.sensors[key] = sensor_states[0]  # This would need to be implemented based on SensorState structure

    # Join extras. Only emit a key when EVERY joined state provides it as a
    # tensor, so the concatenated length equals total num_envs (matching the
    # objects/robots/cameras concat along dim 0) and satisfies
    # TensorState.__post_init__'s extras[key].shape[0] == num_envs check. Without
    # this, parallel get_states().extras was always empty while single-env
    # backends populate it. Non-tensor extras (e.g. mjx 'sites' nested dicts,
    # ContactForces query objects) are dropped here, exactly as the dict-mode
    # round-trip already drops them in state_tensor_to_nested / list_state_to_tensor.
    all_extra_keys: set[str] = set()
    for state in tensor_states:
        if isinstance(state.extras, dict):
            all_extra_keys.update(state.extras.keys())

    for key in all_extra_keys:
        values = [
            state.extras[key] for state in tensor_states if isinstance(state.extras, dict) and key in state.extras
        ]
        if len(values) == len(tensor_states) and all(isinstance(v, torch.Tensor) for v in values):
            rst.extras[key] = torch.cat(values, dim=0)

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


def _warn_action_input_drops_non_position(handler: BaseSimHandler, actions: list) -> None:
    """Warn once when a dict action carries non-position targets dropped here.

    ``action_input_to_tensor`` reads only ``dof_pos_target``; other target
    keys are silently dropped. Dedupe lives on the handler instance via
    ``_action_input_warned_keys`` so hot-path rollouts (one warning per
    (robot, dropped-key) per Python process) don't spam. Tensor / ndarray
    actions skip this path entirely.
    """
    from loguru import logger as _log

    NON_POSITION_KEYS = ("dof_vel_target", "dof_effort_target", "dof_torque")
    seen: set = getattr(handler, "_action_input_warned_keys", set())
    for env_action in actions:
        if not isinstance(env_action, dict):
            continue
        for robot_name, robot_action in env_action.items():
            if not isinstance(robot_action, dict):
                continue
            for key in NON_POSITION_KEYS:
                if robot_action.get(key):
                    cache_key = (robot_name, key)
                    if cache_key in seen:
                        continue
                    seen.add(cache_key)
                    _log.warning(
                        f"action_input_to_tensor: dict action for robot '{robot_name}' includes "
                        f"'{key}' — this helper reads only 'dof_pos_target' and silently drops "
                        f"vel/effort targets. If you need non-position semantics, bypass this "
                        f"helper or pass actions to set_dof_targets directly without conversion."
                    )
    try:
        handler._action_input_warned_keys = seen
    except AttributeError:
        pass


def action_input_to_tensor(
    handler: BaseSimHandler, actions: CompatActionInput, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Normalize supported action inputs into a batched position-target tensor in handler/API order.

    For dict-batch inputs, this helper reads only ``dof_pos_target`` and ignores
    ``dof_vel_target`` / ``dof_effort_target``. Backends that need non-position
    semantics on dict actions must handle that path explicitly instead of routing
    those actions through this helper.

    Warns once per (robot, key) when dict actions contain ``dof_vel_target`` /
    ``dof_effort_target`` — those are silently dropped here, which previously
    looked like the actions were applied to every backend that goes through
    this helper (mujoco / pyrep). Surface the drop so callers can either
    bypass this helper or accept the position-only semantics intentionally.
    """
    if isinstance(actions, torch.Tensor):
        action_tensor = actions.to(device=device, dtype=torch.float32)
    elif isinstance(actions, np.ndarray):
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    elif isinstance(actions, list):
        joint_names_by_robot = {robot.name: handler.get_joint_names(robot.name, sort=True) for robot in handler.robots}
        action_dim = sum(len(joint_names) for joint_names in joint_names_by_robot.values())
        action_tensor = torch.zeros((len(actions), action_dim), dtype=torch.float32, device=device)
        _warn_action_input_drops_non_position(handler, actions)

        for env_id, action in enumerate(actions):
            offset = 0
            for robot in handler.robots:
                joint_names = joint_names_by_robot[robot.name]
                joint_targets = (action.get(robot.name) or {}).get("dof_pos_target") or {}
                for joint_id, joint_name in enumerate(joint_names):
                    if joint_name in joint_targets:
                        action_tensor[env_id, offset + joint_id] = float(joint_targets[joint_name])
                offset += len(joint_names)
    else:
        raise TypeError(f"Unsupported action type: {type(actions)!r}")

    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    elif action_tensor.ndim != 2:
        raise ValueError(f"Expected actions with rank 1 or 2, got shape {tuple(action_tensor.shape)}")

    expected_dim = sum(len(handler.get_joint_names(robot.name, sort=True)) for robot in handler.robots)
    if action_tensor.shape[1] != expected_dim:
        raise ValueError(f"Expected action width {expected_dim}, got {action_tensor.shape[1]}.")

    return action_tensor


def action_input_to_dict_batch(handler: BaseSimHandler, actions: CompatActionInput) -> ActionBatch:
    """Normalize supported action inputs into a batched list of dict actions."""
    if isinstance(actions, list):
        return actions

    action_tensor = action_input_to_tensor(handler, actions, device="cpu").detach().cpu()
    action_batch: ActionBatch = []

    for env_id in range(action_tensor.shape[0]):
        env_action: Action = {}
        offset = 0
        for robot in handler.robots:
            joint_names = handler.get_joint_names(robot.name, sort=True)
            env_action[robot.name] = {
                "dof_pos_target": _dof_tensor_to_dict(
                    action_tensor[env_id, offset : offset + len(joint_names)], joint_names
                )
            }
            offset += len(joint_names)
        action_batch.append(env_action)

    return action_batch


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


def select_envs(state: TensorState, env_ids: list[int]) -> TensorState:
    """Rows ``env_ids`` of every per-env tensor in ``state`` (a new ``TensorState``; names are shared)."""
    import dataclasses

    idx = torch.as_tensor(list(env_ids), dtype=torch.long)

    def take(value):
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > int(idx.max()):
            return value[idx.to(value.device)]
        return value

    def sub(obj):
        if obj is None or not dataclasses.is_dataclass(obj):
            return obj
        return dataclasses.replace(obj, **{f.name: take(getattr(obj, f.name)) for f in dataclasses.fields(obj)})

    return TensorState(
        objects={k: sub(v) for k, v in state.objects.items()},
        robots={k: sub(v) for k, v in state.robots.items()},
        cameras={k: sub(v) for k, v in state.cameras.items()},
        extras={k: take(v) for k, v in (state.extras or {}).items()},
    )


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


def _alloc_state_tensors(n_env: int, n_body: int | None = None, n_jnt: int | None = None, device="cpu"):
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

        objects[name] = ObjectState(root_state=root, body_names=bnames, body_state=body, joint_pos=jpos, joint_vel=jvel)

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
        # The write side (state_tensor_to_nested) only emits "rgb"/"depth" when
        # the corresponding tensor is non-None, so a depth-only or rgb-only
        # camera dict lacks the other key. Index each only when present and pass
        # None otherwise (CameraState accepts None); the both-present path is
        # unchanged.
        cam_dicts = [es["cameras"][cam] for es in env_states if "cameras" in es and cam in es["cameras"]]
        rgb = (
            torch.stack([cd["rgb"] for cd in cam_dicts], dim=0).to(dev) if cam_dicts and "rgb" in cam_dicts[0] else None
        )
        depth = (
            torch.stack([cd["depth"] for cd in cam_dicts], dim=0).to(dev)
            if cam_dicts and "depth" in cam_dicts[0]
            else None
        )
        cameras[cam] = CameraState(rgb=rgb, depth=depth)

    # -------- extras ----------------------------------------------
    for extra_key in extra_names:
        extra_vec = torch.stack(
            [es["extras"][extra_key] for es in env_states if "extras" in es and extra_key in es["extras"]], dim=0
        ).to(dev)
        extras[extra_key] = extra_vec

    return TensorState(objects=objects, robots=robots, cameras=cameras, extras=extras)


def adapt_actions_to_dict(
    handler: BaseSimHandler, actions: CompatActionInput
) -> dict[str, dict[str, dict[str, float]]]:
    """Adapt actions to the format of single env handlers.

    Args:
        handler: The handler of the simulation.
        actions: The actions to adapt.
    """
    action_batch = action_input_to_dict_batch(handler, actions)
    return action_batch[0] if action_batch else {}
