from __future__ import annotations

import importlib.util
import logging
import os
from copy import deepcopy
from pathlib import Path

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.ik_solver import IKSolver
from metasim.utils.math import quat_apply
from roboverse_pack.tasks.pick_place.utils import Utils

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_franka_ik_cfg(robot_cfg, ee_link: str):
    """Build IK config for Franka. IK controls only panda_joint1-7."""
    cfg = deepcopy(robot_cfg)
    cfg.ee_body_name = str(ee_link)

    act_keys = list(getattr(robot_cfg, "actuators", {}).keys())
    if not act_keys:
        return robot_cfg

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    desired_order = arm_joint_names + gripper_joint_names

    if set(act_keys) == set(desired_order):
        cfg.actuators = {jn: cfg.actuators[jn] for jn in desired_order if jn in cfg.actuators}
        cfg.control_type = {jn: cfg.control_type[jn] for jn in desired_order if jn in cfg.control_type}
        cfg.default_joint_positions = {
            jn: cfg.default_joint_positions[jn] for jn in desired_order if jn in cfg.default_joint_positions
        }
        cfg.joint_limits = {jn: cfg.joint_limits[jn] for jn in desired_order if jn in cfg.joint_limits}

    return cfg


@register_task("pick_place.approach", "approach", "pick_place.approach_rand_franka", "approach_rand_franka")
class ApproachRandTaskFranka(RLTaskEnv):
    """Franka approach task."""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bbq_sauce",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq.usda",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            RigidObjCfg(
                name="basket",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/mjcf/basket.xml",
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    # Saved pose source used for:
    # - initial object/robot placement (via `_get_initial_states`)
    # - IK seeding (`_saved_pose_dof_pos`)
    #
    # You can override it at runtime with:
    #   export APPROACH_SAVED_POSE_PY_PATH=/abs/path/to/saved_poses_xxx.py
    DEFAULT_SAVED_POSE_PY_PATH = str(REPO_ROOT / "get_started/output/saved_poses_20260103_175422.py")
    SAVED_POSE_PY_PATH = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", DEFAULT_SAVED_POSE_PY_PATH)
    _SAVED_POSE_PY_PATH_RESOLVED: str | None = None
    _printed_pose_source = False

    EPISODE_HORIZON = 100
    IK_SETTLE_STEPS = 30
    max_episode_steps = EPISODE_HORIZON + IK_SETTLE_STEPS

    IK_BACKEND = "pyroki"
    IK_NUM_SEEDS = 5
    RESIDUAL_STEP_FRAC_OF_RANGE = 0.20
    RESIDUAL_MAX_FRAC_OF_RANGE = 0.05

    BBQ_X_MIN = 0.6
    BBQ_X_MAX = 0.8
    BBQ_Y_RANGE = 0.10
    BBQ_XY_ROT_RANGE_RAD = 1.0
    BASKET_XY_RANGE_M = 0.05

    OTHER_OBJ_MAX_MOVE_M = 0.02
    IGNORE_MOVE_PREFIXES: tuple[str, ...] = ()
    IGNORE_MOVE_NAMES = ("hand_debug_marker",)
    TERMINATION_GRACE_STEPS = 30

    K_REL = 1.0
    K_TANH = 1.0
    W_EXP = 0.2
    W_TANH = 0.2
    K_REL_ROT = 3.0
    K_TANH_ROT = 3.0
    W_EXP_ROT = 0.5
    W_TANH_ROT = 0.5
    W_TIMEOUT_BONUS = 100.0
    K_TIMEOUT_BONUS = 5.0
    INIT_POSE_BONUS_MAX = 20.0
    INIT_POSE_K_POS = 50.0
    INIT_POSE_K_ROT = 2.0

    GRIPPER_CLOSE_FRAMES = 10
    GRIPPER_CLOSE_VALUE = 0.0
    GRIPPER_OPEN_VALUE = 0.04

    @classmethod
    def _load_saved_pose_py(cls) -> dict:
        pose_path = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", cls.SAVED_POSE_PY_PATH)
        p = Path(pose_path)
        if not p.exists():
            raise FileNotFoundError(f"SAVED_POSE_PY_PATH not found: {p}")
        cls._SAVED_POSE_PY_PATH_RESOLVED = str(p)
        mod_name = f"saved_pose_module_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import saved pose module from: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        if not hasattr(mod, "poses"):
            raise ValueError(f"Saved pose module has no `poses` variable: {p}")
        poses = mod.poses
        if not isinstance(poses, dict):
            raise ValueError(f"`poses` must be a dict in: {p}")
        return poses

    def _get_initial_states(self) -> list[dict] | None:
        """Load initial states from saved pose file."""
        poses = self._load_saved_pose_py()
        obj_name = self.object_name
        basket_name = self.basket_name
        robot_name = "franka"
        if "objects" not in poses or "robots" not in poses:
            raise ValueError("Saved pose format error: missing 'objects' or 'robots'.")
        if obj_name not in poses["objects"]:
            raise ValueError(f"Saved pose missing object '{obj_name}'. Keys: {list(poses['objects'].keys())}")
        if basket_name not in poses["objects"]:
            raise ValueError(f"Saved pose missing object '{basket_name}'. Keys: {list(poses['objects'].keys())}")
        if robot_name not in poses["robots"]:
            raise ValueError(f"Saved pose missing robot '{robot_name}'. Keys: {list(poses['robots'].keys())}")

        robot_pos = torch.as_tensor(poses["robots"][robot_name]["pos"], dtype=torch.float32)
        robot_rot = torch.as_tensor(poses["robots"][robot_name]["rot"], dtype=torch.float32)

        obj_pos = torch.as_tensor(poses["objects"][obj_name]["pos"], dtype=torch.float32)
        obj_rot = torch.as_tensor(poses["objects"][obj_name]["rot"], dtype=torch.float32)
        basket_pos = torch.as_tensor(poses["objects"][basket_name]["pos"], dtype=torch.float32)
        basket_rot = torch.as_tensor(poses["objects"][basket_name]["rot"], dtype=torch.float32)
        dof_pos = dict(poses["robots"][robot_name].get("dof_pos", {}))

        init_one = {"objects": {}, "robots": {}}
        init_one["objects"][obj_name] = {"pos": obj_pos, "rot": obj_rot}
        init_one["objects"][basket_name] = {"pos": basket_pos, "rot": basket_rot}

        init_one["robots"][robot_name] = {"pos": robot_pos, "rot": robot_rot, "dof_pos": dof_pos}
        if not self._printed_pose_source:
            logger.info(
                "[approach_rand_franka] SAVED_POSE_PY_PATH=%s, obj_pos=%s",
                self._SAVED_POSE_PY_PATH_RESOLVED or self.SAVED_POSE_PY_PATH,
                obj_pos.tolist(),
            )
            self._printed_pose_source = True
        return [init_one for _ in range(self.num_envs)]

    def __init__(self, scenario, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_envs = int(scenario.num_envs)
        self.object_name = "bbq_sauce"
        self.basket_name = "basket"

        self._target_rel_offset_obj = torch.zeros(3, dtype=torch.float32)
        self._target_rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        self._fixed_rand_ready = False
        self._fixed_obj_delta3 = None
        self._fixed_basket_delta3 = None
        self._ik_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ik_target_joint_pos_env = None
        try:
            robot_cfg_for_ik = _build_franka_ik_cfg(scenario.robots[0], ee_link="panda_hand")
            self._ik_solver = IKSolver(robot_cfg_for_ik, solver=self.IK_BACKEND, use_seed=True)
            self._ik_joint_names = list(getattr(robot_cfg_for_ik, "actuators", {}).keys())
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IK solver backend={self.IK_BACKEND}: {e}") from e
        self._saved_pose_dof_pos = {}

        self._disable_dr = True
        self.robot_name = scenario.robots[0].name
        self._last_action = None
        super().__init__(scenario, device=device)

        self.max_episode_steps = int(self.EPISODE_HORIZON + self.IK_SETTLE_STEPS)

        states = self.handler.get_states()
        hand_pos, hand_quat = self._get_ee_state(states)
        obj_pos = states.objects[self.object_name].root_state[:, 0:3]
        obj_quat = states.objects[self.object_name].root_state[:, 3:7]
        rel_pos0_world = (hand_pos[0] - obj_pos[0]).detach().clone().to(self.device)
        qo0 = obj_quat[0].detach().clone().to(self.device)
        qo0 = qo0 / torch.norm(qo0).clamp(min=1e-9)
        self._target_rel_offset_obj = Utils.quat_rotate(Utils.quat_conjugate(qo0), rel_pos0_world)
        qh0 = hand_quat[0].detach().clone().to(self.device)
        qh0 = qh0 / torch.norm(qh0).clamp(min=1e-9)
        self._target_rel_quat = Utils.quat_mul(qh0, Utils.quat_conjugate(qo0))
        self._target_rel_quat = self._target_rel_quat / torch.norm(self._target_rel_quat).clamp(min=1e-9)
        try:
            poses = self._load_saved_pose_py()
            self._saved_pose_dof_pos = dict(poses.get("robots", {}).get(self.robot_name, {}).get("dof_pos", {}) or {})
        except Exception:
            self._saved_pose_dof_pos = {}

        self._disable_dr = False
        self.reset(env_ids=list(range(self.num_envs)))

    def _prepare_states(self, env_states, env_ids):
        """Apply domain randomization during reset."""
        if self._disable_dr:
            self._ensure_rand_buffers()
            return deepcopy(env_states)

        self._ensure_rand_buffers()
        states = deepcopy(env_states)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        base_pos = self._base_object_init_pos.to(self.device)
        base_quat = self._base_object_init_quat.to(self.device)
        n = int(env_ids_t.numel())
        self._ensure_fixed_randomization()
        delta3 = self._fixed_obj_delta3.index_select(0, env_ids_t).to(self.device)
        dq = self._fixed_obj_delta_quat.index_select(0, env_ids_t).to(self.device)

        new_obj_pos = base_pos.unsqueeze(0) + delta3
        new_obj_quat = Utils.quat_mul(dq.to(dtype=base_quat.dtype), base_quat.unsqueeze(0).expand(n, -1))
        new_obj_quat = new_obj_quat / torch.norm(new_obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        obj_state = states.objects[self.object_name].root_state
        zero_vel = torch.zeros(n, 3, device=self.device, dtype=obj_state.dtype)
        zero_ang = torch.zeros(n, 3, device=self.device, dtype=obj_state.dtype)
        new_obj_root = torch.cat(
            [new_obj_pos.to(dtype=obj_state.dtype), new_obj_quat.to(dtype=obj_state.dtype), zero_vel, zero_ang], dim=-1
        )
        obj_state.index_copy_(0, env_ids_t, new_obj_root)

        self._object_init_pos_env.index_copy_(0, env_ids_t, new_obj_pos.to(dtype=self._object_init_pos_env.dtype))
        self._object_init_quat_env.index_copy_(0, env_ids_t, new_obj_quat.to(dtype=self._object_init_quat_env.dtype))

        base_basket_pos = self._base_basket_init_pos.to(self.device)
        base_basket_quat = self._base_basket_init_quat.to(self.device)
        basket_delta3 = self._fixed_basket_delta3.index_select(0, env_ids_t).to(self.device)

        new_basket_pos = base_basket_pos.unsqueeze(0) + basket_delta3
        basket_state = states.objects[self.basket_name].root_state
        basket_zero_vel = torch.zeros(n, 3, device=self.device, dtype=basket_state.dtype)
        basket_zero_ang = torch.zeros(n, 3, device=self.device, dtype=basket_state.dtype)
        new_basket_root = torch.cat(
            [
                new_basket_pos.to(dtype=basket_state.dtype),
                base_basket_quat.unsqueeze(0).expand(n, -1).to(dtype=basket_state.dtype),
                basket_zero_vel,
                basket_zero_ang,
            ],
            dim=-1,
        )
        basket_state.index_copy_(0, env_ids_t, new_basket_root)

        self._apply_robot_reset_joint_policy(states, env_ids_t)

        return states

    def reset(self, env_ids=None):
        """Reset and sync internal buffers to randomized states."""
        obs, info = RLTaskEnv.reset(self, env_ids=env_ids)
        self._ensure_rand_buffers()

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        st = self.handler.get_states()
        joint_pos = st.robots[self.robot_name].joint_pos.to(self.device).clone()

        if self._last_action is None:
            self._last_action = joint_pos.clone()
        else:
            self._last_action.index_copy_(0, env_ids_t, joint_pos.index_select(0, env_ids_t))

        if self._episode_start_joint_pos is None:
            self._episode_start_joint_pos = joint_pos.clone()
        else:
            self._episode_start_joint_pos.index_copy_(0, env_ids_t, joint_pos.index_select(0, env_ids_t))

        if self._residual_offset is None:
            self._residual_offset = torch.zeros_like(joint_pos)
        else:
            self._residual_offset.index_fill_(0, env_ids_t, 0.0)

        if self._move_check_ready is not None:
            self._move_check_ready.index_fill_(0, env_ids_t, False)

        if not self._disable_dr:
            self._maybe_solve_ik_for_envs(env_ids_t)

        return obs, info

    def step(self, actions):
        """Delta-control with buffers consistent with randomized auto-reset."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        self._ensure_rand_buffers()

        settle = self._episode_steps >= int(self.EPISODE_HORIZON)
        if torch.any(settle) and self._ik_target_joint_pos_env is not None:
            direct_actions = self._ik_target_joint_pos_env.to(self.device, dtype=actions.dtype)
            action_low = self._action_low
            action_high = self._action_high
            if action_low.ndim == 1:
                action_low = action_low.unsqueeze(0)
            if action_high.ndim == 1:
                action_high = action_high.unsqueeze(0)
            direct_actions = torch.clamp(direct_actions, action_low, action_high)
            close_gripper_mask = settle | (self._episode_steps >= (self.max_episode_steps - self.GRIPPER_CLOSE_FRAMES))
            self._apply_gripper_control(direct_actions, close_gripper_mask)
            obs, reward, terminated, time_out, info = RLTaskEnv.step(self, direct_actions)
            states = self.handler.get_states()
            self._last_action = states.robots[self.robot_name].joint_pos.clone()
            return obs, reward, terminated, time_out, info

        H = int(self.EPISODE_HORIZON)
        t = self._episode_steps.to(device=self.device, dtype=torch.float32).clamp(min=0.0, max=float(max(0, H - 1)))
        alpha1 = ((t + 1.0) / float(max(1, H))).unsqueeze(-1)

        if self._episode_start_joint_pos is None:
            st = self.handler.get_states()
            self._episode_start_joint_pos = st.robots[self.robot_name].joint_pos.to(self.device).clone()
        q0 = self._episode_start_joint_pos.to(self.device)
        qT = (
            self._ik_target_joint_pos_env.to(self.device).to(dtype=q0.dtype)
            if self._ik_target_joint_pos_env is not None
            else q0
        )

        q_ref1 = q0 + alpha1 * (qT - q0)

        if self._residual_offset is None:
            self._residual_offset = torch.zeros_like(q_ref1)
        r = self._residual_offset.to(self.device)

        a = torch.clamp(actions, -1.0, 1.0)
        span = (self._action_high - self._action_low).to(self.device)
        eps = float(self.RESIDUAL_STEP_FRAC_OF_RANGE) * span
        r = r + eps.unsqueeze(0) * a
        r_max = float(self.RESIDUAL_MAX_FRAC_OF_RANGE) * span
        r = torch.clamp(r, -r_max.unsqueeze(0), r_max.unsqueeze(0))
        self._residual_offset = r.detach().clone()

        target_actions = q_ref1 + r

        action_low = self._action_low
        action_high = self._action_high
        if action_low.ndim == 1:
            action_low = action_low.unsqueeze(0)
        if action_high.ndim == 1:
            action_high = action_high.unsqueeze(0)
        clamped_actions = torch.clamp(target_actions, action_low, action_high)

        close_gripper = self._episode_steps >= (self.max_episode_steps - self.GRIPPER_CLOSE_FRAMES)
        self._apply_gripper_control(clamped_actions, close_gripper)

        obs, reward, terminated, time_out, info = RLTaskEnv.step(self, clamped_actions)
        states = self.handler.get_states()
        self._last_action = states.robots[self.robot_name].joint_pos.clone()
        return obs, reward, terminated, time_out, info

    def _apply_gripper_control(self, actions: torch.Tensor, mask: torch.Tensor) -> None:
        """Close gripper when mask is True."""
        if not hasattr(self, "_gripper_joint_indices"):
            joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
            self._gripper_joint_indices = [i for i, name in enumerate(joint_names) if "finger" in name.lower()]

        if len(self._gripper_joint_indices) >= 2:
            actions[mask, self._gripper_joint_indices[0]] = self.GRIPPER_CLOSE_VALUE
            actions[mask, self._gripper_joint_indices[1]] = self.GRIPPER_CLOSE_VALUE
        elif len(self._gripper_joint_indices) == 1:
            actions[mask, self._gripper_joint_indices[0]] = self.GRIPPER_CLOSE_VALUE

    def _reward(self, env_states) -> torch.Tensor:
        """Reward: reach saved relative pose + keep object near its per-env initial pose."""
        hand_pos, hand_quat = self._get_ee_state(env_states)  # (B, 3), (B, 4)
        obj_state = env_states.objects[self.object_name].root_state
        obj_pos = obj_state[:, 0:3]
        obj_quat = obj_state[:, 3:7]
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        rel = hand_pos - obj_pos
        rel_off_obj = self._target_rel_offset_obj.to(self.device).unsqueeze(0).expand_as(rel)
        target = Utils.quat_rotate(obj_quat, rel_off_obj)
        err_pos = torch.norm(rel - target, dim=-1)

        r_exp_pos = torch.exp(-float(self.K_REL) * err_pos)
        r_tanh_pos = 1.0 - torch.tanh(float(self.K_TANH) * err_pos)
        reward = float(self.W_EXP) * r_exp_pos + float(self.W_TANH) * r_tanh_pos

        timeout_mask = (self._episode_steps >= int(self.max_episode_steps)).to(dtype=torch.float32, device=self.device)
        terminal_bonus_pos = float(self.W_TIMEOUT_BONUS) * torch.exp(-5 * err_pos)

        hand_quat = hand_quat / torch.norm(hand_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        rel_quat = Utils.quat_mul(hand_quat, Utils.quat_conjugate(obj_quat))
        rel_quat = rel_quat / torch.norm(rel_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        target_rel = self._target_rel_quat.to(self.device).unsqueeze(0).expand_as(rel_quat)
        diff = Utils.quat_mul(target_rel, Utils.quat_conjugate(rel_quat))
        err_rot = Utils.quat_angle(diff)

        terminal_bonus_rot = 30.0 * torch.exp(-5 * err_rot)
        terminal_bonus = terminal_bonus_pos + terminal_bonus_rot
        reward = reward + timeout_mask * terminal_bonus

        self._ensure_rand_buffers()
        pos0 = self._object_init_pos_env.to(self.device, dtype=obj_pos.dtype)
        quat0 = self._object_init_quat_env.to(self.device, dtype=obj_quat.dtype)
        quat0 = quat0 / torch.norm(quat0, dim=-1, keepdim=True).clamp(min=1e-9)

        pos_err0 = torch.norm(obj_pos - pos0, dim=-1)
        dq0 = Utils.quat_mul(obj_quat, Utils.quat_conjugate(quat0))
        rot_err0 = Utils.quat_angle(dq0)

        bonus = (
            float(self.INIT_POSE_BONUS_MAX)
            * torch.exp(-float(self.INIT_POSE_K_POS) * pos_err0)
            * torch.exp(-float(self.INIT_POSE_K_ROT) * rot_err0)
        )
        bonus = torch.clamp(bonus, 0.0, float(self.INIT_POSE_BONUS_MAX))
        term_mask = self._terminated(env_states).to(dtype=torch.float32, device=self.device)
        done_mask = torch.clamp(timeout_mask + term_mask, 0.0, 1.0)
        reward = reward + done_mask * bonus

        return reward

    def _terminated(self, env_states) -> torch.Tensor:
        """Terminate if any checked object moves beyond tolerance."""
        self._ensure_rand_buffers()
        term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        grace = int(self.TERMINATION_GRACE_STEPS)
        steps = self._episode_steps.to(device=self.device, dtype=torch.long)

        need_snapshot = (~self._move_check_ready) & (steps >= grace)
        if torch.any(need_snapshot):
            env_ids_t = torch.nonzero(need_snapshot, as_tuple=False).squeeze(-1)
            for name, pos0 in self._obj_pos0_for_move_check.items():
                if name not in env_states.objects:
                    continue
                pos = env_states.objects[name].root_state[:, 0:3].to(self.device)
                pos0.index_copy_(0, env_ids_t, pos.index_select(0, env_ids_t))
            self._move_check_ready.index_fill_(0, env_ids_t, True)

        ready = self._move_check_ready & (steps > grace)
        if torch.any(ready):
            env_ids_ready = torch.nonzero(ready, as_tuple=False).squeeze(-1)
            thr = float(self.OTHER_OBJ_MAX_MOVE_M)
            for name, pos0 in self._obj_pos0_for_move_check.items():
                if name not in env_states.objects:
                    continue
                pos = env_states.objects[name].root_state[:, 0:3].to(self.device)
                dist = torch.norm(pos.index_select(0, env_ids_ready) - pos0.index_select(0, env_ids_ready), dim=-1)
                moved = dist > thr
                if torch.any(moved):
                    bad_local = torch.nonzero(moved, as_tuple=False).squeeze(-1)
                    bad_envs = env_ids_ready.index_select(0, bad_local)
                    term.index_fill_(0, bad_envs, True)

        if grace > 0:
            term = term & (~(steps < grace))

        return term

    def _observation(self, env_states) -> torch.Tensor:
        """Observation: joint states, hand-object relative position and error."""
        rs = env_states.robots[self.robot_name]
        joint_pos = rs.joint_pos.to(self.device)  # (B, J)
        joint_vel = getattr(rs, "joint_vel", None)
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        else:
            joint_vel = joint_vel.to(self.device)

        obj_state = env_states.objects[self.object_name].root_state.to(self.device)  # (B, 13)
        obj_pos = obj_state[:, 0:3]
        obj_quat = obj_state[:, 3:7]
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        hand_pos, _ = self._get_ee_state(env_states)  # (B, 3)
        hand_pos = hand_pos.to(self.device)

        rel_pos = hand_pos - obj_pos  # (B, 3)
        rel_off_obj = self._target_rel_offset_obj.to(self.device).unsqueeze(0).expand_as(rel_pos)
        target_rel_pos = Utils.quat_rotate(obj_quat, rel_off_obj)  # (B, 3)
        rel_pos_err = rel_pos - target_rel_pos  # (B, 3)

        obs = torch.cat(
            [
                joint_pos,
                joint_vel,
                rel_pos,
                rel_pos_err,
            ],
            dim=-1,
        )
        return obs

    def _ensure_rand_buffers(self) -> None:
        """Lazy-init buffers needed by DR + IK."""
        if getattr(self, "_rand_buffers_ready", False):
            return

        self._action_joint_names = list(getattr(self, "joint_names", self.handler.get_joint_names(self.robot_name)))
        self._action_name_to_idx = {n: i for i, n in enumerate(self._action_joint_names)}

        base_obj = self._initial_states.objects[self.object_name].root_state[0]
        self._base_object_init_pos = base_obj[0:3].detach().clone().to(self.device)
        self._base_object_init_quat = base_obj[3:7].detach().clone().to(self.device)
        self._base_object_init_quat = self._base_object_init_quat / torch.norm(self._base_object_init_quat).clamp(
            min=1e-9
        )

        self._object_init_pos_env = self._base_object_init_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._object_init_quat_env = self._base_object_init_quat.unsqueeze(0).repeat(self.num_envs, 1)

        base_basket = self._initial_states.objects[self.basket_name].root_state[0]
        self._base_basket_init_pos = base_basket[0:3].detach().clone().to(self.device)
        self._base_basket_init_quat = base_basket[3:7].detach().clone().to(self.device)
        self._base_basket_init_quat = self._base_basket_init_quat / torch.norm(self._base_basket_init_quat).clamp(
            min=1e-9
        )

        self._episode_start_joint_pos = None
        self._residual_offset = None

        self._move_check_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obj_pos0_for_move_check = {}
        for oname in list[str](self._initial_states.objects.keys()):
            if any(oname.startswith(p) for p in self.IGNORE_MOVE_PREFIXES):
                continue
            if oname in self.IGNORE_MOVE_NAMES:
                continue
            self._obj_pos0_for_move_check[oname] = torch.zeros(
                self.num_envs, 3, dtype=torch.float32, device=self.device
            )

        limits = self.robot.joint_limits
        self._jl_low = torch.tensor(
            [limits[j][0] for j in self._action_joint_names], dtype=torch.float32, device=self.device
        )
        self._jl_high = torch.tensor(
            [limits[j][1] for j in self._action_joint_names], dtype=torch.float32, device=self.device
        )

        self._rand_buffers_ready = True

    def _sample_yaw_delta_quat(self, n: int, device, dtype, range_rad: float | None = None) -> torch.Tensor:
        """Sample yaw-only delta quaternions around +Z axis."""
        if range_rad is None:
            range_rad = float(self.BBQ_XY_ROT_RANGE_RAD)
        if range_rad <= 0.0:
            q = torch.zeros(n, 4, device=device, dtype=dtype)
            q[:, 0] = 1.0
            return q
        yaw = (torch.rand(n, device=device, dtype=dtype) - 0.5) * float(range_rad)
        half = 0.5 * yaw
        q = torch.zeros(n, 4, device=device, dtype=dtype)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def _ensure_fixed_randomization(self) -> None:
        """Create per-env fixed randomization once."""
        if getattr(self, "_fixed_rand_ready", False) and self._fixed_obj_delta3 is not None:
            return
        self._ensure_rand_buffers()

        x_rand = torch.rand(self.num_envs, device=self.device) * (self.BBQ_X_MAX - self.BBQ_X_MIN) + self.BBQ_X_MIN
        y_rand = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * self.BBQ_Y_RANGE
        delta3 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        base_x = self._base_object_init_pos[0].item()
        delta3[:, 0] = x_rand - base_x
        delta3[:, 1] = y_rand
        delta3[:, 2] = 0.0
        self._fixed_obj_delta3 = delta3

        self._fixed_obj_delta_quat = self._sample_yaw_delta_quat(
            self.num_envs, device=self.device, dtype=torch.float32, range_rad=self.BBQ_XY_ROT_RANGE_RAD
        )

        basket_delta_xy = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 2.0 * self.BASKET_XY_RANGE_M
        basket_delta3 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        basket_delta3[:, 0:2] = basket_delta_xy
        basket_delta3[:, 2] = 0.0
        self._fixed_basket_delta3 = basket_delta3

        self._fixed_rand_ready = True

    def _apply_robot_reset_joint_policy(self, states, env_ids_t: torch.Tensor) -> None:
        """Reset robot joints (zero arm, gripper from saved pose)."""
        if self.robot_name not in states.robots:
            return
        rs = states.robots[self.robot_name]
        if getattr(rs, "joint_pos", None) is None:
            return
        self._ensure_rand_buffers()

        joint_pos = rs.joint_pos.to(self.device)
        n = int(env_ids_t.numel())
        sel = joint_pos.index_select(0, env_ids_t).clone()

        sel = torch.zeros_like(sel)
        lo = self._jl_low.unsqueeze(0).expand(n, -1)
        hi = self._jl_high.unsqueeze(0).expand(n, -1)
        sel = torch.clamp(sel, lo, hi)

        gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        for jn in gripper_joint_names:
            jidx = self._action_name_to_idx.get(jn, None)
            if jidx is None:
                continue
            saved_val = self._saved_pose_dof_pos.get(jn, self.GRIPPER_OPEN_VALUE)
            sel[:, int(jidx)] = float(saved_val)

        sel = torch.clamp(sel, lo, hi)
        rs.joint_pos.index_copy_(0, env_ids_t, sel.to(dtype=rs.joint_pos.dtype))
        if getattr(rs, "joint_vel", None) is not None:
            rs.joint_vel.index_copy_(0, env_ids_t, torch.zeros_like(rs.joint_vel.index_select(0, env_ids_t)))

    def _pose_world_to_base(
        self, root_pos: torch.Tensor, root_quat: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world (pos,quat) into robot base frame (wxyz)."""
        root_quat = root_quat / torch.norm(root_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        inv = Utils.quat_conjugate(root_quat)
        pos_local = Utils.quat_rotate(inv, pos_w - root_pos)
        quat_local = Utils.quat_mul(inv, quat_w)
        quat_local = quat_local / torch.norm(quat_local, dim=-1, keepdim=True).clamp(min=1e-9)
        return pos_local, quat_local

    def _maybe_solve_ik_for_envs(self, env_ids_t: torch.Tensor) -> None:
        """Solve IK targets once per env."""
        self._ensure_rand_buffers()

        if self._ik_target_joint_pos_env is None:
            st0 = self.handler.get_states()
            self._ik_target_joint_pos_env = st0.robots[self.robot_name].joint_pos.to(self.device).clone()

        not_ready = ~self._ik_ready
        mask = not_ready.index_select(0, env_ids_t)
        if not torch.any(mask):
            return
        need = env_ids_t.index_select(0, torch.nonzero(mask, as_tuple=False).squeeze(-1))

        st = self.handler.get_states()
        rs = st.robots[self.robot_name]
        obj = st.objects[self.object_name]

        obj_pos = obj.root_state[:, 0:3].to(self.device)
        obj_quat = obj.root_state[:, 3:7].to(self.device)
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        root_pos = rs.root_state[:, 0:3].to(self.device)
        root_quat = rs.root_state[:, 3:7].to(self.device)

        rel_off_obj = self._target_rel_offset_obj.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        # NOTE: `_target_rel_offset_obj` was computed using `_get_ee_state()`, which returns
        # the gripper-center position (panda_hand + offset). However the IK solver is configured
        # with ee_link="panda_hand", so we must convert the desired gripper-center target back
        # to the panda_hand link target by subtracting the same offset in world frame.
        ee_pos_w = obj_pos + Utils.quat_rotate(obj_quat, rel_off_obj)  # desired gripper-center in world
        target_rel_quat = self._target_rel_quat.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        ee_quat_w = Utils.quat_mul(target_rel_quat, obj_quat)
        ee_quat_w = ee_quat_w / torch.norm(ee_quat_w, dim=-1, keepdim=True).clamp(min=1e-9)

        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=self.device, dtype=ee_pos_w.dtype)  # panda_hand->center
        offset_world = quat_apply(ee_quat_w, offset_local.expand(ee_pos_w.shape[0], -1))
        hand_pos_w = ee_pos_w - offset_world  # desired panda_hand position in world

        ee_pos_local, ee_quat_local = self._pose_world_to_base(root_pos, root_quat, hand_pos_w, ee_quat_w)
        ee_pos_local = ee_pos_local.index_select(0, need)
        ee_quat_local = ee_quat_local.index_select(0, need)

        seed_1 = torch.tensor(
            [[float(self._saved_pose_dof_pos.get(jn, 0.0)) for jn in self._ik_joint_names]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(int(need.numel()), 1)

        k = max(1, int(getattr(self, "IK_NUM_SEEDS", 1)))
        if k == 1:
            q_best, _ = self._ik_solver.solve_ik_batch(ee_pos_local, ee_quat_local, seed_q=seed_1)
        else:
            n = int(need.numel())
            seed_list = [seed_1]
            noise_frac = 0.02
            ik_low = torch.tensor(
                [self.robot.joint_limits[jn][0] for jn in self._ik_joint_names], dtype=torch.float32, device=self.device
            )
            ik_high = torch.tensor(
                [self.robot.joint_limits[jn][1] for jn in self._ik_joint_names], dtype=torch.float32, device=self.device
            )
            ik_span = (ik_high - ik_low).clamp(min=1e-6)
            for _ in range(k - 1):
                eps = (torch.rand(n, int(self._ik_solver.n_dof_ik), device=self.device) - 0.5) * 2.0 * noise_frac
                s = seed_1.clone()
                s[:, : self._ik_solver.n_dof_ik] = torch.clamp(
                    s[:, : self._ik_solver.n_dof_ik] + eps * ik_span[: self._ik_solver.n_dof_ik].unsqueeze(0),
                    ik_low[: self._ik_solver.n_dof_ik].unsqueeze(0),
                    ik_high[: self._ik_solver.n_dof_ik].unsqueeze(0),
                )
                seed_list.append(s)

            seeds = torch.cat(seed_list, dim=0)
            pos_rep = ee_pos_local.repeat(k, 1)
            quat_rep = ee_quat_local.repeat(k, 1)
            q_all, _ = self._ik_solver.solve_ik_batch(pos_rep, quat_rep, seed_q=seeds)
            q_all = q_all.view(k, n, -1).transpose(0, 1)  # (n,k,n_dof_ik)
            seeds_ik = seeds[:, : self._ik_solver.n_dof_ik].view(k, n, -1).transpose(0, 1)  # (n,k,n_dof_ik)
            dist = torch.norm(q_all - seeds_ik, dim=-1)  # (n,k)
            best_idx = torch.argmin(dist, dim=-1)
            q_best = q_all[torch.arange(n, device=self.device), best_idx, :]

        joint_pos_full = rs.joint_pos.to(self.device).index_select(0, need).clone()

        for i in range(int(self._ik_solver.n_dof_ik)):
            jn = self._ik_joint_names[i]
            jidx = self._action_name_to_idx.get(jn, None)
            if jidx is None:
                continue
            joint_pos_full[:, int(jidx)] = q_best[:, i].to(dtype=joint_pos_full.dtype)

        lo = self._jl_low.unsqueeze(0).expand(int(need.numel()), -1)
        hi = self._jl_high.unsqueeze(0).expand(int(need.numel()), -1)
        joint_pos_full = torch.clamp(joint_pos_full, lo, hi)

        self._ik_target_joint_pos_env.index_copy_(0, need, joint_pos_full)
        self._ik_ready.index_fill_(0, need, True)

    def _get_ee_state(self, states):
        """Return EE position/orientation in world."""
        rs = states.robots[self.robot_name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        hand_body_index = rs.body_names.index("panda_hand")
        hand_pos = body_state[:, hand_body_index, 0:3]
        hand_quat = body_state[:, hand_body_index, 3:7]

        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=device, dtype=hand_pos.dtype)
        offset_world = quat_apply(hand_quat, offset_local.expand(hand_pos.shape[0], -1))

        ee_pos_world = hand_pos + offset_world
        ee_quat_world = hand_quat

        return ee_pos_world, ee_quat_world
