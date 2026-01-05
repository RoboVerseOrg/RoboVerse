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


@register_task("track", "approach_track")
class ApproachThenTrackTaskFranka(RLTaskEnv):
    """Single task with 2 stages.

    - stage 0: approach + settle (same spirit as `pick_place.approach_rand_franka`)
    - stage 1: track (gripper EE tracks waypoint markers `traj_marker_0..4`)
    """

    # ---------------- scenario ----------------
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
            *[
                RigidObjCfg(
                    name=f"traj_marker_{i}",
                    urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                    mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                    usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                    scale=0.2,
                    physics=PhysicStateType.RIGIDBODY,
                    enabled_gravity=False,
                    collision_enabled=False,
                )
                for i in range(5)
            ],
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    # ---------------- saved pose ----------------
    DEFAULT_SAVED_POSE_PY_PATH = str(REPO_ROOT / "get_started/output/saved_poses_20260104_143943.py")
    SAVED_POSE_PY_PATH = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", DEFAULT_SAVED_POSE_PY_PATH)

    # ---------------- horizons ----------------
    # stage0: delta control (approach)
    EPISODE_HORIZON = 80
    # stage1: settle to IK target (allow small +/-0.005 action nudges, non-accumulating)
    IK_SETTLE_STEPS = 30
    # stage2: delta control (track)
    TRACK_HORIZON = 100
    max_episode_steps = EPISODE_HORIZON + IK_SETTLE_STEPS + TRACK_HORIZON

    # ---------------- IK ----------------
    IK_BACKEND = "pyroki"
    IK_NUM_SEEDS = 5

    # ---------------- control ----------------
    # Delta control scale for stage0/stage2 (accumulating).
    DELTA_ACTION_SCALE = 0.02
    # During settle (stage1), allow tiny non-accumulating nudges around the settle target.
    SETTLE_NUDGE_SCALE = 0.005
    # Start forcing gripper closed this many steps BEFORE stage2 begins, and keep it closed afterwards.
    CLOSE_GRIPPER_BEFORE_STAGE2_STEPS = 5

    # ---------------- tracking ----------------
    NUM_WAYPOINTS = 5
    TRACK_REACH_THRESHOLD = 0.05
    TRACK_ROT_THRESHOLD_RAD = 0.35

    # ---------------- gripper ----------------
    GRIPPER_CLOSE_FRAMES = 10
    GRIPPER_CLOSE_VALUE = 0.0
    GRIPPER_OPEN_VALUE = 0.04
    # Always force gripper open for the final N frames of each episode (requested for traj saving).
    GRIPPER_FORCE_OPEN_LAST_FRAMES = 5

    # ---------------- domain randomization (aligned with `appoarch.py`) ----------------
    BBQ_X_MIN = 0.6
    BBQ_X_MAX = 0.8
    BBQ_Y_RANGE = 0.10
    BBQ_XY_ROT_RANGE_RAD = 1.0
    BASKET_XY_RANGE_M = 0.05

    @classmethod
    def _load_saved_pose_py(cls) -> dict:
        pose_path = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", cls.SAVED_POSE_PY_PATH)
        p = Path(pose_path)
        if not p.exists():
            raise FileNotFoundError(f"SAVED_POSE_PY_PATH not found: {p}")
        mod_name = f"saved_pose_module_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import saved pose module from: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        poses = getattr(mod, "poses", None)
        if not isinstance(poses, dict):
            raise ValueError(f"`poses` must be a dict in: {p}")
        return poses

    def _get_initial_states(self) -> list[dict] | None:
        """Initial states from saved pose. Markers are aligned as a rigid layout relative to bbq_sauce."""
        poses = self._load_saved_pose_py()
        if "objects" not in poses or "robots" not in poses:
            raise ValueError("Saved pose format error: missing 'objects' or 'robots'.")

        obj_name = "bbq_sauce"
        basket_name = "basket"
        robot_name = "franka"

        if obj_name not in poses["objects"] or basket_name not in poses["objects"] or robot_name not in poses["robots"]:
            raise ValueError("Saved pose missing required keys (bbq_sauce/basket/franka).")

        obj_pos = torch.as_tensor(poses["objects"][obj_name]["pos"], dtype=torch.float32)
        obj_quat = torch.as_tensor(poses["objects"][obj_name]["rot"], dtype=torch.float32)
        obj_quat = obj_quat / torch.norm(obj_quat).clamp(min=1e-9)

        basket_pos = torch.as_tensor(poses["objects"][basket_name]["pos"], dtype=torch.float32)
        basket_quat = torch.as_tensor(poses["objects"][basket_name]["rot"], dtype=torch.float32)
        basket_quat = basket_quat / torch.norm(basket_quat).clamp(min=1e-9)

        robot_pos = torch.as_tensor(poses["robots"][robot_name]["pos"], dtype=torch.float32)
        robot_quat = torch.as_tensor(poses["robots"][robot_name]["rot"], dtype=torch.float32)
        dof_pos = dict(poses["robots"][robot_name].get("dof_pos", {}) or {})

        # Marker layout template: use markers in saved pose if present; otherwise linearly interpolate bbq->basket.
        marker_world = {}
        has_markers = all(f"traj_marker_{i}" in poses["objects"] for i in range(self.NUM_WAYPOINTS))
        if has_markers:
            for i in range(self.NUM_WAYPOINTS):
                m = poses["objects"][f"traj_marker_{i}"]
                marker_world[f"traj_marker_{i}"] = {
                    "pos": torch.as_tensor(m["pos"], dtype=torch.float32),
                    "rot": torch.as_tensor(m["rot"], dtype=torch.float32),
                }
        else:
            for i in range(self.NUM_WAYPOINTS):
                a = float(i) / float(max(1, self.NUM_WAYPOINTS - 1))
                marker_world[f"traj_marker_{i}"] = {
                    "pos": (1 - a) * obj_pos + a * basket_pos,
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                }

        init_one = {"objects": {}, "robots": {}}
        init_one["objects"][obj_name] = {"pos": obj_pos, "rot": obj_quat}
        init_one["objects"][basket_name] = {"pos": basket_pos, "rot": basket_quat}
        for k, v in marker_world.items():
            init_one["objects"][k] = {"pos": v["pos"], "rot": v["rot"]}
        init_one["robots"][robot_name] = {"pos": robot_pos, "rot": robot_quat, "dof_pos": dof_pos}

        return [deepcopy(init_one) for _ in range(int(self.num_envs))]

    def __init__(self, scenario, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_envs = int(scenario.num_envs)
        self.robot_name = scenario.robots[0].name
        self.object_name = "bbq_sauce"
        self.basket_name = "basket"

        # stage: 0=approach+settle, 1=track
        self._stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # legacy discrete waypoint index (kept for backward compatibility; not used in the new
        # "interpolate across TRACK_HORIZON" tracking mode).
        self._stage1_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # IK init
        robot_cfg_for_ik = _build_franka_ik_cfg(scenario.robots[0], ee_link="panda_hand")
        self._ik_solver = IKSolver(robot_cfg_for_ik, solver=self.IK_BACKEND, use_seed=True)
        self._ik_joint_names = list(getattr(robot_cfg_for_ik, "actuators", {}).keys())
        self._ik_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ik_target_joint_pos_env = None

        self._saved_pose_dof_pos = {}
        self._target_rel_offset_obj = torch.zeros(3, dtype=torch.float32)
        self._target_rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        # IMPORTANT: RLTaskEnv.__init__ will call self.reset(), which calls _observation().
        # `_observation()` reads `_waypoints_world/_waypoints_quat_world`, so we must define them
        # before calling `super().__init__()`. We'll fill real values after handler is available.
        self._waypoints_world = torch.zeros(
            self.num_envs, self.NUM_WAYPOINTS, 3, device=self.device, dtype=torch.float32
        )
        self._waypoints_quat_world = torch.zeros(
            self.num_envs, self.NUM_WAYPOINTS, 4, device=self.device, dtype=torch.float32
        )

        # Per-step tracking targets (stage2): interpolated from traj_marker_0..4 and distributed
        # uniformly across TRACK_HORIZON steps.
        self._track_pos_traj = torch.zeros(
            self.num_envs, self.TRACK_HORIZON, 3, device=self.device, dtype=torch.float32
        )
        self._track_quat_traj = torch.zeros(
            self.num_envs, self.TRACK_HORIZON, 4, device=self.device, dtype=torch.float32
        )

        # During RLTaskEnv.__init__(), reset() will be called once.
        # We must keep DR disabled for this bootstrap reset, otherwise the "saved-pose reference"
        # (used to compute target relative EE pose) would be computed on randomized states.
        self._disable_dr = True
        self._last_action = None
        super().__init__(scenario, device=device)

        # compute reference relative pose from saved initial state
        st = self.handler.get_states()
        ee_pos, ee_quat = self._get_ee_state(st)
        obj_pos = st.objects[self.object_name].root_state[:, 0:3]
        obj_quat = st.objects[self.object_name].root_state[:, 3:7]
        qo0 = obj_quat[0].detach().clone().to(self.device)
        qo0 = qo0 / torch.norm(qo0).clamp(min=1e-9)
        rel_pos0_world = (ee_pos[0] - obj_pos[0]).detach().clone().to(self.device)
        self._target_rel_offset_obj = Utils.quat_rotate(Utils.quat_conjugate(qo0), rel_pos0_world)

        qh0 = ee_quat[0].detach().clone().to(self.device)
        qh0 = qh0 / torch.norm(qh0).clamp(min=1e-9)
        self._target_rel_quat = Utils.quat_mul(qh0, Utils.quat_conjugate(qo0))
        self._target_rel_quat = self._target_rel_quat / torch.norm(self._target_rel_quat).clamp(min=1e-9)

        # saved pose dof for IK seed / gripper init
        poses = self._load_saved_pose_py()
        self._saved_pose_dof_pos = dict(poses.get("robots", {}).get(self.robot_name, {}).get("dof_pos", {}) or {})

        # cache waypoints from marker objects (world frame)
        st = self.handler.get_states()
        self._waypoints_world = torch.zeros(
            self.num_envs, self.NUM_WAYPOINTS, 3, device=self.device, dtype=torch.float32
        )
        self._waypoints_quat_world = torch.zeros(
            self.num_envs, self.NUM_WAYPOINTS, 4, device=self.device, dtype=torch.float32
        )
        for i in range(self.NUM_WAYPOINTS):
            mn = f"traj_marker_{i}"
            if mn in st.objects:
                self._waypoints_world[:, i, :] = st.objects[mn].root_state[:, 0:3].to(self.device)
                q = st.objects[mn].root_state[:, 3:7].to(self.device)
                q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
                self._waypoints_quat_world[:, i, :] = q

        # build interpolated track trajectory
        self._rebuild_track_trajectory(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        self._disable_dr = False
        self.reset(env_ids=list(range(self.num_envs)))

    def _prepare_states(self, env_states, env_ids):
        """Apply appoarch-style reset policy: robot default joints + object/basket DR.

        IMPORTANT: traj_markers are defined relative to *basket* (not bbq_sauce).
        """
        if self._disable_dr:
            self._ensure_rand_buffers()
            return deepcopy(env_states)

        self._ensure_rand_buffers()
        states = deepcopy(env_states)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._ensure_fixed_randomization()
        n = int(env_ids_t.numel())

        # -------- object (bbq) xy + yaw --------
        base_pos = self._base_object_init_pos.to(self.device)
        base_quat = self._base_object_init_quat.to(self.device)
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

        # -------- basket xy --------
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

        # -------- markers follow basket (keep rigid layout relative to basket) --------
        new_basket_quat = base_basket_quat.unsqueeze(0).expand(n, -1)
        self._apply_delta_to_markers(states, env_ids_t, new_basket_pos, new_basket_quat)

        # -------- constraint: last waypoint XY must match basket XY --------
        # Keep marker Z and orientation from the marker layout, but force x/y to basket x/y.
        last_mn = f"traj_marker_{self.NUM_WAYPOINTS - 1}"
        if last_mn in states.objects:
            ms = states.objects[last_mn].root_state
            # update only env_ids
            cur = ms.index_select(0, env_ids_t).clone()
            cur[:, 0:2] = new_basket_pos[:, 0:2].to(dtype=cur.dtype)
            # keep z, quat, and keep zero velocities (already zeros in marker update), but be safe:
            cur[:, 7:13] = 0.0
            ms.index_copy_(0, env_ids_t, cur)

        # -------- robot default joints (appoarch-style) --------
        self._apply_robot_reset_joint_policy(states, env_ids_t)

        return states

    # ---------------- stage-aware step ----------------
    def reset(self, env_ids=None):
        """Reset envs and stage state (stage0->stage1) and recompute per-env IK targets."""
        obs, info = RLTaskEnv.reset(self, env_ids=env_ids)
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._stage.index_fill_(0, env_ids_t, 0)
        self._stage1_waypoint_idx.index_fill_(0, env_ids_t, 0)
        self._ik_ready.index_fill_(0, env_ids_t, False)
        if self._last_action is not None:
            # reset delta-control integrator to current joint positions
            st = self.handler.get_states()
            q = st.robots[self.robot_name].joint_pos.to(self.device)
            self._last_action.index_copy_(0, env_ids_t, q.index_select(0, env_ids_t))

        # solve IK targets once per reset (stage0)
        if not self._disable_dr:
            self._maybe_solve_ik_for_envs(env_ids_t)

        # refresh waypoint buffers from marker objects AFTER reset/DR
        st = self.handler.get_states()
        for i in range(self.NUM_WAYPOINTS):
            mn = f"traj_marker_{i}"
            if mn not in st.objects:
                continue
            self._waypoints_world[env_ids_t, i, :] = (
                st.objects[mn].root_state[:, 0:3].to(self.device).index_select(0, env_ids_t)
            )
            q = st.objects[mn].root_state[:, 3:7].to(self.device).index_select(0, env_ids_t)
            q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            self._waypoints_quat_world[env_ids_t, i, :] = q

        # rebuild interpolated per-step targets for stage2
        self._rebuild_track_trajectory(env_ids_t)
        return obs, info

    def step(self, actions):
        """Stage-aware step.

        - stage0: delta action control (accumulating): last_target += actions * DELTA_ACTION_SCALE
        - stage1: settle to IK target, plus tiny non-accumulating nudges: target = qT + actions * SETTLE_NUDGE_SCALE
        - stage2: delta action control (accumulating): last_target += actions * DELTA_ACTION_SCALE
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        self._ensure_rand_buffers()
        steps = self._episode_steps.to(self.device)

        # Determine stage from global episode step (per-env, but same thresholds).
        stage2_start = int(self.EPISODE_HORIZON + self.IK_SETTLE_STEPS)
        stage1_start = int(self.EPISODE_HORIZON)
        stage0 = steps < stage1_start
        stage1 = (steps >= stage1_start) & (steps < stage2_start)
        stage2 = steps >= stage2_start

        # Update stage flag (for reward/obs bookkeeping).
        self._stage = torch.where(stage2, torch.full_like(self._stage, 2), self._stage)
        self._stage = torch.where(stage1, torch.full_like(self._stage, 1), self._stage)
        self._stage = torch.where(stage0, torch.full_like(self._stage, 0), self._stage)

        # Build commanded joint targets per env.
        # - stage0: accumulating delta
        # - stage1: settle to qT + tiny non-accumulating nudge
        # - stage2: accumulating delta
        st = self.handler.get_states()
        q_curr = st.robots[self.robot_name].joint_pos.to(self.device)

        # init last_action buffer
        if self._last_action is None:
            self._last_action = q_curr.clone()

        # joint limits (ordered by joint_names)
        action_low = self._action_low
        action_high = self._action_high
        if action_low.ndim == 1:
            action_low = action_low.unsqueeze(0)
        if action_high.ndim == 1:
            action_high = action_high.unsqueeze(0)

        qT = (
            self._ik_target_joint_pos_env.to(self.device, dtype=q_curr.dtype)
            if self._ik_target_joint_pos_env is not None
            else q_curr
        )
        # Delta control (accumulating) for stage0/stage2, based on LAST ACTION (previous cmd),
        # not on an abstract "target".
        delta = torch.clamp(actions, -1.0, 1.0) * float(self.DELTA_ACTION_SCALE)
        next_cmd = torch.clamp(self._last_action + delta, action_low, action_high)

        # Settle (non-accumulating nudges around target).
        nudge = torch.clamp(actions, -1.0, 1.0) * float(self.SETTLE_NUDGE_SCALE)
        settle_target = torch.clamp(qT + nudge.to(dtype=qT.dtype), action_low, action_high)

        cmd = torch.where(stage1.unsqueeze(-1), settle_target, next_cmd)

        # Gripper policy: from (stage2_start - CLOSE_GRIPPER_BEFORE_STAGE2_STEPS) onward, keep closed always.
        close_from = max(0, stage2_start - int(self.CLOSE_GRIPPER_BEFORE_STAGE2_STEPS))
        close_always_mask = steps >= int(close_from)
        if torch.any(close_always_mask):
            # force both fingers closed regardless of stage
            if not hasattr(self, "_gripper_joint_indices"):
                joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
                self._gripper_joint_indices = [i for i, name in enumerate(joint_names) if "finger" in name.lower()]
            if len(getattr(self, "_gripper_joint_indices", [])) >= 2:
                cmd[close_always_mask, self._gripper_joint_indices[0]] = self.GRIPPER_CLOSE_VALUE
                cmd[close_always_mask, self._gripper_joint_indices[1]] = self.GRIPPER_CLOSE_VALUE
        else:
            # fallback: keep existing end-of-episode close
            close_gripper_mask = steps >= int(self.max_episode_steps - self.GRIPPER_CLOSE_FRAMES)
            self._apply_gripper_control(cmd, close_gripper_mask)

        # Final frames: force OPEN (override any closing policy above).
        open_last = int(getattr(self, "GRIPPER_FORCE_OPEN_LAST_FRAMES", 0) or 0)
        if open_last > 0:
            open_mask = steps >= int(self.max_episode_steps - open_last)
            if torch.any(open_mask):
                if not hasattr(self, "_gripper_joint_indices"):
                    joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
                    self._gripper_joint_indices = [i for i, name in enumerate(joint_names) if "finger" in name.lower()]
                if len(getattr(self, "_gripper_joint_indices", [])) >= 2:
                    cmd[open_mask, self._gripper_joint_indices[0]] = self.GRIPPER_OPEN_VALUE
                    cmd[open_mask, self._gripper_joint_indices[1]] = self.GRIPPER_OPEN_VALUE

        obs, reward, terminated, time_out, info = RLTaskEnv.step(self, cmd)
        # Always update LAST ACTION to the actual commanded joint targets this step
        # (so next delta step is consistent across stage0/stage1/stage2).
        self._last_action = cmd.detach().clone()

        # New tracking mode: targets are time-indexed along TRACK_HORIZON (no discrete waypoint advancement).

        return obs, reward, terminated, time_out, info

    # ---------------- reward / obs ----------------
    def _reward(self, env_states) -> torch.Tensor:
        stage0 = self._stage == 0
        stage2 = self._stage == 2

        # stage0: reach saved relative pose (same metric as approach)
        ee_pos, ee_quat = self._get_ee_state(env_states)
        obj_state = env_states.objects[self.object_name].root_state
        obj_pos = obj_state[:, 0:3]
        obj_quat = obj_state[:, 3:7]
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        rel = ee_pos - obj_pos
        rel_off_obj = self._target_rel_offset_obj.to(self.device).unsqueeze(0).expand_as(rel)
        target_rel_pos = Utils.quat_rotate(obj_quat, rel_off_obj)
        err_pos = torch.norm(rel - target_rel_pos, dim=-1)
        r0 = torch.exp(-1.0 * err_pos)

        ee_quat = ee_quat / torch.norm(ee_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        rel_quat = Utils.quat_mul(ee_quat, Utils.quat_conjugate(obj_quat))
        rel_quat = rel_quat / torch.norm(rel_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        target_rel = self._target_rel_quat.to(self.device).unsqueeze(0).expand_as(rel_quat)
        diff = Utils.quat_mul(target_rel, Utils.quat_conjugate(rel_quat))
        err_rot = Utils.quat_angle(diff)
        r0 = r0 + 0.5 * torch.exp(-3.0 * err_rot)

        # stage2: track interpolated per-step targets (OBJECT pose -> target pose).
        steps = self._episode_steps.to(self.device)
        stage2_start = int(self.EPISODE_HORIZON + self.IK_SETTLE_STEPS)
        t = torch.clamp(steps - stage2_start, 0, int(self.TRACK_HORIZON) - 1).to(torch.long)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        target_pos = self._track_pos_traj[env_ids, t, :]
        target_quat = self._track_quat_traj[env_ids, t, :]

        dist = torch.norm(obj_pos.to(self.device) - target_pos, dim=-1)
        diff_quat = Utils.quat_mul(target_quat, Utils.quat_conjugate(obj_quat))
        ang = Utils.quat_angle(diff_quat)
        r_track = torch.exp(-20.0 * dist) + 0.0 * torch.exp(-5.0 * ang)

        # Only enable tracking reward from stage2; stage1 gets 0 (settle phase).
        r_other = torch.where(stage2, 10 * r_track, torch.zeros_like(r_track))
        return torch.where(stage0, 0.2 * r0, r_other)

    def _observation(self, env_states) -> torch.Tensor:
        # In case super().__init__ triggered reset before we had a chance to fill waypoints,
        # ensure waypoint buffers exist.
        if not hasattr(self, "_waypoints_world") or not hasattr(self, "_waypoints_quat_world"):
            self._waypoints_world = torch.zeros(
                self.num_envs, self.NUM_WAYPOINTS, 3, device=self.device, dtype=torch.float32
            )
            self._waypoints_quat_world = torch.zeros(
                self.num_envs, self.NUM_WAYPOINTS, 4, device=self.device, dtype=torch.float32
            )

        rs = env_states.robots[self.robot_name]
        joint_pos = rs.joint_pos.to(self.device)
        joint_vel = getattr(rs, "joint_vel", None)
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        else:
            joint_vel = joint_vel.to(self.device)

        ee_pos, ee_quat = self._get_ee_state(env_states)
        ee_pos = ee_pos.to(self.device)
        ee_quat = ee_quat.to(self.device)
        ee_quat = ee_quat / torch.norm(ee_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        obj = env_states.objects[self.object_name].root_state.to(self.device)
        obj_pos = obj[:, 0:3]
        obj_quat = obj[:, 3:7]
        obj_quat = obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True).clamp(min=1e-9)

        # stage2 target (object -> per-step interpolated target)
        steps = self._episode_steps.to(self.device)
        stage2_start = int(self.EPISODE_HORIZON + self.IK_SETTLE_STEPS)
        t = torch.clamp(steps - stage2_start, 0, int(self.TRACK_HORIZON) - 1).to(torch.long)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        target_pos = self._track_pos_traj[env_ids, t, :]
        target_quat = self._track_quat_traj[env_ids, t, :]
        obj_to_wp = target_pos - obj_pos
        diff_quat = Utils.quat_mul(target_quat, Utils.quat_conjugate(obj_quat))
        ang = Utils.quat_angle(diff_quat).unsqueeze(-1)

        # stage indicator (0/1/2) as float
        stage_f = self._stage.to(dtype=torch.float32).unsqueeze(-1)
        wp_f = (t.to(dtype=torch.float32) / float(max(1, int(self.TRACK_HORIZON) - 1))).unsqueeze(-1)

        # Keep it simple: joints + ee + object + ee_to_wp + stage
        return torch.cat(
            [
                joint_pos,
                joint_vel,
                ee_pos,
                Utils.quat_to_tan_norm(ee_quat),
                obj_pos,
                Utils.quat_to_tan_norm(obj_quat),
                obj_to_wp,
                ang,
                wp_f,
                stage_f,
            ],
            dim=-1,
        )

    def _rebuild_track_trajectory(self, env_ids_t: torch.Tensor) -> None:
        """Build per-step tracking targets by piecewise-linear interpolation across marker points."""
        if env_ids_t.numel() == 0:
            return

        # Gather marker anchors for these envs
        anchors_p = self._waypoints_world.index_select(0, env_ids_t)  # (n, W, 3)
        anchors_q = self._waypoints_quat_world.index_select(0, env_ids_t)  # (n, W, 4)
        n = int(env_ids_t.numel())
        W = int(self.NUM_WAYPOINTS)
        T = int(self.TRACK_HORIZON)
        if T <= 1 or W <= 1:
            # degenerate: just copy first point
            self._track_pos_traj.index_copy_(0, env_ids_t, anchors_p[:, :1, :].expand(n, T, 3))
            self._track_quat_traj.index_copy_(0, env_ids_t, anchors_q[:, :1, :].expand(n, T, 4))
            return

        # time parameter u in [0, W-1]
        s = torch.linspace(0.0, float(W - 1), T, device=self.device, dtype=torch.float32)  # (T,)
        seg = torch.clamp(s.floor().to(torch.long), 0, W - 2)  # (T,)
        alpha = (s - seg.to(dtype=s.dtype)).clamp(0.0, 1.0)  # (T,)

        # interpolate per t
        pos_out = torch.zeros(n, T, 3, device=self.device, dtype=torch.float32)
        quat_out = torch.zeros(n, T, 4, device=self.device, dtype=torch.float32)
        for ti in range(T):
            i0 = int(seg[ti].item())
            i1 = i0 + 1
            a = alpha[ti].view(1, 1)
            p0 = anchors_p[:, i0, :]
            p1 = anchors_p[:, i1, :]
            pos_out[:, ti, :] = (1.0 - a) * p0 + a * p1

            # nlerp quaternions (sufficient for our marker setup; normalize afterwards)
            q0 = anchors_q[:, i0, :]
            q1 = anchors_q[:, i1, :]
            q = (1.0 - a) * q0 + a * q1
            q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            quat_out[:, ti, :] = q

        self._track_pos_traj.index_copy_(0, env_ids_t, pos_out)
        self._track_quat_traj.index_copy_(0, env_ids_t, quat_out)

    def _terminated(self, env_states) -> torch.Tensor:
        # 合并任务这里先不做复杂 terminate：只按 timeout 结束（与用户需求一致）
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ---------------- helpers ----------------
    def _ensure_rand_buffers(self) -> None:
        if getattr(self, "_rand_buffers_ready", False):
            return
        self._action_joint_names = list(getattr(self, "joint_names", self.handler.get_joint_names(self.robot_name)))
        self._action_name_to_idx = {n: i for i, n in enumerate(self._action_joint_names)}
        limits = self.robot.joint_limits
        self._jl_low = torch.tensor(
            [limits[j][0] for j in self._action_joint_names], dtype=torch.float32, device=self.device
        )
        self._jl_high = torch.tensor(
            [limits[j][1] for j in self._action_joint_names], dtype=torch.float32, device=self.device
        )

        # base pose for DR: read from initial states (env0)
        base_obj = self._initial_states.objects[self.object_name].root_state[0].to(self.device)
        self._base_object_init_pos = base_obj[0:3].detach().clone()
        self._base_object_init_quat = base_obj[3:7].detach().clone()
        self._base_object_init_quat = self._base_object_init_quat / torch.norm(self._base_object_init_quat).clamp(
            min=1e-9
        )

        base_basket = self._initial_states.objects[self.basket_name].root_state[0].to(self.device)
        self._base_basket_init_pos = base_basket[0:3].detach().clone()
        self._base_basket_init_quat = base_basket[3:7].detach().clone()
        self._base_basket_init_quat = self._base_basket_init_quat / torch.norm(self._base_basket_init_quat).clamp(
            min=1e-9
        )

        # marker layout relative to basket (rigid template)
        self._marker_names = [
            f"traj_marker_{i}"
            for i in range(self.NUM_WAYPOINTS)
            if f"traj_marker_{i}" in getattr(self._initial_states, "objects", {})
        ]
        self._base_marker_rel_pos = {}
        self._base_marker_rel_quat = {}
        inv_base_basket_quat = Utils.quat_conjugate(self._base_basket_init_quat)
        for mn in self._marker_names:
            ms = self._initial_states.objects[mn].root_state[0].to(self.device)
            mpos = ms[0:3].detach().clone()
            mquat = ms[3:7].detach().clone()
            mquat = mquat / torch.norm(mquat).clamp(min=1e-9)
            # Define marker pose relative to *basket* in basket-local frame.
            # p_rel = R_basket^{-1} (p_marker - p_basket)
            self._base_marker_rel_pos[mn] = Utils.quat_rotate(inv_base_basket_quat, mpos - self._base_basket_init_pos)
            # q_rel = q_marker * q_basket^{-1}
            self._base_marker_rel_quat[mn] = Utils.quat_mul(mquat, inv_base_basket_quat)

        self._fixed_rand_ready = False
        self._fixed_obj_delta3 = None
        self._fixed_obj_delta_quat = None
        self._fixed_basket_delta3 = None

        self._rand_buffers_ready = True

    def _sample_yaw_delta_quat(self, n: int, device, dtype, range_rad: float) -> torch.Tensor:
        """Sample yaw-only delta quaternions around +Z axis."""
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
        """Create per-env fixed randomization once (same across episodes, like appoarch)."""
        if getattr(self, "_fixed_rand_ready", False) and self._fixed_obj_delta3 is not None:
            return

        # object x in [BBQ_X_MIN, BBQ_X_MAX], y in [-BBQ_Y_RANGE, BBQ_Y_RANGE], yaw in [-range/2, +range/2]
        x_rand = torch.rand(self.num_envs, device=self.device) * (self.BBQ_X_MAX - self.BBQ_X_MIN) + self.BBQ_X_MIN
        y_rand = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * self.BBQ_Y_RANGE
        delta3 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        delta3[:, 0] = x_rand - float(self._base_object_init_pos[0])
        delta3[:, 1] = y_rand
        delta3[:, 2] = 0.0
        self._fixed_obj_delta3 = delta3

        self._fixed_obj_delta_quat = self._sample_yaw_delta_quat(
            self.num_envs, device=self.device, dtype=torch.float32, range_rad=float(self.BBQ_XY_ROT_RANGE_RAD)
        )

        # basket xy in [-range, +range]
        basket_delta_xy = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 2.0 * float(self.BASKET_XY_RANGE_M)
        basket_delta3 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        basket_delta3[:, 0:2] = basket_delta_xy
        basket_delta3[:, 2] = 0.0
        self._fixed_basket_delta3 = basket_delta3

        self._fixed_rand_ready = True

    def _apply_delta_to_markers(
        self,
        states,
        env_ids_t: torch.Tensor,
        new_basket_pos: torch.Tensor,
        new_basket_quat: torch.Tensor,
    ) -> None:
        """Apply basket pose to marker targets (keep marker relative pose to basket)."""
        n = int(env_ids_t.numel())
        for mn in getattr(self, "_marker_names", []):
            if mn not in states.objects:
                continue

            rel_pos0 = self._base_marker_rel_pos[mn].to(self.device)  # (3,) in basket-local frame
            rel_quat0 = self._base_marker_rel_quat[mn].to(self.device)  # (4,) marker relative to basket

            rel_pos = rel_pos0.unsqueeze(0).expand(n, -1)
            mpos = new_basket_pos + Utils.quat_rotate(new_basket_quat, rel_pos)

            mquat = Utils.quat_mul(rel_quat0.unsqueeze(0).expand(n, -1), new_basket_quat)
            mquat = mquat / torch.norm(mquat, dim=-1, keepdim=True).clamp(min=1e-9)

            ms = states.objects[mn].root_state
            zero_vel = torch.zeros(n, 3, device=self.device, dtype=ms.dtype)
            zero_ang = torch.zeros(n, 3, device=self.device, dtype=ms.dtype)
            new_root = torch.cat([mpos.to(dtype=ms.dtype), mquat.to(dtype=ms.dtype), zero_vel, zero_ang], dim=-1)
            ms.index_copy_(0, env_ids_t, new_root)

    def _apply_robot_reset_joint_policy(self, states, env_ids_t: torch.Tensor) -> None:
        """Reset robot joints to default: arm zeros + gripper open (similar to appoarch)."""
        if self.robot_name not in states.robots:
            return
        rs = states.robots[self.robot_name]
        if getattr(rs, "joint_pos", None) is None:
            return
        self._ensure_rand_buffers()

        joint_pos = rs.joint_pos.to(self.device)
        n = int(env_ids_t.numel())

        # start from zeros, then clamp
        q = torch.zeros(n, joint_pos.shape[-1], device=self.device, dtype=joint_pos.dtype)
        lo = self._jl_low.to(dtype=q.dtype).unsqueeze(0).expand(n, -1)
        hi = self._jl_high.to(dtype=q.dtype).unsqueeze(0).expand(n, -1)

        # open gripper
        for jn in ("panda_finger_joint1", "panda_finger_joint2"):
            jidx = self._action_name_to_idx.get(jn, None)
            if jidx is not None:
                q[:, int(jidx)] = float(self.GRIPPER_OPEN_VALUE)

        q = torch.clamp(q, lo, hi)
        rs.joint_pos.index_copy_(0, env_ids_t, q)
        if getattr(rs, "joint_vel", None) is not None:
            rs.joint_vel.index_copy_(0, env_ids_t, torch.zeros_like(rs.joint_vel.index_select(0, env_ids_t)))

    def _apply_gripper_control(self, actions: torch.Tensor, mask: torch.Tensor) -> None:
        if not hasattr(self, "_gripper_joint_indices"):
            joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
            self._gripper_joint_indices = [i for i, name in enumerate(joint_names) if "finger" in name.lower()]
        if len(self._gripper_joint_indices) >= 2:
            actions[mask, self._gripper_joint_indices[0]] = self.GRIPPER_CLOSE_VALUE
            actions[mask, self._gripper_joint_indices[1]] = self.GRIPPER_CLOSE_VALUE

    def _pose_world_to_base(
        self, root_pos: torch.Tensor, root_quat: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor
    ):
        root_quat = root_quat / torch.norm(root_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        inv = Utils.quat_conjugate(root_quat)
        pos_local = Utils.quat_rotate(inv, pos_w - root_pos)
        quat_local = Utils.quat_mul(inv, quat_w)
        quat_local = quat_local / torch.norm(quat_local, dim=-1, keepdim=True).clamp(min=1e-9)
        return pos_local, quat_local

    def _maybe_solve_ik_for_envs(self, env_ids_t: torch.Tensor) -> None:
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
        ee_center_w = obj_pos + Utils.quat_rotate(obj_quat, rel_off_obj)
        target_rel_quat = self._target_rel_quat.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        ee_quat_w = Utils.quat_mul(target_rel_quat, obj_quat)
        ee_quat_w = ee_quat_w / torch.norm(ee_quat_w, dim=-1, keepdim=True).clamp(min=1e-9)

        # Convert gripper-center target -> panda_hand target (IK link) by subtracting the same offset.
        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=self.device, dtype=ee_center_w.dtype)
        offset_world = quat_apply(ee_quat_w, offset_local.expand(ee_center_w.shape[0], -1))
        hand_pos_w = ee_center_w - offset_world

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
            # multi-seed: small noise around seed_1, choose closest solution
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
            seeds_ik = seeds[:, : self._ik_solver.n_dof_ik].view(k, n, -1).transpose(0, 1)
            dist = torch.norm(q_all - seeds_ik, dim=-1)
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
        """Return gripper-center EE pose in world (panda_hand + offset)."""
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
