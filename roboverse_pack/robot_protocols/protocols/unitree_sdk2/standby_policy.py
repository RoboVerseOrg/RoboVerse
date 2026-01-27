from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rootutils
import torch

from roboverse_pack.robot_protocols.core.interfaces import StandbyController
from roboverse_pack.robot_protocols.core.types import SimRobotObservation
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.math_utils import quat_rotate_inverse_wxyz


def _get_gravity_orientation_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Match scripts/unitree_deploy/common/rotation_helper.py:get_gravity_orientation.

    Returns a 3D vector used by Unitree deploy policies (pelvis frame projected gravity).
    """
    qw, qx, qy, qz = quat_wxyz.astype(np.float32, copy=False)
    out = np.zeros((3,), dtype=np.float32)
    out[0] = 2.0 * (-qz * qx + qw * qy)
    out[1] = -2.0 * (qz * qy + qw * qx)
    out[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return out


@dataclass
class StandbyPolicyConfig:
    """Config needed to run a deploy-style policy as a standby controller."""

    control_dt: float
    policy_path: str

    num_actions: int
    num_obs: int
    obs_len_history: int

    # Arrays are in *policy joint order*.
    joint_names: list[str]
    default_angles: np.ndarray

    # Observation scaling.
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: np.ndarray


def load_standby_policy_config(config_path: str) -> StandbyPolicyConfig:
    """Load the same YAML used by scripts/unitree_deploy/deploy_real.py (g1_dof29.yaml)."""
    # Reuse the existing config loader to stay consistent with deploy_real behavior.
    from scripts.unitree_deploy.config import G1Config

    cfg = G1Config(config_path)
    return StandbyPolicyConfig(
        control_dt=float(cfg.control_dt),
        policy_path=str(cfg.policy_path),
        num_actions=int(cfg.num_actions),
        num_obs=int(cfg.num_obs),
        obs_len_history=int(cfg.obs_len_history),
        joint_names=list(cfg.policy_joint_names),
        default_angles=np.asarray(cfg.default_angles, dtype=np.float32),
        ang_vel_scale=float(cfg.ang_vel_scale),
        dof_pos_scale=float(cfg.dof_pos_scale),
        dof_vel_scale=float(cfg.dof_vel_scale),
        action_scale=float(cfg.action_scale),
        cmd_scale=np.asarray(cfg.cmd_scale, dtype=np.float32),
    )


class UnitreeStandbyPolicyController(StandbyController):
    """Standby controller powered by the same TorchScript policy used in deploy_real.py.

    This keeps the robot stable before any meaningful LowCmd is received over DDS.
    """

    def __init__(
        self,
        *,
        sim_dt: float,
        joint_names_sorted: list[str],
        torque_limits_sorted: np.ndarray | None,
        kp_sorted: np.ndarray,
        kd_sorted: np.ndarray,
        config_path: str,
        policy_path: str | None = None,
        device: str = "cpu",
        warmup_time_s: float = 0.0,
    ) -> None:
        self._sim_dt = float(sim_dt)
        self._joint_names_sorted = list(joint_names_sorted)
        self._torque_limits_sorted = (
            np.asarray(torque_limits_sorted, dtype=np.float32) if torque_limits_sorted is not None else None
        )
        kp_sorted = np.asarray(kp_sorted, dtype=np.float32)
        kd_sorted = np.asarray(kd_sorted, dtype=np.float32)
        if kp_sorted.shape != (len(self._joint_names_sorted),) or kd_sorted.shape != (len(self._joint_names_sorted),):
            raise ValueError(
                "kp_sorted/kd_sorted must have shape (num_joints,) in simulator sorted joint order. "
                f"Expected {(len(self._joint_names_sorted),)}, got {kp_sorted.shape} / {kd_sorted.shape}."
            )

        root = rootutils.find_root(search_from=__file__)
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        self._cfg = load_standby_policy_config(str(cfg_path))

        pol_path = Path(policy_path) if policy_path is not None else Path(self._cfg.policy_path)
        if not pol_path.is_absolute():
            pol_path = root / pol_path

        self._device = torch.device(device)
        self._policy = torch.jit.load(str(pol_path), map_location=self._device)
        self._policy.eval()

        # Policy joint order may not match the simulator's sorted joint order.
        policy_names = self._cfg.joint_names
        sim_names = self._joint_names_sorted
        missing = [jn for jn in policy_names if jn not in sim_names]
        if missing:
            raise ValueError(f"Standby policy expects joints {missing}, but simulator joint set is {sim_names}.")
        self._policy_to_sim = np.asarray([sim_names.index(jn) for jn in policy_names], dtype=np.int64)

        # PD gains in policy joint order (match RoboVerse task env control path).
        self._kp_pol = kp_sorted[self._policy_to_sim].astype(np.float32, copy=False)
        self._kd_pol = kd_sorted[self._policy_to_sim].astype(np.float32, copy=False)

        # Basic state for deploy-style observation history.
        self._action = np.zeros((self._cfg.num_actions,), dtype=np.float32)
        self._obs = np.zeros((self._cfg.num_obs,), dtype=np.float32)
        self._obs_queue = deque(
            [self._obs.copy() for _ in range(self._cfg.obs_len_history)],
            maxlen=self._cfg.obs_len_history,
        )

        # Run the policy on the first compute_effort() call.
        self._t_since_update = float(self._cfg.control_dt)
        self._elapsed_s = 0.0
        # Warmup ramps the policy action_scale (it does NOT delay the policy).
        self._warmup_time_s = float(max(0.0, warmup_time_s))
        self._q_des_pol = self._cfg.default_angles.copy()
        self._cached_effort_sorted = np.zeros((len(sim_names),), dtype=np.float32)

    def _build_obs(self, obs: SimRobotObservation) -> np.ndarray:
        # Command 0 (stand still).
        cmd = np.zeros((3,), dtype=np.float32)

        # Body-frame angular velocity and projected gravity from base orientation.
        root = obs.root_state.astype(np.float32, copy=False)
        quat_wxyz = root[3:7].astype(np.float32, copy=False)
        ang_vel_world = root[10:13].astype(np.float32, copy=False)
        ang_vel_body = quat_rotate_inverse_wxyz(quat_wxyz, ang_vel_world)

        gravity_orientation = _get_gravity_orientation_wxyz(quat_wxyz)

        # Joint state in policy order.
        q_sorted = obs.q_sorted.astype(np.float32, copy=False)
        dq_sorted = obs.dq_sorted.astype(np.float32, copy=False)
        q_pol = q_sorted[self._policy_to_sim]
        dq_pol = dq_sorted[self._policy_to_sim]

        # Scaled obs (match deploy_real.py).
        q_obs = (q_pol - self._cfg.default_angles) * self._cfg.dof_pos_scale
        dq_obs = dq_pol * self._cfg.dof_vel_scale

        self._obs[0:3] = cmd * self._cfg.cmd_scale
        self._obs[3:6] = ang_vel_body * self._cfg.ang_vel_scale
        self._obs[6:9] = gravity_orientation

        n = self._cfg.num_actions
        self._obs[9 : 9 + n] = q_obs
        self._obs[9 + n : 9 + 2 * n] = dq_obs
        self._obs[9 + 2 * n : 9 + 3 * n] = self._action

        return self._obs

    def _policy_step(self, obs: SimRobotObservation) -> np.ndarray:
        self._build_obs(obs)
        self._obs_queue.append(self._obs.copy())

        stacked = np.concatenate(list(self._obs_queue), axis=0).astype(np.float32, copy=False)
        obs_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self._device)

        with torch.no_grad():
            act = self._policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32, copy=False)

        # Keep standby aligned with RoboVerse task evaluation (no action smoothing).
        self._action = act.copy()
        return act

    def compute_effort(self, obs: SimRobotObservation) -> np.ndarray:
        """Compute the standby effort based on the standby policy."""
        self._elapsed_s += self._sim_dt
        # Update the policy at its control rate, but apply PD every sim step using the
        # latest desired joint positions (matches RoboVerse task decimation behavior).
        self._t_since_update += self._sim_dt
        if self._t_since_update + 1e-9 >= self._cfg.control_dt:
            self._t_since_update = 0.0
            act = self._policy_step(obs)

            # Policy outputs delta from default angles, in policy joint order.
            scale = float(self._cfg.action_scale)
            if self._warmup_time_s > 0.0:
                alpha = float(np.clip(self._elapsed_s / self._warmup_time_s, 0.0, 1.0))
                scale *= alpha
            self._q_des_pol = self._cfg.default_angles + act * scale

        q_sorted = obs.q_sorted.astype(np.float32, copy=False)
        dq_sorted = obs.dq_sorted.astype(np.float32, copy=False)
        q_pol = q_sorted[self._policy_to_sim]
        dq_pol = dq_sorted[self._policy_to_sim]

        tau_pol = self._kp_pol * (self._q_des_pol - q_pol) + self._kd_pol * (0.0 - dq_pol)

        # Scatter to simulator sorted order.
        tau_sorted = np.zeros((len(self._joint_names_sorted),), dtype=np.float32)
        tau_sorted[self._policy_to_sim] = tau_pol.astype(np.float32, copy=False)

        if self._torque_limits_sorted is not None:
            lim = self._torque_limits_sorted.astype(np.float32, copy=False)
            tau_sorted = np.clip(tau_sorted, -lim, lim)

        self._cached_effort_sorted = tau_sorted.astype(np.float32, copy=False)
        return self._cached_effort_sorted
