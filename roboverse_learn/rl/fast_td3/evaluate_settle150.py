from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure repository root is on sys.path for local package imports
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
from loguru import logger as log
from torch.amp import autocast

from metasim.task.registry import get_task_class
from metasim.utils.demo_util import save_traj_file
from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization


def extract_state_dict(env, scenario, env_idx=0):
    """Extract state dictionary from handler states (similar to teleop_keyboard)."""
    state_dict = {}

    if not hasattr(env, "handler") or env.handler is None:
        log.warning("Handler not available, returning empty state")
        return state_dict

    handler_states = env.handler.get_states(mode="tensor")
    if handler_states is None:
        log.warning("Handler.get_states() returned None")
        return state_dict

    obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
    robot_cfg_dict = {robot.name: robot for robot in scenario.robots}

    if hasattr(handler_states, "objects"):
        for obj_name, obj_state in handler_states.objects.items():
            pos = obj_state.root_state[env_idx, :3].cpu().numpy()
            quat = obj_state.root_state[env_idx, 3:7].cpu().numpy()  # wxyz
            state_entry = {"pos": pos, "rot": quat}
            if obj_state.joint_pos is not None and obj_name in obj_cfg_dict:
                obj_cfg = obj_cfg_dict[obj_name]
                if hasattr(obj_cfg, "actuators") and obj_cfg.actuators is not None:
                    joint_names = sorted(obj_cfg.actuators.keys())
                    joint_positions = obj_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(p) for name, p in zip(joint_names, joint_positions)}
            state_dict[obj_name] = state_entry

    if hasattr(handler_states, "robots"):
        for robot_name, robot_state in handler_states.robots.items():
            pos = robot_state.root_state[env_idx, :3].cpu().numpy()
            quat = robot_state.root_state[env_idx, 3:7].cpu().numpy()  # wxyz
            state_entry = {"pos": pos, "rot": quat}
            if robot_name in robot_cfg_dict:
                robot_cfg = robot_cfg_dict[robot_name]
                if robot_cfg.actuators is not None:
                    joint_names = sorted(robot_cfg.actuators.keys())
                    joint_positions = robot_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(p) for name, p in zip(joint_names, joint_positions)}
            state_dict[robot_name] = state_entry

    return state_dict


def load_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    log.info(f"Loading checkpoint from {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _resolve_policy_and_settle_horizon(env, policy_horizon: int | None, settle_steps: int | None) -> tuple[int, int, int]:
    """Resolve (policy_horizon, settle_steps, total_horizon)."""
    # Prefer explicit args.
    ph = policy_horizon
    ss = settle_steps

    # Fall back to env attributes (ApproachRandTask defines these).
    if ph is None:
        ph = int(getattr(env, "EPISODE_HORIZON", 0) or 0)
    if ss is None:
        ss = int(getattr(env, "IK_SETTLE_STEPS", 0) or 0)

    # Fall back to env.max_episode_steps if caller didn't pass anything meaningful.
    total = int(getattr(env, "max_episode_steps", 0) or 0)
    if ph <= 0 and total > 0:
        ph = total
    if ss < 0:
        ss = 0
    if total <= 0:
        total = ph + ss
    # If env.max_episode_steps is already set, keep it as total horizon (most consistent with termination logic).
    if int(getattr(env, "max_episode_steps", total)) > 0:
        total = int(getattr(env, "max_episode_steps", total))
    return int(ph), int(ss), int(total)


def evaluate(
    env,
    actor,
    obs_normalizer,
    num_episodes: int,
    device: torch.device,
    scenario=None,
    task_name: str = "eval",
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    save_traj: bool = True,
    save_states: bool = True,
    save_every_n_steps: int = 1,
    traj_dir: str = "eval_trajs",
    save_per_env: bool = False,
    episodes_per_env: int = 1,
    # NEW: align with "150 steps policy + 50 steps settle" behavior
    policy_horizon: int | None = None,
    settle_steps: int | None = None,
    dummy_action: str = "zeros",
    save_policy_actions: bool = True,
    # NEW: collection semantics (avoid "存个没完" when early-terminated episodes are filtered out)
    filter_full_horizon_and_above_mean: bool = False,
    count_mode: str = "attempted",  # "attempted" or "saved"
    # NEW: for trajectory generation/replay, ignore early termination (e.g. object touched)
    # so episodes can run to full horizon (including settle window).
    ignore_terminated: bool = False,
) -> dict:
    """
    Eval for staged tasks that have an explicit settle window.

    - stage0 (policy):        t < policy_horizon
    - stage1 (settle/dummy):  policy_horizon <= t < policy_horizon + settle_steps
    - stage2 (policy again):  t >= policy_horizon + settle_steps

    NOTE: We record the executed *dof target* each step using handler.joint_pos_target when available,
    so settle (and any env-internal overrides) are captured in the saved trajectory.
    """
    actor.eval()
    obs_normalizer.eval()

    num_eval_envs = env.num_envs

    # Enable capturing terminal (pre-reset) states from RLTaskEnv.step()
    try:
        env._capture_terminal_states = True
    except Exception:
        pass

    # Optionally disable task termination (keep only timeout). This is useful when you want
    # to SAVE full-horizon trajectories (e.g. 150 policy + 50 settle) even if the task would
    # normally terminate early due to contacts/movement thresholds.
    if bool(ignore_terminated):
        try:
            def _no_term(_states):  # noqa: ANN001
                return torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

            env._terminated = _no_term  # type: ignore[method-assign]
            log.info("[eval_settle150] ignore_terminated=1: patched env._terminated() to always False")
        except Exception as e:
            log.warning(f"[eval_settle150] Failed to patch env._terminated(): {e}")

    if save_per_env:
        episodes_per_env = int(episodes_per_env)
        if episodes_per_env <= 0:
            raise ValueError("episodes_per_env must be >= 1 when save_per_env=True")
        target_saved_episodes = int(num_eval_envs) * episodes_per_env
    else:
        # NOTE:
        # - count_mode="attempted": num_episodes means how many COMPLETED episodes to run (regardless of accept)
        # - count_mode="saved": num_episodes means how many ACCEPTED episodes to save (old evaluate.py behavior)
        count_mode = str(count_mode).lower().strip()
        if count_mode not in ("attempted", "saved"):
            raise ValueError("count_mode must be one of: attempted, saved")
        target_saved_episodes = int(num_episodes)
        if target_saved_episodes <= 0:
            raise ValueError("num_episodes must be >= 1")

    ph, ss, horizon = _resolve_policy_and_settle_horizon(env, policy_horizon, settle_steps)
    log.info(f"[eval_settle150] policy_horizon={ph}, settle_steps={ss}, max_episode_steps={horizon}")
    log.info(
        f"[eval_settle150] filter_full_horizon_and_above_mean={bool(filter_full_horizon_and_above_mean)}, count_mode={count_mode}"
    )

    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    saved_returns: list[float] = []
    saved_lengths: list[int] = []

    if save_traj and scenario is None:
        raise ValueError("scenario must be provided when save_traj=True")

    current_episode_actions: dict[int, list] = {i: [] for i in range(num_eval_envs)}
    current_episode_states: dict[int, list] = {i: [] for i in range(num_eval_envs)} if save_states else {}
    current_episode_init_state: dict[int, dict] = {i: {} for i in range(num_eval_envs)}
    episode_step_count: dict[int, int] = {i: 0 for i in range(num_eval_envs)}
    current_episode_policy_actions: dict[int, list] = {i: [] for i in range(num_eval_envs)} if save_policy_actions else {}

    saved_episodes: list[dict[str, Any]] = []
    saved_by_env: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_eval_envs)}

    ep_return = torch.zeros(num_eval_envs, device=device, dtype=torch.float32)
    ep_len = torch.zeros(num_eval_envs, device=device, dtype=torch.long)  # counts steps within episode

    obs, info = env.reset()
    if save_traj:
        for i in range(num_eval_envs):
            current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)

    max_total_steps = int(max(1, target_saved_episodes) * max(1, horizon) * 200)

    # Resolve robot name once (used for trajectory recording fallbacks).
    robot_name = None
    try:
        if scenario is not None and hasattr(scenario, "robots") and scenario.robots:
            robot_name = scenario.robots[0].name
    except Exception:
        robot_name = None
    if robot_name is None:
        robot_name = getattr(env, "robot_name", None)

    total_steps = 0
    # For count_mode="attempted", we stop after completing this many episodes (across all envs).
    # For count_mode="saved", we stop after saving this many accepted episodes.
    completed_episode_count = 0
    while True:
        if save_per_env:
            if len(saved_episodes) >= target_saved_episodes:
                break
        else:
            if count_mode == "saved":
                if len(saved_episodes) >= target_saved_episodes:
                    break
            else:
                if completed_episode_count >= target_saved_episodes:
                    break
        total_steps += 1
        if total_steps > max_total_steps:
            log.warning(
                "Reached max_total_steps=%d before collecting enough filtered episodes: saved=%d/%d. "
                "Consider relaxing filters.",
                max_total_steps,
                len(saved_episodes),
                target_saved_episodes,
            )
            break

        # Phase masks (per-env, based on ep_len):
        # - stage0: policy
        # - stage1: dummy (settle)
        # - stage2: policy again (for tasks like approach_track that resume control)
        stage0_mask = ep_len < int(ph)
        stage2_start = int(ph) + int(ss)
        stage1_mask = (ep_len >= int(ph)) & (ep_len < stage2_start)
        stage2_mask = ep_len >= stage2_start

        # Build actions:
        # - stage0/stage2: actor(norm_obs)
        # - stage1: dummy actions
        if torch.any(stage0_mask | stage2_mask):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = obs_normalizer(obs)
                policy_actions_tensor = actor(norm_obs).float()
        else:
            policy_actions_tensor = torch.zeros((num_eval_envs, env.num_actions), device=device, dtype=torch.float32)

        # Dummy actions for settle window
        if dummy_action not in ("zeros", "hold_last"):
            raise ValueError("dummy_action must be one of: zeros, hold_last")
        if dummy_action == "zeros":
            dummy = torch.zeros_like(policy_actions_tensor)
        else:
            # hold_last: reuse previous actor output as a stable placeholder
            dummy = policy_actions_tensor.detach().clone()

        policy_mask = stage0_mask | stage2_mask
        actions = torch.where(policy_mask.unsqueeze(-1), policy_actions_tensor, dummy)

        next_obs, rewards, terminated, time_out, infos = env.step(actions)
        dones = terminated | time_out

        ep_return = ep_return + rewards
        ep_len = ep_len + 1

        # Record trajectory data (downsample)
        if save_traj:
            handler_states = None
            if hasattr(env, "handler") and env.handler is not None:
                handler_states = env.handler.get_states(mode="tensor")

            default_joint_names = None
            try:
                if hasattr(env, "joint_names") and env.joint_names:
                    default_joint_names = list(env.joint_names)
                elif (
                    scenario is not None
                    and hasattr(scenario, "robots")
                    and scenario.robots
                    and getattr(scenario.robots[0], "actuators", None)
                ):
                    default_joint_names = sorted(scenario.robots[0].actuators.keys())
            except Exception:
                default_joint_names = None

            for i in range(num_eval_envs):
                if (episode_step_count[i] % save_every_n_steps) == 0:
                    # 1) record executed target dof_pos (what the sim actually commanded)
                    joint_names = None
                    joint_positions = None
                    source = "unknown"

                    if hasattr(env, "_last_action_target") and env._last_action_target is not None:
                        try:
                            joint_positions = env._last_action_target[i].detach().cpu().numpy()
                            joint_names = list(
                                getattr(env, "_action_joint_names", None) or getattr(env, "joint_names", None) or []
                            )
                            if len(joint_names) == 0:
                                joint_names = None
                            source = "env._last_action_target"
                        except Exception:
                            joint_names = None
                            joint_positions = None

                    if joint_positions is None:
                        if robot_name is None:
                            robot_name = getattr(env, "robot_name", None)
                        joint_names = default_joint_names

                    if robot_name is None:
                        if handler_states is not None and hasattr(handler_states, "robots") and len(handler_states.robots) > 0:
                            robot_name = next(iter(handler_states.robots.keys()))

                    if (
                        handler_states is not None
                        and robot_name is not None
                        and hasattr(handler_states, "robots")
                        and robot_name in handler_states.robots
                    ):
                        robot_state = handler_states.robots[robot_name]
                        # IMPORTANT: Prefer *target* joint positions (what was commanded).
                        # Also prefer handler-reported joint name ordering to avoid name/idx mismatch.
                        try:
                            if hasattr(env, "handler") and env.handler is not None and hasattr(env.handler, "get_joint_names"):
                                hn = env.handler.get_joint_names(robot_name)
                                if hn:
                                    joint_names = list(hn)
                        except Exception:
                            pass

                        if getattr(robot_state, "joint_pos_target", None) is not None:
                            joint_positions = robot_state.joint_pos_target[i].cpu().numpy()
                            source = "handler.joint_pos_target"

                    # STRICT: must record target; do NOT fall back to current joint_pos.
                    if joint_positions is None:
                        raise RuntimeError(
                            "Failed to record dof_pos_target: neither env._last_action_target nor "
                            "handler.joint_pos_target is available. Refusing to fall back to joint_pos."
                        )

                    if joint_names is None:
                        joint_names = list(getattr(env, "joint_names", []))

                    action_record = {
                        "dof_pos_target": {name: float(pos) for name, pos in zip(joint_names, joint_positions)},
                        "meta": {
                            "t": int(ep_len[i].item()),
                            "is_settle": bool(int(ep_len[i].item()) > int(ph)),
                            "source": str(source),
                        },
                    }
                    current_episode_actions[i].append(action_record)

                    # 2) optionally record raw policy action (network output) only (helps debug "settle ignored")
                    if save_policy_actions:
                        pa = policy_actions_tensor[i].detach().cpu().numpy().astype(np.float32)
                        current_episode_policy_actions[i].append(
                            {
                                "raw_action": pa,
                                "meta": {
                                    "t": int(ep_len[i].item()),
                                    "used": bool(policy_mask[i].item()),
                                },
                            }
                        )

                    if save_states:
                        current_episode_states[i].append(extract_state_dict(env, scenario, env_idx=i))

                episode_step_count[i] += 1

        done_envs = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_envs.numel() > 0:
            mean_so_far = float(np.mean(completed_returns)) if completed_returns else float("-inf")
            for i_t in done_envs.tolist():
                ret = float(ep_return[i_t].item())
                length = int(ep_len[i_t].item())

                completed_returns.append(ret)
                completed_lengths.append(length)
                completed_episode_count += 1

                if save_per_env:
                    accept = True
                else:
                    if bool(filter_full_horizon_and_above_mean):
                        is_full = length == int(horizon)
                        above_mean = (ret >= mean_so_far) if completed_returns else True
                        accept = bool(is_full and above_mean)
                    else:
                        accept = True

                if accept:
                    saved_returns.append(ret)
                    saved_lengths.append(length)
                    if save_traj and len(current_episode_actions[i_t]) > 0:
                        # Fix last state frame to be true terminal (pre-reset) if available.
                        if save_states and "terminal_states" in infos and current_episode_states[i_t]:
                            try:
                                term = infos["terminal_states"]
                                env_ids = term.get("env_ids")
                                if env_ids is not None:
                                    import numpy as _np

                                    env_ids_np = _np.asarray(env_ids)
                                    idxs = _np.where(env_ids_np == int(i_t))[0]
                                    if idxs.size > 0:
                                        j = int(idxs[0])
                                        fixed = {}
                                        for oname, o in term.get("objects", {}).items():
                                            rs = o.get("root_state")
                                            if rs is None:
                                                continue
                                            rs = rs[j].numpy()
                                            fixed[oname] = {"pos": rs[:3], "rot": rs[3:7]}
                                        for rname, r in term.get("robots", {}).items():
                                            rs = r.get("root_state")
                                            if rs is None:
                                                continue
                                            rs = rs[j].numpy()
                                            entry = {"pos": rs[:3], "rot": rs[3:7]}
                                            jp = r.get("joint_pos")
                                            if jp is not None:
                                                try:
                                                    robot_cfg = next(rr for rr in scenario.robots if rr.name == rname)
                                                    joint_names = sorted(robot_cfg.actuators.keys()) if robot_cfg.actuators else []
                                                except Exception:
                                                    joint_names = []
                                                if joint_names:
                                                    jpos = jp[j].numpy()
                                                    entry["dof_pos"] = {n: float(v) for n, v in zip(joint_names, jpos)}
                                            fixed[rname] = entry
                                        current_episode_states[i_t][-1] = fixed
                            except Exception:
                                pass

                        ep_record = {
                            "init_state": current_episode_init_state[i_t],
                            "actions": current_episode_actions[i_t],
                            "states": current_episode_states[i_t] if save_states else None,
                            "policy_actions": current_episode_policy_actions[i_t] if save_policy_actions else None,
                            "meta": {
                                "env_id": int(i_t),
                                "return": ret,
                                "length": length,
                                "policy_horizon": int(ph),
                                "settle_steps": int(ss),
                                "dummy_action": str(dummy_action),
                            },
                        }
                        saved_episodes.append(ep_record)
                        saved_by_env[int(i_t)].append(ep_record)

                    if len(saved_episodes) >= target_saved_episodes:
                        break

                # reset trackers for this env (env auto-resets inside RLTaskEnv.step)
                ep_return[i_t] = 0.0
                ep_len[i_t] = 0
                episode_step_count[i_t] = 0
                current_episode_actions[i_t] = []
                if save_states:
                    current_episode_states[i_t] = []
                if save_policy_actions:
                    current_episode_policy_actions[i_t] = []
                if save_traj:
                    current_episode_init_state[i_t] = extract_state_dict(env, scenario, env_idx=i_t)

        obs = next_obs

        if save_per_env:
            if all(len(saved_by_env[i]) >= episodes_per_env for i in range(num_eval_envs)):
                break

    if save_traj:
        if len(saved_episodes) > 0:
            robot_name = scenario.robots[0].name
            if save_per_env:
                ordered: list[dict[str, Any]] = []
                for env_id in range(num_eval_envs):
                    ordered.extend(saved_by_env[env_id][:episodes_per_env])
                trajs = {robot_name: ordered}
            else:
                trajs = {robot_name: saved_episodes}
            os.makedirs(traj_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Keep 'v2' in filename so downstream replay utilities can select the right loader by name.
            # (Even if they don't, we also added content-based auto-detection.)
            filename = f"{task_name}_{robot_name}_eval_settle_{timestamp}_v2.pkl"
            filepath = os.path.join(traj_dir, filename)
            save_traj_file(trajs, filepath)
            log.info(f"Trajectories saved to {filepath}")
            log.info(f"  - Saved episodes: {len(saved_episodes)} / target {target_saved_episodes}")
            log.info(f"  - Saved steps: {sum(len(ep['actions']) for ep in saved_episodes)}")
            log.info(f"  - Attempts (completed episodes): {len(completed_returns)}")
        else:
            log.warning("No trajectories matched the filter; nothing saved.")

    stats = {
        "mean_return": float(np.mean(saved_returns)) if saved_returns else 0.0,
        "std_return": float(np.std(saved_returns)) if saved_returns else 0.0,
        "mean_length": float(np.mean(saved_lengths)) if saved_lengths else 0.0,
        "std_length": float(np.std(saved_lengths)) if saved_lengths else 0.0,
        "num_episodes": len(saved_returns),
        "attempted_episodes": len(completed_returns),
        "count_mode": str(count_mode),
        "filter_full_horizon_and_above_mean": bool(filter_full_horizon_and_above_mean),
        "horizon": int(horizon),
        "policy_horizon": int(ph),
        "settle_steps": int(ss),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="FastTD3 Evaluation (explicit settle window after policy horizon)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/track_15000.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override task name (e.g. pick_place.approach_rand_franka). If not set, use checkpoint config['task'].",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
        help="评测 episode 数量。默认按完成的 episode 数计数（count_mode=attempted）。",
    )
    parser.add_argument("--device_rank", type=int, default=0, help="GPU device rank")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: from checkpoint config)",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--simulator",
        type=str,
        default=None,
        help="Override simulator backend (e.g. mujoco / isaacsim). Default: use checkpoint config.",
    )

    # Trajectory saving args
    parser.add_argument("--save_traj", type=int, default=1, help="Save trajectories: 0=no, 1=yes (default)")
    parser.add_argument("--save_states", type=int, default=1, help="Save full states: 0=no, 1=yes (default)")
    parser.add_argument("--save_every_n_steps", type=int, default=1, help="Save every N steps")
    parser.add_argument("--traj_dir", type=str, default="eval_trajs", help="Directory to save trajectories")
    parser.add_argument(
        "--save_per_env",
        type=int,
        default=0,
        help="If 1, save episodes per env in env_id order (no return filter).",
    )
    parser.add_argument("--episodes_per_env", type=int, default=1, help="When save_per_env=1, how many episodes to save")

    # NEW: explicit settle logic controls
    parser.add_argument(
        "--policy_horizon",
        type=int,
        default=None,
        help="Number of steps to query the actor before entering settle window. Default: use env.EPISODE_HORIZON.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=None,
        help="Number of settle steps after policy_horizon. Default: use env.IK_SETTLE_STEPS. If env.max_episode_steps is set, it wins.",
    )
    parser.add_argument(
        "--dummy_action",
        type=str,
        default="zeros",
        choices=["zeros", "hold_last"],
        help="What action to feed env during settle window (policy is ignored anyway).",
    )
    parser.add_argument(
        "--save_policy_actions",
        type=int,
        default=1,
        help="If 1, also save raw policy action vectors alongside executed joint targets.",
    )
    parser.add_argument(
        "--count_mode",
        type=str,
        default="attempted",
        choices=["attempted", "saved"],
        help="attempted: 跑完 N 个 episode 就停；saved: 收集到 N 个被接受的 episode 才停（旧 evaluate.py 行为）。",
    )
    parser.add_argument(
        "--filter_full_horizon_and_above_mean",
        type=int,
        default=0,
        help="若为 1，则只接受 length==env.max_episode_steps 且 return>=历史均值的 episode（可能导致'存个没完'）。",
    )
    parser.add_argument(
        "--ignore_terminated",
        type=int,
        default=0,
        help="若为 1，则忽略 task 的提前终止（只按 timeout 结束），用于生成完整 (policy_horizon + settle_steps) 轨迹。",
    )

    args = parser.parse_args()

    save_traj = bool(args.save_traj)
    save_states = bool(args.save_states)
    save_per_env = bool(args.save_per_env)
    save_policy_actions = bool(args.save_policy_actions)

    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config", {})

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
        torch.cuda.set_device(args.device_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device_rank}")

    log.info(f"Using device: {device}")
    log.info(f"Checkpoint global step: {checkpoint.get('global_step', 'unknown')}")

    task_name = args.task if args.task else config.get("task")
    if not task_name:
        raise ValueError("Task name not provided (use --task) and not found in checkpoint config['task']")

    task_cls = get_task_class(task_name)
    num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 1)

    simulator = args.simulator if args.simulator is not None else config.get("sim", "mujoco")
    scenario = task_cls.scenario.update(
        robots=config.get("robots", ["franka"]),
        simulator=simulator,
        num_envs=num_envs,
        headless=bool(args.headless),
        cameras=[],
    )

    try:
        env = task_cls(scenario, device=device)
    except ModuleNotFoundError as e:
        if "isaaclab" in str(e).lower():
            raise ModuleNotFoundError(
                "缺少依赖 `isaaclab`（IsaacLab）。如果你只是想本地快速跑通脚本，可以加参数 `--simulator mujoco`；"
                "如果你要用 isaacsim 评测，请先安装/配置 IsaacLab。"
            ) from e
        raise

    n_obs = env.num_obs
    n_act = env.num_actions

    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        device=device,
        init_scale=config.get("init_scale", 0.1),
        hidden_dim=config.get("actor_hidden_dim", 256),
    )
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    amp_enabled = config.get("amp", False) and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    log.info(f"Evaluating for {args.num_episodes} episodes (saved/filtered)...")
    log.info(f"  - Save trajectories: {save_traj}")
    log.info(f"  - Save states: {save_states}")
    log.info(f"  - Save policy actions: {save_policy_actions}")
    if save_traj:
        log.info(f"Saving trajectories to {args.traj_dir}/ (every {args.save_every_n_steps} steps)")

    stats = evaluate(
        env=env,
        actor=actor,
        obs_normalizer=obs_normalizer,
        num_episodes=args.num_episodes,
        device=device,
        scenario=scenario,
        task_name=task_name,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        save_traj=save_traj,
        save_states=save_states,
        save_every_n_steps=args.save_every_n_steps,
        traj_dir=args.traj_dir,
        save_per_env=save_per_env,
        episodes_per_env=args.episodes_per_env,
        policy_horizon=args.policy_horizon,
        settle_steps=args.settle_steps,
        dummy_action=args.dummy_action,
        save_policy_actions=save_policy_actions,
        filter_full_horizon_and_above_mean=bool(args.filter_full_horizon_and_above_mean),
        count_mode=args.count_mode,
        ignore_terminated=bool(args.ignore_terminated),
    )

    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"  Episodes (saved): {stats['num_episodes']}")
    log.info(f"  Mean Return: {stats['mean_return']:.4f} ± {stats['std_return']:.4f}")
    log.info(f"  Mean Length: {stats['mean_length']:.4f} ± {stats['std_length']:.4f}")
    log.info(f"  Horizon: {stats['horizon']} (policy {stats['policy_horizon']} + settle {stats['settle_steps']})")
    log.info("=" * 50)

    env.close()


if __name__ == "__main__":
    main()


