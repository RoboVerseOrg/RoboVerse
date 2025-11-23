"""评估脚本：收集成功的lift轨迹

在第一次stable grasp进入lift时记录state，成功lift（保持10帧）后保存traj和state。
循环评估直到收集到指定数量的成功轨迹（默认100条）。
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
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

import torch
import numpy as np
from loguru import logger as log
from torch.amp import autocast
from datetime import datetime

from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import save_traj_file


def extract_state_dict(env, scenario, env_idx=0):
    """Extract state dictionary from handler states.

    Args:
        env: Environment with handler
        scenario: Scenario configuration to get joint names
        env_idx: Environment index to extract state from

    Returns:
        Dictionary containing positions, rotations, and joint positions for all objects and robots
    """
    state_dict = {}

    # Get states from handler (returns TensorState object)
    if not hasattr(env, 'handler') or env.handler is None:
        log.warning("Handler not available, returning empty state")
        return state_dict

    handler_states = env.handler.get_states(mode="tensor")
    if handler_states is None:
        log.warning("Handler.get_states() returned None")
        return state_dict

    # Create lookup dicts for configurations
    obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
    robot_cfg_dict = {robot.name: robot for robot in scenario.robots}

    # Extract object states
    if hasattr(handler_states, 'objects'):
        for obj_name, obj_state in handler_states.objects.items():
            pos = obj_state.root_state[env_idx, :3].cpu().numpy()  # [x, y, z]
            quat = obj_state.root_state[env_idx, 3:7].cpu().numpy()  # [w, x, y, z]

            state_entry = {
                "pos": pos,
                "rot": quat,
            }

            # Add joint positions if the object has joints
            if obj_state.joint_pos is not None and obj_name in obj_cfg_dict:
                obj_cfg = obj_cfg_dict[obj_name]
                if hasattr(obj_cfg, "actuators") and obj_cfg.actuators is not None:
                    # Joint names are sorted alphabetically (standard in handlers)
                    joint_names = sorted(obj_cfg.actuators.keys())
                    joint_positions = obj_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

            state_dict[obj_name] = state_entry

    # Extract robot states
    if hasattr(handler_states, 'robots'):
        for robot_name, robot_state in handler_states.robots.items():
            pos = robot_state.root_state[env_idx, :3].cpu().numpy()  # [x, y, z]
            quat = robot_state.root_state[env_idx, 3:7].cpu().numpy()  # [w, x, y, z]

            state_entry = {
                "pos": pos,
                "rot": quat,
            }

            # Add joint positions for robot
            if robot_name in robot_cfg_dict:
                robot_cfg = robot_cfg_dict[robot_name]
                if robot_cfg.actuators is not None:
                    # Joint names are sorted alphabetically (standard in handlers)
                    joint_names = sorted(robot_cfg.actuators.keys())
                    joint_positions = robot_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

            state_dict[robot_name] = state_entry

    return state_dict


def tensor_to_list(data):
    """Recursively convert tensors to lists/numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def evaluate_lift_collection(
    env,
    actor,
    obs_normalizer,
    target_count: int,
    device: torch.device,
    scenario=None,
    task_name: str = "eval",
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    traj_dir: str = "eval_trajs",
    state_dir: str = "eval_states",
    lift_stable_frames: int = 10,
) -> dict:
    """
    评估并收集成功的lift轨迹。

    Args:
        env: 环境
        actor: 策略网络
        obs_normalizer: 观察归一化器
        target_count: 目标收集的成功轨迹数量
        device: 设备
        scenario: 场景配置
        task_name: 任务名称
        amp_enabled: 是否启用自动混合精度
        amp_device_type: AMP设备类型
        amp_dtype: AMP数据类型
        traj_dir: 轨迹保存目录
        state_dir: 状态保存目录
        lift_stable_frames: lift需要保持的帧数（默认10帧）

    Returns:
        包含统计信息的字典
    """
    actor.eval()
    obs_normalizer.eval()

    num_eval_envs = env.num_envs
    collected_trajs = []  # 收集的成功轨迹列表
    collected_states = []  # 收集的成功状态列表

    # 为每个环境跟踪状态
    # lift_start_state: 进入lift时的状态（每个env一个）
    lift_start_state = {}  # Dict: env_id -> state dict
    lift_frame_count = {}  # Dict: env_id -> lift保持的帧数
    in_lift_phase = {}  # Dict: env_id -> 是否在lift阶段
    recording_traj = {}  # Dict: env_id -> 是否正在记录轨迹（从进入lift开始）

    # 初始化跟踪变量
    for i in range(num_eval_envs):
        lift_start_state[i] = None
        lift_frame_count[i] = 0
        in_lift_phase[i] = False
        recording_traj[i] = False

    # 当前episode的轨迹记录（从episode开始到当前）
    current_episode_actions = {}  # Dict: env_id -> current episode actions
    current_episode_states = {}  # Dict: env_id -> current episode states
    current_episode_init_state = {}  # Dict: env_id -> init state

    for i in range(num_eval_envs):
        current_episode_actions[i] = []
        current_episode_states[i] = []
        current_episode_init_state[i] = None

    episodes_completed = 0
    current_returns = torch.zeros(num_eval_envs, device=device)
    current_lengths = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

    obs, info = env.reset()

    # 记录初始状态
    for i in range(num_eval_envs):
        current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)

    max_steps_per_episode = env.max_episode_steps
    max_total_steps = max_steps_per_episode * 10000  # 最多运行10000个episode

    log.info(f"开始收集lift轨迹，目标数量: {target_count}")

    for step in range(max_total_steps):
        # 如果已收集足够的轨迹，停止
        if len(collected_trajs) >= target_count:
            log.info(f"已收集 {len(collected_trajs)} 条成功轨迹，达到目标数量 {target_count}")
            break

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())
        dones = terminated | time_out

        # 获取handler states用于记录
        handler_states = None
        if hasattr(env, 'handler') and env.handler is not None:
            handler_states = env.handler.get_states(mode="tensor")

        # 处理每个环境
        for i in range(num_eval_envs):
            if done_masks[i]:
                continue

            # 获取当前状态信息
            grasp_success = infos.get("grasp_success", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]
            lift_active = infos.get("lift_active", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]

            # 始终记录当前步骤的action和state（从episode开始）
            robot_name = scenario.robots[0].name
            joint_names = sorted(scenario.robots[0].actuators.keys())

            if handler_states is not None and hasattr(handler_states, 'robots') and robot_name in handler_states.robots:
                robot_state = handler_states.robots[robot_name]
                joint_positions = robot_state.joint_pos[i].cpu().numpy()
            else:
                robot_state = obs.robots[robot_name]
                joint_positions = robot_state.joint_pos[i].cpu().numpy()

            action_record = {
                "dof_pos_target": {name: float(pos) for name, pos in zip(joint_names, joint_positions)},
            }

            # 记录到当前episode轨迹（从episode开始就记录）
            current_episode_actions[i].append(action_record)
            current_state = extract_state_dict(env, scenario, env_idx=i)
            current_episode_states[i].append(current_state)

            # 检测第一次进入lift阶段（grasp成功且lift激活）
            if grasp_success and lift_active and not in_lift_phase[i]:
                # 第一次进入lift，记录当前状态（这是进入lift时的state）
                in_lift_phase[i] = True
                lift_start_state[i] = extract_state_dict(env, scenario, env_idx=i)
                lift_frame_count[i] = 1
                recording_traj[i] = True  # 标记正在记录轨迹

                log.info(f"[Env {i}] 进入lift阶段 (grasp成功且lift激活)")

            # 如果已经在lift阶段，更新帧计数
            elif in_lift_phase[i]:
                if lift_active and grasp_success:
                    lift_frame_count[i] += 1

                    # 如果lift保持足够长时间，保存轨迹
                    if lift_frame_count[i] >= lift_stable_frames:
                        # 成功！保存轨迹和状态
                        # 使用完整的episode轨迹（从episode开始到lift成功）
                        traj_data = {
                            "init_state": current_episode_init_state[i],
                            "actions": current_episode_actions[i],
                            "states": current_episode_states[i],
                        }

                        # 转换为可序列化格式
                        traj_data_serializable = tensor_to_list(traj_data)
                        state_data_serializable = tensor_to_list(lift_start_state[i])

                        collected_trajs.append(traj_data_serializable)
                        collected_states.append(state_data_serializable)

                        log.info(
                            f"[Env {i}] 成功收集第 {len(collected_trajs)} 条轨迹 "
                            f"(lift保持 {lift_frame_count[i]} 帧, 总步数: {len(current_episode_actions[i])})"
                        )

                        # 重置该环境的跟踪变量（但不清空current_episode，因为episode可能还在继续）
                        lift_start_state[i] = None
                        lift_frame_count[i] = 0
                        in_lift_phase[i] = False
                        recording_traj[i] = False

                        # 如果已收集足够的轨迹，可以提前终止这个环境
                        if len(collected_trajs) >= target_count:
                            done_masks[i] = True
                else:
                    # lift中断（grasp失败或lift不活跃），重置
                    lift_frame_count[i] = 0
                    if not grasp_success:
                        in_lift_phase[i] = False
                        recording_traj[i] = False

        # 更新episode统计
        active_mask = ~done_masks
        current_returns = torch.where(active_mask, current_returns + rewards, current_returns)
        current_lengths = torch.where(active_mask, current_lengths + 1, current_lengths)

        # 检查episode是否结束
        newly_done = dones & ~done_masks
        if newly_done.any():
            for i in range(num_eval_envs):
                if newly_done[i]:
                    episodes_completed += 1

                    # 重置该环境的跟踪变量
                    lift_start_state[i] = None
                    lift_frame_count[i] = 0
                    in_lift_phase[i] = False
                    recording_traj[i] = False
                    current_episode_actions[i] = []
                    current_episode_states[i] = []
                    current_episode_init_state[i] = None
                    current_returns[i] = 0
                    current_lengths[i] = 0

            done_masks = torch.logical_or(done_masks, dones)

        # 如果所有环境都结束了，重置
        if done_masks.all():
            done_masks.fill_(False)
            obs, info = env.reset()

            # 重置所有跟踪变量
            for i in range(num_eval_envs):
                lift_start_state[i] = None
                lift_frame_count[i] = 0
                in_lift_phase[i] = False
                recording_traj[i] = False
                current_episode_actions[i] = []
                current_episode_states[i] = []
                current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)
        else:
            obs = next_obs

    # 保存收集的轨迹和状态
    if len(collected_trajs) > 0:
        # 创建输出目录
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        # 组织轨迹数据
        robot_name = scenario.robots[0].name
        trajs = {robot_name: collected_trajs}

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_filename = f"{task_name}_{robot_name}_lift_{len(collected_trajs)}trajs_{timestamp}_v2.pkl"
        state_filename = f"{task_name}_{robot_name}_lift_states_{len(collected_states)}states_{timestamp}.pkl"

        traj_filepath = os.path.join(traj_dir, traj_filename)
        state_filepath = os.path.join(state_dir, state_filename)

        # 保存轨迹
        save_traj_file(trajs, traj_filepath)
        log.info(f"轨迹已保存到: {traj_filepath}")
        log.info(f"  - 轨迹数量: {len(collected_trajs)}")
        log.info(f"  - 总步数: {sum(len(traj['actions']) for traj in collected_trajs)}")

        # 保存状态
        with open(state_filepath, "wb") as f:
            pickle.dump(collected_states, f)
        log.info(f"状态已保存到: {state_filepath}")
        log.info(f"  - 状态数量: {len(collected_states)}")
    else:
        log.warning("未收集到任何成功的轨迹")

    # 计算统计信息
    stats = {
        "collected_count": len(collected_trajs),
        "target_count": target_count,
        "episodes_completed": episodes_completed,
        "success_rate": len(collected_trajs) / episodes_completed if episodes_completed > 0 else 0.0,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='FastTD3 Lift轨迹收集评估')
    parser.add_argument('--checkpoint', type=str, default='models/pick_place.approach_grasp_simple_1210000.pt',
                       help='Checkpoint文件路径')
    parser.add_argument('--target_count', type=int, default=100,
                       help='目标收集的成功轨迹数量（默认: 100）')

    parser.add_argument('--device_rank', type=int, default=0,
                       help='GPU设备rank')
    parser.add_argument('--num_envs', type=int, default=None,
                       help='并行环境数量（默认: 从checkpoint config读取）')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行')

    parser.add_argument('--traj_dir', type=str, default='eval_trajs',
                       help='轨迹保存目录')
    parser.add_argument('--state_dir', type=str, default='eval_states',
                       help='状态保存目录')
    parser.add_argument('--lift_stable_frames', type=int, default=10,
                       help='lift需要保持的帧数（默认: 10）')

    args = parser.parse_args()

    # 加载checkpoint
    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    # 从checkpoint获取配置
    config = checkpoint.get("config", {})

    # 根据可用性覆盖设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
        torch.cuda.set_device(args.device_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device_rank}")

    log.info(f"使用设备: {device}")
    log.info(f"Checkpoint global step: {checkpoint.get('global_step', 'unknown')}")

    # 获取任务配置
    task_name = config.get("task")
    if not task_name:
        raise ValueError("Checkpoint config中未找到任务名称")

    # 设置环境
    task_cls = get_task_class(task_name)
    num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 1)

    scenario = task_cls.scenario.update(
        robots=config.get("robots", ["franka"]),
        simulator=config.get("sim", "mujoco"),
        num_envs=num_envs,
        headless=args.headless,
        cameras=[],  # 不需要渲染
    )

    env = task_cls(scenario, device=device)

    # 获取维度
    n_obs = env.num_obs
    n_act = env.num_actions

    # 创建actor和normalizer
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        device=device,
        init_scale=config.get("init_scale", 0.1),
        hidden_dim=config.get("actor_hidden_dim", 256),
    )

    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    # 加载权重
    actor.load_state_dict(checkpoint["actor_state_dict"])
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    # 设置AMP
    amp_enabled = config.get("amp", False) and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    # 运行评估
    log.info(f"开始收集lift轨迹...")
    log.info(f"  - 目标数量: {args.target_count}")
    log.info(f"  - Lift稳定帧数: {args.lift_stable_frames}")
    log.info(f"  - 轨迹保存目录: {args.traj_dir}")
    log.info(f"  - 状态保存目录: {args.state_dir}")

    stats = evaluate_lift_collection(
        env=env,
        actor=actor,
        obs_normalizer=obs_normalizer,
        target_count=args.target_count,
        device=device,
        scenario=scenario,
        task_name=task_name,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        traj_dir=args.traj_dir,
        state_dir=args.state_dir,
        lift_stable_frames=args.lift_stable_frames,
    )

    # 打印结果
    log.info("=" * 50)
    log.info("评估结果:")
    log.info(f"  收集的轨迹数: {stats['collected_count']}")
    log.info(f"  目标数量: {stats['target_count']}")
    log.info(f"  完成的episode数: {stats['episodes_completed']}")
    log.info(f"  成功率: {stats['success_rate']:.2%}")
    log.info("=" * 50)

    # 清理
    env.close()


if __name__ == "__main__":
    main()
