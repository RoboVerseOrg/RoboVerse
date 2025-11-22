# 分阶段训练说明 (Stage-based Training Guide)

## 概述

Pick and Place 任务被拆分为两个阶段进行训练：

1. **Stage 1: Approach & Grasp & Lift** (`pick_place_approach_grasp`)
   - 学习接近物体、抓取物体、并抬起物体
   - 成功条件：物体被抓取并抬起超过阈值高度

2. **Stage 2: Trajectory Tracking** (`pick_place_track`)
   - 学习在抓取物体的同时跟踪轨迹点
   - 从 Stage 1 的 checkpoint 初始化

## 使用方法

### Stage 1 训练

```bash
# 使用 Stage 1 配置文件训练
python roboverse_learn/rl/fast_td3/train.py \
    --config roboverse_learn/rl/fast_td3/configs/pick_place_stage1.yaml
```

Stage 1 会从头开始训练，学习基本的抓取和抬起技能。

### Stage 2 训练

```bash
# 使用 Stage 2 配置文件训练（会自动加载 Stage 1 checkpoint）
python roboverse_learn/rl/fast_td3/train.py \
    --config roboverse_learn/rl/fast_td3/configs/pick_place_stage2.yaml
```

Stage 2 会从 Stage 1 的 checkpoint 加载权重，然后继续训练轨迹跟踪技能。

**注意**：在 `pick_place_stage2.yaml` 中，需要设置 `checkpoint_path` 指向 Stage 1 的 checkpoint 文件路径。

## 配置文件说明

### Stage 1 配置 (`pick_place_stage1.yaml`)

- `task: "pick_place_approach_grasp"` - 使用 Stage 1 task
- `checkpoint_path: null` - 从头开始训练
- Reward 包括：
  - `gripper_approach`: 接近物体的奖励
  - `gripper_close`: 接近物体的奖励
  - `grasp_success`: 成功抓取的奖励（一次性奖励，+50）
  - `lift_success`: 成功抬起的奖励（一次性奖励，+100）

### Stage 2 配置 (`pick_place_stage2.yaml`)

- `task: "pick_place_track"` - 使用 Stage 2 task
- `checkpoint_path: "/path/to/stage1/checkpoint.pt"` - 从 Stage 1 checkpoint 加载
- Reward 包括：
  - `tracking_approach`: 接近轨迹点的奖励
  - `tracking_progress`: 到达轨迹点的奖励（一次性奖励，+150）
  - `grasp_maintain`: 维持抓取的奖励

## Task 实现细节

### Stage 1: `PickPlaceApproachGrasp`

- **Reward 函数**：
  - `_reward_gripper_approach`: 鼓励接近物体
  - `_reward_gripper_close`: 鼓励接近物体
  - `_reward_robot_target_qpos`: 鼓励保持目标关节位置
  - `_reward_grasp_success`: 成功抓取时给予一次性奖励（在 `step()` 中添加）
  - `_reward_lift_success`: 成功抬起时给予一次性奖励（在 `step()` 中添加）

- **成功条件**：
  - 物体被抓取（手闭合且接近物体）
  - 物体被抬起超过阈值高度（默认 0.85m）

### Stage 2: `PickPlaceTrack`

- **Reward 函数**：
  - `_reward_trajectory_tracking`: 跟踪轨迹点的奖励（继承自基类）
  - `_reward_grasp_maintain`: 维持抓取的奖励

- **初始化**：
  - 如果提供了 Stage 1 checkpoint 路径，会在 `reset()` 时尝试将物体初始化为已抓取状态
  - 训练脚本会自动从 checkpoint 加载 actor 和 obs_normalizer 的权重

## Checkpoint 加载

训练脚本 (`train.py`) 会自动处理 checkpoint 加载。在 Stage 2 配置文件中设置 `checkpoint_path` 后，训练脚本会自动：

1. 加载 Stage 1 checkpoint 中的网络权重
2. 初始化 actor 和 obs_normalizer 的权重
3. 继续训练 Stage 2

```python
# train.py 中的加载逻辑
if cfg("checkpoint_path"):
    torch_checkpoint = torch.load(f"{cfg('checkpoint_path')}", map_location=device, weights_only=False)
    actor.load_state_dict(torch_checkpoint["actor_state_dict"])
    obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
    critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
    qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
    qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
    global_step = torch_checkpoint["global_step"]
```

**注意**：
- 如果 Stage 1 和 Stage 2 的观察空间维度不同（例如，Stage 2 增加了轨迹点信息），PyTorch 的 `load_state_dict` 可能会报错。在这种情况下，可以修改训练脚本使用 `strict=False`。
- Stage 2 task 会通过环境变量 `STAGE2_START_WITH_GRASP` 控制是否从已抓取状态开始（默认为 `true`）。如果设置为 `false`，task 会从头开始学习抓取和跟踪。

## 自定义配置

可以在配置文件中调整以下参数：

### Stage 1

- `grasp_check_distance`: 抓取检查距离（默认 0.12m）
- `lift_height_threshold`: 抬起高度阈值（默认 0.85m）
- `grasp_success`: 抓取成功奖励（默认 50.0）
- `lift_success`: 抬起成功奖励（默认 100.0）

### Stage 2

- `reach_threshold`: 到达轨迹点的阈值（默认 0.10m）
- `tracking_approach`: 接近轨迹点的奖励权重（默认 4.0）
- `tracking_progress`: 到达轨迹点的奖励权重（默认 150.0）
- `grasp_maintain`: 维持抓取的奖励权重（默认 1.0）

## 评估

可以使用 `evaluate.py` 评估训练好的模型：

```bash
# 评估 Stage 1
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint /path/to/stage1/checkpoint.pt \
    --num_episodes 10

# 评估 Stage 2
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint /path/to/stage2/checkpoint.pt \
    --num_episodes 10
```

