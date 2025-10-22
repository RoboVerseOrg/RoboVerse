# TaskViserWrapper 使用说明

TaskViserWrapper 是一个包装器，用于为 RLTaskEnv 添加实时 Viser 可视化功能。

## 主要特性

- 自动设置 Viser 可视化
- 实时更新机器人和物体状态
- 只渲染第一个环境（避免多环境复杂性）
- 透明代理所有环境属性和方法
- 错误处理：可视化失败不影响训练

## 使用方法

### 1. 基本使用

```python
from metasim.task.registry import get_task_class
from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

# 创建环境
task_cls = get_task_class('reach_origin')
scenario = task_cls.scenario.update(
    robots=['franka'],
    simulator='mujoco',
    num_envs=1024,
    headless=False,  # 启用渲染以支持 Viser
    cameras=[]
)

env = task_cls(scenario, device='cuda')

# 包装环境以启用可视化
viser_env = TaskViserWrapper(env, port=8080)

# 正常使用环境
obs = viser_env.reset()
for _ in range(100):
    actions = policy(obs)  # 你的策略
    obs, reward, terminated, timeout, info = viser_env.step(actions)
    if terminated.any() or timeout.any():
        obs = viser_env.reset()

viser_env.close()
```

### 2. 与 fast_td3 集成

在 fast_td3 中启用可视化：

```bash
# 启用 Viser 可视化
python get_started/rl/fast_td3/1_fttd3.py --viser-port 8080

# 不使用可视化（默认）
python get_started/rl/fast_td3/1_fttd3.py
```

配置示例：
```python
CONFIG = {
    "sim": "mujoco",
    "robots": ["franka"],
    "task": "reach_origin",
    "headless": True,
    "viser_port": 8080,  # 设置 > 0 启用可视化
    # ... 其他配置
}
```

## 工作原理

1. **初始化时**：
   - 创建 ViserVisualizer 实例
   - 下载必要的 URDF 文件
   - 可视化场景中的所有机器人和物体
   - 设置摄像头控制

2. **运行时**：
   - 每次 `reset()` 和 `step()` 后更新可视化
   - 提取第一个环境的状态
   - 更新所有机器人和物体的位置和姿态

3. **错误处理**：
   - 可视化失败不影响训练
   - 静默处理导入错误

## 环境属性代理

TaskViserWrapper 透明代理所有环境属性：

```python
wrapper = TaskViserWrapper(env)
print(wrapper.num_envs)      # 代理到 env.num_envs
print(wrapper.num_actions)   # 代理到 env.num_actions
print(wrapper.num_obs)       # 代理到 env.num_obs
print(wrapper.action_space)  # 代理到 env.action_space
```

## 技术细节

- 使用与 `viser_demo.py` 相同的状态提取逻辑
- 支持所有模拟器后端（MuJoCo、Isaac Sim 等）
- 只渲染第一个环境以保持性能
- 自动处理多环境张量的索引

