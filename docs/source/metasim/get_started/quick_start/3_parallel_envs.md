# 3. Parallel Envs
In this tutorial, we will show you how to use MetaSim to run multiple environments in parallel.

## Common Usage

```bash
python get_started/3_parallel_envs.py  --sim <simulator> --num_envs <num_envs>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will run multiple environments in parallel and it will automatically record a video.


### Examples

#### IsaacSim
```bash
python get_started/3_parallel_envs.py  --sim isaacsim --num_envs 4
```

#### Isaac Gym
```bash
python get_started/3_parallel_envs.py  --sim isaacgym --num_envs 4
```

#### Genesis
```bash
python get_started/3_parallel_envs.py  --sim genesis --num_envs 4
```
Note that we find the `headless` mode of Genesis is not stable. So we recommend using the `non-headless` mode.

#### Mujoco
```bash
# For mac users, replace python with mjpython
python get_started/3_parallel_envs.py  --sim mujoco --num_envs 4 --headless
```
**If you are on mac**, please avoid running this task without the `headless` tag.

We can open multiple environments at the same time.

<video width="50%" autoplay loop muted playsinline>
    <source src="https://roboverse.wiki/_static/standard_output/3_parallel_envs_demo.mp4" type="video/mp4">
</video>

You will get the following videos:

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/3_parallel_envs_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Lab</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/3_parallel_envs_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/3_parallel_envs_genesis.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Genesis</p>
        </div>
    </div>

</div>

## Code Highlights

**Parallel Environment Setup**: Simply change `num_envs` to enable parallel execution:
```python
scenario = ScenarioCfg(
    robots=["franka"],
    simulator=args.sim,
    num_envs=4,  # Enable 4 parallel environments
    headless=args.headless,
)
handler = get_handler(scenario)
```

**Simulator Handling**:
- **Native parallel simulators** (IsaacSim, IsaacGym, Genesis): Directly use handler with multiple environments
- **Single-environment simulators** (MuJoCo): Automatically wrapped with ParallelHandler for multi-threaded scheduling
- **Same API**: All simulators provide identical interface regardless of underlying implementation
