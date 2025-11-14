from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import tyro
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.render import RenderCfg


@dataclass
class Args:
    render: RenderCfg = field(default_factory=RenderCfg)
    """Renderer options"""
    task: str = "pick_butter"
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "isaacsim", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    # max_demo_idx: int | None = None
    # """Maximum number of demos to collect, None for all demos"""
    num_demo_success: int | None = None
    """Target number of successful demos to collect"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    custom_save_dir: str | None = None
    """Custom base path for saving demos. If None, use default structure."""
    scene: str | None = None
    """Scene name"""
    run_all: bool = True
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""
    renderer: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"

    ## Domain randomization options
    enable_randomization: bool = False
    """Enable domain randomization during demo collection"""
    randomize_materials: bool = True
    """Enable material randomization (when randomization is enabled)"""
    randomize_lights: bool = False
    """Enable light randomization (when randomization is enabled)"""
    randomize_cameras: bool = True
    """Enable camera randomization (when randomization is enabled)"""
    randomize_physics: bool = True
    """Enable physics (mass/friction/pose) randomization using ObjectRandomizer"""
    randomization_frequency: Literal["per_demo", "per_episode"] = "per_demo"
    """When to apply randomization: per_demo (once at start) or per_episode (every episode)"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    def __post_init__(self):
        assert self.run_all or self.run_unfinished or self.run_failed, (
            "At least one of run_all, run_unfinished, or run_failed must be True"
        )
        # if self.max_demo_idx is None:
        #     self.max_demo_idx = math.inf
        if self.num_demo_success is None:
            self.num_demo_success = 100
        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        # Validate randomization settings
        if self.enable_randomization:
            if not (
                self.randomize_materials or self.randomize_lights or self.randomize_cameras or self.randomize_physics
            ):
                log.warning("Randomization enabled but no randomization types selected, disabling randomization")
                self.enable_randomization = False

        log.info(f"Args: {self}")

        # Log randomization settings
        if self.enable_randomization:
            log.info("=" * 60)
            log.info("DOMAIN RANDOMIZATION CONFIGURATION")
            log.info(f"  Materials: {'✓' if self.randomize_materials else '✗'}")
            log.info(f"  Lights: {'✓' if self.randomize_lights else '✗'}")
            log.info(f"  Cameras: {'✓' if self.randomize_cameras else '✗'}")
            log.info(f"  Physics: {'✓' if self.randomize_physics else '✗'} (ObjectRandomizer)")
            log.info(f"  Frequency: {self.randomization_frequency}")
            log.info(f"  Seed: {self.randomization_seed if self.randomization_seed else 'Random'}")
            log.info("=" * 60)


args = tyro.cli(Args)

import multiprocessing as mp
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
from tqdm.rich import tqdm_rich as tqdm

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu

rootutils.setup_root(__file__, pythonpath=True)

# Import randomization components (after rootutils setup)
try:
    from roboverse_pack.randomization import (
        CameraPresets,
        CameraRandomizer,
        LightPresets,
        LightRandomizer,
        MaterialPresets,
        MaterialRandomizer,
        ObjectPresets,
        ObjectRandomizer,
    )

    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Randomization components not available: {e}")
    RANDOMIZATION_AVAILABLE = False


def log_randomization_result(
    randomizer_type: str, obj_name: str, property_name: str, before_value, after_value, unit: str = ""
):
    """Log randomization results in a consistent format."""
    if hasattr(before_value, "cpu"):
        before_str = str(before_value.cpu().numpy().round(3) if hasattr(before_value, "numpy") else before_value)
    else:
        before_str = str(before_value)

    if hasattr(after_value, "cpu"):
        after_str = str(after_value.cpu().numpy().round(3) if hasattr(after_value, "numpy") else after_value)
    else:
        after_str = str(after_value)

    log.info(f"  [{randomizer_type}] {obj_name}.{property_name}: {before_str} -> {after_str} {unit}")


def log_randomization_header(randomizer_name: str, description: str = ""):
    """Log a consistent header for randomization sections."""
    log.info("=" * 50)
    if description:
        log.info(f"{randomizer_name}: {description}")
    else:
        log.info(randomizer_name)


class DomainRandomizationManager:
    """Manages domain randomization for demo collection with unified interface."""

    def __init__(self, args: Args, scenario, handler):
        self.args = args
        self.scenario = scenario
        self.handler = handler
        self.randomizers = []
        self._demo_count = 0

        # Early validation
        if not self._validate_setup():
            return

        log_randomization_header("DOMAIN RANDOMIZATION SETUP", "Initializing randomizers")
        self._setup_randomizers()
        log.info(f"Setup complete: {len(self.randomizers)} randomizers ready")

    def _validate_setup(self) -> bool:
        """Validate if randomization can be set up."""
        if not self.args.enable_randomization:
            log.info("Domain randomization disabled")
            return False

        if not RANDOMIZATION_AVAILABLE:
            log.warning("Domain randomization requested but components not available")
            return False

        return True

    def _setup_randomizers(self):
        """Initialize all randomizers based on configuration."""
        seed = self.args.randomization_seed
        self._setup_reproducibility(seed)

        # Setup each randomization type symmetrically
        if self.args.randomize_materials:
            self._setup_material_randomizers(seed)

        if self.args.randomize_lights:
            self._setup_light_randomizers(seed)

        if self.args.randomize_cameras:
            self._setup_camera_randomizers(seed)

        if self.args.randomize_physics:
            self._setup_physics_randomizers(seed)

    def _setup_reproducibility(self, seed: int | None):
        """Setup global reproducibility if seed is provided."""
        if seed is not None:
            log.info(f"Setting up reproducible randomization with seed: {seed}")
            torch.manual_seed(seed)
            import numpy as np

            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _setup_material_randomizers(self, seed: int | None):
        """Setup material randomizers for all objects."""
        objects = getattr(self.scenario, "objects", [])
        if not objects:
            log.info("  No objects found for material randomization")
            return

        log.info(f"  Setting up material randomizers for {len(objects)} objects")
        for obj in objects:
            obj_name = obj.name
            config = self._get_material_config(obj_name)

            randomizer = MaterialRandomizer(config, seed=seed)
            randomizer.bind_handler(self.handler)
            self.randomizers.append(randomizer)
            log.info(f"    Added MaterialRandomizer for {obj_name}")

    def _setup_light_randomizers(self, seed: int | None):
        """Setup light randomizers for all lights."""
        from metasim.scenario.lights import DiskLightCfg, DomeLightCfg, SphereLightCfg

        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights found for light randomization")
            return

        log.info(f"  Setting up light randomizers for {len(lights)} lights")
        for light in lights:
            light_name = getattr(light, "name", f"light_{len(self.randomizers)}")

            if isinstance(light, DomeLightCfg):
                config = LightPresets.dome_ambient(light_name, randomization_mode="combined")
            elif isinstance(light, (SphereLightCfg, DiskLightCfg)):
                config = LightPresets.sphere_ceiling_light(light_name, randomization_mode="combined")
            else:
                log.warning(f"Unknown light type for {light_name}, using sphere_ceiling_light preset")
                config = LightPresets.sphere_ceiling_light(light_name, randomization_mode="combined")

            randomizer = LightRandomizer(config, seed=seed)
            randomizer.bind_handler(self.handler)
            self.randomizers.append(randomizer)
            log.info(f"    Added LightRandomizer for {light_name}")

    def _setup_camera_randomizers(self, seed: int | None):
        """Setup camera randomizers for all cameras."""
        cameras = getattr(self.scenario, "cameras", [])
        if not cameras:
            log.info("  No cameras found for camera randomization")
            return

        log.info(f"  Setting up camera randomizers for {len(cameras)} cameras")
        for camera in cameras:
            camera_name = getattr(camera, "name", f"camera_{len(self.randomizers)}")
            config = CameraPresets.surveillance_camera(camera_name, randomization_mode="combined")

            randomizer = CameraRandomizer(config, seed=seed)
            randomizer.bind_handler(self.handler)
            self.randomizers.append(randomizer)
            log.info(f"    Added CameraRandomizer for {camera_name}")

    def _get_material_config(self, obj_name: str):
        """Get appropriate material configuration based on object type."""
        obj_lower = obj_name.lower()
        if "cube" in obj_lower:
            return MaterialPresets.mdl_family_object(obj_name, family="metal", randomization_mode="combined")
        elif "sphere" in obj_lower:
            return MaterialPresets.rubber_object(obj_name, randomization_mode="combined")
        else:
            return MaterialPresets.mdl_family_object(obj_name, family="wood", randomization_mode="combined")

    def _setup_physics_randomizers(self, seed: int | None):
        """Setup unified ObjectRandomizers for robots and objects."""
        robots = getattr(self.scenario, "robots", [])
        objects = getattr(self.scenario, "objects", [])

        self._setup_object_randomizers(robots, objects, seed)

    def _setup_object_randomizers(self, robots: list, objects: list, seed: int | None):
        """Setup unified ObjectRandomizers for all physical entities."""
        log.info("  Setting up ObjectRandomizers for physics randomization")

        # Robot randomization
        if robots:
            robot_name = robots[0] if isinstance(robots[0], str) else robots[0].name
            robot_randomizer = ObjectRandomizer(ObjectPresets.robot_base(robot_name), seed=seed)
            robot_randomizer.bind_handler(self.handler)
            self.randomizers.append(robot_randomizer)
            log.info(f"    Added ObjectRandomizer for robot {robot_name}")

        # Object randomization
        if objects:
            for obj in objects:
                obj_name = obj.name
                config = self._get_object_physics_config(obj_name)

                obj_randomizer = ObjectRandomizer(config, seed=seed)
                obj_randomizer.bind_handler(self.handler)
                self.randomizers.append(obj_randomizer)
                log.info(f"    Added ObjectRandomizer for {obj_name}")

        if not robots and not objects:
            log.info("    No robots or objects found for physics randomization")

    def _get_object_physics_config(self, obj_name: str):
        """Get appropriate physics configuration based on object type."""
        obj_lower = obj_name.lower()
        if "cube" in obj_lower:
            return ObjectPresets.grasping_target(obj_name)
        elif "sphere" in obj_lower:
            return ObjectPresets.bouncy_object(obj_name)
        else:
            return ObjectPresets.physics_only(obj_name)

    def randomize_for_demo(self, demo_idx: int):
        """Apply randomization for a new demo."""
        if not self._should_randomize(demo_idx):
            return

        log_randomization_header("DOMAIN RANDOMIZATION", f"Demo {demo_idx}")

        # Apply all randomizers and collect statistics
        stats = self._apply_all_randomizers()

        # Log summary
        self._log_randomization_summary(stats)
        self._demo_count += 1

    def _should_randomize(self, demo_idx: int) -> bool:
        """Check if randomization should be applied for this demo."""
        if not self.args.enable_randomization or not self.randomizers:
            return False

        return self.args.randomization_frequency == "per_demo" or (
            self.args.randomization_frequency == "per_episode" and demo_idx == 0
        )

    def _apply_all_randomizers(self) -> dict[str, int]:
        """Apply all randomizers and return statistics."""
        stats = {"ObjectRandomizer": 0, "MaterialRandomizer": 0, "LightRandomizer": 0, "CameraRandomizer": 0}

        for randomizer in self.randomizers:
            try:
                obj_name = self._get_randomizer_target_name(randomizer)
                randomizer_type = type(randomizer).__name__

                # Apply randomization
                randomizer()
                stats[randomizer_type] = stats.get(randomizer_type, 0) + 1
                log.info(f"  Applied {randomizer_type} for {obj_name}")

            except Exception as e:
                log.warning(f"  {type(randomizer).__name__} failed for {obj_name}: {e}")

        return stats

    def _get_randomizer_target_name(self, randomizer) -> str:
        """Extract target object name from randomizer configuration."""
        if not hasattr(randomizer, "cfg"):
            return "unknown"

        cfg = randomizer.cfg
        if hasattr(cfg, "obj_name"):
            return cfg.obj_name
        elif hasattr(cfg, "light_name"):
            return cfg.light_name
        elif hasattr(cfg, "camera_name"):
            return cfg.camera_name
        else:
            return "unknown"

    def _log_randomization_summary(self, stats: dict[str, int]):
        """Log a summary of applied randomizers."""
        applied_types = [f"{name}: {count}" for name, count in stats.items() if count > 0]
        if applied_types:
            log.info(f"Applied randomizers: {', '.join(applied_types)}")
        else:
            log.info("No randomizers were applied")


def get_actions(all_actions, env, demo_idxs: list[int], robot: RobotCfg):
    action_idxs = env._episode_steps

    actions = []
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs)):
        if action_idx < len(all_actions[demo_idx]):
            action = all_actions[demo_idx][action_idx]
        else:
            action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int]) -> list[bool]:
    action_idxs = env._episode_steps
    run_out = [action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs)]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        # Only start checking after minimum steps
        if step >= min_steps:
            if prev_state is not None:
                # Check if key states are stable (focus on articulated objects)
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            # Check DOF positions for articulated objects
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                # Additional validation: check if we're stable at the RIGHT state
                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        # We're stable but at wrong state - force more simulation
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False
                        # Continue simulating to let physics settle properly

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:  # Stable for 2 consecutive steps at correct state
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    # Final validation if we ran out of steps
    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    # Final state refresh
    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True  # Can't validate, assume correct

    # Focus on articulated objects which are most prone to reset issues
    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True  # No critical objects to validate

    tolerance = 5e-3  # Reasonable tolerance for DOF positions

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        # Check DOF positions for articulated objects (most critical for demo consistency)
        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                # Log the specific difference for debugging
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def force_reset_to_state(env, state, env_id):
    """Force reset environment to specific state with validation."""
    env.reset(states=[state], env_ids=[env_id])
    # Pass expected state for validation
    ensure_clean_state(env.handler, expected_state=state)
    # Reset episode counter AFTER stabilization to ensure demo starts from action 0
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


class DemoCollector:
    def __init__(self, handler, robot_cfg, task_desc=""):
        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.robot_cfg = robot_cfg
        self.task_desc = task_desc
        self.cache: dict[int, list[dict]] = {}
        self.save_request_queue = mp.Queue()
        self.save_proc = mp.Process(target=save_demo_mp, args=(self.save_request_queue, robot_cfg, task_desc))
        self.save_proc.start()

        TaskName = args.task
        if args.custom_save_dir:
            self.base_save_dir = args.custom_save_dir
        else:
            additional_str = f"-{args.cust_name}" if args.cust_name else ""
            self.base_save_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    def create(self, demo_idx: int, data_dict: dict):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]

    def add(self, demo_idx: int, data_dict: dict):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))

    def save(self, demo_idx: int, status: str):
        assert demo_idx in self.cache
        assert status in ["success", "failed"], f"Invalid status: {status}"

        save_dir = os.path.join(self.base_save_dir, status, f"demo_{demo_idx:04d}")
        if os.path.exists(os.path.join(save_dir, "status.txt")):
            os.remove(os.path.join(save_dir, "status.txt"))

        os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving demo {demo_idx} to {save_dir}")

        ## Option 1: Save immediately, blocking and slower

        from metasim.utils.save_util import save_demo

        save_demo(save_dir, self.cache[demo_idx], self.robot_cfg, self.task_desc)

        if status == "failed":
            with open(os.path.join(save_dir, "status.txt"), "w") as f:
                f.write(status)

        ## Option 2: Save in a separate process, non-blocking, not friendly to KeyboardInterrupt
        # self.save_request_queue.put({"demo": self.cache[demo_idx], "save_dir": save_dir})

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]

    def final(self):
        """
        Finalize collector:
        - Save any remaining cached demos (mark them as 'failed' so they are persisted)
        - Clear the cache
        - Signal the save process to exit and join it
        """
        # If there are any remaining demos in cache, save them as 'failed' to persist data
        if self.cache:
            log.warning(f"Finalizing: {len(self.cache)} unfinished demo(s) found in cache. Saving them as 'failed'.")
        for demo_idx in list(self.cache.keys()):
            try:
                log.info(f"Finalizing: saving unfinished demo {demo_idx} as failed")
                # save will create directories and write status.txt for failed demos
                self.save(demo_idx, status="failed")
            except Exception as e:
                log.error(f"Failed to save unfinished demo {demo_idx} during finalization: {e}")
            try:
                # ensure we remove it from cache even if save failed
                self.delete(demo_idx)
            except Exception as e:
                log.error(f"Failed to delete demo {demo_idx} from cache during finalization: {e}")

        # signal the background save process to exit and join
        try:
            self.save_request_queue.put(None)  # signal to save_demo_mp to exit
            self.save_proc.join()
        except Exception as e:
            log.error(f"Error while shutting down save process: {e}")

        # ensure cache is empty (no assert, just log if something remains)
        if self.cache:
            log.error("Collector finalization completed but cache is not empty.")
        else:
            log.info("Collector finalization completed and cache is empty.")


def should_skip(log_dir: str, demo_idx: int):
    demo_name = f"demo_{demo_idx:04d}"
    success_path = os.path.join(log_dir, "success", demo_name, "status.txt")
    failed_path = os.path.join(log_dir, "failed", demo_name, "status.txt")

    if args.run_all:
        return False

    if args.run_unfinished:
        if not os.path.exists(success_path) and not os.path.exists(failed_path):
            return False
        return True

    if args.run_failed:
        if os.path.exists(success_path):
            return is_status_success(log_dir, demo_idx)
        return False

    return True


def is_status_success(log_dir: str, demo_idx: int) -> bool:
    demo_name = f"demo_{demo_idx:04d}"
    status_path = os.path.join(log_dir, "success", demo_name, "status.txt")

    if os.path.exists(status_path):
        return open(status_path).read().strip() == "success"
    return False


class DemoIndexer:
    def __init__(self, save_root_dir: str, start_idx: int, end_idx: int, pbar: tqdm):
        self.save_root_dir = save_root_dir
        self._next_idx = start_idx
        self.end_idx = end_idx
        self.pbar = pbar
        self._skip_if_should()

    @property
    def next_idx(self):
        return self._next_idx

    def _skip_if_should(self):
        while should_skip(self.save_root_dir, self._next_idx):
            global global_step, tot_success, tot_give_up
            if is_status_success(self.save_root_dir, self._next_idx):
                tot_success += 1
            else:
                tot_give_up += 1
            self.pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
            self.pbar.update(1)
            log.info(f"Demo {self._next_idx} already exists, skipping...")
            self._next_idx += 1

    def move_on(self):
        self._next_idx += 1
        self._skip_if_should()


def main():
    global global_step, tot_success, tot_give_up
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    robot = get_robot(args.robot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    # Initialize domain randomization manager
    randomization_manager = DomainRandomizationManager(args, scenario, env.handler)
    ## Data
    assert os.path.exists(env.traj_filepath), f"Trajectory file does not exist: {env.traj_filepath}"
    init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)

    tot_demo = len(all_actions)
    if args.split == "train":
        init_states = init_states[: int(tot_demo * 0.9)]
        all_actions = all_actions[: int(tot_demo * 0.9)]
        all_states = all_states[: int(tot_demo * 0.9)]
    elif args.split == "val" or args.split == "test":
        init_states = init_states[int(tot_demo * 0.9) :]
        all_actions = all_actions[int(tot_demo * 0.9) :]
        all_states = all_states[int(tot_demo * 0.9) :]

    n_demo = len(all_actions)
    log.info(f"Collecting from {args.split} split, {n_demo} out of {tot_demo} demos")

    ########################################################
    ## Main
    ########################################################
    # if args.max_demo_idx > n_demo:
    #     log.warning(
    #         f"Max demo {args.max_demo_idx} is greater than the number of demos in the dataset {n_demo}, using {n_demo}"
    #     )
    # max_demo = min(args.max_demo_idx, n_demo)
    max_demo = n_demo
    try_num = args.retry_num + 1

    ## Demo collection state machine:
    ## CollectingDemo -> Success -> FinalizeDemo -> NextDemo
    ## CollectingDemo -> Timeout -> Retry/GiveUp -> NextDemo

    ## Setup
    # Get task description from environment
    task_desc = getattr(env, "task_desc", "")
    collector = DemoCollector(env.handler, robot, task_desc)
    # pbar = tqdm(total=max_demo - args.demo_start_idx, desc="Collecting demos")
    pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")

    ## State variables
    failure_count = [0] * env.handler.num_envs
    steps_after_success = [0] * env.handler.num_envs
    finished = [False] * env.handler.num_envs
    TaskName = args.task

    if args.cust_name is not None:
        additional_str = f"-{args.cust_name}"
    else:
        additional_str = ""

    if args.custom_save_dir:
        save_root_dir = args.custom_save_dir
    else:
        save_root_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    demo_indexer = DemoIndexer(
        save_root_dir=save_root_dir,
        start_idx=args.demo_start_idx,
        end_idx=max_demo,
        pbar=pbar,
    )
    demo_idxs = []
    for demo_idx in range(env.handler.num_envs):
        demo_idxs.append(demo_indexer.next_idx)
        demo_indexer.move_on()
    log.info(f"Initialize with demo idxs: {demo_idxs}")

    ## Apply initial randomization
    for env_id, demo_idx in enumerate(demo_idxs):
        randomization_manager.randomize_for_demo(demo_idx)

    ## Reset to initial states
    obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])

    ## Wait for environment to stabilize after reset (before counting demo steps)
    # For initial setup, we can't validate individual states easily, so just ensure stability
    ensure_clean_state(env.handler)

    ## Reset episode step counters AFTER stabilization
    if hasattr(env, "_episode_steps"):
        for env_id in range(env.handler.num_envs):
            env._episode_steps[env_id] = 0

    ## Now record the clean, stabilized initial state
    obs = env.handler.get_states()
    obs = state_tensor_to_nested(env.handler, obs)
    for env_id, demo_idx in enumerate(demo_idxs):
        log.info(f"Starting Demo {demo_idx} in Env {env_id}")
        collector.create(demo_idx, obs[env_id])

    ## Main Loop
    while not all(finished):
        if tot_success >= args.num_demo_success:
            log.info(f"Reached target number of successful demos ({args.num_demo_success}). Stopping collection.")
            break

        if demo_indexer.next_idx >= max_demo:
            log.warning(f"Reached maximum demo index ({max_demo}). Stopping collection.")
            break

        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
        actions = get_actions(all_actions, env, demo_idxs, robot)
        obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, obs)
        run_out = get_run_out(all_actions, env, demo_idxs)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id])

        for env_id in success.nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            if steps_after_success[env_id] == 0:
                log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                tot_success += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

            if not run_out[env_id] and steps_after_success[env_id] < args.tot_steps_after_success:
                steps_after_success[env_id] += 1
            else:
                steps_after_success[env_id] = 0
                collector.save(demo_idx, status="success")
                collector.delete(demo_idx)

                if demo_indexer.next_idx < max_demo:
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    log.info(f"Transitioning Env {env_id}: Demo {demo_idx} to Demo {new_demo_idx}")

                    randomization_manager.randomize_for_demo(new_demo_idx)
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                    run_out[env_id] = False
                else:
                    finished[env_id] = True

        for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
            collector.save(demo_idx, status="failed")
            collector.delete(demo_idx)
            failure_count[env_id] += 1

            if failure_count[env_id] < try_num:
                log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                randomization_manager.randomize_for_demo(demo_idx)
                force_reset_to_state(env, init_states[demo_idx], env_id)

                obs = env.handler.get_states()
                obs = state_tensor_to_nested(env.handler, obs)
                collector.create(demo_idx, obs[env_id])
            else:
                log.error(f"Demo {demo_idx} failed too many times, giving up")
                failure_count[env_id] = 0
                tot_give_up += 1
                # pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                if demo_indexer.next_idx < max_demo:
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    randomization_manager.randomize_for_demo(new_demo_idx)
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                else:
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
