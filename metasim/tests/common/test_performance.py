"""Performance and memory management tests for simulator handlers.

These tests measure and verify performance characteristics like simulation speed,
memory usage, and scalability across different simulators.
"""

import gc
import os
import time
import tracemalloc

import numpy as np
import psutil
import pytest


class TestPerformance:
    """Test suite for performance and memory management."""

    @pytest.mark.performance
    def test_simulation_throughput(self, simulator_handler, simple_robot_urdf, performance_monitor):
        """Test simulation throughput (steps per second)."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        for _ in range(10):
            action = np.zeros(1)
            simulator_handler.step(action)

        num_steps = 100
        actions = np.zeros((num_steps, 1))

        performance_monitor.start()

        for action in actions:
            simulator_handler.step(action)

        metrics = performance_monitor.stop()

        throughput = num_steps / metrics["duration"]

        print(f"\nSimulator: {simulator_handler.simulator_name}")
        print(f"Throughput: {throughput:.2f} steps/second")
        print(f"Time per step: {1000 * metrics['duration'] / num_steps:.2f} ms")

        assert throughput > 100, f"Throughput too low: {throughput:.2f} steps/s"

        assert metrics["memory_used"] < 50, f"Memory grew by {metrics['memory_used']:.2f} MB during test"

    @pytest.mark.performance
    def test_reset_performance(self, simulator_handler, simple_robot_urdf):
        """Test reset operation performance."""
        simulator_handler.load_robot(simple_robot_urdf)

        simulator_handler.reset()

        num_resets = 50
        reset_times = []

        for _ in range(num_resets):
            start = time.time()
            simulator_handler.reset()
            end = time.time()
            reset_times.append(end - start)

        avg_reset_time = np.mean(reset_times)
        std_reset_time = np.std(reset_times)
        max_reset_time = np.max(reset_times)

        print("\nReset Performance:")
        print(f"Average: {avg_reset_time * 1000:.2f} ms")
        print(f"Std Dev: {std_reset_time * 1000:.2f} ms")
        print(f"Max: {max_reset_time * 1000:.2f} ms")

        assert avg_reset_time < 0.1, f"Reset too slow: {avg_reset_time * 1000:.2f} ms"

        assert std_reset_time < 0.05, f"Reset time too variable: {std_reset_time * 1000:.2f} ms"

    @pytest.mark.performance
    def test_state_operations_performance(self, simulator_handler, simple_robot_urdf):
        """Test performance of state get/set operations."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        num_ops = 100

        start = time.time()
        for _ in range(num_ops):
            state = simulator_handler.get_states()
        get_time = time.time() - start

        start = time.time()
        for _ in range(num_ops):
            simulator_handler.set_states(state)
        set_time = time.time() - start

        print("\nState Operations Performance:")
        print(f"get_states: {get_time / num_ops * 1000:.2f} ms per call")
        print(f"set_states: {set_time / num_ops * 1000:.2f} ms per call")

        assert get_time / num_ops < 0.01, f"get_states too slow: {get_time / num_ops * 1000:.2f} ms"
        assert set_time / num_ops < 0.01, f"set_states too slow: {set_time / num_ops * 1000:.2f} ms"

    @pytest.mark.performance
    def test_memory_usage_over_time(self, simulator_handler, simple_robot_urdf):
        """Test memory usage stability over extended simulation."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        process = psutil.Process(os.getpid())

        memory_samples = []
        sample_interval = 100
        num_samples = 10

        for i in range(num_samples):
            for _ in range(sample_interval):
                action = np.array([0.1 * np.sin(i)])
                simulator_handler.step(action)

            mem_info = process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
            memory_samples.append(memory_mb)

            gc.collect()

        memory_growth = memory_samples[-1] - memory_samples[0]
        avg_growth_per_sample = memory_growth / (num_samples - 1)

        print("\nMemory Usage Over Time:")
        print(f"Initial: {memory_samples[0]:.2f} MB")
        print(f"Final: {memory_samples[-1]:.2f} MB")
        print(f"Total Growth: {memory_growth:.2f} MB")
        print(f"Growth Rate: {avg_growth_per_sample:.2f} MB per {sample_interval} steps")

        assert avg_growth_per_sample < 1.0, f"Memory leak detected: {avg_growth_per_sample:.2f} MB growth per sample"

    @pytest.mark.performance
    def test_rendering_performance(self, simulator_handler, simple_robot_urdf):
        """Test rendering performance if supported."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        try:
            image = simulator_handler.render()
            if image is None:
                pytest.skip("Rendering not supported in current mode")
        except NotImplementedError:
            pytest.skip("Rendering not implemented")

        num_frames = 30
        render_times = []

        for _ in range(num_frames):
            start = time.time()
            image = simulator_handler.render()
            end = time.time()
            render_times.append(end - start)

            simulator_handler.simulate()

        avg_render_time = np.mean(render_times)
        fps = 1.0 / avg_render_time

        print("\nRendering Performance:")
        print(f"Average render time: {avg_render_time * 1000:.2f} ms")
        print(f"FPS: {fps:.2f}")

        assert fps > 30, f"Rendering too slow: {fps:.2f} FPS"

    @pytest.mark.performance
    def test_multi_object_scaling(self, simulator_handler, simple_cube_urdf):
        """Test performance scaling with number of objects."""
        object_counts = [1, 5, 10, 20]
        step_times = []

        for num_objects in object_counts:
            simulator_handler.reset()

            for i in range(num_objects):
                obj_id = simulator_handler.load_object(simple_cube_urdf)
                x = (i % 5) * 0.3
                y = (i // 5) * 0.3
                simulator_handler.set_pose(obj_id, np.array([x, y, 0.5]), np.array([1, 0, 0, 0]))

            num_steps = 50
            start = time.time()

            for _ in range(num_steps):
                simulator_handler.simulate()

            elapsed = time.time() - start
            avg_step_time = elapsed / num_steps
            step_times.append(avg_step_time)

            print(f"\nWith {num_objects} objects: {avg_step_time * 1000:.2f} ms/step")

        scaling_factor = step_times[-1] / step_times[0]
        object_factor = object_counts[-1] / object_counts[0]

        print(f"\nScaling: {scaling_factor:.2f}x time for {object_factor:.0f}x objects")

        assert scaling_factor < 0.8 * object_factor, (
            f"Poor scaling: {scaling_factor:.2f}x for {object_factor:.0f}x objects"
        )

    @pytest.mark.performance
    def test_action_batch_performance(self, simulator_handler, simple_robot_urdf):
        """Test performance with batched vs individual actions."""
        num_robots = 5
        robot_ids = []

        for i in range(num_robots):
            robot_id = simulator_handler.load_robot(simple_robot_urdf)
            robot_ids.append(robot_id)
            try:
                base_pos = np.array([i * 2.0, 0, 0])
                simulator_handler.set_pose(robot_id, base_pos, np.array([1, 0, 0, 0]))
            except:
                pass

        simulator_handler.reset()

        num_steps = 50
        individual_start = time.time()

        for _ in range(num_steps):
            for i, robot_id in enumerate(robot_ids):
                action = np.array([0.1 * np.sin(i)])
                try:
                    simulator_handler.step(action, robot_id=robot_id)
                except TypeError:
                    simulator_handler.step(action)
                    break

        individual_time = time.time() - individual_start

        simulator_handler.reset()
        batch_start = time.time()

        for step in range(num_steps):
            actions = np.array([0.1 * np.sin(i) for i in range(num_robots)])
            try:
                simulator_handler.step(actions)
            except:
                pytest.skip("Batched actions not supported")

        batch_time = time.time() - batch_start

        print("\nAction Batching Performance:")
        print(f"Individual: {individual_time:.2f}s for {num_steps * num_robots} actions")
        print(f"Batched: {batch_time:.2f}s for {num_steps} batched actions")
        print(f"Speedup: {individual_time / batch_time:.2f}x")

        assert batch_time < individual_time, "Batching should be faster than individual actions"

    @pytest.mark.performance
    def test_memory_profiling(self, simulator_handler, simple_robot_urdf):
        """Profile memory allocation patterns."""
        tracemalloc.start()

        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        snapshot1 = tracemalloc.take_snapshot()

        for i in range(100):
            action = np.array([0.1 * np.cos(i * 0.1)])
            simulator_handler.step(action)

            if i % 10 == 0:
                state = simulator_handler.get_states()
                simulator_handler.set_states(state)

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        print("\nTop 10 memory allocations:")
        for stat in top_stats[:10]:
            print(f"{stat}")

        tracemalloc.stop()

        total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        total_mb = total_allocated / 1024 / 1024

        print(f"\nTotal allocated during test: {total_mb:.2f} MB")

        assert total_mb < 100, f"Excessive memory allocation: {total_mb:.2f} MB"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_stability(self, simulator_handler, simple_robot_urdf):
        """Test stability over very long simulations."""
        simulator_handler.load_robot(simple_robot_urdf)
        simulator_handler.reset()

        num_steps = 10000
        checkpoint_interval = 1000

        checkpoints = []

        for i in range(num_steps):
            action = np.array([0.5 * np.sin(i * 0.01)])
            simulator_handler.step(action)

            if i % checkpoint_interval == 0:
                state = simulator_handler.get_states()
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                checkpoints.append({"step": i, "memory_mb": memory_mb, "state_valid": self._is_state_valid(state)})

        print("\nLong-running simulation checkpoints:")
        for cp in checkpoints:
            print(f"Step {cp['step']}: {cp['memory_mb']:.2f} MB, Valid: {cp['state_valid']}")

        assert all(cp["state_valid"] for cp in checkpoints), "Invalid state detected during long run"

        memory_growth = checkpoints[-1]["memory_mb"] - checkpoints[0]["memory_mb"]
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.2f} MB"

    def _is_state_valid(self, state) -> bool:
        """Check if state contains valid values."""
        if isinstance(state, dict):
            for value in state.values():
                if isinstance(value, np.ndarray):
                    if not np.all(np.isfinite(value)):
                        return False
        elif isinstance(state, np.ndarray):
            if not np.all(np.isfinite(state)):
                return False
        return True
