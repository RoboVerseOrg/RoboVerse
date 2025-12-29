# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script for exporting ONNX models.

Usage:
    python export_onnx.py --task walk_agibot_a2_dof12 --resume <log_dir> --checkpoint <checkpoint>
"""

from __future__ import annotations

import os
import copy
import re

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_learn.rl.unitree_rl.helper import (
    get_args,
    get_log_dir,
    get_load_path,
    make_objects,
    make_robots,
)
from roboverse_learn.rl.unitree_rl.runners import MasterRunner


class ExportedPolicy(torch.nn.Module):
    """Policy wrapper class for ONNX export (non-recurrent networks)."""

    def __init__(self, actor):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()

    def forward(self, observations):
        return self.actor(observations)


class ExportedPolicyLSTM(torch.nn.Module):
    """Policy wrapper class for ONNX export (with LSTM).

    Note: For ONNX export, LSTM states need to be passed as inputs and outputs.
    This wrapper ensures ONNX can properly handle LSTM with explicit state inputs.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        # Extract LSTM parameters
        lstm = actor_critic.memory_a.rnn
        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.num_layers = lstm.num_layers
        self.bias = lstm.bias
        self.batch_first = lstm.batch_first
        self.dropout = lstm.dropout
        self.bidirectional = lstm.bidirectional

        # Create a new LSTM layer with the same parameters
        # This ensures clean state for ONNX export
        self.memory = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).cpu()

        # Copy weights from original LSTM
        self.memory.load_state_dict(lstm.state_dict())

    def forward(self, observations, hidden_state, cell_state):
        """Forward pass for ONNX export.

        Args:
            observations: Observations with shape (batch_size, obs_dim)
            hidden_state: Hidden state with shape (num_layers, batch_size, hidden_size)
            cell_state: Cell state with shape (num_layers, batch_size, hidden_size)

        Returns:
            (actions, new_hidden_state, new_cell_state)
        """
        # LSTM requires sequence input, so add sequence dimension
        # Input shape: (seq_len, batch_size, input_size) = (1, batch_size, obs_dim)
        obs_seq = observations.unsqueeze(0)
        # Explicitly pass hidden_state and cell_state as tuple
        # This is critical for ONNX to recognize them as inputs
        lstm_out, (h, c) = self.memory(obs_seq, (hidden_state, cell_state))
        # Remove sequence dimension: (1, batch_size, hidden_size) -> (batch_size, hidden_size)
        out = lstm_out.squeeze(0)
        actions = self.actor(out)
        return actions, h, c


def prepare(args):
    """Prepare environment and model for export."""
    task_cls = get_task_class(args.task)
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    overrides = {
        "num_envs": args.num_envs,
        "simulator": args.sim,
        "headless": args.headless,
    }

    if args.robots:
        overrides["robots"] = make_robots(args.robots)
        overrides["cameras"] = [
            camera
            for robot in overrides["robots"]
            if hasattr(robot, "cameras")
            for camera in getattr(robot, "cameras", [])
        ]

    if args.objects:
        overrides["objects"] = make_objects(args.objects)

    scenario.update(**overrides)

    device = "cpu" if args.sim == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    master_runner = MasterRunner(
        task_cls=task_cls,
        scenario=scenario,
        log_path=args.resume,
        lib_name="rsl_rl",
        device=device,
    )

    return master_runner


def export_onnx(args):
    """Export ONNX model."""
    # Prepare environment and model
    master_runner = prepare(args)
    name_0 = list(master_runner.runners.keys())[0]

    if not args.resume:
        raise ValueError("Please provide --resume argument to specify model path.")

    # Load model
    log_dir = get_log_dir(task_name=master_runner.task_name, now=args.resume)
    master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)

    runner_0 = master_runner.runners[name_0]
    actor_critic = runner_0.runner.alg.policy

    # Get action dimensions
    # Try multiple ways to get num_actions
    num_actions = None

    # Method 1: From actor.mlp (most common)
    if hasattr(actor_critic, "actor"):
        actor = actor_critic.actor
        if hasattr(actor, "mlp"):
            # Try direct indexing first (MLP might be a Sequential or ModuleList)
            try:
                # Try to access the last layer directly
                if hasattr(actor.mlp, "__getitem__"):
                    # Try to find the last Linear layer by iterating backwards
                    for i in range(len(actor.mlp) - 1, -1, -1):
                        layer = actor.mlp[i]
                        if isinstance(layer, torch.nn.Linear):
                            num_actions = layer.out_features
                            break
            except (TypeError, AttributeError):
                pass

            # If direct indexing didn't work, try children()
            if num_actions is None:
                try:
                    mlp_children = list(actor.mlp.children())
                    for layer in reversed(mlp_children):
                        if isinstance(layer, torch.nn.Linear):
                            num_actions = layer.out_features
                            break
                except (TypeError, AttributeError):
                    pass

    # Method 2: Try to find any Linear layer in actor that outputs actions
    if num_actions is None and hasattr(actor_critic, "actor"):
        actor = actor_critic.actor
        # Try to find the last Linear layer in the entire actor module
        try:
            for name, module in reversed(list(actor.named_modules())):
                if isinstance(module, torch.nn.Linear):
                    num_actions = module.out_features
                    break
        except (TypeError, AttributeError):
            pass

    # Method 3: Try from environment if available
    if num_actions is None and hasattr(runner_0, "env"):
        env = runner_0.env
        if hasattr(env, "num_actions"):
            num_actions = env.num_actions

    if num_actions is None:
        # Debug: print actor structure
        print(f"Debug: actor_critic type: {type(actor_critic)}")
        if hasattr(actor_critic, "actor"):
            print(f"Debug: actor type: {type(actor_critic.actor)}")
            print(f"Debug: actor attributes: {dir(actor_critic.actor)}")
            if hasattr(actor_critic.actor, "mlp"):
                print(f"Debug: mlp type: {type(actor_critic.actor.mlp)}")
                print(f"Debug: mlp: {actor_critic.actor.mlp}")
        raise ValueError("Cannot determine action dimensions. Tried multiple methods.")

    # Get observation dimensions directly from model structure
    # For LSTM models, input is single step observation (num_obs_single)
    # For non-LSTM models, input is stacked observations (num_obs_single * (obs_len_history + 1))
    if hasattr(actor_critic, "memory_a"):
        # LSTM model: input is single step observation
        if hasattr(actor_critic.memory_a, "rnn"):
            # Get from LSTM input dimension
            num_obs = actor_critic.memory_a.rnn.input_size
        else:
            raise ValueError("Cannot determine observation dimensions for LSTM model.")
    else:
        # Non-LSTM model: input is stacked observations
        if hasattr(actor_critic, "actor") and hasattr(actor_critic.actor, "mlp"):
            # Get from actor network input dimension
            first_layer = list(actor_critic.actor.mlp.children())[0]
            num_obs = first_layer.in_features
        else:
            raise ValueError("Cannot determine observation dimensions.")

    # Create exported policy
    if hasattr(actor_critic, "memory_a"):
        # Policy with LSTM
        exported_policy = ExportedPolicyLSTM(actor_critic)
    else:
        # Regular policy
        exported_policy = ExportedPolicy(actor_critic.actor)

    exported_policy.eval()
    exported_policy.to("cpu")

    # Create export directory
    root_path = os.path.join(log_dir, "exported_onnx")
    os.makedirs(root_path, exist_ok=True)

    # Generate filename
    checkpoint_num = None
    if args.checkpoint != -1:
        checkpoint_num = args.checkpoint
    else:
        # Extract checkpoint number from model path
        model_path = get_load_path(load_root=log_dir, checkpoint=args.checkpoint)
        model_filename = os.path.basename(model_path)
        match = re.search(r"model_(\d+)\.pt", model_filename)
        if match:
            checkpoint_num = match.group(1)

    # Extract run_name from resume path (if any)
    run_name_for_file = None
    if args.resume:
        # Try to extract run_name from path
        # Path format is usually: outputs/unitree_rl/{task_name}/{timestamp}_{run_name}
        parts = args.resume.split("/")
        if len(parts) > 0:
            last_part = parts[-1]
            # Try to match timestamp format: YYYY_MMDD_HHMMSS{run_name}
            match = re.search(r"\d{4}_\d{4}_\d{6}(.+)", last_part)
            if match:
                run_name_for_file = match.group(1)

    # Generate filename
    if run_name_for_file and checkpoint_num:
        file_name = f"{run_name_for_file}_ckpt{checkpoint_num}.onnx"
    elif run_name_for_file:
        file_name = f"{run_name_for_file}.onnx"
    elif checkpoint_num:
        file_name = f"{args.task.split('_')[0]}_ckpt{checkpoint_num}.onnx"
    else:
        file_name = f"{args.task.split('_')[0]}_policy.onnx"

    path = os.path.join(root_path, file_name)

    # Create example input
    example_input = torch.randn(1, num_obs)

    # Export ONNX model
    print(f"Exporting ONNX model to: {path}")
    print(f"Input dimension: {num_obs}")

    try:
        if hasattr(actor_critic, "memory_a"):
            # LSTM model: need to pass states as inputs and outputs
            num_layers = exported_policy.memory.num_layers
            hidden_size = exported_policy.memory.hidden_size
            example_hidden = torch.zeros(num_layers, 1, hidden_size)
            example_cell = torch.zeros(num_layers, 1, hidden_size)

            # For LSTM export, we need to ensure ONNX recognizes hidden_state and cell_state as inputs
            # Export with explicit input/output names
            print(f"Exporting LSTM model with:")
            print(f"  - Input shape: {example_input.shape}")
            print(f"  - Hidden state shape: {example_hidden.shape}")
            print(f"  - Cell state shape: {example_cell.shape}")

            torch.onnx.export(
                exported_policy,  # Model
                (example_input, example_hidden, example_cell),  # Model example input
                path,  # Model output path
                export_params=True,  # Export model parameters
                opset_version=11,  # ONNX opset version
                do_constant_folding=True,  # Optimize constant folding
                input_names=["input", "hidden_state", "cell_state"],  # Model input names
                output_names=["output", "new_hidden_state", "new_cell_state"],  # Model output names
                # Fixed batch_size=1, no dynamic axes
                training=torch.onnx.TrainingMode.EVAL,  # Ensure eval mode
            )

            # Verify the exported model
            try:
                import onnx
                model = onnx.load(path)
                print(f"\nExported ONNX model inputs:")
                for inp in model.graph.input:
                    print(f"  - {inp.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
                print(f"\nExported ONNX model outputs:")
                for out in model.graph.output:
                    print(f"  - {out.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]}")
            except ImportError:
                print("Note: onnx package not available, skipping model verification")
        else:
            # Regular model: only observations as input
            torch.onnx.export(
                exported_policy,  # Model
                example_input,  # Model example input
                path,  # Model output path
                export_params=True,  # Export model parameters
                opset_version=11,  # ONNX opset version
                do_constant_folding=True,  # Optimize constant folding
                input_names=["input"],  # Model input names
                output_names=["output"],  # Model output names
                # Fixed batch_size=1, no dynamic axes
            )
        print(f"Successfully exported ONNX model to: {path}")
        print("\n" + "=" * 60)
        print("Model Input/Output Dimensions:")
        print("=" * 60)
        if hasattr(actor_critic, "memory_a"):
            # LSTM model
            num_layers = exported_policy.memory.num_layers
            hidden_size = exported_policy.memory.hidden_size
            print(f"Inputs:")
            print(f"  - observations: (batch_size, {num_obs})")
            print(f"  - hidden_state: ({num_layers}, batch_size, {hidden_size})")
            print(f"  - cell_state: ({num_layers}, batch_size, {hidden_size})")
            print(f"Outputs:")
            print(f"  - actions: (batch_size, {num_actions})")
            print(f"  - new_hidden_state: ({num_layers}, batch_size, {hidden_size})")
            print(f"  - new_cell_state: ({num_layers}, batch_size, {hidden_size})")
        else:
            # Regular model
            print(f"Input:")
            print(f"  - observations: (batch_size, {num_obs})")
            print(f"Output:")
            print(f"  - actions: (batch_size, {num_actions})")
        print("=" * 60)
    except Exception as e:
        print(f"Error exporting ONNX model: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    args = get_args()
    if args.checkpoint is None:
        args.checkpoint = -1
    export_onnx(args)

