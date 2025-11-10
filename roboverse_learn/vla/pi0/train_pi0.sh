#!/bin/bash

# One-click training script for π models on RoboVerse
# This script automates the complete workflow from data conversion to training

set -e  # Exit on error

# ============================================================================
# Configuration - Modify these variables as needed
# ============================================================================

# Input/Output paths
INPUT_DEMO_ROOT="${INPUT_DEMO_ROOT:-./roboverse_demo}"
HF_REPO_ID="${HF_REPO_ID:-your_hf_name/your_repo_name}"
CONFIG_NAME="${CONFIG_NAME:-pi05_roboverse_lora}"
EXP_NAME="${EXP_NAME:-roboverse_pi05_lora}"

# OpenPI paths
ROBOVERSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${ROBOVERSE_ROOT}/third_party/openpi}"

# Training parameters
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"
OVERWRITE_DATA="${OVERWRITE_DATA:-false}"
OVERWRITE_TRAINING="${OVERWRITE_TRAINING:-false}"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH"
        exit 1
    fi
}

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

One-click training script for π models on RoboVerse demonstrations.

Options:
    -i, --input DIR           Input RoboVerse demo directory (default: ./roboverse_demo)
    -r, --repo-id ID          HuggingFace repo ID (default: your_hf_name/your_repo_name)
    -c, --config NAME         Training config name (default: pi05_roboverse_lora)
    -e, --exp-name NAME       Experiment name (default: roboverse_pi05_lora)
    -o, --openpi-root DIR     OpenPI installation directory (default: ../../../third_party/openpi)
    --overwrite-data          Overwrite existing LeRobot dataset
    --overwrite-training      Overwrite existing training checkpoint
    --skip-data-conversion    Skip data conversion step
    --skip-norm-stats         Skip norm stats computation
    --skip-registration       Skip policy and config registration
    -h, --help                Show this help message

Examples:
    # Basic usage
    $0 -i ./my_demos -r myname/myrepo

    # With custom config
    $0 -i ./my_demos -r myname/myrepo -c pi0_roboverse_lora -e my_experiment

    # Skip data conversion if already done
    $0 --skip-data-conversion -c pi05_roboverse_lora

Environment Variables:
    XLA_MEM_FRACTION          XLA memory fraction (default: 0.9)
    HF_LEROBOT_HOME           LeRobot cache directory (default: ~/.cache/huggingface/lerobot)

EOF
    exit 1
}

SKIP_DATA_CONVERSION=false
SKIP_NORM_STATS=false
SKIP_REGISTRATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DEMO_ROOT="$2"
            shift 2
            ;;
        -r|--repo-id)
            HF_REPO_ID="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        -e|--exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        -o|--openpi-root)
            OPENPI_ROOT="$2"
            shift 2
            ;;
        --overwrite-data)
            OVERWRITE_DATA=true
            shift
            ;;
        --overwrite-training)
            OVERWRITE_TRAINING=true
            shift
            ;;
        --skip-data-conversion)
            SKIP_DATA_CONVERSION=true
            shift
            ;;
        --skip-norm-stats)
            SKIP_NORM_STATS=true
            shift
            ;;
        --skip-registration)
            SKIP_REGISTRATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "Pre-flight Checks"

echo "Configuration:"
echo "  Input demo root:       $INPUT_DEMO_ROOT"
echo "  HuggingFace repo ID:   $HF_REPO_ID"
echo "  Config name:           $CONFIG_NAME"
echo "  Experiment name:       $EXP_NAME"
echo "  OpenPI root:           $OPENPI_ROOT"
echo "  XLA memory fraction:   $XLA_MEM_FRACTION"
echo ""

# Check if uv is installed
check_command uv

# Check if input directory exists (only if not skipping data conversion)
if [ "$SKIP_DATA_CONVERSION" = false ]; then
    if [ ! -d "$INPUT_DEMO_ROOT" ]; then
        echo "Error: Input demo directory does not exist: $INPUT_DEMO_ROOT"
        exit 1
    fi
fi

# Check if OpenPI is installed
if [ ! -d "$OPENPI_ROOT" ]; then
    echo "Error: OpenPI directory not found at: $OPENPI_ROOT"
    echo "Please install OpenPI first:"
    echo "  cd $ROBOVERSE_ROOT/third_party"
    echo "  git clone https://github.com/physical-intelligence/openpi.git"
    echo "  cd openpi"
    echo "  GIT_LFS_SKIP_SMUDGE=1 uv sync"
    echo "  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
    exit 1
fi

# Check if roboverse_policy.py exists in OpenPI
POLICY_FILE="$OPENPI_ROOT/src/openpi/policies/roboverse_policy.py"
if [ ! -f "$POLICY_FILE" ]; then
    echo "Warning: roboverse_policy.py not found in OpenPI policies directory"
    echo "Please ensure you have copied roboverse_policy.py to:"
    echo "  $OPENPI_ROOT/src/openpi/policies/"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "All checks passed!"

# ============================================================================
# Step 0: Register RoboVerse policy and config in OpenPI
# ============================================================================

if [ "$SKIP_REGISTRATION" = false ]; then
    print_header "Step 0: Registering RoboVerse policy and config"

    # Copy roboverse_policy.py to OpenPI
    POLICY_SOURCE="$ROBOVERSE_ROOT/roboverse_learn/vla/pi0/roboverse_policy.py"
    POLICY_DEST="$OPENPI_ROOT/src/openpi/policies/roboverse_policy.py"

    if [ -f "$POLICY_DEST" ]; then
        echo "roboverse_policy.py already exists in OpenPI policies"
    else
        echo "Copying roboverse_policy.py to OpenPI..."
        cp "$POLICY_SOURCE" "$POLICY_DEST"
        echo "✓ roboverse_policy.py copied"
    fi

    # Check and register data config in config.py
    CONFIG_FILE="$OPENPI_ROOT/src/openpi/training/config.py"

    if grep -q "LeRobotRoboVerseDataConfig" "$CONFIG_FILE"; then
        echo "LeRobotRoboVerseDataConfig already registered in config.py"
    else
        echo "Adding LeRobotRoboVerseDataConfig to config.py..."

        # Create a Python script to inject the config
        cat > /tmp/inject_config.py << 'PYTHON_EOF'
import sys

config_file = sys.argv[1]
repo_id = sys.argv[2]

# Read the config snippet
config_snippet = '''

# ============================================================================
# RoboVerse Data Configuration (Auto-generated by train_pi0.sh)
# ============================================================================

@dataclasses.dataclass(frozen=True)
class LeRobotRoboVerseDataConfig(DataConfigFactory):
    """Data configuration for RoboVerse demonstrations in LeRobot format."""

    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        from openpi.policies import roboverse_policy

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[roboverse_policy.RoboVerseInputs(model_type=model_config.model_type)],
            outputs=[roboverse_policy.RoboVerseOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(-2, 7)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

'''

        # Add training configs
        train_configs = f'''

# ============================================================================
# RoboVerse Training Configurations (Auto-generated by train_pi0.sh)
# ============================================================================

TrainConfig(
    name="pi05_roboverse_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRoboVerseDataConfig(
        repo_id="{repo_id}",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=30_000,
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
)

TrainConfig(
    name="pi0_roboverse_lora",
    model=pi0_config.Pi0Config(
        pi05=False,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRoboVerseDataConfig(
        repo_id="{repo_id}",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=30_000,
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
)

'''

        # Read original file
        with open(config_file, 'r') as f:
            content = f.read()

        # Backup original file
        with open(config_file + '.backup', 'w') as f:
            f.write(content)

        # Append our configs at the end
        with open(config_file, 'a') as f:
            f.write(config_snippet)
            f.write(train_configs)

        print(f"✓ Configuration injected into {config_file}")
        print(f"✓ Backup saved to {config_file}.backup")

PYTHON_EOF

        python /tmp/inject_config.py "$CONFIG_FILE" "$HF_REPO_ID"
        rm /tmp/inject_config.py

        echo "✓ LeRobotRoboVerseDataConfig registered"
        echo "✓ Training configs registered (pi05_roboverse_lora, pi0_roboverse_lora)"
    fi

    echo "Registration completed!"
else
    print_header "Step 0: Skipping registration (--skip-registration)"
fi

# ============================================================================
# Step 1: Convert RoboVerse demos to LeRobot dataset
# ============================================================================

if [ "$SKIP_DATA_CONVERSION" = false ]; then
    print_header "Step 1: Converting RoboVerse demos to LeRobot dataset"

    OVERWRITE_FLAG=""
    if [ "$OVERWRITE_DATA" = true ]; then
        OVERWRITE_FLAG="--overwrite"
    fi

    cd "$ROBOVERSE_ROOT"

    echo "Running conversion script..."
    uv run roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
        --input-root "$INPUT_DEMO_ROOT" \
        --repo-id "$HF_REPO_ID" \
        $OVERWRITE_FLAG

    echo "Data conversion completed!"
else
    print_header "Step 1: Skipping data conversion (--skip-data-conversion)"
fi

# ============================================================================
# Step 2: Compute normalization statistics
# ============================================================================

if [ "$SKIP_NORM_STATS" = false ]; then
    print_header "Step 2: Computing normalization statistics"

    cd "$OPENPI_ROOT"

    echo "Running compute_norm_stats.py for config: $CONFIG_NAME"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

    echo "Norm stats computation completed!"
else
    print_header "Step 2: Skipping norm stats computation (--skip-norm-stats)"
fi

# ============================================================================
# Step 3: Launch training
# ============================================================================

print_header "Step 3: Launching training"

cd "$OPENPI_ROOT"

TRAINING_FLAGS=""
if [ "$OVERWRITE_TRAINING" = true ]; then
    TRAINING_FLAGS="--overwrite"
fi

echo "Starting training with:"
echo "  Config: $CONFIG_NAME"
echo "  Experiment name: $EXP_NAME"
echo "  XLA memory fraction: $XLA_MEM_FRACTION"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION \
    uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    $TRAINING_FLAGS

# ============================================================================
# Training Completed
# ============================================================================

print_header "Training completed successfully!"

echo "Next steps:"
echo "  1. Check training logs and metrics"
echo "  2. Evaluate the trained checkpoint using:"
echo "     cd $OPENPI_ROOT"
echo "     uv run scripts/serve_policy.py policy:checkpoint \\"
echo "       --policy.config=$CONFIG_NAME \\"
echo "       --policy.dir=<checkpoint_path>"
echo ""
echo "  3. In a separate terminal, run evaluation:"
echo "     cd $ROBOVERSE_ROOT"
echo "     python roboverse_learn/vla/pi0/pi_eval.py \\"
echo "       --task <task_name> --robot franka --sim mujoco \\"
echo "       --policy-host localhost --policy-port 8000"
echo ""
