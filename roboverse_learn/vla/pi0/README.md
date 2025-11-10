# Using π Models on RoboVerse

This guide walks through the end-to-end workflow for training / fine-tuning the openpi π-family models (π₀, π₀.₅, π₀-FAST) on RoboVerse demonstrations.


## Quick Start (One-Click Training)

### Run Training

For a streamlined training experience, use the provided automated script:

```bash
# Basic usage
./roboverse_learn/vla/pi0/train_pi0.sh \
  -i ./roboverse_demo \
  -r your_hf_name/your_repo_name

# With custom configuration
./roboverse_learn/vla/pi0/train_pi0.sh \
  -i ./your_demos \
  -r your_hf_name/your_repo_name \
  -c pi05_roboverse_lora \
  -e your_experiment_name

# Skip data conversion if already done
./roboverse_learn/vla/pi0/train_pi0.sh \
  --skip-data-conversion \
  -c pi05_roboverse_lora
```

**Options:**
- `-i, --input DIR`: Input RoboVerse demo directory
- `-r, --repo-id ID`: HuggingFace repo ID
- `-c, --config NAME`: Training config name (default: `pi05_roboverse_lora`)
- `-e, --exp-name NAME`: Experiment name
- `--overwrite-data`: Overwrite existing LeRobot dataset
- `--overwrite-training`: Overwrite existing training checkpoint
- `--skip-data-conversion`: Skip data conversion if already completed
- `--skip-norm-stats`: Skip normalization statistics computation
- `-h, --help`: Show help message

The script automates:
1. Data conversion from RoboVerse to LeRobot format
2. Normalization statistics computation
3. Model training with proper environment setup

**What the script does:**
1. **Automatic registration**: Copies `roboverse_policy.py` to OpenPI and injects data/training configs into `config.py`
2. **Data conversion**: Converts RoboVerse demos to LeRobot format
3. **Normalization stats**: Computes statistics needed for training
4. **Training launch**: Starts the training process with proper environment setup

**Prerequisites:**
- OpenPI must be installed (see section 1 below)
- The script will automatically handle policy and config registration

### Complete Example Workflow

Here's a complete example showing how to train π₀.₅ on your demonstrations:

```bash
# 1. First-time setup: Install OpenPI (one-time only)
cd third_party
git clone https://github.com/physical-intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../..

# 2. Collect or prepare your demonstrations
# (Your demos should be in RoboVerse format with metadata.json and rgb.mp4)

# 3. Run the complete training pipeline
./roboverse_learn/vla/pi0/train_pi0.sh \
  -i ./my_robot_demos \
  -r your_hf_username/robot_task_demos \
  -c pi05_roboverse_lora \
  -e my_first_experiment

# 4. Monitor training progress
# Training logs and checkpoints will be saved in the OpenPI directory

# 5. Evaluate your trained model (in a separate terminal)
# Terminal 1: Start policy server
cd third_party/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_roboverse_lora \
  --policy.dir=<path_to_checkpoint>

# Terminal 2: Run evaluation
python roboverse_learn/vla/pi0/pi_eval.py \
  --task PickCube \
  --robot franka \
  --sim mujoco \
  --policy-host localhost \
  --policy-port 8000
```

---

## Manual Setup (Step-by-Step)

If you prefer more control or need to customize the training configuration beyond what the automated script provides, follow these detailed steps:

## 1. Clone and install openpi

1. Clone openpi under `third_party/`:
   ```bash
   cd third_party
   git clone https://github.com/physical-intelligence/openpi.git
   ```
2. Install dependencies as instructed in the openpi README (we recommend `uv`):
   ```bash
   cd openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   ```

## 2. Convert RoboVerse demos into a LeRobot dataset

We provide `roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py`. The script iterates through every episode under `roboverse_demo`, reads `metadata.json` and `rgb.mp4`, and writes a LeRobot dataset where `joint_qpos` becomes `state`, `joint_qpos_target` becomes `actions`, and `task_desc` is stored as the prompt.

Install the required dependencies:
```bash
uv pip install lerobot imageio-ffmpeg
```

Run the conversion:
```bash
uv run roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root <your_roboverse_demo> \
  --repo-id <your_hf_name>/<repo_name> \
  --overwrite
```
The dataset will be written to `$HF_LEROBOT_HOME/<repo-id>` (defaults to `~/.cache/huggingface/lerobot`).

## 3. Register the RoboVerse policy and data config

Inside the openpi repo:

1. Copy `roboverse_policy.py` into `openpi/src/openpi/policies/`. This policy maps RoboVerse images/state/actions to the format expected by the π models.
2. Add a new data config to `openpi/src/openpi/training/config.py`:

   ```python
   @dataclasses.dataclass(frozen=True)
   class LeRobotRoboVerseDataConfig(DataConfigFactory):
       extra_delta_transform: bool = True

       @override
       def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
           repack_transform = _transforms.Group(
               inputs=[
                   _transforms.RepackTransform(
                       {
                           "observation/image": "image",
                           # "observation/wrist_image": "wrist_image",  # RoboVerse has a single view now
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
   ```

   *Set `extra_delta_transform=False` if your actions are already absolute joint positions and you do not want to compute deltas.*

## 4. Define a TrainConfig and launch fine-tuning

Add a training config in `training/config.py`. Example (π₀.₅ + LoRA):

```python
TrainConfig(
    name="pi05_roboverse_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRoboVerseDataConfig(
        repo_id="<your_hf_name>/<repo_name>",
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
```

Start training:
```bash
cd ~/codes/RoboVerse/third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_roboverse_lora 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_roboverse_lora --exp-name=roboverse_pi05_lora --overwrite
```

To fine-tune π₀ or π₀-FAST, switch the `model` field to `Pi0Config`/`Pi0FASTConfig` variants and adapt the LoRA settings accordingly.

## 5. Evaluate the trained checkpoint

  1. Start the policy server from *inside* the openpi repo (pointing to whatever checkpoint you want to
  test; here we use iteration 6000):

     ```bash
     cd ~/codes/openpi
     uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi05_roboverse_lora \
       --policy.dir=<your_checkpoint_path>

  2. In a separate terminal, launch the RoboVerse evaluation client:
     ```bash
     cd ~/codes/RoboVerse
     python roboverse_learn/vla/pi0/pi_eval.py \
       --task <task_name> --robot franka --sim mujoco \
       --policy-host localhost --policy-port 8000
     ```
   

        You can shrink the command frequency by supplying --actions-per-call N (e.g., --actions-per-call 5
  executes five cached commands before querying the server again), and other options such as --max_steps,
  --num_episodes, or --output-dir.
  3. After each run, a metrics JSON and an episode video will appear in pi_eval_output/ (for example
  pi_eval_output/episode_001.mp4). Review the MP4 to check the rollout qualitatively.

## 6. Troubleshooting

### Feature Type 'List' Not Found Error

If you encounter the following error:
```
ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf', 'VideoFrame']
```

**Solution:** Add the following monkey-patch to `openpi/training/dataloader.py`:
```python
# Monkey-patch to fix 'List' feature type error in old datasets
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass
# End of monkey-patch
```

See: https://github.com/Physical-Intelligence/openpi/issues/561

### UV Environment Issues

If you encounter errors or version incompatibilities with the `uv` environment, you can use a `conda` environment instead and run commands with `python` directly rather than `uv run`.

### Environment Setup Recommendations

**Separate Environments:** Use Python 3.11 to create a dedicated π environment. We recommend maintaining two separate environments:

- **RoboVerse environment**: For collecting demonstrations and running evaluations
- **π environment**: For training and inference with π models

**During evaluation:**
1. Install `openpi_client` in your RoboVerse environment: `pip install openpi_client`
2. Use the π environment to serve the policy and generate action chunks
3. Use the RoboVerse environment to execute actions and evaluate performance