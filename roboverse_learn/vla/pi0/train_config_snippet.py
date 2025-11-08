# Training configuration snippet to add to openpi/src/openpi/training/config.py
# This file is used by train_pi0.sh to automatically register training configs

from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi.policies import pi0_config


# π₀.₅ with LoRA configuration
TrainConfig(
    name="pi05_roboverse_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRoboVerseDataConfig(
        repo_id="<your_hf_name>/<repo_name>",  # Replace with your HuggingFace repo ID
        base_config=_config.DataConfig(prompt_from_task=True),
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


# π₀ with LoRA configuration
TrainConfig(
    name="pi0_roboverse_lora",
    model=pi0_config.Pi0Config(
        pi05=False,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRoboVerseDataConfig(
        repo_id="<your_hf_name>/<repo_name>",  # Replace with your HuggingFace repo ID
        base_config=_config.DataConfig(prompt_from_task=True),
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
