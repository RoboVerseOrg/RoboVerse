"""WandB logging helpers.

Run setup_wandb(hyperparam_dict, ...) to initialize wandb logging.
See default_wandb_config() for a list of available configurations.

We recommend the following workflow (see examples/mujoco/d4rl_iql.py for a more full example):

    from ml_collections import config_flags
    from jaxrl_m.wandb import setup_wandb, default_wandb_config
    import wandb

    # This line allows us to change wandb config flags from the command line
    config_flags.DEFINE_config_dict('wandb', default_wandb_config(), lock_config=False)

    ...
    def main(argv):
        hyperparams = ...
        setup_wandb(hyperparams, **FLAGS.wandb)

        # Log metrics as you wish now
        wandb.log({'metric': 0.0}, step=0)


With the following setup, you may set wandb configurations from the command line, e.g.
    python main.py --wandb.project=my_project --wandb.group=my_group --wandb.offline
"""

import datetime
import tempfile
import time

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from ml_collections.config_dict import FieldReference


def get_flag_dict():
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def default_wandb_config():
    config = ml_collections.ConfigDict()
    config.offline = False  # Syncs online or not?
    config.project = "jaxrl_m"  # WandB Project Name
    config.entity = FieldReference(None, field_type=str)  # Which entity to log as (default: your own user)

    group_name = FieldReference(None, field_type=str)  # Group name
    config.exp_prefix = group_name  # Group name (deprecated, but kept for backwards compatibility)
    config.group = group_name  # Group name

    experiment_name = FieldReference(None, field_type=str)  # Experiment name
    config.name = experiment_name  # Run name (will be formatted with flags / variant)
    config.exp_descriptor = experiment_name  # Run name (deprecated, but kept for backwards compatibility)

    config.unique_identifier = ""  # Unique identifier for run (will be automatically generated unless provided)
    config.random_delay = 0  # Random delay for wandb.init (in seconds)
    return config


def setup_wandb(
    hyperparam_dict,
    entity=None,
    project="jaxrl_m",
    group=None,
    name=None,
    unique_identifier="",
    offline=False,
    random_delay=0,
    **additional_init_kwargs,
):
    """Utility for setting up wandb logging (based on Young's simplesac):

    Arguments:
        - hyperparam_dict: dict of hyperparameters for experiment
        - offline: bool, whether to sync online or not
        - project: str, wandb project name
        - entity: str, wandb entity name (default is your user)
        - group: str, Group name for wandb
        - name: str, Experiment name for wandb (formatted with FLAGS & hyperparameter_dict)
        - unique_identifier: str, Unique identifier for wandb (default is timestamp)
        - random_delay: float, Random delay for wandb.init (in seconds) to avoid collisions
        - additional_init_kwargs: dict, additional kwargs to pass to wandb.init
    Returns:
        - wandb.run

    """
    if "exp_descriptor" in additional_init_kwargs:
        # Remove deprecated exp_descriptor
        additional_init_kwargs.pop("exp_descriptor")
        additional_init_kwargs.pop("exp_prefix")

    if not unique_identifier:
        if random_delay:
            time.sleep(np.random.uniform(0, random_delay))
        unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if name is not None:
        name = name.format(**{**get_flag_dict(), **hyperparam_dict})

    if group is not None and name is not None:
        experiment_id = f"{group}_{name}_{unique_identifier}"
    elif name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    else:
        experiment_id = None

    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=hyperparam_dict,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        id=experiment_id,
        name=name,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
        ),
        mode="offline" if offline else "online",
        save_code=True,
    )

    init_kwargs.update(additional_init_kwargs)
    run = wandb.init(**init_kwargs)

    wandb.config.update(get_flag_dict())

    wandb_config = dict(
        exp_prefix=group,
        exp_descriptor=name,
        experiment_id=experiment_id,
    )
    wandb.config.update(wandb_config)
    return run
