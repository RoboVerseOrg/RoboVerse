## Design Philosophy

We aim to follow a **single-file, modular design philosophy** inspired by libraries such as Hugging Face's *transformerss* and *diffusers* (https://huggingface.co/blog/transformers-design-philosophy). In this approach, each specific algorithm implementation is encapsulated within its own folder and file, allowing it to evolve independently without affecting other parts of the codebase.

At the same time, to maintain consistency, enable unified management, and support unit testing, we define common base classes (e.g., `BaseRunner`, `BaseModel`). These base classes provide standardized interfaces for training, evaluation, and model behavior, ensuring that all algorithms conform to the same core structure while remaining self-contained and easily extensible.

> Refer to the implementation style in the *IL* branch of *diffusion_policy*. Very soon you'll see it in *main*.

### 1. BaseRunner (Core Execution Interface)

**Purpose**: Encapsulate the overall logic for training, evaluation, checkpoint saving/loading, etc.

**Usage**: All algorithm runners should subclass `BaseRunner` and implement their own `train()` and `evaluate()` methods.

**Key methods**:

- `train()`: Core training loop, must be implemented by each algorithm.
- `evaluate()`: Core evaluation logic, must be implemented by each algorithm.
- `save_checkpoint(path)`: Save model weights.
- `load_checkpoint(path)`: Load model weights.
- `run()`: High-level entry point (default is to run training and evaluation, can be overridden).

---

### 2. BaseModel (Core Model Interface)

**Purpose**: Define the common structure for models and loss computation. All algorithm-specific models should inherit from this class.

**Key methods**:

- `forward(obs)`: Forward inference to produce actions or distributions.
- `compute_loss(batch)`: Compute the loss given a data batch (algorithm-specific implementation).

---

### 3. Dataset Management

- Each algorithm can define its own dataset class as needed.
- All dataset classes should follow a consistent interface (compatible with PyTorch `Dataset`).
- Dataset-related configurations will be managed in `dataset_config.yaml`.

---

## Configuration Management (Hydra + YAML)

To keep parameters clean, flexible, and modular, we use Hydra and separate YAML files for different concerns. For merging your own policy into roboverse, please write your configuration in the following format. The `<policy>` symbol below is a placeholder and should be replaced with your desired name.

### configs/model_config/\<policy\>_model.yaml

Configuration for model architectures and hyperparameters:

```yaml
model_name: "diffusion_policy"
hidden_dim: 256
num_layers: 4
use_attention: true
dropout: 0.1
...
```
---
### configs/dataset_config/\<policy\>_dataset.yaml

Dataset-related configuration:

```yaml
dataset_name: "robosuite_pnp"
data_path: "/path/to/dataset"
batch_size: 64
num_workers: 8
shuffle: true
...
```

---

### configs/train_config/\<policy\>_train.yaml

Training hyperparameters:

```yaml
epochs: 100
lr: 0.0001
weight_decay: 0.00001
checkpoint_dir: "./checkpoints"
log_interval: 10
...
```

---

### `configs/eval_config/\<policy\>_eval.yaml`

Evaluation configurations:

```yaml
eval_episodes: 10
render: false
save_video: true
output_dir: "./eval_results"
...
```

---

### `configs/\<policy\>_runner.yaml`

Entrance to the configuration:

```yaml
defaults:
  - _self_
  - dataset_config: <policy>_dataset
  - model_config: <policy>_model
  - eval_config: <policy>_eval
  - train_config: <policy>_train

# Remaining params
```

---

### Example Main Script

```python
import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
sys.path.insert(0, project_root)
from roboverse_learn.base.base_runner import BaseRunner

abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("configs").absolute())
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(config_path=abs_config_path, version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    runner: BaseRunner = cls(cfg)
    runner.run()

if __name__ == "__main__":
    main()
```

---

### Unit Testing Strategy

Each runner and model should be independently unit-tested to verify correctness and ensure modularity.

- **Runner tests**: Validate `train()` and `evaluate()` logic, check loss values, and confirm model updates.
- **Model tests**: Check output shapes of `forward()` and correctness of `compute_loss()`.
- **Config tests**: Ensure YAML configs load correctly and pass parameters as intended.

---

### Project Structure

```csharp
roboverse_learn/
├── base
│   ├── base_runner.py        # BaseRunner definition
│   ├── base_model.py         # BaseModel definition
│   ├── base_dataset.py       # BaseDataset definition
│   ├── base_eval_runner.py   # BaseEvalRunner definition
├── Runner/
│   ├── dp_runner.py          # Diffusion Policy Runner
│   ├── fm_runner.py          # Flow matching Runner
│   ├── act_runner.py         # ACT Runner
│   └── ...
├── models/
│   ├── act_model.py
│   ├── fm_model.py
│   ├── diffusion_model.py
│   └── ...
├── datasets/
│   ├── robot_image_dataset.py
│   └── ...
├── configs/
│   ├── model_config/
│   │   ├── diffusion_policy_model.yaml
│   ├── dataset_config/
│   │   ├── robot_image_dataset.yaml
│   ├── train_config/
│   │   ├── diffusion_policy_train.yaml
│   └── eval_config/
│       ├── diffusion_policy_eval.yaml
├── utils/
│   ├── common/                 # Common utils
│   └── diffusion_policy/       # Policy specific utils
├── tests/
│   ├── test_base_runner.py
│   ├── test_base_model.py
│   └── ...
└── main.py                   # Project entry point
```