# Development Tips

This section contains helpful tips and best practices for working with RoboVerse and its dependencies.

---

## Topics

| Topic | Description |
|-------|-------------|
| [Hugging Face](huggingface) | Managing assets and models via Hugging Face Hub |
| [Git Submodules](git_submodule) | Working with external dependencies as submodules |
| [Git LFS](git_lfs) | Managing large files with Git Large File Storage |

---

## Quick Tips

### Environment Management

```bash
# Create a dedicated conda environment
conda create -n roboverse python=3.10
conda activate roboverse

# Install in development mode
pip install -e ".[dev]"
```

### Debugging Simulations

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run in non-headless mode to visualize
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="mujoco",
    headless=False,  # Show GUI for debugging
)
```

### Performance Profiling

```python
import torch

# Enable CUDA profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    for _ in range(100):
        handler.simulate()
        
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

```{toctree}
:titlesonly:
:hidden:

huggingface
git_submodule
git_lfs
```
