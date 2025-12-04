# RoboVerse Imitation Learning (IL) Policies

## Example Usage

Pick a policy folder and follow its README for setup and usage.

Example:

```bash
# From the repo root
cd roboverse_learn/il/dp   # or fm/, vita/ depending on the policy
pip install -r requirements.txt
cd ../../..

# Run policy training and evaluation (example: diffusion policy, DiT backbone)
bash roboverse_learn/il/il_run.sh --task_name_set close_box --algo_choose ddpm_dit
```

We keep each policy as self-contained as possible (code, dependencies, docs) and only share the minimum common abstractions.

## Troubleshooting

```bash
# Fix potential package version issues
bash roboverse_learn/il/il_setup.sh
```

## Supported Algorithms

| Name | Policy | Backbone | Model Config | Ref |
| --- | --- | --- | --- | --- |
| `ddpm_dit` | Diffusion Policy (DDPM) | DiT | `model_config/ddpm_dit_model.yaml` | [1], [5] |
| `fm_dit` | Flow Matching | DiT | `model_config/fm_dit_model.yaml` | [6], [5] |
| `vita` | VITA Policy | MLP | `model_config/vita_model.yaml` | [7] |
| `ddpm_unet` | Diffusion Policy (DDPM) | UNet | `model_config/ddpm_model.yaml` | [1], [4] |
| `ddim_unet` | Diffusion Policy (DDIM) | UNet | `model_config/ddim_model.yaml` | [2], [4] |
| `fm_unet` | Flow Matching | UNet | `model_config/fm_unet_model.yaml` | [6] |
| `score_unet` | Score-Based Model | UNet | `model_config/score_model.yaml` | [3], [4] |

### References

1. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models." (2020).  
2. Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." (2021).  
3. Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." (2021).  
4. Chi, Cheng, et al. "Diffusion Policy: Diffusion Models for Robotic Manipulation." (2023).  
5. Peebles, William, and Jun-Yan Zhu. "DiT: Diffusion Models with Transformers." (2023).  
6. Lipman, Yaron, et al. "Flow Matching for Generative Modeling." (2023).  
7. Gao, Dechen, et al. "VITA: Vision-to-Action Flow Matching Policy." (2025).
