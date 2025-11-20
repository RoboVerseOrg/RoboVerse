# Flow Matching and Diffusion Based IL Policies

## 1. Install

```bash
cd roboverse_learn/il/utils/diffusion_policy

pip install -e .

cd ../../../../

pip install pandas wandb
```

Register for a Weights & Biases (wandb) account to obtain an API key.

## 2. Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## 3. Train and eval

```bash
./roboverse_learn/il/dp/dp_run.sh
```

### 3.1 Train only

```bash
train_enable=True
eval_enable=False
```

### 3.2 Eval only

```bash
train_enable=False
eval_enable=True
```

## Supported Algorithms

| Algorithm | Backbone | Model Config | Ref |
| --- | --- | --- | --- |
| Diffusion Policy (DDPM) | DiT | `model_config/ddpm_dit_model.yaml` | [1], [5] |
| Flow Matching | DiT | `model_config/fm_dit_model.yaml` | [6], [5] |
| VITA Policy | MLP | `model_config/vita_model.yaml` | [7] |
| Diffusion Policy (DDPM) | UNet | `model_config/ddpm_model.yaml` | [1], [4] |
| Diffusion Policy (DDIM) | UNet | `model_config/ddim_model.yaml` | [2], [4] |
| Flow Matching | UNet | `model_config/fm_unet_model.yaml` | [6] |
| Score-Based Model | UNet | `model_config/score_model.yaml` | [3], [4] |

### References

1. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models." (2020).  
2. Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." (2021).  
3. Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." (2021).  
4. Chi, Cheng, et al. "Diffusion Policy: Diffusion Models for Robotic Manipulation." (2023).  
5. Peebles, William, and Jun-Yan Zhu. "DiT: Diffusion Models with Transformers." (2023).  
6. Lipman, Yaron, et al. "Flow Matching for Generative Modeling." (2023).  
7. Gao, Dechen, et al. "VITA: Vision-to-Action Flow Matching Policy." (2025).
