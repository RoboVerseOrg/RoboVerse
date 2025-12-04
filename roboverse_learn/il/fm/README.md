# Flow Matching Policies (IL)

Flow Matching variants (UNet and DiT) live here and use the shared IL runners under `il/dp/`.

## Install

```bash
cd roboverse_learn/il/dp
pip install -r requirements.txt
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## Train and eval

Use the shared driver and point it at a Flow Matching model:

```bash
# Choose one: fm_dit_model (DiT backbone) or fm_unet_model (UNet backbone)
export algo_model="fm_dit_model"

./roboverse_learn/il/dp/dp_run.sh
```

Inside `dp_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.

## References

- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
- William Peebles and Jun-Yan Zhu, "DiT: Diffusion Models with Transformers." (2023).
