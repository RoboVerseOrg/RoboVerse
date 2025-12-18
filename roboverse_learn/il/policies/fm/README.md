# Flow Matching Policies (IL)

Flow Matching variants (UNet and DiT) live here and use the shared IL runners under `il/policies/fm/`.

## Install

```bash
cd roboverse_learn/il/policies/fm
pip install -r requirements.txt
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## Train and eval

```bash
bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name fm_unet # or fm_dit
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.

## References

- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
- William Peebles and Jun-Yan Zhu, "DiT: Diffusion Models with Transformers." (2023).
