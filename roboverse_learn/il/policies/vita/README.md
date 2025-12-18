# VITA Policy (IL)

VITA is a vision-to-action Flow Matching policy built on the shared IL runners under `il/policies/vita/`.

## Install

```bash
cd roboverse_learn/il/policies/vita
pip install -r requirements.txt
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## Train and eval

```bash
bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name vita
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.
## References

- Dechen Gao et al., "VITA: Vision-to-Action Flow Matching Policy." (2025).
- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
