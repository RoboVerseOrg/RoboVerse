# VITA Policy (IL)

VITA is a vision-to-action Flow Matching policy built on the shared IL runners under `il/dp/`.

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

Use the shared driver and select the VITA model:

```bash
export algo_model="vita_model"
./roboverse_learn/il/dp/dp_run.sh
```

Inside `dp_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.

## References

- Dechen Gao et al., "VITA: Vision-to-Action Flow Matching Policy." (2025).
- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
