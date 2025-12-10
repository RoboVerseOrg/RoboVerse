# ACT: Action Chunking with Transformers

## 1. Install
```bash
cd roboverse_learn/il/policies/act/detr && pip install -e .

cd ../../../../../
```

## 2. Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## 3. Train and eval

Move back to RoboVerse/ and run the following training script.
Similar to diffusion policy, it first stores the expert data in zarr format, and then trains a policy. You can configure joint or end effector control.
```bash
./roboverse_learn/il/policies/act/act_run.sh
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
