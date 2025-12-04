# Diffusion Policy

## 1. Install

```bash
cd roboverse_learn/il/dp
pip install -r requirements.txt
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
