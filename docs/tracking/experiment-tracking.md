# Experiment Tracking

The seed supports three tracking tools. All are optional — enable what fits your project.

## Overview

| Tool | Best for | Run server? | Cost |
|---|---|---|---|
| TensorBoard | Watching one run live | No (reads files) | Free, local |
| MLflow | Comparing many runs, model registry | Yes (local) | Free, self-hosted |
| W&B | Team collaboration, cloud dashboard | No (cloud) | Free tier |

## TensorBoard (default: on)

Logs train/val loss, accuracy, and learning rate per epoch. Good for monitoring a single run in real time.

```bash
# Start training in one terminal
python src/train.py

# View in a second terminal
tensorboard --logdir outputs/

# Open http://localhost:6006
```

TensorBoard reads event files from all runs under `outputs/` — you can compare multiple runs on the same chart.

To disable:

```bash
python src/train.py logging.tensorboard=false
```

## MLflow (default: off)

Tracks params, metrics, and artifacts across all your runs in a searchable UI. Useful when you're running 10+ experiments and need to compare them.

### Setup

MLflow is already installed. Start the tracking server once before training:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Enable per run

```bash
python src/train.py logging.mlflow=true
```

Or set permanently in an experiment config:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
logging:
  mlflow: true
```

### What gets logged

- All config params (model name, LR, batch size, epochs, seed)
- Per-epoch train/val loss and accuracy
- Best checkpoint as an artifact
- Git hash

### Querying runs

In the MLflow UI you can filter runs by any logged param or metric. From Python:

```python
import mlflow

runs = mlflow.search_runs(
    filter_string="params.model_name = 'resnet50' and metrics.val_acc > 0.9"
)
print(runs[["params.training.lr", "metrics.val_acc"]])
```

## W&B (default: off)

Cloud-hosted experiment tracking with team sharing, rich dashboards, and sweep support. Good for collaborative research projects.

### Setup

```bash
pip install wandb
wandb login   # one-time, creates ~/.netrc
```

### Enable per run

```bash
python src/train.py logging.wandb=true
```

### What gets logged

Same as MLflow — params, metrics, checkpoints — but viewable in the cloud at wandb.ai and shareable with your team via a link.

## Choosing between them

- **Just watching training** → TensorBoard only
- **Solo research, many local runs** → TensorBoard + MLflow
- **Team research, want to share results** → TensorBoard + W&B
- **Engineering project** → TensorBoard only (keep it simple)
