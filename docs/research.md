# Research Projects

Research projects are hypothesis-driven. You're running many experiments, comparing approaches, and need to reproduce results weeks later. This guide covers how to use the seed for that workflow.

## What makes it research

- You have a question: "does augmentation help on this dataset?" or "which backbone performs better?"
- You run 10–100+ experiments with different configs
- Results need to be reproducible and comparable across runs
- Output is a report, paper, or model published to HuggingFace Hub

## Recommended setup

Enable MLflow for cross-run tracking:

```bash
python src/train.py logging.mlflow=true
```

Or set it permanently in your project's config override:

```yaml
# configs/experiment/my_research.yaml
# @package _global_
logging:
  mlflow: true
  tensorboard: true

training:
  seed: 42
  epochs: 50
```

Then run:

```bash
python src/train.py +experiment=my_research
```

## Running multiple experiments

The simplest approach is to vary one param at a time from the CLI and let Hydra timestamp each run:

```bash
# Vary learning rate
python src/train.py training.lr=0.01 logging.mlflow=true
python src/train.py training.lr=0.001 logging.mlflow=true
python src/train.py training.lr=0.0001 logging.mlflow=true
```

All runs appear in the MLflow UI for comparison. See [experiment-tracking.md](experiment-tracking.md) for how to launch it.

For larger sweeps, use **Hydra multirun**:

```bash
python src/train.py --multirun training.lr=0.01,0.001,0.0001 training.epochs=20,50
```

This runs all combinations (6 runs here) sequentially and logs each to MLflow.

## Reproducibility checklist

Every run already saves automatically:

- `outputs/<date>/<time>/config.yaml` — exact config used
- `outputs/<date>/<time>/git_hash.txt` — code state
- `outputs/<date>/<time>/metrics.json` — per-epoch metrics
- `outputs/<date>/<time>/best_model.pt` — best checkpoint

For a result to be reproducible, also document:

- [ ] Dataset version / download source (add to your dataset config as a comment)
- [ ] Environment: `conda env export > environment_locked.yml`
- [ ] Random seed explicitly set (`training.seed` in config)
- [ ] Model weights shared to HuggingFace Hub (see [../checkpoints/README.md](../checkpoints/README.md))

## Running multiple seeds

To report robust results, run 3–5 seeds and report mean ± std:

```bash
python src/train.py training.seed=0 logging.mlflow=true
python src/train.py training.seed=1 logging.mlflow=true
python src/train.py training.seed=2 logging.mlflow=true
```

Or with multirun:

```bash
python src/train.py --multirun training.seed=0,1,2,3,4
```

## Sharing results

- **Model weights** → HuggingFace Hub (see [../checkpoints/README.md](../checkpoints/README.md))
- **Plots and figures** → `results/` directory (gitignored by default; `git add -f` to commit specific ones)
- **Experiment logs** → MLflow tracking server or exported CSV from the MLflow UI
