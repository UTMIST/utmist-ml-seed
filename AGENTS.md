# UTMIST ML Seed

Hydra-driven ML experiment launcher. Supports classification (ResNet/CIFAR-10) and detection (YOLOv8).

## Key commands

```bash
# Train (default: ResNet50 on CIFAR-10)
python src/train.py

# Override from CLI
python src/train.py model=yolo_v8 training.epochs=20 training.lr=0.0005

# Apply experiment preset
python src/train.py +experiment=long_run

# Evaluate checkpoint
python src/evaluate.py checkpoint=path/to/best_model.pt

# TensorBoard (view training progress)
tensorboard --logdir outputs/

# Lint
ruff check src/ tests/

# Test
pytest tests/ -v
```

## Architecture

- Single entry point: `src/train.py` (Hydra)
- Config composition: `configs/config.yaml` → model + dataset + experiment overrides
- Model registry: `src/models.py` — `build_model(cfg)` dispatches on `cfg.name`
- Dataset registry: `src/data.py` — `build_dataset(cfg)` dispatches on `cfg.name`
- YOLO uses ultralytics' built-in `.train()`, not the standard training loop

## Adding a model

1. Add config: `configs/model/<name>.yaml` with `name:` field
2. Add builder: `_build_<name>(cfg)` in `src/models.py`
3. Register in `build_model()` dispatch

## Adding a dataset

1. Add config: `configs/dataset/<name>.yaml` with `name:` field
2. Add builder: `_build_<name>(cfg)` returning `(train_ds, val_ds)` in `src/data.py`
3. Register in `build_dataset()` dispatch

## Conventions

- All hyperparameters live in YAML configs, never hardcoded
- Functions follow `build_*()` / `_build_*()` naming
- Hydra manages output directories (`outputs/<date>/<time>/`)
- Seeds set via `src/utils.py:set_seed()` — covers python, numpy, torch
