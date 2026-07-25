# Configuration

`configs/config.yaml` sets defaults. Override anything from the CLI — no code changes needed:

```bash
python src/train.py training.lr=0.01 training.batch_size=64
python src/train.py model=yolo_v8 dataset=cifar10
python src/train.py +experiment=long_run   # apply a preset
```

Hydra saves the full resolved config alongside every run for reproducibility.
