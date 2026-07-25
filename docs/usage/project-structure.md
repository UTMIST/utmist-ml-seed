# Project Structure

```
configs/
├── config.yaml          # Base config — training defaults + model/dataset selection
├── model/
│   ├── resnet50.yaml    # Classification (working)
│   └── yolo_v8.yaml     # Detection (working)
├── dataset/
│   └── cifar10.yaml
└── experiment/
    └── long_run.yaml    # Example override preset

src/
├── train.py             # Entry point (Hydra)
├── evaluate.py
├── models.py            # Model registry: build_model(cfg)
├── data.py              # Dataset registry: build_dataset(cfg)
└── utils.py             # Seeds, device, checkpoints

tests/
└── test_smoke.py        # Config loading, forward pass, seed tests

data/                    # Auto-downloaded datasets (gitignored)
outputs/                 # Per-run: config.yaml, metrics.json, best_model.pt (gitignored)
checkpoints/             # Named/shared weights — see checkpoints/README.md
results/                 # Curated plots and reports (gitignored)
```
