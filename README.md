# UTMIST ML Seed

A Hydra-driven ML experiment launcher for UTMIST projects. Ships with working ResNet (classification) and YOLO (detection) examples. Copy the pattern to add your own models and datasets.

## Prerequisites

- **Miniconda** — [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Git LFS** — `brew install git-lfs` / `sudo apt install git-lfs`, then `git lfs install`

## Setup

```bash
git clone https://github.com/UTMIST/utmist-ml-seed.git
cd utmist-ml-seed

# Conda (recommended)
conda env create -f environment.yml
conda activate utmist-ml

# or venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train (default: ResNet50 on CIFAR-10)
python src/train.py

# Switch model or override any param from CLI
python src/train.py model=yolo_v8
python src/train.py training.epochs=20 training.lr=0.0005

# Apply an experiment preset
python src/train.py +experiment=long_run

# Evaluate a checkpoint
python src/evaluate.py checkpoint=outputs/<date>/<time>/best_model.pt

# Monitor with TensorBoard (in a second terminal)
tensorboard --logdir outputs/

# Lint / test
ruff check src/ tests/
pytest tests/ -v
```

Every run saves a full config snapshot, metrics, and best checkpoint to `outputs/<date>/<time>/` automatically.

## Project structure

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

## Adding a model

1. `configs/model/my_model.yaml` — add a YAML with a `name:` field and any hyperparams
2. `src/models.py` — add `_build_my_model(cfg)` and register it in `build_model()`
3. Run: `python src/train.py model=my_model`

## Adding a dataset

1. `configs/dataset/my_dataset.yaml` — add a YAML with a `name:` field and `path:`
2. `src/data.py` — add `_build_my_dataset(cfg)` returning `(train_ds, val_ds)` and register it in `build_dataset()`
3. Run: `python src/train.py dataset=my_dataset`

## Installing new packages

```bash
# Conda — install then export so teammates get it
pip install <package>
conda env export --from-history > environment.yml

# Sync from someone else's updated environment.yml
conda env update -f environment.yml --prune
```

Add permanent packages to `pyproject.toml` under the appropriate group (`ml`, `cv`, `dev`) and install with `pip install -e ".[ml]"`.

## Config system

`configs/config.yaml` sets defaults. Override anything from the CLI — no code changes needed:

```bash
python src/train.py training.lr=0.01 training.batch_size=64
python src/train.py model=yolo_v8 dataset=cifar10
python src/train.py +experiment=long_run   # apply a preset
```

Hydra saves the full resolved config alongside every run for reproducibility.

## License

MIT
