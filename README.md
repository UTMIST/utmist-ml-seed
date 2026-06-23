# UTMIST ML Seed

A Hydra-driven experiment launcher for ML projects at the University of Toronto Machine Intelligence Student Team (UTMIST). Supports classification (ResNet) and detection (YOLO) out of the box, with a clean config system for adding new models and datasets.

## Prerequisites

### 1. Install Miniconda

Download and install Miniconda (lightweight version of Anaconda):

- **macOS (Apple Silicon):**
  ```bash
  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
  bash Miniconda3-latest-MacOSX-arm64.sh
  ```

- **macOS (Intel):**
  ```bash
  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  bash Miniconda3-latest-MacOSX-x86_64.sh
  ```

- **Linux:**
  ```bash
  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```

- **Windows:**
  Download the installer from https://docs.conda.io/en/latest/miniconda.html, run it, then open "Anaconda Prompt" from the Start menu.

After installing, restart your terminal and verify:
```bash
conda --version
```

### 2. Install Git LFS

This repo uses Git LFS for large files (model weights, datasets, images).

- **macOS:** `brew install git-lfs`
- **Ubuntu/Debian:** `sudo apt install git-lfs`
- **Windows:** Download from https://git-lfs.github.com

Then run once:
```bash
git lfs install
```

### 3. Clone the repo

```bash
git clone https://github.com/UTMIST/utmist-ml-seed.git
cd utmist-ml-seed
```

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate utmist-ml
```

### Option B: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running an experiment

```bash
# Default: ResNet50 on CIFAR-10, 10 epochs
python src/train.py

# Override from CLI
python src/train.py training.epochs=20 training.lr=0.0005

# Switch model
python src/train.py model=yolo_v8

# Use an experiment preset
python src/train.py +experiment=long_run
```

Outputs (config snapshot, metrics, checkpoints) are saved automatically to `outputs/<date>/<time>/` by Hydra.

## Evaluate a checkpoint

```bash
python src/evaluate.py checkpoint=outputs/2026-06-21/12-30-00/best_model.pt
```

## Project structure

```
utmist-ml-seed/
├── configs/
│   ├── config.yaml            # Base config (Hydra entry)
│   ├── model/                 # Model configs
│   │   ├── resnet50.yaml
│   │   ├── yolo_v8.yaml
│   │   └── transformer.yaml   # Template
│   ├── dataset/               # Dataset configs
│   │   ├── cifar10.yaml
│   │   └── custom.yaml        # Template
│   └── experiment/            # Override presets
│       └── long_run.yaml
├── src/
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation entry point
│   ├── models.py              # Model registry
│   ├── data.py                # Dataset + dataloaders
│   └── utils.py               # Seeds, checkpoints, device
├── notebooks/
│   └── exploration.ipynb      # Interactive walkthrough
├── colab/
│   └── setup.ipynb            # Google Colab quickstart
├── docker/
│   ├── Dockerfile             # Optional GPU/server container
│   └── docker-compose.yml
├── data/                      # Datasets (gitignored, auto-downloaded)
├── outputs/                   # Hydra run outputs (gitignored)
├── checkpoints/               # Saved model weights (gitignored)
├── results/
│   ├── figures/               # Plots and visualizations
│   ├── metrics/               # Exported metric summaries
│   └── reports/               # Write-ups and analysis
├── logs/                      # TensorBoard / W&B logs (gitignored)
├── tests/
│   └── test_smoke.py
├── CLAUDE.md                  # AI coding assistant context
├── pyproject.toml
├── requirements.txt
├── environment.yml
└── .github/workflows/ci.yml
```

## Adding a new model

1. Create `configs/model/my_model.yaml`:
   ```yaml
   name: my_model
   num_classes: 10
   # your params here
   ```

2. Add a builder function in `src/models.py`:
   ```python
   def _build_my_model(cfg):
       # your model construction
       return model
   ```

3. Register it in `build_model()`:
   ```python
   if cfg.name == "my_model":
       return _build_my_model(cfg)
   ```

4. Run it:
   ```bash
   python src/train.py model=my_model
   ```

## Adding a new dataset

1. Create `configs/dataset/my_dataset.yaml`:
   ```yaml
   name: my_dataset
   path: ./data/my_dataset
   num_classes: 5
   augment: true
   ```

2. Add a builder function in `src/data.py`:
   ```python
   def _build_my_dataset(cfg):
       # return (train_dataset, val_dataset)
   ```

3. Register it in `build_dataset()`:
   ```python
   if cfg.name == "my_dataset":
       return _build_my_dataset(cfg)
   ```

4. Run it:
   ```bash
   python src/train.py dataset=my_dataset
   ```

## How the config system works

This repo uses [Hydra](https://hydra.cc/) for configuration management. The key ideas:

- **`configs/config.yaml`** is the base config with defaults for model, dataset, and training
- **Model/dataset configs** are composed via Hydra's defaults list
- **Experiment configs** override any base values (use `+experiment=name` to apply)
- **CLI overrides** take highest priority: `python src/train.py training.lr=0.01`
- Every run saves a complete config snapshot to the output directory for reproducibility

## Docker (optional)

For shared GPU servers or reproducible environments:

```bash
# Build and run training
docker compose -f docker/docker-compose.yml run train

# Or with config overrides
docker compose -f docker/docker-compose.yml run train python src/train.py training.epochs=20 model=yolo_v8
```

Docker is not required for local development — use conda or venv instead.

## Tests

```bash
pytest tests/ -v
```

## License

MIT
