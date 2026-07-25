# Engineering Projects

Engineering projects are pipeline-driven. You have a specific model to build, a dataset to handle, and something to deliver — a working classifier, a detection system, an inference API. This guide covers how to use the seed for that workflow.

## What makes it engineering

- You have a deliverable: "a model that classifies X with >90% accuracy"
- You run a handful of experiments to tune, then lock the config
- Reliability and reproducibility of the pipeline matter more than exploring many approaches
- Output is a deployed model, an API, or an integration into a larger system

## Recommended setup

The defaults are already engineering-appropriate. TensorBoard is on, no experiment tracking overhead:

```bash
python src/train.py
```

Once you've settled on a config, lock it in an experiment file so anyone can reproduce the exact run:

```yaml
# configs/experiment/production.yaml
# @package _global_
training:
  epochs: 30
  lr: 0.001
  batch_size: 64
  seed: 42

logging:
  tensorboard: true
  mlflow: false
```

```bash
python src/train.py +experiment=production
```

## Typical workflow

1. **Explore** — run a few experiments with different LRs/architectures to find what works
2. **Lock** — commit your best experiment config to `configs/experiment/`
3. **Train final model** — run with the locked config, save the checkpoint
4. **Share weights** — push to HuggingFace Hub (see [checkpoints/README.md](../../checkpoints/README.md))
5. **Evaluate** — run `src/evaluate.py` and log the final metrics

## Adding your data pipeline

Replace CIFAR-10 with your dataset (see [adding a dataset](../usage/adding-a-dataset.md)). For engineering projects, data preprocessing is often the most important part — invest time here:

- Clean and validate inputs before training
- Use Parquet for tabular data (see [data formats](../data/formats.md))
- Add a data validation step before your `build_dataset()` function

## CI / automated testing

The seed ships with a basic GitHub Actions CI that lints and runs smoke tests. For engineering projects, extend it to include:

```yaml
# .github/workflows/ci.yml — add after existing jobs
  integration-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: pip install hydra-core omegaconf pytest numpy
      # Run a real 1-epoch training to catch pipeline regressions
      - run: python src/train.py training.epochs=1 training.device=cpu
```

## Inference

After training, evaluate the model on a held-out test set:

```bash
python src/evaluate.py checkpoint=outputs/<date>/<time>/best_model.pt
```

For serving predictions in a larger system, load the checkpoint directly:

```python
from src.models import build_model
from src.utils import load_checkpoint
from omegaconf import OmegaConf

cfg = OmegaConf.load("outputs/<date>/<time>/config.yaml")
model = build_model(cfg.model)
load_checkpoint("checkpoints/best_model.pt", model)
model.eval()
```
