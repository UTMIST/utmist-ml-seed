# notebooks/

Interactive Jupyter notebooks for exploration and prototyping.

## Files

- `exploration.ipynb` — Walkthrough of the full pipeline: load config, build model, visualize data, train for 2 epochs, plot loss curve.

## Usage

```bash
jupyter notebook notebooks/exploration.ipynb
```

Uses Hydra's compose API (not CLI) so configs work inside notebooks.
