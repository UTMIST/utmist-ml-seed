# Data Guide

## Directory layout

```
data/
├── cifar-10-batches-py/   # auto-downloaded by torchvision
├── custom/                # your custom dataset
│   ├── train/
│   ├── val/
│   └── test/
└── raw/                   # unprocessed source data
```

`data/` is gitignored. Never commit data files.

## Where to get data

- **Small standard datasets** (CIFAR-10, MNIST, ImageNet subsets) — auto-downloaded by torchvision on first run
- **Custom datasets** — download manually, place in `data/`, document the source in your dataset config as a comment
- **Large datasets** (ImageNet, COCO) — store on a shared drive or cluster filesystem, set `dataset.path` in your config to that location

## Recommended format: Parquet

For tabular or structured data, use Parquet instead of CSV. It's faster, smaller, and preserves column types.

### Save

```python
import pandas as pd

df = pd.DataFrame({"feature1": [1, 2, 3], "label": ["cat", "dog", "cat"]})
df.to_parquet("data/my_dataset.parquet")
```

### Load

```python
import pandas as pd

# Single file
df = pd.read_parquet("data/my_dataset.parquet")

# Specific columns only (fast — Parquet is columnar)
df = pd.read_parquet("data/my_dataset.parquet", columns=["feature1", "label"])

# Partitioned dataset (multiple files in a folder)
df = pd.read_parquet("data/my_dataset/")
```

### Convert from CSV

```python
df = pd.read_csv("data/raw/data.csv")
df.to_parquet("data/processed/data.parquet")
```

### Why Parquet over CSV

| | CSV | Parquet |
|---|---|---|
| Read speed | Slow (text parsing) | Fast (columnar, binary) |
| File size | Large | Small (compressed) |
| Column types | Lost (all strings) | Preserved |
| Column selection | Reads entire file | Reads only what you ask for |

## Other formats

| Format | Read | When to use |
|---|---|---|
| Excel `.xlsx` | `pd.read_excel("file.xlsx")` | Data from non-technical collaborators |
| HDF5 `.h5` | `pd.read_hdf("file.h5")` | Large arrays, research datasets |
| NumPy `.npy` / `.npz` | `np.load("file.npy")` | Pre-processed tensor data |
| CSV | `pd.read_csv("file.csv")` | Small files, one-off imports only |

## Adding a custom dataset to the training pipeline

1. Create `configs/dataset/my_dataset.yaml`:

```yaml
name: my_dataset
path: ./data/my_dataset
num_classes: 5
augment: true
```

2. Add a builder in `src/data.py`:

```python
def _build_my_dataset(cfg):
    # Load your data, return (train_dataset, val_dataset)
    df = pd.read_parquet(cfg.path)
    # ... wrap in a torch Dataset
    return train_ds, val_ds
```

3. Register it in `build_dataset()`:

```python
if cfg.name == "my_dataset":
    return _build_my_dataset(cfg)
```

4. Train:

```bash
python src/train.py dataset=my_dataset
```
