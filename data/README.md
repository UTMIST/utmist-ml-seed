# data/

Raw and processed datasets. This directory is gitignored — do not commit data files.

## Convention

```
data/
├── cifar-10-batches-py/   # auto-downloaded by torchvision
├── custom/                # your custom dataset
│   ├── train/
│   ├── val/
│   └── test/
└── raw/                   # unprocessed source data
```

## Recommended data format: Parquet (via PyArrow)

For tabular or structured data, use **Parquet** instead of CSV. It's faster to read, smaller on disk, and preserves column types.

### Saving data as Parquet

```python
import pandas as pd

df = pd.DataFrame({"feature1": [1, 2, 3], "label": ["cat", "dog", "cat"]})

# Save — uses pyarrow backend automatically
df.to_parquet("data/my_dataset.parquet")
```

### Loading Parquet data

```python
import pandas as pd

# Single file
df = pd.read_parquet("data/my_dataset.parquet")

# Specific columns only (fast — Parquet is columnar)
df = pd.read_parquet("data/my_dataset.parquet", columns=["feature1", "label"])

# Multiple files / partitioned dataset
df = pd.read_parquet("data/my_dataset/")
```

### Converting CSV to Parquet

```python
df = pd.read_csv("data/raw/data.csv")
df.to_parquet("data/processed/data.parquet")
```

### Why Parquet over CSV

| | CSV | Parquet |
|---|---|---|
| Read speed | Slow (parsed line-by-line) | Fast (columnar, zero-copy) |
| File size | Large (text) | Small (compressed binary) |
| Column types | Lost (everything is strings) | Preserved (int, float, datetime) |
| Column selection | Must read entire file | Reads only requested columns |

### Other supported formats

- **Excel** (`.xlsx`): `pd.read_excel("file.xlsx")` — requires `openpyxl`
- **HDF5** (`.h5`): `pd.read_hdf("file.h5")` — requires `h5py`, common in research
- **CSV**: `pd.read_csv("file.csv")` — use for small files or one-off imports

## Where to store data

- **Small datasets** (CIFAR-10, MNIST): auto-downloaded by torchvision into this folder
- **Medium datasets**: download manually, place here, document the source in your experiment config
- **Large datasets** (ImageNet, COCO): store on a shared drive or cluster filesystem, symlink into this folder
- **Google Drive** (Colab): mount and symlink, or set `dataset.path` in your config to the Drive path
